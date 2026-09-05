"""
Training.

    fit()       epoch-based, for the parallel corpus
    pretrain()  step-based, mixed precision, resumable - for the large corpus

Teacher forcing throughout: the (tgt_in, tgt_out) offset is built into the
batch, so nothing is sliced here.
"""

import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from .config import TransformerConfig
from .evaluate import bleu
from .inference import translate_corpus


def make_criterion(pad_idx: int, label_smoothing: float = 0.1):
    """Padding excluded from the loss. Smoothing keeps logits off extremes."""
    return nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)


def make_optimizer(model, lr: float = 1e-3, weight_decay: float = 0.01):
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay
    )


def make_scheduler(optimizer, warmup: int = 400):
    """
    Linear warmup then inverse-sqrt decay, peaking at 1.0 when warmup ends.

    Warmup matters because Adam's variance estimate is unreliable early.
    """
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup:
            return step / warmup
        return (warmup / step) ** 0.5

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def token_accuracy(logits, tgt_out, pad_idx: int) -> float:
    """Fraction of non-padding positions predicted correctly."""
    mask = tgt_out != pad_idx
    correct = (logits.argmax(-1) == tgt_out) & mask
    return (correct.sum() / mask.sum()).item()


def train_step(model, batch, criterion, optimizer, scheduler=None, clip: float = 1.0):
    """One forward/backward/update. Returns (loss, accuracy)."""
    model.train()
    src, tgt_in, tgt_out = batch

    logits = model(src, tgt_in)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Clipping keeps a single bad batch from taking a huge step.
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item(), token_accuracy(logits, tgt_out, criterion.ignore_index)


def train(
    model,
    get_batch,
    steps: int = 2000,
    lr: float = 1e-3,
    warmup: int = 400,
    label_smoothing: float = 0.1,
    clip: float = 1.0,
    log_every: int = 200,
    verbose: bool = True,
):
    """
    Args:
        get_batch: callable returning a fresh (src, tgt_in, tgt_out) triple

    Returns:
        history dict with "step", "loss", "acc", "lr" lists
    """
    criterion = make_criterion(model.pad_idx, label_smoothing)
    optimizer = make_optimizer(model, lr)
    scheduler = make_scheduler(optimizer, warmup)

    history = {"step": [], "loss": [], "acc": [], "lr": []}

    for step in range(1, steps + 1):
        loss, acc = train_step(model, get_batch(), criterion, optimizer, scheduler, clip)

        history["step"].append(step)
        history["loss"].append(loss)
        history["acc"].append(acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if verbose and (step % log_every == 0 or step == 1):
            print(f"step {step:>5}   loss {loss:6.4f}   acc {acc:5.1%}")

    return history


def overfit_batch(model, batch, steps: int = 300, lr: float = 1e-3,
                  log_every: int = 50, verbose: bool = True):
    """
    Train on ONE batch repeatedly; loss must collapse to ~0.

    If a model cannot memorise one batch the problem is a bug, not the data.
    Smoothing off - it would put a floor under the loss.
    """
    return train(
        model,
        get_batch=lambda: batch,
        steps=steps,
        lr=lr,
        warmup=1,
        label_smoothing=0.0,
        log_every=log_every,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Training on a real corpus: epochs, validation, checkpointing, early stopping
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    """Average loss and token accuracy over a loader, weighted by real tokens."""
    model.eval()
    total_loss = total_correct = total_tokens = 0

    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)

        mask = tgt_out != criterion.ignore_index
        n = mask.sum().item()

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total_loss += loss.item() * n
        total_correct += (((logits.argmax(-1) == tgt_out) & mask).sum().item())
        total_tokens += n

    return total_loss / total_tokens, total_correct / total_tokens


def save_checkpoint(path, model, config, epoch, score):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": asdict(config),
         "epoch": epoch, "score": score},
        path,
    )


def load_checkpoint(path, model=None, device=None):
    """Returns (model, checkpoint). Builds the model from the saved config if needed."""
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    if model is None:
        from .transformer import build_model
        model = build_model(TransformerConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    if device is not None:
        model = model.to(device)
    return model, checkpoint


def fit(
    model,
    config,
    train_loader,
    valid_loader,
    valid_source,
    valid_target,
    vocab,
    device,
    epochs: int = 30,
    lr: float = 1e-3,
    warmup: int = 400,
    label_smoothing: float = 0.1,
    clip: float = 1.0,
    patience: int = 5,
    checkpoint_path: str = "checkpoints/best.pt",
    baseline: float = None,
):
    """
    Train up to `epochs`, keeping the best-validation-BLEU checkpoint.

    Selection is on BLEU, not loss - they diverge, and BLEU is what the task
    is judged on.

    Args:
        valid_source/valid_target: raw strings, for scoring translations
        baseline: copy-baseline BLEU, printed alongside
    """
    criterion = make_criterion(model.pad_idx, label_smoothing)
    optimizer = make_optimizer(model, lr)
    scheduler = make_scheduler(optimizer, warmup)

    history = {"epoch": [], "train_loss": [], "valid_loss": [],
               "train_acc": [], "valid_acc": [], "bleu": []}
    best_bleu, best_epoch, stale = -1.0, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = running_acc = seen = 0

        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            loss, acc = train_step(model, batch, criterion, optimizer, scheduler, clip)
            running_loss += loss
            running_acc += acc
            seen += 1

        train_loss, train_acc = running_loss / seen, running_acc / seen
        valid_loss, valid_acc = evaluate_loss(model, valid_loader, criterion, device)

        hypotheses = translate_corpus(model, valid_source, vocab, device)
        score = bleu(hypotheses, valid_target)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        history["bleu"].append(score)

        marker = ""
        if score > best_bleu:
            best_bleu, best_epoch, stale = score, epoch, 0
            save_checkpoint(checkpoint_path, model, config, epoch, score)
            marker = "  <- best"
        else:
            stale += 1

        note = "" if baseline is None else f" (baseline {baseline:.2f})"
        print(f"epoch {epoch:>3}  train {train_loss:.3f}/{train_acc:5.1%}  "
              f"valid {valid_loss:.3f}/{valid_acc:5.1%}  BLEU {score:6.2f}{note}{marker}")

        if stale >= patience:
            print(f"\nno improvement for {patience} epochs - stopping")
            break

    print(f"\nbest BLEU {best_bleu:.2f} at epoch {best_epoch}  ->  {checkpoint_path}")
    return history


# ---------------------------------------------------------------------------
# Pretraining: step-based, mixed precision, resumable
#
# fit() above is epoch-based, which is right for 18k sentence pairs and wrong
# for 200M words - an "epoch" there is hours long and tells you nothing. These
# functions train for a fixed number of STEPS and survive a cloud session dying
# partway through.
# ---------------------------------------------------------------------------

def save_training_state(path, model, config, step, optimizer=None, scheduler=None,
                        scaler=None, metrics=None):
    """
    Full state, not just weights.

    Resuming from weights alone resets Adam's momentum and restarts the LR
    schedule at warmup - training then takes badly scaled steps into trained
    weights. Nothing crashes; it just gets worse.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "config": asdict(config),
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    # atomic write: a session dying mid-save cannot corrupt the checkpoint
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_training_state(path, model=None, optimizer=None, scheduler=None,
                        scaler=None, device=None):
    """Restore everything saved by save_training_state. Returns (model, state)."""
    state = torch.load(path, map_location=device or "cpu", weights_only=False)

    if model is None:
        from .transformer import build_model
        model = build_model(TransformerConfig(**state["config"]))
    model.load_state_dict(state["model"])
    if device is not None:
        model = model.to(device)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    return model, state


def load_pretrained(model, path, device=None):
    """Weights only - the pretraining optimizer state is not wanted here."""
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device) if device is not None else model


def pretrain(
    model,
    config,
    loader,
    device,
    steps: int = 100_000,
    lr: float = 3e-4,
    warmup: int = 4000,
    label_smoothing: float = 0.0,
    clip: float = 1.0,
    accumulate: int = 1,
    amp: bool = True,
    log_every: int = 100,
    checkpoint_path: str = "checkpoints/pretrain.pt",
    checkpoint_every: int = 2000,
    resume: bool = True,
):
    """
    Train for `steps` optimiser updates over an endlessly cycled loader.

    Args:
        accumulate: micro-batches per update; effective batch = loader batch x this
        amp: mixed precision, ~2x faster on a T4
        resume: pick up from checkpoint_path if it exists

    Returns:
        history dict; tokens_per_sec counts SOURCE tokens, not target
    """
    criterion = make_criterion(model.pad_idx, label_smoothing)
    optimizer = make_optimizer(model, lr)
    scheduler = make_scheduler(optimizer, warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    start_step = 0
    if resume and Path(checkpoint_path).exists():
        _, state = load_training_state(checkpoint_path, model, optimizer,
                                       scheduler, scaler, device)
        start_step = state["step"]
        print(f"resumed from {checkpoint_path} at step {start_step:,}")

    history = {"step": [], "loss": [], "perplexity": [], "lr": [], "tokens_per_sec": []}
    model.train()

    def batches():
        while True:
            for batch in loader:
                yield batch

    stream = batches()
    running_loss = running_tokens = 0
    window_start = time.time()

    for step in range(start_step + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        for _ in range(accumulate):
            src, tgt_in, tgt_out = (t.to(device, non_blocking=True) for t in next(stream))

            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                loss = loss / accumulate

            scaler.scale(loss).backward()

            running_loss += loss.item() * accumulate
            # source tokens = corpus consumed; targets are only ~15% of that
            running_tokens += int((src != model.pad_idx).sum())

        # unscale before clipping, or the threshold applies to scaled gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % log_every == 0:
            elapsed = time.time() - window_start
            mean_loss = running_loss / (log_every * accumulate)
            tokens_per_sec = running_tokens / max(elapsed, 1e-9)
            ppl = math.exp(min(mean_loss, 20))

            history["step"].append(step)
            history["loss"].append(mean_loss)
            history["perplexity"].append(ppl)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["tokens_per_sec"].append(tokens_per_sec)

            eta = (steps - step) / log_every * elapsed
            print(f"step {step:>7,}/{steps:,}  loss {mean_loss:6.3f}  ppl {ppl:8.1f}  "
                  f"{tokens_per_sec/1000:5.1f}k corpus tok/s  eta {eta/3600:4.1f}h")

            running_loss = running_tokens = 0
            window_start = time.time()

        if checkpoint_every and step % checkpoint_every == 0:
            save_training_state(
                checkpoint_path, model, config, step, optimizer, scheduler, scaler,
                metrics={"loss": history["loss"][-1] if history["loss"] else None},
            )

    save_training_state(checkpoint_path, model, config, steps, optimizer, scheduler, scaler)
    print(f"\ndone -> {checkpoint_path}")
    return history


@torch.no_grad()
def perplexity(model, loader, device, max_batches: int = 50) -> float:
    """Validation perplexity, weighted by real tokens."""
    criterion = make_criterion(model.pad_idx, label_smoothing=0.0)
    model.eval()

    total_loss = total_tokens = 0
    for i, (src, tgt_in, tgt_out) in enumerate(loader):
        if i >= max_batches:
            break
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        n = int((tgt_out != model.pad_idx).sum())
        total_loss += criterion(logits.reshape(-1, logits.size(-1)),
                                tgt_out.reshape(-1)).item() * n
        total_tokens += n

    model.train()
    return math.exp(min(total_loss / max(total_tokens, 1), 20))
