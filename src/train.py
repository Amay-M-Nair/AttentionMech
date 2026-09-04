"""
Training loop.

Teacher forcing: the decoder is fed the correct previous tokens rather than its
own predictions, so every position trains in parallel. The (tgt_in, tgt_out)
offset is built into the batch, so nothing is sliced here.
"""

import torch
import torch.nn as nn


def make_criterion(pad_idx: int, label_smoothing: float = 0.1):
    """
    Padding is excluded from the loss entirely.

    Label smoothing spreads a little probability mass off the correct token,
    which stops the model driving logits to extremes and generalising worse.
    """
    return nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)


def make_optimizer(model, lr: float = 1e-3, weight_decay: float = 0.01):
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay
    )


def make_scheduler(optimizer, warmup: int = 400):
    """
    Linear warmup, then inverse-square-root decay, as a multiplier on the base
    learning rate. Peaks at 1.0 exactly when warmup ends.

    Warmup matters because Adam's second-moment estimate is unreliable in the
    first few steps; a full-size update then can wreck the initialisation.
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
    Train on ONE batch repeatedly. Loss must collapse towards zero.

    The cheapest possible check that the plumbing is right: if a model cannot
    memorise a single batch, the problem is a bug - a mask, a shift, a detached
    branch - not the data or the hyperparameters. No label smoothing here,
    since it puts a floor under the loss and hides exactly what we want to see.
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
