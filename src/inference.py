"""
Decoding: greedy and beam search.

The encoder runs once, the decoder once per token.
"""

import torch
import torch.nn.functional as F

from .config import EOS_IDX, PAD_IDX, SOS_IDX
from .masking import make_pad_mask


def step_logprobs(model, tokens, memory, src_mask):
    """Log-probabilities for the next token. Shared by greedy and beam."""
    output, _, _ = model.decode(tokens, memory, src_mask=src_mask)
    return F.log_softmax(model.generator(output[:, -1]), dim=-1)


@torch.no_grad()
def greedy_decode(model, src, max_len: int = None,
                  sos_idx: int = SOS_IDX, eos_idx: int = EOS_IDX) -> torch.Tensor:
    """
    Args:
        src: (batch, src_len) source token indices
        max_len: generation cap; defaults to source length plus headroom

    Returns:
        (batch, generated_len) token indices, SOS stripped, padded after EOS.
    """
    model.eval()
    device = src.device
    batch = src.size(0)
    if max_len is None:
        max_len = src.size(1) + 10

    src_mask = make_pad_mask(src, model.pad_idx)
    memory, _ = model.encode(src, src_mask)

    tgt = torch.full((batch, 1), sos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_len):
        # The whole prefix is re-run each step. Wasteful, but correct, and the
        # sequences here are short. A KV cache is the optimisation later.
        next_token = step_logprobs(model, tgt, memory, src_mask).argmax(-1)

        # Once a sequence has emitted EOS, keep it padded.
        next_token = torch.where(
            finished, torch.full_like(next_token, model.pad_idx), next_token
        )
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        finished |= next_token == eos_idx
        if finished.all():
            break

    return tgt[:, 1:]


@torch.no_grad()
def cross_attention_map(model, src, tgt_in, layer: int = -1, head: int = None):
    """
    Cross-attention weights for one teacher-forced pass.

    Args:
        layer: which decoder layer, -1 for the last
        head: which head, or None to average across heads

    Returns:
        (batch, tgt_len, src_len) if head is None, else the same shape for that head.

    On a trained copy model this should be close to a diagonal; on reverse, an
    anti-diagonal.
    """
    model.eval()
    _, cross_attns = model(src, tgt_in, return_attn=True)
    attn = cross_attns[layer]
    return attn.mean(1) if head is None else attn[:, head]


def strip_special(row, eos_idx: int = EOS_IDX, pad_idx: int = PAD_IDX) -> list:
    """Take one sequence up to its first EOS, dropping padding."""
    out = []
    for token in row.tolist():
        if token == eos_idx:
            break
        if token != pad_idx:
            out.append(token)
    return out


def exact_match(pred, gold, eos_idx: int = EOS_IDX, pad_idx: int = PAD_IDX) -> float:
    """Fraction of sequences reproduced exactly, ignoring EOS and padding."""
    hits = sum(
        strip_special(p, eos_idx, pad_idx) == strip_special(g, eos_idx, pad_idx)
        for p, g in zip(pred, gold)
    )
    return hits / len(pred)


@torch.no_grad()
def translate_corpus(model, lines, vocab, device=None, batch_size: int = 128,
                     max_len: int = None, beam_size: int = 1,
                     length_penalty: float = 0.6) -> list:
    """
    Translate many sentences at once, sorted by length to limit padding.

    Returns translations aligned with `lines`.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    encoded = [vocab.encode(line) for line in lines]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))

    results = [None] * len(encoded)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        rows = [torch.tensor(encoded[i] + [EOS_IDX]) for i in chunk]

        width = max(len(r) for r in rows)
        src = torch.full((len(rows), width), PAD_IDX, dtype=torch.long)
        for j, row in enumerate(rows):
            src[j, : len(row)] = row

        cap = max_len if max_len is not None else width + 10
        if beam_size > 1:
            predictions = beam_search_decode(model, src.to(device), beam_size=beam_size,
                                             max_len=cap, length_penalty=length_penalty)
        else:
            predictions = greedy_decode(model, src.to(device), max_len=cap)

        for j, i in enumerate(chunk):
            results[i] = vocab.decode(predictions[j])

    return results


@torch.no_grad()
def beam_search_decode(model, src, beam_size: int = 5, max_len: int = None,
                       length_penalty: float = 0.6, sos_idx: int = SOS_IDX,
                       eos_idx: int = EOS_IDX) -> torch.Tensor:
    """
    Beam search over a batch of sentences.

    Length penalty (GNMT): score / ((5 + L) / 6) ** alpha. Log-probs are sums
    over tokens, so they favour short output; the penalty removes that bias.
    Measured best here is alpha ~1.5, not the usual 0.6.

    Args:
        src: (batch, src_len)

    Returns:
        (batch, generated_len) - best beam per sentence, padded after EOS.
    """
    model.eval()
    device = src.device
    batch, K = src.size(0), beam_size
    if max_len is None:
        max_len = src.size(1) + 10

    src_mask = make_pad_mask(src, model.pad_idx)
    memory, _ = model.encode(src, src_mask)

    # Every sentence is repeated K times so all beams decode in one pass.
    src_len, d_model = memory.size(1), memory.size(2)
    memory = memory.unsqueeze(1).expand(batch, K, src_len, d_model).reshape(batch * K, src_len, d_model)
    src_mask = src_mask.unsqueeze(1).expand(batch, K, 1, 1, src_len).reshape(batch * K, 1, 1, src_len)

    tokens = torch.full((batch * K, 1), sos_idx, dtype=torch.long, device=device)

    # Only beam 0 starts alive. Without this every beam holds the same prefix
    # and the first top-k returns K copies of one token.
    scores = torch.full((batch, K), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)

    finished = torch.zeros(batch * K, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logprobs = step_logprobs(model, tokens, memory, src_mask)
        vocab = logprobs.size(-1)

        # A finished beam may only extend with PAD, and gains no further score,
        # so it stays comparable against beams still running.
        if finished.any():
            logprobs[finished] = float("-inf")
            logprobs[finished, model.pad_idx] = 0.0

        candidates = (scores.unsqueeze(1) + logprobs).view(batch, K * vocab)
        scores, flat = candidates.topk(K, dim=-1)

        beam = torch.div(flat, vocab, rounding_mode="floor")
        token = flat % vocab

        # Reorder the prefixes to match the beams that were kept.
        reorder = (torch.arange(batch, device=device).unsqueeze(1) * K + beam).view(-1)
        tokens = torch.cat([tokens[reorder], token.view(-1, 1)], dim=1)
        scores = scores.view(-1)
        finished = finished[reorder] | (token.view(-1) == eos_idx)

        if finished.all():
            break

    generated = tokens[:, 1:]

    # Length = position of the first EOS (inclusive), else everything produced.
    is_eos = generated == eos_idx
    has_eos = is_eos.any(dim=1)
    first_eos = torch.where(has_eos, is_eos.float().argmax(dim=1) + 1,
                            torch.full_like(has_eos, generated.size(1), dtype=torch.long))
    lengths = first_eos.float().clamp(min=1)

    normalised = (scores / (((5.0 + lengths) / 6.0) ** length_penalty)).view(batch, K)
    best = normalised.argmax(dim=-1)
    pick = torch.arange(batch, device=device) * K + best

    return generated[pick]
