"""
Greedy decoding.

Training runs the whole target through at once. Generation cannot - the input
at step t is the model's own output from step t-1. So the encoder runs once and
the decoder runs once per token.

Beam search comes later; greedy is enough to see whether the model learned.
"""

import torch

from .config import EOS_IDX, PAD_IDX, SOS_IDX
from .masking import make_pad_mask


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
        output, _, _ = model.decode(tgt, memory, src_mask=src_mask)
        next_token = model.generator(output[:, -1]).argmax(-1)

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
