"""
Attention masks. True = may attend, False = blocked.

Shapes are broadcast-friendly against scores of (batch, heads, q_len, k_len),
so one masked_fill covers every attention type.
"""

import torch


def make_pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Block attention *to* padding positions.

    seq: (batch, seq_len) -> (batch, 1, 1, seq_len)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device=None) -> torch.Tensor:
    """
    Block attention to future positions.

    -> (1, 1, size, size), lower-triangular. Row = query, column = key.
    """
    ones = torch.ones(size, size, dtype=torch.bool, device=device)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


def make_decoder_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Both restrictions at once: no padding, no future.

    tgt: (batch, tgt_len) -> (batch, 1, tgt_len, tgt_len)
    """
    pad_mask = make_pad_mask(tgt, pad_idx)                   # (B, 1, 1, T)
    causal_mask = make_causal_mask(tgt.size(1), tgt.device)  # (1, 1, T, T)
    return pad_mask & causal_mask                            # (B, 1, T, T)
