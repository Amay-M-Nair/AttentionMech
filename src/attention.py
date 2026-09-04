"""
Multi-head attention.

One class covers all three uses - encoder self-attention, decoder causal
self-attention, and cross-attention - since they differ only in what is passed
as query/key/value and which mask. See docs/architecture.md#attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention across several parallel heads."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)"""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)"""
        batch, _, seq_len, _ = x.shape
        # contiguous() because transpose only changes indexing, and view needs
        # a contiguous buffer
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, q_len, d_model)
            key:   (batch, kv_len, d_model)
            value: (batch, kv_len, d_model)
            mask:  bool, broadcastable to (batch, heads, q_len, kv_len)

        Returns:
            output:       (batch, q_len, d_model)
            attn_weights: (batch, heads, q_len, kv_len)

        q_len and kv_len may differ; in cross-attention they usually do.
        """
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        # scaled by head_dim, not d_model - after the split that is the d_k
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # mask before softmax. finfo.min rather than -inf so a fully masked
        # row yields zeros instead of NaN
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = self.dropout(attn_weights) @ V

        # weights returned undropped - they are for inspection
        return self.W_o(self._merge_heads(attn_out)), attn_weights
