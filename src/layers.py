"""
Encoder and decoder blocks.

Pre-norm throughout: x = x + sublayer(norm(x)), rather than the paper's
post-norm. See docs/architecture.md#layers.
"""

import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise MLP: d_model -> d_ff -> d_model."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Self-attention, then feed-forward. Two sublayers."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Args:
            x:        (batch, src_len, d_model)
            src_mask: (batch, 1, 1, src_len)

        Returns:
            x:            unchanged shape
            attn_weights: (batch, heads, src_len, src_len)
        """
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed, normed, normed, src_mask)
        x = x + self.dropout(attn_out)

        x = x + self.dropout(self.feed_forward(self.norm2(x)))

        return x, attn_weights


class DecoderLayer(nn.Module):
    """Causal self-attention, cross-attention, then feed-forward. Three sublayers."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        """
        Args:
            x:        (batch, tgt_len, d_model) - target so far
            memory:   (batch, src_len, d_model) - encoder output
            tgt_mask: (batch, 1, tgt_len, tgt_len)
            src_mask: (batch, 1, 1, src_len)

        Returns:
            x:                  unchanged shape
            self_attn_weights:  (batch, heads, tgt_len, tgt_len)
            cross_attn_weights: (batch, heads, tgt_len, src_len)
        """
        normed = self.norm1(x)
        self_out, self_attn_weights = self.self_attn(normed, normed, normed, tgt_mask)
        x = x + self.dropout(self_out)

        # queries from the decoder, keys/values from the encoder.
        # masked with src_mask: source padding is what must be hidden here
        normed = self.norm2(x)
        cross_out, cross_attn_weights = self.cross_attn(normed, memory, memory, src_mask)
        x = x + self.dropout(cross_out)

        x = x + self.dropout(self.feed_forward(self.norm3(x)))

        return x, self_attn_weights, cross_attn_weights
