"""Sinusoidal positional encoding. See docs/architecture.md#positional-encoding."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Adds fixed position signals to token embeddings. Not learned."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(-2i/d_model) computed in log space to stay well-conditioned
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        # slice div_term in case d_model is odd
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        # buffer, not parameter: moves with .to(device), never optimised
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> same shape."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
