"""Model hyperparameters. See docs/architecture.md#config."""

from dataclasses import dataclass

import torch


@dataclass
class TransformerConfig:
    """Defaults are smaller than the paper's 512/8/6 - see docs."""

    vocab_size: int = 10000
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 5000
    pad_idx: int = 0

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
