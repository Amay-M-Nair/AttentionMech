"""
Toy sequence tasks: copy and reverse.

Random integer sequences, so the model can be checked for the ability to learn
before any tokeniser or real corpus exists. Lengths vary within a batch, which
is deliberate - it exercises padding, and padding is where mask bugs hide.

Each batch is a (src, tgt_in, tgt_out) triple:

    src      content + EOS            what the encoder reads
    tgt_in   SOS + target             what the decoder is fed (teacher forcing)
    tgt_out  target + EOS             what the decoder is graded against

tgt_in and tgt_out are the same sequence offset by one, so predicting position
t of tgt_in should produce position t of tgt_out.
"""

import torch

from .config import EOS_IDX, NUM_SPECIAL, PAD_IDX, SOS_IDX

TASKS = ("copy", "reverse")


def make_batch(
    batch_size: int = 32,
    vocab_size: int = 50,
    min_len: int = 4,
    max_len: int = 10,
    task: str = "copy",
    device=None,
    generator: torch.Generator = None,
):
    """
    Args:
        vocab_size: total vocabulary including the special tokens
        min_len/max_len: content length range, sampled per sequence
        task: "copy" or "reverse"
        generator: pass a seeded torch.Generator for a reproducible batch

    Returns:
        src:     (batch, src_len)
        tgt_in:  (batch, tgt_len)
        tgt_out: (batch, tgt_len)
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if vocab_size <= NUM_SPECIAL:
        raise ValueError(f"vocab_size must exceed {NUM_SPECIAL} special tokens")

    lengths = torch.randint(min_len, max_len + 1, (batch_size,), generator=generator)

    sequences = [
        torch.randint(NUM_SPECIAL, vocab_size, (int(n),), generator=generator)
        for n in lengths
    ]

    src_rows, tgt_in_rows, tgt_out_rows = [], [], []
    for seq in sequences:
        target = seq if task == "copy" else torch.flip(seq, dims=[0])

        src_rows.append(torch.cat([seq, torch.tensor([EOS_IDX])]))
        tgt_in_rows.append(torch.cat([torch.tensor([SOS_IDX]), target]))
        tgt_out_rows.append(torch.cat([target, torch.tensor([EOS_IDX])]))

    src = pad_stack(src_rows)
    tgt_in = pad_stack(tgt_in_rows)
    tgt_out = pad_stack(tgt_out_rows)

    if device is not None:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

    return src, tgt_in, tgt_out


def pad_stack(rows, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """Stack variable-length 1-D tensors into (batch, max_len), right-padded."""
    width = max(len(r) for r in rows)
    out = torch.full((len(rows), width), pad_idx, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def batch_stream(steps: int, **kwargs):
    """Yield `steps` fresh batches. Every step sees unseen data."""
    for _ in range(steps):
        yield make_batch(**kwargs)


def describe(src, tgt_in, tgt_out, index: int = 0) -> str:
    """One example rendered as three aligned rows, for eyeballing the shift."""
    def row(t):
        return " ".join(f"{int(v):>3}" for v in t[index].tolist())

    return (
        f"src     {row(src)}\n"
        f"tgt_in  {row(tgt_in)}\n"
        f"tgt_out {row(tgt_out)}"
    )
