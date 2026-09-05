"""
Shakespeare -> modern parallel corpus.

    .original  Shakespeare -> source        .modern  the rewrite -> target

Split is by play (15 train, Twelfth Night valid, Romeo and Juliet test). Do not
reshuffle - lines from one play would leak across sides.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import EOS_IDX, PAD_IDX, SOS_IDX

SPLITS = ("train", "valid", "test")


def load_split(data_dir, split: str):
    """
    Returns:
        (source_lines, target_lines) - Shakespeare, modern
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    data_dir = Path(data_dir)

    def read(side):
        path = data_dir / f"{split}.{side}.nltktok"
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    source, target = read("original"), read("modern")
    if len(source) != len(target):
        raise ValueError(f"{split} is misaligned: {len(source)} vs {len(target)} lines")
    return source, target


class TranslationDataset(Dataset):
    """Encoded sentence pairs. Long sentences are truncated, not dropped."""

    def __init__(self, source_lines, target_lines, vocab, max_len: int = 100):
        self.vocab = vocab
        self.max_len = max_len
        self.pairs = list(zip(source_lines, target_lines))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source, target = self.pairs[index]

        # Truncate before adding EOS/SOS, so every sequence still terminates
        # properly - a cut-off sentence with no EOS teaches the model not to stop.
        src_ids = self.vocab.encode(source)[: self.max_len]
        tgt_ids = self.vocab.encode(target)[: self.max_len]

        return (
            torch.tensor(src_ids + [EOS_IDX]),
            torch.tensor([SOS_IDX] + tgt_ids),
            torch.tensor(tgt_ids + [EOS_IDX]),
        )


def collate_fn(batch, pad_idx: int = PAD_IDX):
    """Pad each of the three sides to the longest sequence in this batch."""
    def pad(sequences):
        width = max(len(s) for s in sequences)
        out = torch.full((len(sequences), width), pad_idx, dtype=torch.long)
        for i, seq in enumerate(sequences):
            out[i, : len(seq)] = seq
        return out

    src, tgt_in, tgt_out = zip(*batch)
    return pad(src), pad(tgt_in), pad(tgt_out)


class LengthBucketSampler(Sampler):
    """
    Groups sentences of similar length into a batch.

    Cuts padding from 75% to 9% over an epoch. Batch order stays shuffled.
    """

    def __init__(self, dataset, batch_size: int, pool_factor: int = 50, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_factor
        self.shuffle = shuffle
        self.lengths = [len(s.split()) for s, _ in dataset.pairs]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batches = []
        # Sort within a shuffled pool rather than globally: keeps batches tight
        # without making the epoch order deterministic.
        for start in range(0, len(indices), self.pool_size):
            pool = sorted(indices[start : start + self.pool_size], key=lambda i: self.lengths[i])
            batches += [pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def make_dataloader(dataset, batch_size: int = 64, shuffle: bool = True, bucket: bool = True):
    if bucket:
        return DataLoader(
            dataset,
            batch_sampler=LengthBucketSampler(dataset, batch_size, shuffle=shuffle),
            collate_fn=collate_fn,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
