"""
Latin -> English parallel corpus.

    .la  Latin -> source        .en  English -> target

Splits come from the dataset and are not reshuffled. Built by latin_data.py.
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
        (source_lines, target_lines) - Latin, English
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    data_dir = Path(data_dir)

    def read(side):
        path = data_dir / f"{split}.{side}"
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    source, target = read("la"), read("en")
    if len(source) != len(target):
        raise ValueError(f"{split} is misaligned: {len(source)} vs {len(target)} lines")
    return source, target


class TranslationDataset(Dataset):
    """Encoded sentence pairs. Long sentences are truncated, not dropped."""

    def __init__(self, source_lines, target_lines, vocab, max_len: int = 100):
        self.max_len = max_len

        # Encoded once, here. The previous version tokenised inside __getitem__,
        # so every epoch re-ran SentencePiece over the whole corpus - 188k
        # encodes per epoch at 94k pairs, on the main process, in series with
        # the GPU.
        #
        # Truncate before adding EOS/SOS, so every sequence still terminates
        # properly - a cut-off sentence with no EOS teaches the model not to stop.
        self.source = [vocab.encode(line)[: max_len] for line in source_lines]
        self.target = [vocab.encode(line)[: max_len] for line in target_lines]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        src_ids, tgt_ids = self.source[index], self.target[index]
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

    Bucketing is on encoded length, not word count: token length is what gets
    padded, and Latin's fertility of 1.84 means the two are not proportional.
    """

    def __init__(self, dataset, batch_size: int, pool_factor: int = 50, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_factor
        self.shuffle = shuffle
        self.lengths = [len(ids) for ids in dataset.source]

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
