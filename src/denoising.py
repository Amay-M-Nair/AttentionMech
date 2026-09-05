"""
Span corruption (T5) - the pretraining objective.

    original   The cat sat on the mat and purred
    encoder    The <extra_id_0> on the mat <extra_id_1> purred
    decoder    <extra_id_0> cat sat <extra_id_1> and <extra_id_2>

Unlike BERT-style masking it produces a source and a target, so it trains an
encoder-decoder unchanged. Raffel et al. 2020, section 3.1.4.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EOS_IDX, SOS_IDX

NOISE_DENSITY = 0.15      # fraction of tokens masked
MEAN_SPAN_LENGTH = 3.0    # average masked span, in tokens


def _random_segmentation(num_items: int, num_segments: int, rng) -> np.ndarray:
    """Split `num_items` into `num_segments` positive lengths, at random."""
    if num_segments >= num_items:
        return np.ones(num_items, dtype=np.int64)

    cuts = rng.choice(num_items - 1, num_segments - 1, replace=False) + 1
    cuts.sort()
    return np.diff(np.concatenate([[0], cuts, [num_items]]))


def corrupt_spans(tokens, sentinel_ids, noise_density: float = NOISE_DENSITY,
                  mean_span_length: float = MEAN_SPAN_LENGTH, rng=None):
    """
    Args:
        tokens: 1-D sequence of token ids
        sentinel_ids: the <extra_id_N> ids, in order

    Returns:
        (encoder_input, decoder_target) as lists of ids
    """
    rng = rng or np.random.default_rng()
    tokens = np.asarray(tokens)
    length = len(tokens)

    num_noise = int(round(length * noise_density))
    num_noise = min(max(num_noise, 1), length - 1)

    num_spans = int(round(num_noise / mean_span_length))
    num_spans = min(max(num_spans, 1), num_noise, len(sentinel_ids))

    noise_lengths = _random_segmentation(num_noise, num_spans, rng)
    keep_lengths = _random_segmentation(length - num_noise, num_spans, rng)

    encoder_input, decoder_target = [], []
    cursor = 0
    for i in range(num_spans):
        keep, noise = int(keep_lengths[i]), int(noise_lengths[i])

        encoder_input.extend(tokens[cursor : cursor + keep].tolist())
        cursor += keep

        encoder_input.append(sentinel_ids[i])
        decoder_target.append(sentinel_ids[i])
        decoder_target.extend(tokens[cursor : cursor + noise].tolist())
        cursor += noise

    # whatever is left over stays visible to the encoder
    encoder_input.extend(tokens[cursor:].tolist())

    # closing sentinel: tells the model the reconstruction is finished
    decoder_target.append(sentinel_ids[min(num_spans, len(sentinel_ids) - 1)])

    return encoder_input, decoder_target


class DenoisingDataset(Dataset):
    """
    Corrupted (src, tgt_in, tgt_out) triples, same convention as dataset.py.

    A fresh corruption is drawn per fetch, so a second epoch is not a repeat.
    Set `seed` for determinism.
    """

    def __init__(self, token_source, sentinel_ids,
                 noise_density: float = NOISE_DENSITY,
                 mean_span_length: float = MEAN_SPAN_LENGTH,
                 seed: int = None):
        self.tokens = token_source
        self.sentinel_ids = list(sentinel_ids)
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index) if self.seed is not None else self._rng

        encoder_input, target = corrupt_spans(
            self.tokens[index], self.sentinel_ids,
            self.noise_density, self.mean_span_length, rng,
        )

        return (
            torch.tensor(encoder_input + [EOS_IDX], dtype=torch.long),
            torch.tensor([SOS_IDX] + target, dtype=torch.long),
            torch.tensor(target + [EOS_IDX], dtype=torch.long),
        )
