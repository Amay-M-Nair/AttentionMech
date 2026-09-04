"""
Word-level vocabulary, shared between source and target.

One vocabulary for both sides, not two. Shakespeare and modern English are the
same language - 91% of source tokens also appear on the target side - so a word
should carry the same index wherever it shows up. That is also what makes the
tied embeddings in transformer.py pay off: copying a word through the model
becomes nearly free.

The corpus is already tokenised (NLTK, in the .nltktok files), so splitting on
whitespace is all that is needed here.
"""

import json
from collections import Counter
from pathlib import Path

from .config import EOS_IDX, PAD_IDX, SOS_IDX, SPECIALS, UNK_IDX


class Vocab:
    """Maps tokens to indices and back. Specials occupy the first slots."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def build(cls, *corpora, min_freq: int = 2, max_size: int = None) -> "Vocab":
        """
        Args:
            *corpora: iterables of lines; pass source AND target to share one vocab
            min_freq: tokens rarer than this become <unk>
            max_size: optional cap, keeping the most frequent

        Rare words are dropped on purpose. A word seen once cannot be learned
        from one example, and keeping it wastes an embedding row - the <unk>
        replacement trick at inference recovers most of them by copying from
        the source anyway.
        """
        counts = Counter(
            token for corpus in corpora for line in corpus for token in line.split()
        )

        kept = [t for t, c in counts.most_common() if c >= min_freq]
        if max_size is not None:
            kept = kept[: max_size - len(SPECIALS)]

        return cls(list(SPECIALS) + kept)

    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def encode(self, line: str, add_sos: bool = False, add_eos: bool = False) -> list:
        """Tokens to indices. Unknown tokens map to <unk>."""
        ids = [self.stoi.get(t, UNK_IDX) for t in line.split()]
        if add_sos:
            ids = [SOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids, keep_specials: bool = False) -> str:
        """Indices back to a string, stopping at <eos>."""
        tokens = []
        for i in ids:
            i = int(i)
            if i == EOS_IDX and not keep_specials:
                break
            if i in (PAD_IDX, SOS_IDX) and not keep_specials:
                continue
            tokens.append(self.itos[i])
        return " ".join(tokens)

    def unk_rate(self, corpus) -> float:
        """Fraction of tokens in `corpus` that this vocabulary cannot represent."""
        total = unknown = 0
        for line in corpus:
            for token in line.split():
                total += 1
                unknown += token not in self.stoi
        return unknown / total if total else 0.0

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
