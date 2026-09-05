"""
SentencePiece subword tokenizer.

Replaces the word-level vocabulary in vocab.py, which left 3.65% of decoder
output as <unk> and turned long sentences into unreadable soup. Subwords remove
<unk> entirely - any word can be spelled from smaller pieces - and let related
forms share structure, so `thou`, `thee` and `thy` stop being three unrelated
rows in a lookup table.

TOKENIZATION IS NOT EMBEDDING. This file only splits text into ids. Those ids
are then looked up in nn.Embedding (a static vector per token, the word2vec-like
part) and only become CONTEXTUAL after passing through the encoder's attention
layers. No word2vec or GloVe is involved anywhere: the embedding table is
learned from scratch during pretraining, which is both simpler and better,
since GloVe is word-level and this vocabulary is not.

The interface matches Vocab in vocab.py - encode / decode / __len__ - so
TranslationDataset and translate_corpus work against either one unchanged.
"""

from pathlib import Path

import sentencepiece as spm

from .config import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX

# Direction tokens, for the bidirectional and multilingual work later.
TO_MODERN = "<2modern>"
TO_SHAKESPEARE = "<2shakespeare>"
DIRECTION_TOKENS = (TO_MODERN, TO_SHAKESPEARE)

NUM_SENTINELS = 100


def sentinel(i: int) -> str:
    """Name of the i-th span-corruption sentinel."""
    return f"<extra_id_{i}>"


SENTINELS = tuple(sentinel(i) for i in range(NUM_SENTINELS))


class SPMTokenizer:
    """Subword tokenizer with the same surface as `Vocab`."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.model_path = str(model_path)
        self.sentinel_ids = [self.sp.piece_to_id(s) for s in SENTINELS]

    @classmethod
    def train(cls, input_files, model_prefix, vocab_size: int = 32000,
              model_type: str = "unigram", character_coverage: float = 0.9999,
              input_sentence_size: int = 5_000_000):
        """
        Train on one or more plain-text files.

        Args:
            input_files: path or list of paths, one sentence/paragraph per line
            input_sentence_size: sample this many lines rather than reading all
                of a 200M-word corpus into memory; SentencePiece shuffles first

        Special ids are pinned to the values in config.py. SentencePiece leaves
        pad disabled by default, which would silently shift every other id and
        corrupt every checkpoint written before the change.

        Sentinels and direction tokens are registered as user-defined symbols so
        they survive tokenisation as single units.
        """
        prefix = Path(model_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(input_files, (str, Path)):
            input_files = [input_files]

        spm.SentencePieceTrainer.train(
            input=",".join(str(f) for f in input_files),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            pad_id=PAD_IDX,
            bos_id=SOS_IDX,
            eos_id=EOS_IDX,
            unk_id=UNK_IDX,
            user_defined_symbols=list(DIRECTION_TOKENS) + list(SENTINELS),
        )
        return cls(prefix.with_suffix(".model"))

    def __len__(self):
        return self.sp.get_piece_size()

    @property
    def itos(self):
        return [self.sp.id_to_piece(i) for i in range(len(self))]

    def token_id(self, token: str) -> int:
        """Id of a whole token, e.g. a direction token or a sentinel."""
        return self.sp.piece_to_id(token)

    def encode(self, line: str, add_sos: bool = False, add_eos: bool = False) -> list:
        ids = self.sp.encode(line.strip(), out_type=int)
        if add_sos:
            ids = [SOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids, keep_specials: bool = False) -> str:
        """
        Ids back to text, stopping at <eos>.

        Trained on the already-tokenised parallel corpus alongside Gutenberg, so
        it reproduces that corpus's spacing exactly (`speak'st .` keeps the space
        before the period). That is what keeps BLEU comparable with every score
        recorded before this change - verify it with a round-trip before
        trusting any number.
        """
        kept = []
        for i in ids:
            i = int(i)
            if i == EOS_IDX and not keep_specials:
                break
            if i in (PAD_IDX, SOS_IDX) and not keep_specials:
                continue
            kept.append(i)
        return self.sp.decode(kept)

    def unk_rate(self, corpus) -> float:
        """Should be ~0. Subwords can spell anything; worth asserting."""
        total = unknown = 0
        for line in corpus:
            ids = self.encode(line)
            total += len(ids)
            unknown += sum(i == UNK_IDX for i in ids)
        return unknown / total if total else 0.0

    def fertility(self, corpus) -> float:
        """Subword pieces per whitespace word - how much the text got split."""
        pieces = words = 0
        for line in corpus:
            pieces += len(self.encode(line))
            words += len(line.split())
        return pieces / words if words else 0.0

    def round_trip_failures(self, corpus) -> list:
        """Lines that do not decode back to themselves. Should be ~empty."""
        return [line for line in corpus if self.decode(self.encode(line)) != line]
