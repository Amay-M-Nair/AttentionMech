"""
SentencePiece subword tokenizer. Same encode/decode/__len__ surface as Vocab,
so the datasets and decoders work against either.

Tokenisation only - ids become vectors in nn.Embedding, and contextual only
after the encoder.
"""

from pathlib import Path

import sentencepiece as spm

from .config import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX

class SPMTokenizer:
    """Subword tokenizer. Only the four specials in config.py are reserved."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.model_path = str(model_path)

    @classmethod
    def train(cls, input_files, model_prefix, vocab_size: int = 8000,
              model_type: str = "unigram", character_coverage: float = 1.0,
              input_sentence_size: int = 5_000_000):
        """
        Train on one or more plain-text files, one sentence per line.

        Pass BOTH languages to get a joint vocabulary - a shared vocabulary is
        what lets the tied embedding serve encoder and decoder alike, and Latin
        and English share an alphabet and a great deal of word stock.

        Special ids are pinned to config.py - SentencePiece disables pad by
        default, which would shift every other id.

        byte_fallback spends 256 slots so that ANY character can be encoded, as
        raw bytes when nothing better exists. Without it, a character absent
        from the training text becomes <unk> and the text cannot round-trip -
        and the tokenizer must be trained on train only, so that case is real.
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
            byte_fallback=True,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            pad_id=PAD_IDX,
            bos_id=SOS_IDX,
            eos_id=EOS_IDX,
            unk_id=UNK_IDX,
        )
        return cls(prefix.with_suffix(".model"))

    def __len__(self):
        return self.sp.get_piece_size()

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

        Trained on the parallel corpus too, so it reproduces that corpus's
        spacing exactly - which is what keeps BLEU comparable.
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
