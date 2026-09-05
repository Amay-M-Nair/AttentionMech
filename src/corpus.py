"""
Gutenberg pretraining corpus, streamed from `sedthh/gutenberg_english`.

Tokenise once into a flat uint16 file and memory-map it: later epochs cost
nothing and the dataloader never touches the network.
"""

import re

GUTENBERG = "sedthh/gutenberg_english"
TEXT_COLUMN = "TEXT"

# Gutenberg boilerplate, transcriber notes, and chapter furniture.
_BOILERPLATE = re.compile(
    r"(\*\*\*\s*(START|END) OF TH(IS|E) PROJECT GUTENBERG.*?\*\*\*"
    r"|\[Illustration[^\]]*\]"
    r"|\[Footnote[^\]]*\]"
    r"|\[Transcriber'?s? Note[^\]]*\])",
    re.IGNORECASE | re.DOTALL,
)
_UNDERSCORES = re.compile(r"_([^_\n]{1,80})_")   # _italics_ markup
_WHITESPACE = re.compile(r"[ \t]+")
_BLANKLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip Gutenberg boilerplate and normalise whitespace."""
    text = _BOILERPLATE.sub(" ", text)
    text = _UNDERSCORES.sub(r"\1", text)
    text = text.replace("\r", "")
    text = _WHITESPACE.sub(" ", text)
    return _BLANKLINES.sub("\n\n", text).strip()


def stream_gutenberg(max_words: int = 200_000_000, split: str = "train",
                     min_chars: int = 2000, progress_every: int = 200,
                     dataset: str = GUTENBERG, text_column: str = TEXT_COLUMN):
    """Yield cleaned book texts until `max_words` have been produced."""
    from datasets import load_dataset

    stream = load_dataset(dataset, split=split, streaming=True)

    words = books = 0
    for record in stream:
        text = clean_text(record.get(text_column, "") or "")
        if len(text) < min_chars:
            continue

        yield text

        books += 1
        words += text.count(" ") + 1
        if progress_every and books % progress_every == 0:
            print(f"  {books:>6} books   {words / 1e6:6.1f}M words")
        if words >= max_words:
            break

    print(f"  done: {books} books, {words / 1e6:.1f}M words")




# ---------------------------------------------------------------------------
# Tokenised corpus on disk
#
# Streaming and tokenising 200M words takes minutes, and pretraining reads the
# corpus more than once. So do it ONCE into a flat array of token ids and
# memory-map it afterwards: epoch 2 costs nothing, the run is resumable, and
# the dataloader never touches the network.
#
# uint16 holds ids up to 65535, so a 32k vocabulary fits in 2 bytes per token -
# 200M tokens is ~400 MB on disk.
# ---------------------------------------------------------------------------

import numpy as np

TOKEN_DTYPE = np.uint16


def build_token_file(texts, tokenizer, path, max_tokens: int = None,
                     progress_every: int = 5_000_000) -> int:
    """
    Tokenise a stream of documents into one flat binary file.

    Returns:
        number of tokens written
    """
    if len(tokenizer) > np.iinfo(TOKEN_DTYPE).max + 1:
        raise ValueError(
            f"vocab {len(tokenizer)} does not fit in {TOKEN_DTYPE.__name__}"
        )

    written = next_report = 0
    with open(path, "wb") as f:
        for text in texts:
            ids = tokenizer.encode(text)
            if not ids:
                continue
            np.asarray(ids, dtype=TOKEN_DTYPE).tofile(f)
            written += len(ids)

            if progress_every and written >= next_report:
                print(f"  {written / 1e6:6.1f}M tokens")
                next_report += progress_every
            if max_tokens and written >= max_tokens:
                break

    print(f"  wrote {written / 1e6:.1f}M tokens -> {path}")
    return written


def load_token_file(path):
    """Memory-map a token file. Never loads the whole thing into RAM."""
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")


class TokenFileDataset:
    """
    Fixed-length windows over a memory-mapped token file.

    Windows cross document boundaries - standard for pretraining.
    """

    def __init__(self, path, seq_len: int = 256):
        self.tokens = load_token_file(path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, index):
        start = index * self.seq_len
        return np.asarray(self.tokens[start : start + self.seq_len], dtype=np.int64)

    def close(self):
        """Release the memmap. On Windows the file cannot be deleted until this."""
        if getattr(self, "tokens", None) is not None:
            self.tokens._mmap.close()
            self.tokens = None
