"""
Pretraining corpus: Project Gutenberg literature.

Phase 4 failed because 474k words cannot teach a model English. This module
supplies ~200M words of it.

Source is `sedthh/gutenberg_english` on HuggingFace - the English Gutenberg
corpus as parquet shards. We STREAM a subset rather than downloading the whole
thing: `streaming=True` pulls shards lazily, so a 200M-word subset costs a few
hundred MB of traffic and no disk.

(The better-known PG-19 release is unusable here - it ships as a loading script,
which `datasets` 5.x dropped support for.)

Why literature rather than Wikipedia: every token is stylistically relevant -
prose, dialogue, narrative voice - and the collection already contains
Shakespeare, the King James Bible, Marlowe and Milton, so Early Modern English
arrives without a separate domain-adaptation stage.

The trade is noise. Gutenberg text carries OCR artefacts, transcriber notes and
licence boilerplate, so `clean_text` earns its place.
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
    """Strip Gutenberg furniture and normalise whitespace."""
    text = _BOILERPLATE.sub(" ", text)
    text = _UNDERSCORES.sub(r"\1", text)
    text = text.replace("\r", "")
    text = _WHITESPACE.sub(" ", text)
    return _BLANKLINES.sub("\n\n", text).strip()


def stream_gutenberg(max_words: int = 200_000_000, split: str = "train",
                     min_chars: int = 2000, progress_every: int = 200,
                     dataset: str = GUTENBERG, text_column: str = TEXT_COLUMN):
    """
    Yield cleaned book texts until `max_words` have been produced.

    Args:
        max_words: stop once this many whitespace words have been yielded
        min_chars: skip anything shorter than this after cleaning
        progress_every: print a progress line every N books, 0 to silence

    Yields:
        str - one cleaned book at a time
    """
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


def to_chunks(texts, tokenizer, seq_len: int = 256, drop_last: bool = True):
    """
    Turn a stream of documents into fixed-length token sequences.

    Documents are concatenated and sliced at `seq_len`, rather than padding each
    one - padding a 200M-word corpus would waste an enormous amount of compute.
    Chunks therefore cross sentence and document boundaries, which is standard
    for pretraining and harmless: the model is learning language, not documents.

    Yields:
        list[int] - token ids of length seq_len
    """
    buffer = []
    for text in texts:
        buffer.extend(tokenizer.encode(text))
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            buffer = buffer[seq_len:]

    if buffer and not drop_last:
        yield buffer


def write_lines(texts, path, max_words: int = None):
    """
    Dump a corpus stream to a plain text file, one paragraph per line.

    Used to build the SentencePiece training input, which wants a file rather
    than a generator.
    """
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            for paragraph in text.split("\n\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                f.write(paragraph + "\n")
                written += paragraph.count(" ") + 1
            if max_words and written >= max_words:
                break
    return written


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

    Chunks cross sentence and document boundaries. That is standard for
    pretraining and harmless - the model is learning language, not documents.
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
