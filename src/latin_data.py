"""
Latin -> English corpus.

Source: grosenthal/latin_english_translation on HuggingFace. Splits are already
made (99,343 / 1,014 / 1,014), so we never reshuffle them.

About a third is the Vulgate Bible; the rest is a long tail of classical authors
- Cato, Tertullian, Sallust, Catullus, Cicero, Pliny the Younger, Varro. The
per-row source is kept alongside the text so evaluation can separate formulaic
scripture from real prose.

ONE DEFECT WORTH KNOWING: ~0.3% of rows have Latin in the English column - two
different Vulgate versions aligned to each other rather than Latin to English:

    la: et clamaverunt ad Dominum in tribulatione sua de angustiis eorum salvavit eos
    en: et clamaverunt ad Dominum cum tribularentur et de necessitatibus eorum liberavit

Undetectable once training starts, so it is filtered here.
"""

from pathlib import Path

DATASET = "grosenthal/latin_english_translation"
SPLITS = ("train", "valid", "test")

# If the English side contains none of these, it is not English.
EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "was", "that", "for", "with",
    "he", "his", "they", "you", "it", "not", "but", "are", "have", "from",
    "this", "who", "which", "we", "be", "on", "at", "by", "as", "had",
    "her", "she", "them", "their", "him", "all", "will", "would", "there",
}


def looks_english(text: str) -> bool:
    return bool({w.strip(".,;:!?\"'()").lower() for w in text.split()} & EN_STOPWORDS)


def keep(la: str, en: str) -> bool:
    """One row survives if both sides are present, differ, and the target is English."""
    la, en = la.strip(), en.strip()
    return bool(la) and bool(en) and la != en and looks_english(en)


def build(data_dir, dataset: str = DATASET, valid_frac: float = None,
          test_frac: float = None, seed: int = 0, verbose: bool = True) -> dict:
    """
    Download, filter, and write aligned line files.

    Writes per split:
        {split}.la    Latin,   one sentence per line
        {split}.en    English, aligned line for line
        {split}.src   the work each line came from, for per-source evaluation

    Args:
        valid_frac/test_frac: re-split all rows at these fractions instead of
            using the dataset's own splits. Safe here only because the provided
            split is RANDOM - every work in valid and test also appears in
            train, so pooling leaks nothing that was not already leaked. Do not
            do this on a corpus split by document.
        seed: fixes the shuffle, so the split is reproducible

    Returns {split: {"kept": n, "dropped": n}}.
    """
    import random

    from datasets import load_dataset

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(dataset)

    def clean(rows):
        out = []
        for row in rows:
            la, en = row["la"].strip(), row["en"].strip()
            if not keep(la, en):
                continue
            # newlines would break the one-sentence-per-line format
            out.append((" ".join(la.split()), " ".join(en.split()),
                        Path(row["file"].replace("\\", "/")).stem))
        return out

    cleaned = {split: clean(raw[split]) for split in SPLITS}
    dropped = {split: len(raw[split]) - len(cleaned[split]) for split in SPLITS}

    if valid_frac is not None or test_frac is not None:
        pool = [r for split in SPLITS for r in cleaned[split]]

        # The dataset contains duplicate rows. Pooling scatters the copies, so
        # the same pair can land in train AND test - contamination that inflates
        # the score silently. Deduplicate before splitting, not after.
        seen, unique = set(), []
        for row in pool:
            key = (row[0], row[1])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        duplicates = len(pool) - len(unique)
        pool = unique
        random.Random(seed).shuffle(pool)

        n = len(pool)
        n_valid = int(n * (valid_frac or 0))
        n_test = int(n * (test_frac or 0))
        cleaned = {
            "valid": pool[:n_valid],
            "test": pool[n_valid:n_valid + n_test],
            "train": pool[n_valid + n_test:],
        }
        total_dropped = sum(dropped.values())
        dropped = {"train": total_dropped, "valid": 0, "test": 0}
        if verbose:
            print(f"  removed {duplicates:,} duplicate pairs")
            print(f"  re-split {n:,} rows at "
                  f"{1 - (valid_frac or 0) - (test_frac or 0):.0%}/"
                  f"{valid_frac or 0:.0%}/{test_frac or 0:.0%} (seed {seed})")

    counts = {}
    for split in SPLITS:
        rows = cleaned[split]
        for i, suffix in enumerate(("la", "en", "src")):
            (data_dir / f"{split}.{suffix}").write_text(
                "\n".join(r[i] for r in rows) + "\n", encoding="utf-8")

        counts[split] = {"kept": len(rows), "dropped": dropped[split]}
        if verbose:
            print(f"  {split:<6} {len(rows):>6,} rows")

    if verbose and not (valid_frac or test_frac):
        print(f"  dropped {sum(dropped.values()):,} rows in filtering")

    return counts
