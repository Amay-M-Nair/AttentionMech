"""
Evaluation metrics.

## Why tokenize="none"

sacreBLEU normally applies its own tokenizer so that scores are comparable
across projects. Our corpus is ALREADY tokenised (the .nltktok files), and
running a second tokeniser over it would split things twice and quietly shift
the score.

So every BLEU call here passes tokenize="none", which means plain whitespace
splitting. The important part is not which setting is used but that the SAME
setting is used everywhere - the copy baseline, validation during training, and
the final test number. Mixing them produces numbers that cannot be compared,
which is one of the easiest ways to fool yourself about progress.
"""

import sacrebleu


def bleu(hypotheses, references) -> float:
    """
    Corpus BLEU over already-tokenised text.

    Args:
        hypotheses: list of predicted strings
        references: list of reference strings, one per hypothesis
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    # force=True silences sacreBLEU's "you forgot to detokenize" warning - here
    # the text is tokenised deliberately, which is exactly what it complains about.
    return sacrebleu.corpus_bleu(
        hypotheses, [references], tokenize="none", force=True
    ).score


def copy_baseline(source, target) -> float:
    """
    BLEU for echoing the input unchanged.

    This is the number to beat. Source and target here are the same language
    sharing most of their vocabulary, so doing nothing already scores well - a
    model below this line is worse than useless, and one only slightly above it
    has learned almost nothing.
    """
    return bleu(source, target)


def summarise(hypotheses, references, source=None) -> dict:
    """BLEU alongside the baseline, so the score is never read in isolation."""
    result = {"bleu": bleu(hypotheses, references)}
    if source is not None:
        result["copy_baseline"] = copy_baseline(source, references)
        result["gain"] = result["bleu"] - result["copy_baseline"]
    return result
