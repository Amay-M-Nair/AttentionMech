"""
BLEU, always reported against the copy baseline.

tokenize="none" everywhere: the corpus is already tokenised, and the setting
must match across the baseline, validation and the final test number.
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
