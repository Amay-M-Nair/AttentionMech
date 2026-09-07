"""
BLEU, always reported against the copy baseline.

Standard sacreBLEU tokenisation - the corpus is natural text, so scores are
comparable with published work. What matters is that the same setting is used
for the baseline, for validation, and for the final test number.
"""

import sacrebleu


def bleu(hypotheses, references) -> float:
    """
    Corpus BLEU.

    Args:
        hypotheses: list of predicted strings
        references: list of reference strings, one per hypothesis
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} references")
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def copy_baseline(source, target) -> float:
    """
    BLEU for echoing the input unchanged.

    This is the number to beat. Source and target here are the same language
    sharing most of their vocabulary, so doing nothing already scores well - a
    model below this line is worse than useless, and one only slightly above it
    has learned almost nothing.
    """
    return bleu(source, target)
