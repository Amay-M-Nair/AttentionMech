"""
Transformer from scratch.

Architecture - read in this order, each depends only on the ones above it:

    config.py         hyperparameters, special token ids
    masking.py        which positions each token may attend to
    positional.py     word-order signal
    attention.py      multi-head attention
    layers.py         encoder / decoder blocks
    transformer.py    the full model

Data:

    spm_tokenizer.py  SentencePiece subwords (current)
    vocab.py          word-level vocabulary (Phase 3, kept for the Phase 4 baseline)
    corpus.py         Gutenberg pretraining corpus
    dataset.py        parallel corpus, bucketed batching
    toy_data.py       copy / reverse tasks

Training and evaluation:

    train.py          teacher forcing, fit loop, checkpointing
    inference.py      greedy + beam search, corpus translation
    evaluate.py       BLEU against the copy baseline
"""
