"""
Transformer from scratch.

Read the modules in this order - each depends only on the ones above it:

    config.py      hyperparameters
    masking.py     which positions each token may attend to
    positional.py  word-order signal
    attention.py   multi-head attention
    layers.py      encoder / decoder blocks
    transformer.py the full model

How and why any of it works: docs/architecture.md
"""
