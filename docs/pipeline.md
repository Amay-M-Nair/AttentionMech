# Pipeline

Shakespeare → modern English translation, built on the architecture from *Attention Is All You Need*.

---

## Overview

```
                    ┌─────────────────────────────────────────┐
                    │  STAGE 0  ·  TOKENIZER                  │
                    │  SentencePiece unigram, 32k joint vocab │
                    └────────────────────┬────────────────────┘
                                         │
      ┌──────────────────────────────────┼──────────────────────────────────┐
      │                                  │                                  │
      ▼                                  ▼                                  ▼
┌───────────┐                    ┌───────────────┐                  ┌──────────────┐
│ WikiText  │                    │  Gutenberg    │                  │ No Fear      │
│  ~103M w  │                    │  Early Modern │                  │ Shakespeare  │
│           │                    │  ~3-5M words  │                  │ 18,395 pairs │
└─────┬─────┘                    └───────┬───────┘                  └──────┬───────┘
      │                                  │                                 │
      ▼                                  ▼                                 ▼
┌───────────────────┐          ┌───────────────────┐          ┌───────────────────┐
│ STAGE 1           │          │ STAGE 2           │          │ STAGE 3           │
│ PRETRAIN          │─weights─▶│ DOMAIN ADAPT      │─weights─▶│ FINE-TUNE         │
│                   │          │                   │          │                   │
│ span corruption   │          │ span corruption   │          │ seq2seq,          │
│ 3 epochs          │          │ archaic English   │          │ teacher forcing   │
│ ~5-6 h  (T4)      │          │ ~1 h             │          │ ~10 min           │
│                   │          │                   │          │                   │
│ learns: words     │          │ learns: thou/     │          │ learns: the       │
│ + syntax          │          │ dost/-eth/-est    │          │ style mapping     │
└───────────────────┘          └───────────────────┘          └─────────┬─────────┘
                                                                        │
                                                                        ▼
                                                        ┌───────────────────────────┐
                                                        │ STAGE 4  ·  DECODE        │
                                                        │ beam 4-5, length α ≈ 1.5  │
                                                        └─────────────┬─────────────┘
                                                                      │
                                        ┌─────────────────────────────┴──────────────┐
                                        ▼                                            ▼
                          ┌───────────────────────────┐              ┌───────────────────────────┐
                          │ STAGE 5  ·  EVALUATE      │              │ STAGE 6  ·  BOOK PIPELINE │
                          │ BLEU vs copy baseline     │              │ parse → segment → batch   │
                          │ by-length · unk · dup     │              │ → decode → reassemble     │
                          └───────────────────────────┘              └───────────────────────────┘
```

---

## Stages

| # | Stage | Input | Output | Time | Status |
|---|---|---|---|---|---|
| 0 | Tokenizer | all corpora | `spm32k.model` | 2 min | config validated |
| 1 | Pretrain | ~200–300M words | base weights | 5–6 h | to build |
| 2 | Domain adapt | 3–5M archaic words | adapted weights | ~1 h | optional |
| 3 | Fine-tune | 18,395 pairs | translator | 10 min | **built** |
| 4 | Decode | trained model | translations | — | **built** |
| 5 | Evaluate | translations | BLEU + diagnostics | — | **built** |
| 6 | Book pipeline | a play | modern play | — | to build |

Times are for a Colab/Kaggle T4 or P100.

---

### Stage 0 — Tokenizer

SentencePiece unigram, **32k joint vocab**, trained on all corpora together. Special ids pinned to `pad=0, sos=1, eos=2, unk=3`.

Validated at 8k on the parallel corpus: `<unk>` **5.69% → 0.01%**, 2923/2924 exact round-trip, fertility 1.10–1.18 pieces per word.

```
thou      → ▁thou
speak'st  → ▁speak + ' + st
sweeting  → ▁sweet + ing
Montague  → ▁Mo + n + t + a + g + u + e     (was <unk>)
```

### Stage 1 — Pretrain

**Objective:** span corruption — mask contiguous spans in the encoder input, reconstruct them in the decoder. This is what teaches word meanings and syntax; they are not separable stages.

**Model:** d_model 384–512, 6 encoder + 6 decoder layers, 8 heads, d_ff 1536–2048 → **30–60M params**.

**Corpus:** WikiText-103 (~103M words) + a Gutenberg literature subset (~100–200M).

> Chinchilla-optimal for 300M tokens is ~15M params, so 60M is under-trained. Normal at this scale, and still transformative versus training on 18k pairs alone.

### Stage 2 — Domain adaptation *(optional)*

Same objective, archaic text only. Free and pre-1800:

| Source | Words |
|---|---|
| Shakespeare, complete works | ~885k |
| King James Bible | ~790k |
| Marlowe, Jonson, Spenser, Milton | ~1–2M |

~15× more archaic English than the parallel corpus contains.

### Stage 3 — Fine-tune

Teacher forcing on the 18,395 pairs. Decoder input `[<sos>] + tgt[:-1]`, labels `tgt[1:] + [<eos>]`.

Split is **by play** — 15 train, *Twelfth Night* valid, *Romeo and Juliet* test. Never reshuffle; lines from one play would leak across sides.

### Stage 4 — Decode

Beam 4–5, GNMT length penalty **α ≈ 1.5**.

> α=1.5, not the textbook 0.6 — measured. The model under-generates, so it needs a stronger length correction. At 0.6 beam gave +0.77 BLEU; at 1.5, +1.40.

### Stage 5 — Evaluate

| Metric | Why |
|---|---|
| BLEU vs **copy baseline** | echoing the input already scores 19.22; BLEU alone is meaningless here |
| BLEU by source length | overall BLEU hides collapse on long sentences |
| `<unk>` rate | each one is a guaranteed miss that breaks surrounding n-grams |
| Adjacent duplicate rate | catches "then then" decoding loops |

### Stage 6 — Book pipeline

```
raw play
  → classify lines (heading / stage direction / SPEAKER: / dialogue)
  → segment dialogue into sentences
  → split anything over max_len on ; : ,
  → sort by length, batch, beam decode
  → restore order, rejoin, detokenise
  → reassemble with original structure
```

---

## Current results

| | valid | test |
|---|---|---|
| **Copy baseline** | **15.97** | **19.22** |
| From-scratch, 8.1M params, no pretraining | 14.20 | 15.07 |
| From-scratch, 4.0M params | 13.67 | 14.77 |

Measured diagnostics on the 8.1M model:

| Source tokens | n | Model | Copy |
|---|---|---|---|
| 1–5 | 368 | 25.84 | 28.12 |
| 6–10 | 550 | 18.87 | 24.33 |
| 11–15 | 252 | 17.94 | 22.03 |
| 16–25 | 200 | 10.32 | 15.44 |
| 26+ | 92 | 6.18 | 10.11 |

- `<unk>` **3.65%** of output tokens
- adjacent duplicates **0.85%** vs 0.02% in references
- copies **61.8%** of tokens from source; references copy 54.3% — it over-copies, and its edits are wrong

**Why Stage 1 exists:** 474,478 words of parallel data against 8.1M parameters is 0.06 words per parameter. The rule of thumb is ~20. That model needs ~160M words — a **340× shortfall**. No architecture change closes that; only pretraining does.

---

## Repository

```
src/
  config.py         hyperparameters, special token ids
  masking.py        padding + causal masks
  positional.py     sinusoidal encoding
  attention.py      multi-head attention (all three uses)
  layers.py         encoder / decoder blocks, pre-norm
  transformer.py    full model, tied embeddings, pointer-generator
  vocab.py          word-level vocabulary
  dataset.py        corpus loading, bucketed batching
  train.py          teacher forcing, fit loop, checkpointing
  inference.py      greedy + beam search, corpus translation
  evaluate.py       BLEU, copy baseline
  toy_data.py       copy/reverse tasks

notebooks/
  00_transformer_foundation.ipynb   architecture, verified
  01_toy_copy_task.ipynb            proves it learns
  02_shakespeare_data.ipynb         corpus + baseline
  03_train_shakespeare.ipynb        training

data/       corpus + vocab.json
checkpoints/
```

---

## Rules that keep the numbers honest

1. Never reshuffle the by-play split.
2. Tune decoding on **validation**; touch test once.
3. Recompute the copy baseline in whatever tokenization the model outputs.
4. Report BLEU **next to** that baseline, never alone.
5. Overfit one batch before every long run.
