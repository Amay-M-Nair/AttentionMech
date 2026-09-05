# Pipeline

Shakespeare → modern English translation, built on the architecture from *Attention Is All You Need*, then pretrained so it has enough language to work with.

---

## Why there is a pretraining stage

Phase 4 trained the architecture directly on the parallel corpus and **lost to echoing the input**: test BLEU 15.07 against a 19.22 copy baseline. ([full analysis](phase4_findings.md))

Not a bug — a data shortfall, stateable exactly:

| | |
|---|---|
| Parallel corpus | **474,478 words** |
| Model | 8,111,872 params |
| Words per parameter | **0.06** |
| Rule of thumb (Chinchilla) | ~20 |
| **Shortfall** | **~340×** |

The model had to learn what English words mean, how English sentences are built, *and* the Shakespeare→modern mapping — all from 18,395 sentence pairs. Pretraining moves the first two off the parallel corpus, so the 18k pairs only have to teach the third.

---

## Overview

```
   ┌──────────────────────┐              ┌──────────────────────┐
   │  Gutenberg English   │              │  No Fear Shakespeare │
   │  ~200M words         │              │  18,395 pairs        │
   │  (streamed)          │              │  split by play       │
   └──────────┬───────────┘              └───────────┬──────────┘
              │                                      │
              ▼                                      │
   ┌──────────────────────────────────────┐          │
   │  PHASE 5 · TOKENIZER          ✅     │          │
   │  SentencePiece unigram, 32k          │◀─────────┤ trained on both,
   │  + 100 sentinels, + 2 direction      │          │ so the corpus
   │  unk 5.69% → 0.00%, 0 RT failures    │          │ round-trips exactly
   └──────────┬───────────────────────────┘          │
              │                                      │
              ▼                                      │
   ┌──────────────────────────────────────┐          │
   │  PHASE 6 · INFRASTRUCTURE            │          │
   │  span corruption · AMP               │          │
   │  grad accumulation · resume          │          │
   └──────────┬───────────────────────────┘          │
              │                                      │
              ▼                                      │
   ┌──────────────────────────────────────┐          │
   │  PHASE 7 · PRETRAIN        ~0.9 h    │          │
   │  37M params, seq 256                 │          │
   │  learns words + syntax               │          │
   └──────────┬───────────────────────────┘          │
              │ weights                              │
              ▼                                      ▼
   ┌──────────────────────────────────────────────────────────┐
   │  PHASE 8 · FINE-TUNE + EVALUATE            ~30 min       │
   │  teacher forcing · beam 4-5, α ≈ 1.5                     │
   │  TARGET: beat 19.22                                      │
   └──────────┬───────────────────────────────────────────────┘
              │
      ┌───────┴────────┐
      ▼                ▼
┌───────────┐   ┌──────────────┐
│ PHASE 9   │   │  PHASE 10    │
│ book      │   │  multilingual│
│ pipeline  │   │  6 languages │
└───────────┘   └──────────────┘
```

---

## Phases

| # | Phase | Deliverable | Time | Status |
|---|---|---|---|---|
| 1 | Model foundation | architecture, verified | — | ✅ |
| 2 | Prove it learns | toy copy/reverse, diagonal attention | — | ✅ |
| 3 | Real data | corpus, vocab, **copy baseline 19.22** | — | ✅ |
| 4 | Train from scratch | 15.07 — below baseline | — | ✅ negative result |
| 5 | Tokenizer + corpus | SentencePiece 32k, Gutenberg stream | ~45 min | ✅ |
| 6 | Infrastructure | span corruption, AMP, resume | ~1 h | ✅ |
| **7** | **Pretrain** | a model that knows English | ~0.9 h | **next** |
| 8 | Fine-tune + evaluate | **beat 19.22** | ~30 min | |
| 9 | Book pipeline | translate a whole play | ~2 h | |
| 10 | Multilingual | de/fr/es/hi/ml/ta | later | |

Times assume a Kaggle T4 or P100, and Phase 7 is measured (24.5k corpus tok/s on
an RTX 3050, ~2.5x that on a T4) rather than estimated. **~1.5 h from here to a
real number.**

---

### Phase 5 — Tokenizer and corpus ✅

`src/spm_tokenizer.py` · `src/corpus.py` · `data/spm32k.model`

SentencePiece unigram, 32k joint vocab, trained on Gutenberg **and** the parallel corpus so the latter round-trips exactly. Special ids pinned to `pad=0, sos=1, eos=2, unk=3`; sentinels at 6–105; direction tokens at 4–5.

Corpus is `sedthh/gutenberg_english` — English Gutenberg as parquet, streamed rather than downloaded. *(PG-19 is unusable: it ships as a loading script, dropped by `datasets` 5.x.)*

**Gate results:**

| | word-level (Phase 3) | subword (Phase 5) |
|---|---|---|
| Round-trip exact | — | **0 failures / 8,924** |
| `<unk>` on test source | 5.69% | **0.0000%** |
| Fertility (pieces/word) | — | **1.07–1.08** |
| Vocabulary | 10,119 words | 32,000 subwords |

```
thou      → ▁thou              hath      → ▁hath
speak'st  → ▁speak + ' + st    sweeting  → ▁sweet + ing
Montague  → ▁Montague          wherefore → ▁wherefore
```

`Montague` was 7 pieces at a 8k trial and is 1 here — Gutenberg contains Shakespeare.

### Phase 6 — Infrastructure ✅

`src/denoising.py` · `pretrain()` in `src/train.py`

**Span corruption** (T5's objective): mask 15% of tokens in spans of mean length 3. Encoder sees each span replaced by one sentinel; decoder reconstructs only the missing spans.

```
original   The cat sat on the mat and purred
encoder    The <extra_id_0> on the mat <extra_id_1> purred
decoder    <extra_id_0> cat sat <extra_id_1> and <extra_id_2>
```

Filling blanks forces word meanings *and* syntax out of raw text, with no labels. They are not separable stages — one objective teaches both.

Four gaps in the current trainer:

| Gap | Why |
|---|---|
| Step-based `pretrain()` | `fit()` is epoch-based; epochs are meaningless over 200M words |
| Mixed precision | **~2× faster** on a T4 |
| Gradient accumulation | effective batch 128+ without the VRAM |
| **Resume with optimizer state** | `save_checkpoint` stores only weights; a resumed run would restart the LR schedule |

**Gate results:** corruption verified over 300 random sequences (sentinel counts, length reconstruction, no invented tokens); density 0.148 vs 0.150 target; loss 8.23 → 5.12 on a real batch; resume continues without a reset spike.

### Phase 7 — Pretrain

d_model 384, 6 encoder + 6 decoder layers, 8 heads, d_ff 1536, vocab 32k → **~37M params**, seq len 256. Scaling is a config change; the architecture is untouched.

**Gate:** overfit one batch first — a 37M model reaches loss ~0.7 at 93% accuracy in 600 steps. Then perplexity falling from ~32,000; checkpoint every 1,000 steps with optimizer state; finally mask a span in a held-out sentence and confirm the fill is plausible English.

### Phase 8 — Fine-tune and evaluate

Load Phase 7 weights, teacher forcing on the 18k pairs. Split stays **by play** — 15 train, *Twelfth Night* valid, *Romeo and Juliet* test.

Decode with beam 4–5 and length penalty **α ≈ 1.5** — measured, not the textbook 0.6.

**Gate**, against the Phase 4 numbers:

| Metric | Phase 4 | Target |
|---|---|---|
| Test BLEU | 15.07 | **> 19.22** |
| BLEU at 26+ tokens | 6.18 | > 10.11 |
| `<unk>` in output | 3.65% | ~0% |
| Adjacent duplicates | 0.85% | ~0.02% |

### Phases 9–10

**9 — Book pipeline.** Classify lines (heading / stage direction / `SPEAKER:` / dialogue), segment, split long lines on `; : ,`, batch-decode, reassemble with the original structure.

**10 — Multilingual.** Joint SentencePiece across en/de/fr/es/hi/ml/ta, `<2xx>` as the first decoder token, temperature-balanced sampling so low-resource pairs are not drowned.

---

## Tokenization vs embeddings

Two different things. SentencePiece does only the first.

```
"Thy wit is bitter"
  ├─ 1. TOKENIZE (SentencePiece)   →  [▁thy, ▁wit, ▁is, ▁bitter]
  ├─ 2. LOOKUP IDS                 →  [412, 891, 25, 3301]
  ├─ 3. nn.Embedding(32000, 384)   →  4 × 384 vectors    ← STATIC, word2vec-like
  ├─ 4. + positional encoding
  └─ 5. 6 encoder layers           →  4 × 384 vectors    ← CONTEXTUAL
```

**No word2vec or GloVe.** GloVe is word-level and this vocabulary is subword; learned jointly, embeddings live in the space the attention layers expect; and step 5 dominates anyway. Step 3 gives `bitter` one fixed vector forever — after the encoder, `bitter` in "bitter apple" and "bitter enemy" differ, because self-attention mixed in the context. That is what attention buys.

---

## Repository

```
src/
  config.py         hyperparameters, special token ids
  masking.py        padding + causal masks
  positional.py     sinusoidal encoding
  attention.py      multi-head attention (all three uses)
  layers.py         encoder / decoder blocks, pre-norm
  transformer.py    full model, tied embeddings

  spm_tokenizer.py  SentencePiece subwords            ← Phase 5
  corpus.py         Gutenberg streaming + token file  ← Phase 5
  vocab.py          word-level vocab (Phases 3-4)
  dataset.py        parallel corpus, bucketed batching
  toy_data.py       copy / reverse tasks

  train.py          teacher forcing, fit loop, checkpointing
  inference.py      greedy + beam search, corpus translation
  evaluate.py       BLEU against the copy baseline

notebooks/
  00_transformer_foundation.ipynb   architecture, verified
  01_toy_copy_task.ipynb            proves it learns
  02_shakespeare_data.ipynb         corpus + copy baseline
  03_pretrain.ipynb                 pretraining run (Kaggle)

docs/
  pipeline.md            this file
  phase4_findings.md     why from-scratch failed
  notes.md               week 1-2 working notes
```

---

## Rules that keep the numbers honest

1. Never reshuffle the by-play split.
2. Tune decoding on **validation**; touch test once.
3. Recompute the copy baseline in whatever tokenization the model outputs.
4. Report BLEU **next to** that baseline, never alone.
5. Overfit one batch before every long run.
