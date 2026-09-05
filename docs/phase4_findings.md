# Phase 4 findings — from-scratch training

**Result: the model lost to doing nothing.** This is the measurement that justifies pretraining.

Notebook: [`03_train_shakespeare.ipynb`](../notebooks/03_train_shakespeare.ipynb)

---

## Headline

| | valid | test |
|---|---|---|
| **Copy baseline** (echo the input unchanged) | **15.97** | **19.22** |
| Trained model | 14.20 | **15.07** |
| Gap | −1.77 | **−4.15** |

Config: d_model 256, 4 heads, 3 encoder + 3 decoder layers, d_ff 1024, dropout 0.2, label smoothing 0.1, AdamW + warmup, **8,111,872 params**. Trained 36 epochs, early-stopped on validation BLEU, greedy decoding.

The overfit-one-batch gate passed first (loss 9.25 → 0.61, 99.6% accuracy), so the pipeline was correct. The failure is not a bug.

---

## Diagnostics

### It loses at every sentence length

| Source tokens | n | Model | Copy | Delta |
|---|---|---|---|---|
| 1–5 | 368 | 25.84 | 28.12 | −2.28 |
| 6–10 | 550 | 18.87 | 24.33 | −5.46 |
| 11–15 | 252 | 17.94 | 22.03 | −4.10 |
| 16–25 | 200 | 10.32 | 15.44 | −5.12 |
| 26+ | 92 | 6.18 | 10.11 | −3.93 |

Absolute quality collapses **4×** from short to long. Short sentences are merely *closest* to the baseline, not good.

### `<unk>` destroys long sentences

**3.65%** of output tokens were `<unk>`. Each is a guaranteed miss that also breaks the n-grams around it. The longest test sentence (60 tokens) decoded to:

> The `<unk>` of them will be `<unk>`, and the work of their fortunes, which will `<unk>`, which of their fortunes have to `<unk>`, which we could have made the `<unk>` of our ears.

### Repetition loops

Adjacent duplicate tokens: **0.85%** against **0.02%** in the references — 40× the human rate. Visible as "and then then we'll agree" in sample output. A greedy-decoding pathology.

### It over-copies, and its edits are wrong

| Per-sentence token copy rate | valid | test |
|---|---|---|
| Reference translations | 50.5% | 54.3% |
| Model | **66.1%** | **61.8%** |

The model echoes the source *more* than a good translation should, and still scores below pure copying. So the problem is not too little copying — **when it deviates from the source, it deviates wrongly.** Its edits break more n-grams than they earn.

Other decoding health checks were fine: length ratio 1.02 (no truncation or rambling), sentence-level copying calibrated (6.2% verbatim vs 6.8% for references). Only 5.7% of sentences matched the reference exactly.

### It overfits before reaching the baseline

| Epoch | Train loss/acc | Valid loss/acc | BLEU |
|---|---|---|---|
| 6 | 4.19 / 42.5% | 4.38 / 39.4% | 7.15 |
| 21 | 3.22 / 56.7% | **4.110** / 45.6% | 12.50 |
| 28 | 3.01 / 59.7% | 4.148 / 45.7% | **13.34** |
| 36 | 2.83 / 62.7% | 4.187 / 45.8% | 13.03 |

Validation loss bottoms at epoch 21 and rises after, while training loss keeps falling. The train/valid accuracy gap widens from 3 points to **17**.

---

## Follow-up experiments

**Smaller model made it worse.** d_model 192, 2 layers, 4.0M params: valid 13.67, test **14.77**, `<unk>` up to 4.21%, 26+ tokens down to 5.12. So overfitting was **not** the binding constraint — removing capacity cost more than it saved.

**Beam search helps, but the textbook length penalty is wrong here.** Tuned on validation:

| beam | α | BLEU | avg length |
|---|---|---|---|
| greedy | — | 11.69 | 10.8 |
| 4 | 0.6 | 12.46 | 9.5 |
| 4 | **1.5** | **13.05** | 10.4 |
| 5 | **1.5** | **13.09** | 10.3 |

**+1.4 BLEU at α≈1.5**, not the usual 0.6 — this model under-generates, so it needs a stronger length correction. Beam also cut repetition 1.41% → 0.34%, a 4× reduction, while barely moving BLEU. That says the bottleneck is the model's probability distribution, not the search over it.

---

## Why it failed

| | |
|---|---|
| Parallel corpus | **474,478 words** |
| Model | 8,111,872 params |
| Words per parameter | **0.06** |
| Rule of thumb (Chinchilla) | ~20 |
| Data needed for this model | ~160M words |
| **Shortfall** | **~340×** |

The model had to learn three things at once from 18,395 sentence pairs:

1. What English words mean
2. How English sentences are built
3. The Shakespeare→modern style mapping

Concretely: `nn.Embedding(10119, 256)` is **2.6M parameters learned from 474k word occurrences**. Thousands of those rows saw two or three examples. A 256-dimensional vector cannot be learned from two examples — which is also the direct cause of the 3.65% `<unk>`.

No architectural change fixes this, and three attempts confirmed it: a smaller model was worse, beam search bought 1.4 points, and a pointer-generator was built but never going to close a 4-point gap on its own.

**The fix is to learn 1 and 2 from a large unlabeled corpus first, so the 18k pairs only have to teach 3.** That is Phases 5–8.

---

## What carries forward

- **Copy baseline 19.22 (test), 15.97 (valid)** — still the number to beat. Computed on raw text, so it survives the tokenizer change.
- **Beam α ≈ 1.5**, not 0.6.
- **Select on validation BLEU, not validation loss** — they diverge; loss bottomed at epoch 21 while BLEU peaked at 28.
- **Overfit one batch before every long run.**
- The diagnostics in this document are the comparison set for every later phase.
