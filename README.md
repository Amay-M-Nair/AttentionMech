# AttentionMech — a Latin→English translator, built from scratch

A transformer encoder–decoder implemented from the ground up following
*Attention Is All You Need* — no `nn.Transformer`, no pretrained weights — and
trained to translate Latin into English.

**Test BLEU 18.87 against a copy baseline of 0.31.** Every component, from the
attention mechanism to the beam search, is in `src/`.

---

## Result

Test set decoded once, at the end, with the beam and length penalty chosen on
validation. 3,003 sentences, fp32.

| | BLEU | chrF |
|---|---|---|
| copy baseline (echo the Latin) | 0.31 | 19.74 |
| **model** | **18.87** | **42.78** |

Decoding: beam 8, length penalty α = 1.5, tuned on 500 held-out validation
sentences. Checkpoint from epoch 60, validation BLEU 18.48.

### Where it works, and where it doesn't

The corpus is about a third Vulgate scripture and two thirds classical prose.
Splitting the test set on that line is the most informative thing in this
report:

| split | n | copy | BLEU | chrF |
|---|---|---|---|---|
| Vulgate | 984 | 0.45 | **40.37** | 61.00 |
| classical (251 works) | 2,019 | 0.19 | **10.36** | 37.16 |

A **four-fold gap**. Scripture is formulaic, repetitive, and heavily represented;
Cicero, Tacitus and Apuleius are none of those things. A single overall number
hides this completely, which is the argument for reporting the split.

Performance also decays with sentence length, as expected for a model without
a copy mechanism translating a free-word-order language:

| source tokens | n | copy | BLEU | chrF |
|---|---|---|---|---|
| 1–20 | 729 | 0.74 | 26.01 | 46.38 |
| 21–35 | 1,019 | 0.39 | 25.89 | 47.13 |
| 36–55 | 733 | 0.11 | 18.84 | 42.90 |
| 56+ | 522 | 0.17 | 11.04 | 38.76 |

Full translations for every test sentence: `outputs/test_predictions.en`.

---

## Why Latin, and not the original plan

The project began as a Shakespeare→modern-English translator. Eight phases
established that the task was unwinnable, and the reason is worth recording.

**Echoing the input unchanged scores 19.22 BLEU**, because source and target are
the same language sharing 91.5% of their tokens. Nothing built ever beat it:

| | test BLEU |
|---|---|
| **copy baseline** | **19.22** |
| from scratch, 8.1M params | 15.07 |
| smaller model, 4.0M | 14.77 |
| pretrained on 272M words, then fine-tuned | 15.18 |

Pretraining fixed every diagnosed symptom — `<unk>` 3.65% → 0.000%, adjacent
duplicates 0.85% → 0.09%, long-sentence BLEU 6.18 → 7.29 — and moved the score
by **+0.11**. It made the model fluent without making it accurate, and on a task
where copying scores 19.22, a confident wrong rewrite costs more than a clumsy
one. chrF confirmed rather than rescued it: 40.86 for copying against 31.34 for
the model.

There was also no more data: 18,395 pairs is essentially every Shakespeare
modernisation that exists.

Latin removes both problems. The copy baseline is **0.31**, so the architecture
can finally be *evaluated*. The same score means opposite things in the two
tasks — 15 was a failure at Shakespeare and would be a clear success here.

The architecture was never the problem. It was proven on toy copy/reverse tasks
in Phase 2 and carried into the Latin work unchanged.

---

## Architecture

`src/` — every file is hand-written and small enough to read in one sitting.

| module | contents |
|---|---|
| `masking.py` | padding and causal masks |
| `positional.py` | sinusoidal position encoding |
| `attention.py` | multi-head attention — one class serving encoder self-, decoder causal self-, and cross-attention |
| `layers.py` | encoder and decoder blocks, pre-norm residuals |
| `transformer.py` | the full model, tied embeddings |
| `spm_tokenizer.py` | SentencePiece wrapper, special ids pinned |
| `dataset.py` | parallel corpus, length-bucketed batching |
| `latin_data.py` | download, filter, deduplicate, split |
| `train.py` | teacher forcing, `fit()`, checkpointing, early stopping |
| `inference.py` | greedy and beam search with GNMT length penalty |
| `evaluate.py` | BLEU and chrF, always against the copy baseline |

Final configuration — **19,619,328 parameters**:

```
d_model 384 · 8 heads · 4 encoder + 4 decoder layers · d_ff 1536
vocab 8,000 (joint) · dropout 0.3 · label smoothing 0.1 · max_len 128
```

Teacher forcing convention, built into the batch rather than sliced in the loop:

```
src     = tokens + <eos>
tgt_in  = <sos> + tokens
tgt_out = tokens + <eos>
```

---

## Pipeline

### L1 — Data · `src/latin_data.py`

[`grosenthal/latin_english_translation`](https://huggingface.co/datasets/grosenthal/latin_english_translation),
filtered and re-split.

Two silent defects were found and fixed:

- **0.3% of rows have Latin in the English column** — two Vulgate versions
  aligned to each other. Filtered by requiring at least one English stopword on
  the target side.
- **Duplicate pairs across splits.** Deduplication happens *before* splitting;
  doing it after leaves 20 pairs in both train and eval, quietly inflating every
  score that follows.

| split | rows |
|---|---|
| train | 94,099 |
| valid | 3,003 |
| test | 3,003 |

`{split}.src` records the source work per row — this is what makes the
Vulgate/classical breakdown possible.

### L2 — Tokenizer · `notebooks/02_latin_tokenizer.ipynb`

Joint SentencePiece unigram over both languages, **8,000 pieces**.

| | |
|---|---|
| round-trip failures | **0 / 12,012** |
| `<unk>` rate | **0.0000%** both languages |
| fertility | Latin **1.84**, English **1.39** |

8k rather than 32k on purpose: at ~94k pairs a 32k vocabulary means most
subwords are seen a handful of times, and an untrainable embedding table is
itself a cause of overfitting. Small vocabularies are standard in low-resource
NMT for this reason.

`byte_fallback` earned its 256 slots — without it one test line failed to
round-trip on a stray `|` absent from all 94,099 training lines. The tokenizer
must be trained on train only, so that failure mode is structural, not a fluke.

Latin's higher fertility is what inflection predicts: one Latin word carries
case, number and gender that English spreads across several.

### L3 — Training · `notebooks/03_latin_train.ipynb`

Selection on validation **BLEU**, not loss — the two diverge, and BLEU is what
gets reported. Early stopping on patience 8. Every long run is preceded by an
overfit-one-batch gate, the check that caught every wiring bug in this project.

Two runs, and the difference between them is one hyperparameter:

| | run 1 | run 2 |
|---|---|---|
| `warmup` | 400 | **4000** |
| epochs | 40 | 60 |
| **validation BLEU** | 14.51 | **18.48** |

The scheduler is inverse-sqrt: `lr = base × sqrt(warmup / step)`. With
`warmup=400` and 2,941 steps per epoch, the rate had decayed to **2.9e-5** by
epoch 40 while BLEU was still climbing — the model was rate-limited, not
converged. `warmup=4000` holds it ~3.2× higher throughout and bought **+4
BLEU**.

`dropout` was deliberately left at 0.3 as a control. Changing three things at
once would have explained nothing.

### L4 — Evaluation · `notebooks/04_latin_evaluate.ipynb`

Beam size swept at fixed penalty, then the penalty at the winning beam — ten
decodes instead of twenty-five, and the two interact weakly.

| beam (α=1.0) | BLEU | | α (beam 8) | BLEU |
|---|---|---|---|---|
| 1 | 17.65 | | 0.6 | 18.15 |
| 4 | 18.31 | | 1.0 | 18.33 |
| **8** | **18.33** | | **1.5** | **18.49** |

α ≈ 1.5, not the textbook 0.6 — matching what was measured on the Shakespeare
model. Beam beyond 4 buys almost nothing (+0.02).

Test was then decoded **once**, in fp32 so the number is exactly reproducible.

---

## Out-of-domain: Newton's *Principia*

The model was pointed at the Definitions and Laws of Motion from the *Principia*
(1687, public domain). It fails, and the failure is diagnostic.

```
Lex I   Corpus omne perseverare in statu suo quiescendi vel movendi
        uniformiter in directum, nisi quatenus a viribus impressis
        cogitur statum illum mutare.

model   The Corpus is forced to continue in his own state, or to move on
        the right, unless it is compelled by any strength to change the
        statue from the strength of the body.
```

Occurrences of Newton's vocabulary in the 1.96M-word training corpus:

```
centripeta 0    parallelogrammi 0    diagonalem 0
proportionalem 0    reactionem 0    quantitas 3    densitate 14
                    …but dominus 3,057, deus 2,219, rex 1,741
```

The words the physics is *made of* are absent, so they shatter into subwords the
model never learned to compose. And the Vulgate bias is visible in the failures —
given nothing to go on, the model guesses in a biblical register: *vis
centripeta* becomes "**Scripture**".

This is not the model underperforming its measurement. 18.87 on held-out data
from its own distribution is real; 1687 mathematical Neo-Latin is a different
distribution. It is exactly what a test set drawn from the same source as
training cannot tell you.

Full output: `outputs/newton_principia.md`.

---

## Performance engineering

Training started at ~12 minutes per epoch and finished at 3–4. Four changes,
each measured:

| change | why |
|---|---|
| **Mixed precision** in `fit()` | `autocast` + `GradScaler`, ~2× on training and decoding alike |
| **Pre-encode the dataset** | `__getitem__` was calling SentencePiece on every access — 188k encodes *per epoch*, on the main process, in series with the GPU |
| **Accumulate metrics on the GPU** | `.item()` twice per step is 5,882 synchronisation stalls per epoch |
| **Bucket by token length** | batches were grouped by word count while padding is by *token* count; at Latin's fertility of 1.84 the two are not proportional |

Validation was also bounded. Beam search holds `batch × beam` sequences at once,
so the batch scales inversely with the beam; and `translate_corpus` truncates
sources to the length the model trained on. Without that, validation fed the
model 372-token sequences when training had capped it at 128 — the source of a
CUDA OOM that training itself, peaking at 0.90 GB, never came close to.

---

## Layout

```
src/          the model, data, training, decoding, metrics
notebooks/    00 foundation · 01 toy tasks · 02 tokenizer · 03 train · 04 evaluate
data/         Latin/English splits + spm8k tokenizer
outputs/      results, curves, all 3,003 test translations
docs/         pipeline notes, paper summary
checkpoints/  weights (gitignored)
```

## Running it

```bash
pip install -r requirements.txt
python -c "from src.latin_data import build; build('data')"
```

Then `02` → `03` → `04` in order. The notebooks find the repo by walking up from
the kernel's working directory, so they run from anywhere, and `03` clones and
installs its own dependencies on Kaggle.

Training takes ~2.5 h on a T4. Evaluation takes 15–30 min on a laptop GPU.

---

## What would improve it

Ordered by expected value, and each one follows from a measurement above.

1. **Lower the dropout.** Two runs, 100 epochs, and validation loss never turned
   upward — the model has never once overfit. `dropout=0.3` was insurance
   against a Shakespeare-style collapse that never came, and it is now costing
   capacity. 0.15 is the obvious next trial.
2. **Train longer.** Patience 8 never fired in either run; both stopped at their
   epoch cap while still improving. The terminal slope was ~0.05 BLEU/epoch.
3. **A KV cache in `step_logprobs`.** The decoder re-runs the entire prefix at
   every generation step, making decoding quadratic in output length. This is
   the single largest speed win available and would make wider beams cheap.
4. **Attack the classical/Vulgate gap directly** — it is 40.37 vs 10.36, and
   the overall number is mostly the easy third. Oversampling classical prose, or
   simply more of it, targets what is actually broken.
5. **More data.** 94k pairs is small. The levers beyond it are a larger parallel
   corpus or back-translation of monolingual Latin.

Not worth doing: a bigger vocabulary, or a bigger model, without more data
first. Both were tried on the Shakespeare task and neither moved the number.
