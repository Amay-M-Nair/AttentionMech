"""
Phase 1: The Transformer Foundation

Walks through every piece of the model, smallest first, and checks each one
before moving on. No data and no training yet - this only answers one
question: is the architecture correct?

Run it:
    python notebooks/04_transformer_foundation.py
    python notebooks/04_transformer_foundation.py --no-plots
"""

import os
import sys

# Make `src` importable when running this file directly from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHOW_PLOTS = "--no-plots" not in sys.argv

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.config import TransformerConfig, get_device
from src.layers import DecoderLayer, EncoderLayer, FeedForward
from src.masking import make_causal_mask, make_decoder_mask, make_pad_mask
from src.positional import PositionalEncoding
from src.transformer import Transformer, build_model


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


torch.manual_seed(0)  # same numbers every run, so you can compare across runs

print("Phase 1: Transformer Foundation")
print("Device:", get_device())


# ---------------------------------------------------------------------------
section("PART 1: Masks - who is allowed to look at whom")
# ---------------------------------------------------------------------------

# Two sentences. 0 is <pad>, so sentence 1 is 3 real words then 2 pads.
tokens = torch.tensor([
    [5, 8, 2, 9, 4],   # 5 real words
    [7, 3, 6, 0, 0],   # 3 real words, 2 padding
])
print("Token indices (0 = <pad>):")
print(tokens)

pad_mask = make_pad_mask(tokens, pad_idx=0)
print("\nPadding mask, shape", tuple(pad_mask.shape), " (batch, 1, 1, seq_len)")
print("The two size-1 dimensions are there to broadcast over heads and queries.")
print("Sentence 0:", pad_mask[0, 0, 0].int().tolist())
print("Sentence 1:", pad_mask[1, 0, 0].int().tolist(), " <- last two blocked")

causal_mask = make_causal_mask(5)
print("\nCausal mask, shape", tuple(causal_mask.shape), " (1, 1, seq_len, seq_len)")
print("Row = who is looking, column = what they can see:")
for i, row in enumerate(causal_mask[0, 0].int().tolist()):
    print(f"  position {i} sees {row}")
print("Lower-triangular: nobody sees the future.")

dec_mask = make_decoder_mask(tokens, pad_idx=0)
print("\nDecoder mask (padding AND causal), shape", tuple(dec_mask.shape))
print("Sentence 1 - notice columns 3 and 4 are blocked for EVERY row,")
print("on top of the triangle:")
for i, row in enumerate(dec_mask[1, 0].int().tolist()):
    print(f"  position {i}: {row}")


# ---------------------------------------------------------------------------
section("PART 2: Positional encoding - where the sense of order comes from")
# ---------------------------------------------------------------------------

d_model = 64
pos_enc = PositionalEncoding(d_model, max_len=100, dropout=0.0)

dummy = torch.zeros(1, 20, d_model)     # zeros, so we see the encoding alone
encoded = pos_enc(dummy)
print("Input shape :", tuple(dummy.shape))
print("Output shape:", tuple(encoded.shape), " (unchanged - it is an addition)")

pe = pos_enc.pe[0]  # (max_len, d_model)
print("\nValue range: [%.3f, %.3f]  <- always within [-1, 1]" % (pe.min(), pe.max()))
print("So embeddings scaled by sqrt(d_model) are never swamped by it.")

print("\nIs it in parameters()? ", any(p is pos_enc.pe for p in pos_enc.parameters()))
print("It is a registered buffer: moves with .to(device), saved in state_dict,")
print("but never updated by the optimiser.")

sim = torch.nn.functional.cosine_similarity(pe[5].unsqueeze(0), pe[:20], dim=-1)
print("\nCosine similarity of position 5 against positions 0-19:")
print("  ", " ".join(f"{v:.2f}" for v in sim.tolist()))
print("Peaks at 1.00 for position 5 and falls off nearby - which is exactly")
print("the signal the model needs to recover distance between words.")

plt.figure(figsize=(10, 4))
plt.imshow(pe[:50].T.numpy(), cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="value")
plt.xlabel("Position in sentence")
plt.ylabel("Embedding dimension")
plt.title("Sinusoidal Positional Encoding\n(fast waves at the top, slow at the bottom)")
plt.tight_layout()
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()


# ---------------------------------------------------------------------------
section("PART 3: Multi-head attention - one class, three jobs")
# ---------------------------------------------------------------------------

batch, src_len, tgt_len, d_model, num_heads = 2, 5, 7, 64, 4
mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
mha.eval()  # turn dropout off so the numbers below are exact

print(f"d_model={d_model}, num_heads={num_heads}, head_dim={mha.head_dim}")
print("Each head works in 16 dimensions; 4 x 16 = 64, the arithmetic is the")
print("same as one 64-dim head, but each head can specialise.")

src = torch.randn(batch, src_len, d_model)
tgt = torch.randn(batch, tgt_len, d_model)

# --- Job 1: encoder self-attention (padding mask only) ---
out, weights = mha(src, src, src, pad_mask)
print("\n[1] ENCODER SELF-ATTENTION   Q = K = V = source")
print("    output ", tuple(out.shape), " <- same shape as the input")
print("    weights", tuple(weights.shape), " (batch, heads, query, key)")
print("    rows sum to 1?", torch.allclose(weights.sum(-1), torch.ones(batch, num_heads, src_len)))
blocked = weights[1, :, :, 3:]
print("    attention paid to sentence 1's padding: %.10f  <- exactly zero" % blocked.max())

# --- Job 2: decoder self-attention (causal mask) ---
tgt_causal = make_causal_mask(tgt_len)
out2, weights2 = mha(tgt, tgt, tgt, tgt_causal)
print("\n[2] DECODER SELF-ATTENTION   Q = K = V = target, causal mask")
print("    output ", tuple(out2.shape))
upper = torch.triu(weights2[0, 0], diagonal=1)
print("    weight above the diagonal: %.10f  <- the future is invisible" % upper.max())

# --- Job 3: cross-attention (different lengths) ---
out3, weights3 = mha(tgt, src, src, pad_mask)
print("\n[3] CROSS-ATTENTION          Q = target, K = V = source")
print("    query length %d, key length %d  <- DIFFERENT, and that is the point" % (tgt_len, src_len))
print("    output ", tuple(out3.shape), " <- follows the QUERY length")
print("    weights", tuple(weights3.shape), " (batch, heads, tgt_len, src_len)")
print("    This is the alignment matrix: for each target word, how much of each")
print("    source word it used. Visualising it later shows the translation.")

print("\nThe same object served all three. Only the arguments changed.")


# ---------------------------------------------------------------------------
section("PART 4: Encoder and decoder blocks")
# ---------------------------------------------------------------------------

d_ff = 256
ff = FeedForward(d_model, d_ff, dropout=0.0)
print("FeedForward: %d -> %d -> %d" % (d_model, d_ff, d_model))
print("  output shape:", tuple(ff(src).shape), " (per position, independently)")

enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.0).eval()
enc_out, enc_w = enc_layer(src, pad_mask)
print("\nEncoderLayer - 2 sublayers (self-attn, feed-forward)")
print("  in ", tuple(src.shape), "-> out", tuple(enc_out.shape))

dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout=0.0).eval()
dec_out, self_w, cross_w = dec_layer(tgt, enc_out, tgt_causal, pad_mask)
print("\nDecoderLayer - 3 sublayers (causal self-attn, CROSS-attn, feed-forward)")
print("  in ", tuple(tgt.shape), "-> out", tuple(dec_out.shape))
print("  self-attention weights :", tuple(self_w.shape), " target looking at itself")
print("  cross-attention weights:", tuple(cross_w.shape), " target looking at source")

print("\nBoth blocks preserve shape, which is what lets them stack N deep.")


# ---------------------------------------------------------------------------
section("PART 5: The full model")
# ---------------------------------------------------------------------------

config = TransformerConfig(vocab_size=10000)
model = build_model(config)
print(config)

src_tokens = torch.tensor([
    [4, 17, 25, 9, 33],
    [8, 12, 40, 0, 0],   # padded
])
tgt_tokens = torch.tensor([
    [1, 6, 19, 27, 14, 8],   # 1 = <sos>
    [1, 22, 31, 0, 0, 0],
])

model.eval()
with torch.no_grad():
    logits, cross_attns = model(src_tokens, tgt_tokens, return_attn=True)

print("\nsrc", tuple(src_tokens.shape), "+ tgt", tuple(tgt_tokens.shape))
print("-> logits", tuple(logits.shape), " (batch, tgt_len, vocab_size)")
print("One score per vocabulary word, at every target position.")
print("Raw logits, NOT softmaxed - CrossEntropyLoss does that itself.")

print("\nCross-attention returned: %d tensors, one per decoder layer" % len(cross_attns))
print("  each", tuple(cross_attns[0].shape), " (batch, heads, tgt_len, src_len)")

print("\n-- Weight tying --")
tied = model.src_embed.weight.data_ptr() == model.generator.weight.data_ptr()
print("src_embed and generator share memory:", tied)
print("src_embed and tgt_embed are the same object:", model.src_embed is model.tgt_embed)
print("Untied, the two matrices would cost an extra %s parameters." %
      f"{config.vocab_size * config.d_model:,}")

print("\n-- Parameter count --")
for name, count in model.count_parameters().items():
    print(f"  {name:<14} {count:>12,}")


# ---------------------------------------------------------------------------
section("PART 6: Does gradient reach everything?")
# ---------------------------------------------------------------------------

# The shape checks above would all still pass if some branch were accidentally
# detached from the graph. This is the check that catches that.
model.train()
logits = model(src_tokens, tgt_tokens)

# Predict the NEXT token: position t of the output is graded against token t+1.
loss = nn.CrossEntropyLoss(ignore_index=config.pad_idx)(
    logits[:, :-1].reshape(-1, config.vocab_size),
    tgt_tokens[:, 1:].reshape(-1),
)
loss.backward()

print("Loss: %.4f" % loss.item())
print("Untrained, so it should sit near ln(vocab_size) = %.4f" %
      torch.log(torch.tensor(float(config.vocab_size))).item())

missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
print("\nParameters with no gradient:", missing if missing else "none - every "
      "part of the model is connected")

section("Phase 1 complete")
print("""
Built and verified:
  masking.py     padding and causal masks, broadcast-shaped
  positional.py  sinusoidal encoding, a buffer rather than a parameter
  attention.py   multi-head attention serving all three roles
  layers.py      encoder (2 sublayer) and decoder (3 sublayer) blocks
  transformer.py the assembled model with tied embeddings

The architecture runs and gradients flow. What it does NOT yet prove is that
it can learn - that needs the toy copy task, which comes next, along with the
tokeniser, dataset and training loop.
""")
