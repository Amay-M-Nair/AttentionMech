"""
The full encoder-decoder transformer.

    source tokens                       target tokens (shifted right)
         |                                       |
    embed x sqrt(d_model)               embed x sqrt(d_model)
    + positional encoding               + positional encoding
         |                                       |
    N x EncoderLayer                    N x DecoderLayer  <-- reads encoder
         |                                       |             output here
      LayerNorm  ----- memory ----->          LayerNorm
                                                 |
                                            Linear -> vocab logits

See docs/architecture.md#the-full-model.
"""

import math

import torch.nn as nn

from .config import TransformerConfig
from .layers import DecoderLayer, EncoderLayer
from .masking import make_decoder_mask, make_pad_mask
from .positional import PositionalEncoding


class Encoder(nn.Module):
    """Stack of encoder layers. Turns the source sentence into `memory`."""

    def __init__(self, embed, pos_encoding, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.embed = embed
        self.pos_encoding = pos_encoding
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        # pre-norm leaves the final residual sum unnormalised
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """
        Args:
            src:      (batch, src_len) token indices
            src_mask: (batch, 1, 1, src_len)

        Returns:
            memory:       (batch, src_len, d_model)
            attn_weights: list, one per layer
        """
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        attn_weights = []
        for layer in self.layers:
            x, weights = layer(x, src_mask)
            attn_weights.append(weights)

        return self.norm(x), attn_weights


class Decoder(nn.Module):
    """Stack of decoder layers. Generates the target, consulting `memory`."""

    def __init__(self, embed, pos_encoding, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.embed = embed
        self.pos_encoding = pos_encoding
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        """
        Args:
            tgt:      (batch, tgt_len) token indices
            memory:   (batch, src_len, d_model) encoder output
            tgt_mask: (batch, 1, tgt_len, tgt_len)
            src_mask: (batch, 1, 1, src_len)

        Returns:
            x:           (batch, tgt_len, d_model)
            self_attns:  list, one per layer
            cross_attns: list, one per layer - the source/target alignments
        """
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        self_attns, cross_attns = [], []
        for layer in self.layers:
            x, self_w, cross_w = layer(x, memory, tgt_mask, src_mask)
            self_attns.append(self_w)
            cross_attns.append(cross_w)

        return self.norm(x), self_attns, cross_attns


class Transformer(nn.Module):
    """
    Encoder-decoder transformer with a single shared vocabulary.

    Source embedding, target embedding and output projection are all the same
    matrix - see docs/architecture.md#weight-tying.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.pad_idx = config.pad_idx

        self.src_embed = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_idx
        )
        self.tgt_embed = self.src_embed  # alias - the same object, not a copy

        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_len, config.dropout
        )

        self.encoder = Encoder(
            self.src_embed, self.pos_encoding, config.d_model,
            config.num_heads, config.d_ff, config.num_layers, config.dropout,
        )
        self.decoder = Decoder(
            self.tgt_embed, self.pos_encoding, config.d_model,
            config.num_heads, config.d_ff, config.num_layers, config.dropout,
        )

        self.generator = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_parameters()

        # tie AFTER init, so the shared matrix keeps the embedding's values
        self.generator.weight = self.src_embed.weight

    def _init_parameters(self):
        """Xavier uniform on every matrix; 1-D tensors keep their defaults."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # xavier overwrote the zeroed <pad> row that padding_idx set up
        self.src_embed.weight.data[self.pad_idx].zero_()

    def make_masks(self, src, tgt):
        return make_pad_mask(src, self.pad_idx), make_decoder_mask(tgt, self.pad_idx)

    def encode(self, src, src_mask=None):
        """
        Run the encoder alone - generation encodes once, then decodes
        repeatedly.
        """
        if src_mask is None:
            src_mask = make_pad_mask(src, self.pad_idx)
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None):
        """Run the decoder alone, against an already-computed memory."""
        if tgt_mask is None:
            tgt_mask = make_decoder_mask(tgt, self.pad_idx)
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def forward(self, src, tgt, return_attn=False):
        """
        Args:
            src: (batch, src_len) source token indices
            tgt: (batch, tgt_len) target token indices, already shifted right
            return_attn: also return the cross-attention weights

        Returns:
            logits:      (batch, tgt_len, vocab_size), raw - not softmaxed
            cross_attns: list of (batch, heads, tgt_len, src_len), if requested
        """
        src_mask, tgt_mask = self.make_masks(src, tgt)

        memory, _ = self.encoder(src, src_mask)
        output, _, cross_attns = self.decoder(tgt, memory, tgt_mask, src_mask)
        logits = self.generator(output)

        if return_attn:
            return logits, cross_attns
        return logits

    def count_parameters(self):
        """Trainable parameters by component; tied weights counted once."""
        counts, seen = {}, set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            component = name.split(".")[0]
            counts[component] = counts.get(component, 0) + p.numel()
        counts["TOTAL"] = sum(counts.values())
        return counts


def build_model(config: TransformerConfig = None, **overrides) -> Transformer:
    """build_model(vocab_size=5000, num_layers=2)"""
    if config is None:
        config = TransformerConfig(**overrides)
    elif overrides:
        config = TransformerConfig(**{**config.__dict__, **overrides})
    return Transformer(config)
