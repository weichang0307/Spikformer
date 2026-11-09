"""SpikingTransformer: Spikformer-style Spiking Self Attention (SSA) model for
causal next-token prediction (the Associative Recall task in train.py, or
character-level language modelling in train_char.py), built by simulating `SpikingBlock`s for
T discrete time steps and rate-decoding the output (mean spike count over T).
L1LayerNorm is used throughout. The combined token and positional embeddings
are normalized and fed as a constant current into an input LIF neuron
(identity encoding), which converts them to spikes before the blocks.

With norm_table=True every L1LayerNorm uses a per-position denominator
lookup table instead of the per-token mean absolute deviation (see
modules.L1LayerNorm and finetune.py), so the hardware needs no divider.
attn_lif and pow2_gamma make the blocks multiplier-free (see modules.py).
pos_embed="sinusoid" replaces the learned position embedding with the fixed
sinusoidal encoding of Vaswani et al. (2017), held as a buffer of the same
name and shape so checkpoints and the integer model treat it alike.
Use `model_from_config` to rebuild the model a run was trained with.

Self-attention is causal (autoregressive): position i may only attend
to positions <= i. This is what lets an induction-head-style mechanism
(attend to a previous occurrence of the current token, copy whatever
followed it) form during training.

forward() accepts `return_attn=True` to additionally return a list (one
entry per layer) of attention matrices of shape (B, H, L, L), for
mechanistic analysis (e.g. plotting how the QK attention pattern evolves
during training). These are the raw spike QK^T overlap counts, averaged over
the T simulated time steps.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import L1LayerNorm, SpikingBlock
from neurons import LIFNeuron


def attention_mask(length: int, round_len: int = 0, device=None) -> torch.Tensor:
    """(L, L) bool, True = masked: future positions, and (if round_len > 0)
    every position at or after round_len, so attention only reads round 1."""
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
    if round_len > 0:
        mask[:, round_len:] = True
    return mask


def sinusoid_table(length: int, dim: int) -> torch.Tensor:
    """(length, dim) sinusoidal position encoding: sin on even dims, cos on odd."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * freq)
    table[:, 1::2] = torch.cos(position * freq)
    return table


class SpikingTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, depth: int = 2,
                 num_heads: int = 4, T: int = 32, max_len: int = 64,
                 norm_table: bool = False, attn_lif: bool = False, pow2_gamma: bool = False,
                 attn_proj: bool = True, mlp_layers: int = 2, attn_norm: bool = False,
                 pos_embed: str = "learned", learnable_scale: bool = False, mask_round_len: int = 0,
                 head_scale: bool = False, head_norm: bool = False, attn_norm_eps: float = 1e-6,
                 mlp_hidden: int = 0):
        super().__init__()
        self.T = T
        self.mask_round_len = mask_round_len  # > 0: attention may only look at positions < this
        self.embed = nn.Embedding(vocab_size, dim)
        if pos_embed == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed == "sinusoid":
            self.register_buffer("pos_embed", sinusoid_table(max_len, dim)[None])
        else:
            raise ValueError(f"unknown pos_embed {pos_embed!r}")

        table_len = max_len if norm_table else None
        self.input_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
        self.input_lif = LIFNeuron()
        self.blocks = nn.ModuleList([
            SpikingBlock(dim, num_heads, table_len=table_len, attn_lif=attn_lif,
                         pow2_gamma=pow2_gamma, attn_proj=attn_proj, mlp_layers=mlp_layers,
                         attn_norm=attn_norm, learnable_scale=learnable_scale,
                         head_scale=head_scale, head_norm=head_norm, attn_norm_eps=attn_norm_eps,
                         mlp_hidden=mlp_hidden)
            for _ in range(depth)
        ])
        self.head = nn.Linear(dim, vocab_size)
        self.multistep = True           # all T steps of a block in one call (same result, faster)
        self.checkpoint_blocks = False  # recompute block activations in backward (less memory)

    def norms(self) -> list:
        """All L1 norms in network order (the order calibration must follow)."""
        norms = [self.input_norm]
        for blk in self.blocks:
            a = blk.attn
            norms += [a.q_norm, a.k_norm, a.v_norm] + ([a.attn_norm] if a.attn_norm is not None else [])
            norms += [a.proj_norm] if a.attn_proj else []
            norms += [blk.mlp.norm1] + ([blk.mlp.norm2] if blk.mlp.layers == 2 else [])
        return norms

    def reset_state(self):
        self.input_lif.reset_state()
        for blk in self.blocks:
            blk.reset_state()

    def forward(self, tokens: torch.Tensor, return_attn: bool = False, attn_mask: torch.Tensor = None,
                attn_rows: torch.Tensor = None):
        """attn_mask: optional (L, L) bool mask (True = masked) replacing the causal one,
        e.g. the identity complement to measure how much the model uses its context.
        attn_rows: with return_attn, the maps hold only these query rows and keep their gradient."""
        B, L = tokens.shape
        x0 = self.input_norm(self.embed(tokens) + self.pos_embed[:, :L])
        causal_mask = attn_mask if attn_mask is not None else attention_mask(L, self.mask_round_len, tokens.device)

        self.reset_state()
        if self.multistep:
            return self._forward_multistep(x0, causal_mask, return_attn, attn_rows)
        acc = 0.0
        attn_acc = [0.0] * len(self.blocks) if return_attn else None

        for _ in range(self.T):
            x = self.input_lif(x0)
            for i, blk in enumerate(self.blocks):
                if return_attn:
                    x, attn = blk(x, attn_mask=causal_mask, return_attn=True, attn_rows=attn_rows)
                    attn_acc[i] = attn_acc[i] + attn
                else:
                    x = blk(x, attn_mask=causal_mask)
            acc = acc + x
        out = acc / self.T  # rate-coded readout
        logits = self.head(out)

        if return_attn:
            attn_maps = [a / self.T for a in attn_acc]  # time-averaged spike attention pattern
            return logits, attn_maps
        return logits

    def _forward_multistep(self, x0, causal_mask, return_attn, attn_rows=None):
        """Same computation, block by block over all T steps at once: the T copies of the input
        are stacked along the batch dimension, every LIF loops over the steps internally."""
        T, B = self.T, x0.shape[0]
        lifs = [m for m in self.modules() if isinstance(m, LIFNeuron)]
        for m in lifs:
            m.steps = T
        try:
            x = self.input_lif(x0.repeat(T, 1, 1))                      # (T * B, L, D)
            attn_maps = []
            for blk in self.blocks:
                if return_attn:
                    x, attn = blk(x, attn_mask=causal_mask, return_attn=True, attn_rows=attn_rows)
                    attn_maps.append(attn.view(T, B, *attn.shape[1:]).mean(0))
                else:
                    x = blk(x, attn_mask=causal_mask, checkpoint=self.checkpoint_blocks and self.training)
        finally:
            for m in lifs:
                m.steps = None
        logits = self.head(x.view(T, B, *x.shape[1:]).mean(0))      # rate-coded readout
        return (logits, attn_maps) if return_attn else logits


def model_from_config(config: dict, max_len: int) -> nn.Module:
    """The model a run's config.json describes (weights not loaded)."""
    model = _model_from_config(config, max_len)
    if config.get("attn_threshold_heads", 0) > 0:
        for blk in model.blocks:
            blk.attn.enable_learned_threshold(config["attn_threshold_heads"], config.get("attn_threshold_init", 2.5))
    if config.get("model_type") != "ann" and config.get("dropout", 0.0) > 0 and hasattr(model, "blocks"):
        for blk in model.blocks:   # spike dropout (only active in training mode)
            blk.drop.p = config["dropout"]
    if config.get("attn_square_heads", 0) > 0:
        for blk in model.blocks:
            blk.attn.enable_square(config["attn_square_heads"])
    return model


def _model_from_config(config: dict, max_len: int) -> nn.Module:
    """The model a run's config.json describes (weights not loaded)."""
    if config.get("model_type") == "ann":
        return ANNTransformer(config["vocab_size"], dim=config["dim"], depth=config["depth"],
                              num_heads=config["num_heads"], max_len=max_len,
                              mlp_hidden=config.get("mlp_hidden", 0), dropout=config.get("dropout", 0.0))
    vocab = config["vocab_size"] if "vocab_size" in config else config["num_keys"] + config["num_values"]
    return SpikingTransformer(
        vocab, dim=config["dim"], depth=config["depth"],
        num_heads=config["num_heads"], T=config["T"], max_len=max_len,
        norm_table=config.get("norm_table", False), attn_lif=config.get("attn_lif", False),
        pow2_gamma=config.get("pow2_gamma", False), attn_proj=config.get("attn_proj", True),
        mlp_layers=config.get("mlp_layers", 2), attn_norm=config.get("attn_norm", False),
        pos_embed=config.get("pos_embed", "learned"), learnable_scale=config.get("learnable_scale", False),
        mask_round_len=2 * config["num_kv_pairs"] if config.get("mask_round1") else 0,  # AR runs only
        head_scale=config.get("head_scale", False), head_norm=config.get("head_norm", False),
        attn_norm_eps=config.get("attn_norm_eps", 1e-6), mlp_hidden=config.get("mlp_hidden", 0),
    )


class ANNSelfAttention(nn.Module):
    """Causal softmax self-attention with separate Q / K / V / output projections,
    named like SpikingSelfAttention's linears so weights can be copied across."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q_linear, self.k_linear, self.v_linear = (nn.Linear(dim, dim) for _ in range(3))
        self.proj_linear = nn.Linear(dim, dim)
        self.dropout = dropout
        # attributes read by health.param_stats / attention_and_context
        self.scale, self.log2_scale, self.attn_lif = self.head_dim ** -0.5, None, None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attn: bool = False,
                attn_rows: torch.Tensor = None):
        B, L, D = x.shape
        q, k, v = (f(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                   for f in (self.q_linear, self.k_linear, self.v_linear))
        keep = None if attn_mask is None else ~attn_mask  # attn_mask: True = masked (as in the spiking model)
        if return_attn:  # explicit softmax so the attention pattern can be returned
            scores = (q @ k.transpose(-2, -1)) * self.scale
            scores = scores.masked_fill(attn_mask, float("-inf")) if attn_mask is not None else scores
            attn = scores.float().softmax(-1).to(q.dtype)
            out = attn @ v
        else:
            drop = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=drop,
                                                 is_causal=attn_mask is None)
        out = self.proj_linear(out.transpose(1, 2).reshape(B, L, D))
        if not return_attn:
            return out
        return out, (attn.detach() if attn_rows is None else attn[:, :, attn_rows])


class ANNBlock(nn.Module):
    """Pre-LayerNorm GPT block: x + attn(LN(x)), then x + MLP(LN(x)) with GELU."""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = ANNSelfAttention(dim, num_heads, dropout)
        self.mlp = nn.Module()
        self.mlp.fc1, self.mlp.fc2 = nn.Linear(dim, mlp_hidden), nn.Linear(mlp_hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, return_attn=False, attn_rows=None):
        a = self.attn(self.ln1(x), attn_mask=attn_mask, return_attn=return_attn, attn_rows=attn_rows)
        a, attn = a if return_attn else (a, None)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp.fc2(F.gelu(self.mlp.fc1(self.ln2(x)))))
        return (x, attn) if return_attn else x


class ANNTransformer(nn.Module):
    """Non-spiking GPT baseline with the SpikingTransformer's dimensions (embedding,
    learned positions, depth, width, heads, MLP width, head), for a same-size
    comparison and as a teacher / initialiser for the spiking model. Parameter
    names match where the layers correspond (embed, pos_embed, blocks.i.attn.
    {q,k,v,proj}_linear, blocks.i.mlp.fc1/fc2, head). T is accepted and ignored."""

    def __init__(self, vocab_size: int, dim: int = 512, depth: int = 12, num_heads: int = 8, max_len: int = 1024,
                 mlp_hidden: int = 0, dropout: float = 0.0, T: int = 1):
        super().__init__()
        self.T = T  # unused: the training loop sets it per evaluation
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ANNBlock(dim, num_heads, mlp_hidden or 4 * dim, dropout) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        for name, p in self.named_parameters():  # GPT-2 initialisation
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * depth) if name.endswith(("proj_linear.weight", "fc2.weight")) else 0.02
                nn.init.normal_(p, std=std)
            elif name.endswith("bias"):
                nn.init.zeros_(p)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def reset_state(self):
        pass

    def forward(self, tokens: torch.Tensor, return_attn: bool = False, attn_mask: torch.Tensor = None,
                attn_rows: torch.Tensor = None):
        L = tokens.shape[1]
        x = self.drop(self.embed(tokens) + self.pos_embed[:, :L])
        if return_attn and attn_mask is None:
            attn_mask = attention_mask(L, device=tokens.device)
        maps = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, attn_mask=attn_mask, return_attn=True, attn_rows=attn_rows)
                maps.append(a)
            else:
                x = blk(x, attn_mask=attn_mask)
        logits = self.head(self.ln_f(x))
        return (logits, maps) if return_attn else logits
