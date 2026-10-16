"""SpikingTransformer: Spikformer-style Spiking Self Attention (SSA) model for
the (causal) Associative Recall task, built by simulating `SpikingBlock`s for
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
                 pos_embed: str = "learned", learnable_scale: bool = False, mask_round_len: int = 0):
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
                         attn_norm=attn_norm, learnable_scale=learnable_scale)
            for _ in range(depth)
        ])
        self.head = nn.Linear(dim, vocab_size)

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

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        B, L = tokens.shape
        x0 = self.input_norm(self.embed(tokens) + self.pos_embed[:, :L])
        causal_mask = attention_mask(L, self.mask_round_len, tokens.device)

        self.reset_state()
        acc = 0.0
        attn_acc = [0.0] * len(self.blocks) if return_attn else None

        for _ in range(self.T):
            x = self.input_lif(x0)
            for i, blk in enumerate(self.blocks):
                if return_attn:
                    x, attn = blk(x, attn_mask=causal_mask, return_attn=True)
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


def model_from_config(config: dict, max_len: int) -> SpikingTransformer:
    """The SpikingTransformer a run's config.json describes (weights not loaded)."""
    return SpikingTransformer(
        config["num_keys"] + config["num_values"], dim=config["dim"], depth=config["depth"],
        num_heads=config["num_heads"], T=config["T"], max_len=max_len,
        norm_table=config.get("norm_table", False), attn_lif=config.get("attn_lif", False),
        pow2_gamma=config.get("pow2_gamma", False), attn_proj=config.get("attn_proj", True),
        mlp_layers=config.get("mlp_layers", 2), attn_norm=config.get("attn_norm", False),
        pos_embed=config.get("pos_embed", "learned"), learnable_scale=config.get("learnable_scale", False),
        mask_round_len=2 * config["num_kv_pairs"] if config.get("mask_round1") else 0,
    )
