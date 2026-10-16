"""Spiking Self Attention (SSA) block from Spikformer (Zhou et al., 2023),
plus a matching spiking MLP block. Q, K, V are binary spike tensors produced
by Linear -> L1LayerNorm -> LIF, so the attention matmuls are between
{0,1} tensors and no softmax/normalization is needed:

    SSA(Q, K, V) = SN( scale * (Q @ K^T) @ V )

Two options make the blocks multiplier-free in hardware (both off by
default, so older checkpoints behave as trained):

- attn_lif: an LIF neuron on the attention output, before proj (as in the
  original Spikformer), so proj's input is binary spikes like every other
  linear layer in the block.
- attn_norm: an L1LayerNorm on the attention output before attn_lif, so the
  LIF sees each token's attention rescaled to its working range instead of
  saturating on large overlap counts. Requires attn_lif.
- learnable_scale: the attention scale is a learned scalar per block,
  2^log2_scale (initialised to 1/8), instead of the fixed 1/8.
- attn_proj=False: drop proj (and its norm and LIF) entirely; the attention
  LIF's spikes go straight into the residual sum and on to the MLP. Requires
  attn_lif.
- pow2_gamma: every L1LayerNorm scale gamma is used as sign(gamma) *
  2^round(log2|gamma|) in the forward pass (straight-through gradient), so the
  norm's gamma multiply is a shift.

All neurons here are stateful (see neurons.LIFNeuron) and must be simulated
for T time steps by the caller (see models.SpikingTransformer).
"""

import torch
import torch.nn as nn

from neurons import LIFNeuron


class L1LayerNorm(nn.Module):
    """LayerNorm variant that centers by the mean like standard LayerNorm,
    but scales by the mean absolute deviation (L1) instead of the standard
    deviation (L2) -- cheaper in hardware (no square root).

    With `table_len` set, the norm also owns a `mad_table` buffer: one
    denominator per sequence position, replacing the per-token mad so the
    hardware needs no divider (just a per-position reciprocal lookup). It is
    filled by `finetune.calibrate_mad_tables` and only used when
    `use_table` is True; it is saved in the state_dict.
    """

    def __init__(self, dim: int, eps: float = 1e-6, table_len: int = None,
                 pow2_gamma: bool = False):
        super().__init__()
        self.eps = eps
        self.pow2_gamma = pow2_gamma
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.use_table = table_len is not None
        if table_len is not None:
            self.register_buffer("mad_table", torch.ones(table_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - x.mean(dim=-1, keepdim=True)
        if self.use_table:
            mad = self.mad_table[: x.shape[-2], None]  # (L, 1), broadcast over batch and features
        else:
            mad = x_centered.abs().mean(dim=-1, keepdim=True)
        return x_centered / (mad + self.eps) * self.gamma() + self.bias

    def gamma(self) -> torch.Tensor:
        """The scale used in the forward pass: gamma, or gamma rounded to a signed
        power of two (straight-through gradient) when pow2_gamma is set."""
        if not self.pow2_gamma:
            return self.weight
        w = self.weight
        rounded = torch.sign(w) * torch.pow(2.0, torch.round(torch.log2(w.abs().clamp_min(1e-8))))
        return w + (rounded - w).detach()


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False,
                 table_len: int = None, attn_lif: bool = False, pow2_gamma: bool = False,
                 attn_proj: bool = True, attn_norm: bool = False, learnable_scale: bool = False):
        super().__init__()
        assert attn_lif or not attn_norm, "attn_norm feeds attn_lif"
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert attn_proj or attn_lif, "without proj, attn_lif must produce the block's spikes"
        self.attn_proj = attn_proj
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125  # fixed scale, as in Spikformer (inputs are already unit-scale spikes)
        self.log2_scale = nn.Parameter(torch.tensor(-3.0)) if learnable_scale else None

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
        self.q_lif = LIFNeuron()

        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
        self.k_lif = LIFNeuron()

        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
        self.v_lif = LIFNeuron()

        self.attn_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma) if attn_norm else None
        self.attn_lif = LIFNeuron() if attn_lif else None

        if attn_proj:
            self.proj_linear = nn.Linear(dim, dim)
            self.proj_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
            self.proj_lif = LIFNeuron()
        else:
            self.proj_lif = None

    def reset_state(self):
        for m in (self.q_lif, self.k_lif, self.v_lif, self.attn_lif, self.proj_lif):
            if m is not None:
                m.reset_state()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attn: bool = False):
        # x: (B, L, D) input current for the current time step
        # attn_mask: (L, L) bool, True at positions that must be masked out
        # (e.g. a causal mask forbidding attention to future positions)
        B, L, D = x.shape

        q = self.q_lif(self.q_norm(self.q_linear(x)))
        k = self.k_lif(self.k_norm(self.k_linear(x)))
        v = self.v_lif(self.v_norm(self.v_linear(x)))

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = q @ k.transpose(-2, -1)          # (B, H, L, L) integer spike-overlap counts
        if attn_mask is not None:
            # spikes are non-negative counts (no softmax), so masking is a
            # plain zero-out rather than the -inf used before a softmax
            attn = attn.masked_fill(attn_mask, 0.0)
        scale = self.scale if self.log2_scale is None else torch.pow(2.0, self.log2_scale)
        out = (attn @ v) * scale                # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)
        if self.attn_norm is not None:
            out = self.attn_norm(out)
        if self.attn_lif is not None:
            out = self.attn_lif(out)            # binary spikes into proj

        if self.attn_proj:
            out = self.proj_lif(self.proj_norm(self.proj_linear(out)))
        if return_attn:
            return out, attn.detach()
        return out


class SpikingMLP(nn.Module):
    """layers=2: fc1 (dim -> hidden) and fc2 (hidden -> dim), each Linear -> norm -> LIF.
    layers=1: a single fc1 (dim -> dim) -> norm -> LIF."""

    def __init__(self, dim: int, hidden_dim: int, table_len: int = None, pow2_gamma: bool = False,
                 layers: int = 2):
        super().__init__()
        assert layers in (1, 2), "the MLP has 1 or 2 layers"
        self.layers = layers
        width = hidden_dim if layers == 2 else dim
        self.fc1 = nn.Linear(dim, width)
        self.norm1 = L1LayerNorm(width, table_len=table_len, pow2_gamma=pow2_gamma)
        self.lif1 = LIFNeuron()

        if layers == 2:
            self.fc2 = nn.Linear(hidden_dim, dim)
            self.norm2 = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
            self.lif2 = LIFNeuron()

    def reset_state(self):
        self.lif1.reset_state()
        if self.layers == 2:
            self.lif2.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lif1(self.norm1(self.fc1(x)))
        if self.layers == 2:
            x = self.lif2(self.norm2(self.fc2(x)))
        return x


class SpikingBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, table_len: int = None,
                 attn_lif: bool = False, pow2_gamma: bool = False, attn_proj: bool = True,
                 mlp_layers: int = 2, attn_norm: bool = False, learnable_scale: bool = False):
        super().__init__()
        self.attn = SpikingSelfAttention(dim, num_heads, table_len=table_len, attn_lif=attn_lif,
                                         pow2_gamma=pow2_gamma, attn_proj=attn_proj, attn_norm=attn_norm,
                                         learnable_scale=learnable_scale)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio, table_len=table_len, pow2_gamma=pow2_gamma,
                              layers=mlp_layers)

    def reset_state(self):
        self.attn.reset_state()
        self.mlp.reset_state()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attn: bool = False):
        if return_attn:
            attn_out, attn = self.attn(x, attn_mask=attn_mask, return_attn=True)
            x = x + attn_out
            x = x + self.mlp(x)
            return x, attn
        x = x + self.attn(x, attn_mask=attn_mask)
        x = x + self.mlp(x)
        return x
