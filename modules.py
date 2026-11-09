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
- head_scale: with learnable_scale, one learned scale per head instead of per
  block. Under a shared attn_norm a per-block scale cancels out, but per-head
  scales change how loud each head is inside the shared normalisation.
- head_norm: attn_norm takes its mean and MAD over each head's features
  separately, so a head with large overlap counts cannot shrink the others
  below the LIF threshold.
- attn_norm_eps: the constant added to attn_norm's MAD (default 1e-6). A
  per-head MAD over only head_dim features can be tiny for a nearly constant
  head; a larger floor (e.g. 1/16) keeps such heads quiet instead of blowing
  their noise (and their gradients) up to full size.
- attn_proj=False: drop proj (and its norm and LIF) entirely; the attention
  LIF's spikes go straight into the residual sum and on to the MLP. Requires
  attn_lif.
- pow2_gamma: every L1LayerNorm scale gamma is used as sign(gamma) *
  2^round(log2|gamma|) in the forward pass (straight-through gradient), so the
  norm's gamma multiply is a shift.

All neurons here are stateful (see neurons.LIFNeuron) and must be simulated
for T time steps by the caller (see models.SpikingTransformer).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

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
                 pow2_gamma: bool = False, groups: int = 1):
        super().__init__()
        assert dim % groups == 0, "dim must be divisible by groups"
        assert groups == 1 or table_len is None, "the per-position MAD table supports groups=1 only"
        self.eps = eps
        self.pow2_gamma = pow2_gamma
        self.mad_pow2 = None  # "round" / "floor": divide by a power of two (a shift) instead of the exact MAD
        self.groups = groups  # > 1: mean and MAD per group of dim // groups features (e.g. per head)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.use_table = table_len is not None
        if table_len is not None:
            self.register_buffer("mad_table", torch.ones(table_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups > 1:
            g = x.view(*x.shape[:-1], self.groups, -1)
            g = g - g.mean(dim=-1, keepdim=True)
            g = g / self._divisor(g.abs().mean(dim=-1, keepdim=True) + self.eps)
            return g.reshape(x.shape) * self.gamma() + self.bias
        x_centered = x - x.mean(dim=-1, keepdim=True)
        if self.use_table:
            mad = self.mad_table[: x.shape[-2], None]  # (L, 1), broadcast over batch and features
        else:
            mad = x_centered.abs().mean(dim=-1, keepdim=True)
        return x_centered / self._divisor(mad + self.eps) * self.gamma() + self.bias

    def _divisor(self, d: torch.Tensor) -> torch.Tensor:
        """The denominator: as is, or (mad_pow2 = "round" / "floor") replaced by a power of two, so the divide
        is a shift in hardware -- 2^round(log2 d) (within a factor sqrt 2) or 2^floor(log2 d) (within 2x)."""
        if self.mad_pow2 is None:
            return d
        e = torch.log2(d.clamp_min(1e-12))
        e = torch.round(e) if self.mad_pow2 == "round" else torch.floor(e)
        return d + (torch.pow(2.0, e) - d).detach()   # straight-through gradient

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
                 attn_proj: bool = True, attn_norm: bool = False, learnable_scale: bool = False,
                 head_scale: bool = False, head_norm: bool = False, attn_norm_eps: float = 1e-6):
        super().__init__()
        assert attn_lif or not attn_norm, "attn_norm feeds attn_lif"
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert attn_proj or attn_lif, "without proj, attn_lif must produce the block's spikes"
        self.attn_proj = attn_proj
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125  # fixed scale, as in Spikformer (inputs are already unit-scale spikes)
        self.count_threshold = 0  # > 0: Q.K overlap counts below this are set to 0 (per time step)
        self.thr_heads = 0        # > 0 after enable_learned_threshold(): heads [0, thr_heads) use a learned threshold
        self.thr_raw = None
        self.square_heads = 0     # > 0 after enable_square(): heads [0, square_heads) weight keys by count^2
        assert learnable_scale or not head_scale, "head_scale needs learnable_scale"
        assert attn_norm or not head_norm, "head_norm needs attn_norm"
        self.log2_scale = None
        if learnable_scale:  # 2^s: one per block, or one per head with head_scale
            self.log2_scale = nn.Parameter(torch.full((num_heads,) if head_scale else (), -3.0))

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
        self.q_lif = LIFNeuron()

        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
        self.k_lif = LIFNeuron()

        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
        self.v_lif = LIFNeuron()

        self.attn_norm = (L1LayerNorm(dim, eps=attn_norm_eps, table_len=table_len, pow2_gamma=pow2_gamma,
                                      groups=num_heads if head_norm else 1) if attn_norm else None)
        self.attn_lif = LIFNeuron() if attn_lif else None

        if attn_proj:
            self.proj_linear = nn.Linear(dim, dim)
            self.proj_norm = L1LayerNorm(dim, table_len=table_len, pow2_gamma=pow2_gamma)
            self.proj_lif = LIFNeuron()
        else:
            self.proj_lif = None

    def enable_learned_threshold(self, heads: int, init: float = 2.5):
        """Learned per-head count threshold: in heads [0, heads) a key is kept only if its Q.K overlap count is
        >= theta_h = 1 + softplus(raw_h) (theta >= 1; theta <= 1 keeps every overlapping key). Forward: hard
        mask; backward: the derivative of sigmoid((count - theta) / 0.5), so theta and the counts near it learn.
        thr_blend (0..1, a buffer so it can change every step without recompiling) mixes the masked and the plain
        counts while the threshold is introduced."""
        self.thr_heads = heads
        raw = math.log(math.expm1(max(init - 1.0, 1e-3)))     # softplus^-1(init - 1)
        self.thr_raw = nn.Parameter(torch.full((self.num_heads,), raw))
        self.register_buffer("thr_blend", torch.ones(()), persistent=False)

    def enable_square(self, heads: int):
        """Heads [0, heads) weight each key by its squared Q.K overlap count instead of the count: a smooth
        winner-take-more (a 4-bit match counts 16, a 1-bit match 1) with no cut and no learned parameter.
        sq_blend (0..1, a buffer) mixes count and count^2 while it is introduced."""
        self.square_heads = heads
        self.register_buffer("sq_blend", torch.ones(()), persistent=False)

    def _apply_square(self, attn: torch.Tensor) -> torch.Tensor:
        n = self.square_heads
        c = attn[:, :n]
        c = c + self.sq_blend.to(c.dtype) * (c * c - c)
        return torch.cat([c, attn[:, n:]], dim=1)

    def threshold(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.thr_raw)

    def _apply_learned_threshold(self, attn: torch.Tensor) -> torch.Tensor:
        n = self.thr_heads
        c = attn[:, :n]
        th = self.threshold()[:n].view(1, n, 1, 1).to(c.dtype)
        soft = torch.sigmoid((c - th) / 0.5)
        gate = soft + ((c >= th).to(c.dtype) - soft).detach()
        c = c + self.thr_blend.to(c.dtype) * (c * gate - c)
        return torch.cat([c, attn[:, n:]], dim=1)

    def reset_state(self):
        for m in (self.q_lif, self.k_lif, self.v_lif, self.attn_lif, self.proj_lif):
            if m is not None:
                m.reset_state()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attn: bool = False,
                attn_rows: torch.Tensor = None):
        # x: (B, L, D) input current for the current time step
        # attn_rows: with return_attn, return only these query rows of the map, *with* gradient
        # (for an attention-distillation loss); otherwise the full map is returned detached
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
        if self.count_threshold > 0:
            attn = attn * (attn >= self.count_threshold)
        if self.thr_heads > 0:
            attn = self._apply_learned_threshold(attn)
        if self.square_heads > 0:
            attn = self._apply_square(attn)
        scale = self.scale if self.log2_scale is None else torch.pow(2.0, self.log2_scale)
        if torch.is_tensor(scale) and scale.dim() == 1:
            scale = scale.view(-1, 1, 1)        # per head, broadcast over (L, head_dim)
        out = (attn @ v) * scale                # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)
        if self.attn_norm is not None:
            out = self.attn_norm(out)
        if self.attn_lif is not None:
            out = self.attn_lif(out)            # binary spikes into proj

        if self.attn_proj:
            out = self.proj_lif(self.proj_norm(self.proj_linear(out)))
        if return_attn:
            return out, (attn.detach() if attn_rows is None else attn[:, :, attn_rows])
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
                 mlp_layers: int = 2, attn_norm: bool = False, learnable_scale: bool = False,
                 head_scale: bool = False, head_norm: bool = False, attn_norm_eps: float = 1e-6,
                 mlp_hidden: int = 0):
        super().__init__()
        self.attn = SpikingSelfAttention(dim, num_heads, table_len=table_len, attn_lif=attn_lif,
                                         pow2_gamma=pow2_gamma, attn_proj=attn_proj, attn_norm=attn_norm,
                                         learnable_scale=learnable_scale, head_scale=head_scale,
                                         head_norm=head_norm, attn_norm_eps=attn_norm_eps)
        self.mlp = SpikingMLP(dim, mlp_hidden or dim * mlp_ratio, table_len=table_len, pow2_gamma=pow2_gamma,
                              layers=mlp_layers)
        # spike dropout on both branch outputs before the residual add (training only; p set by the trainer).
        # Inverted dropout: kept spikes are scaled by 1 / (1 - p) while training, so the residual stream - read
        # out by the head without a norm - has the same expected scale; at inference spikes stay binary.
        self.drop = nn.Dropout(0.0)

    def reset_state(self):
        self.attn.reset_state()
        self.mlp.reset_state()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attn: bool = False,
                checkpoint: bool = False, attn_rows: torch.Tensor = None):
        if return_attn:
            attn_out, attn = self.attn(x, attn_mask=attn_mask, return_attn=True, attn_rows=attn_rows)
            x = x + self.drop(attn_out)
            x = x + self.drop(self.mlp(x))
            return x, attn
        if checkpoint:  # multi-step mode only: the block is stateless within one call
            steps = self.attn.q_lif.steps  # the recomputation runs in backward, so it sets multi-step mode itself
            return torch.utils.checkpoint.checkpoint(self._forward_steps, x, attn_mask, steps, use_reentrant=False)
        return self._forward(x, attn_mask)

    def _forward_steps(self, x: torch.Tensor, attn_mask: torch.Tensor, steps: int):
        lifs = [m for m in self.modules() if isinstance(m, LIFNeuron)]
        previous = [m.steps for m in lifs]
        for m in lifs:
            m.steps = steps
        try:
            return self._forward(x, attn_mask)
        finally:
            for m, p in zip(lifs, previous):
                m.steps = p

    def _forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.drop(self.attn(x, attn_mask=attn_mask))
        x = x + self.drop(self.mlp(x))
        return x
