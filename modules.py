"""Spiking Self Attention (SSA) block from Spikformer (Zhou et al., 2023),
plus a matching spiking MLP block. Q, K, V are binary spike tensors produced
by Linear -> LayerNorm -> LIF, so the attention matmuls are between {0,1}
tensors and no softmax/normalization is needed:

    SSA(Q, K, V) = SN( scale * (Q @ K^T) @ V )

All neurons here are stateful (see neurons.LIFNeuron) and must be simulated
for T time steps by the caller (see models.SpikingTransformer).
"""

import torch
import torch.nn as nn

from neurons import LIFNeuron


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125  # fixed scale, as in Spikformer (inputs are already unit-scale spikes)

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(dim)
        self.q_lif = LIFNeuron()

        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_norm = nn.LayerNorm(dim)
        self.k_lif = LIFNeuron()

        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_norm = nn.LayerNorm(dim)
        self.v_lif = LIFNeuron()

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_norm = nn.LayerNorm(dim)
        self.proj_lif = LIFNeuron()

    def reset_state(self):
        for m in (self.q_lif, self.k_lif, self.v_lif, self.proj_lif):
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
        out = (attn @ v) * self.scale           # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)

        out = self.proj_lif(self.proj_norm(self.proj_linear(out)))
        if return_attn:
            return out, attn.detach()
        return out


class SpikingMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lif1 = LIFNeuron()

        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.lif2 = LIFNeuron()

    def reset_state(self):
        self.lif1.reset_state()
        self.lif2.reset_state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lif1(self.norm1(self.fc1(x)))
        x = self.lif2(self.norm2(self.fc2(x)))
        return x


class VanillaSelfAttentionBlock(nn.Module):
    """Standard post-LN transformer block (same architecture as
    nn.TransformerEncoderLayer with norm_first=False and a GELU FFN), built
    from nn.MultiheadAttention directly so per-head attention weights can be
    retrieved for analysis -- nn.TransformerEncoderLayer's fast path does not
    expose them."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, dim * mlp_ratio)
        self.linear2 = nn.Linear(dim * mlp_ratio, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attn: bool = False):
        attn_out, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask, need_weights=return_attn, average_attn_weights=False
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        if return_attn:
            return x, attn_weights.detach()
        return x


class SpikingBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = SpikingSelfAttention(dim, num_heads)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio)

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
