"""Two models trained on the same (causal) Associative Recall task for
comparison:

- VanillaTransformer: standard softmax multi-head self-attention (baseline).
- SpikingTransformer: Spikformer-style Spiking Self Attention (SSA), built
  by simulating `SpikingBlock`s for T discrete time steps and rate-decoding
  the output (mean spike count over T), with selectable L1/L2/RMS
  normalization and identity/rate input encoding. The combined token and
  positional embeddings are converted to spikes before entering the blocks.

Both use causal (autoregressive) self-attention: position i may only attend
to positions <= i. This is what lets an induction-head-style mechanism
(attend to a previous occurrence of the current token, copy whatever
followed it) form during training.

Both models' forward() accept `return_attn=True` to additionally return a
list (one entry per layer) of attention matrices of shape (B, H, L, L), for
mechanistic analysis (e.g. plotting how the QK attention pattern evolves
during training). For SpikingTransformer these are the raw spike QK^T
overlap counts, averaged over the T simulated time steps.
"""

import torch
import torch.nn as nn

from modules import SpikingBlock, VanillaSelfAttentionBlock, make_encoder, make_norm
from neurons import LIFNeuron


class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, depth: int = 2,
                 num_heads: int = 4, max_len: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([VanillaSelfAttentionBlock(dim, num_heads) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        B, L = tokens.shape
        x = self.embed(tokens) + self.pos_embed[:, :L]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tokens.device)

        attn_maps = [] if return_attn else None
        for blk in self.blocks:
            if return_attn:
                x, attn = blk(x, attn_mask=causal_mask, return_attn=True)
                attn_maps.append(attn)
            else:
                x = blk(x, attn_mask=causal_mask)

        logits = self.head(x)
        return (logits, attn_maps) if return_attn else logits


class SpikingTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64, depth: int = 2,
                 num_heads: int = 4, T: int = 4, max_len: int = 64,
                 norm_type: str = "l1", encoder_type: str = "identity"):
        super().__init__()
        self.T = T
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.input_norm = make_norm(dim, norm_type)
        self.encoder = make_encoder(encoder_type)
        self.encoder_type = encoder_type
        self.input_lif = LIFNeuron()
        self.blocks = nn.ModuleList([
            SpikingBlock(dim, num_heads, norm_type=norm_type) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, vocab_size)

    def reset_state(self):
        self.input_lif.reset_state()
        for blk in self.blocks:
            blk.reset_state()

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        B, L = tokens.shape
        x0 = self.input_norm(self.embed(tokens) + self.pos_embed[:, :L])
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)

        self.reset_state()
        acc = 0.0
        attn_acc = [0.0] * len(self.blocks) if return_attn else None

        for _ in range(self.T):
            x = self.encoder(x0)
            if self.encoder_type == "identity":
                x = self.input_lif(x)
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
