"""Training-health metrics for the spiking char LM, logged by train_char.py at
every evaluation and plotted by dashboard.html / plot_curves.py.

Per LIF neuron (keyed by module name, e.g. "blocks.2.attn.k_lif"):
  fire_rate   mean spike probability over time steps, tokens and features
  fire_dead   fraction of features that never fired on the whole val set
  fire_sat    fraction of features firing on >= 90% of (token, time step) pairs
  surrogate   mean surrogate gradient d spike / d v (peak alpha / 4 at threshold);
              near 0 means almost no gradient flows through that neuron

Per block ("input", "block i", "head"):
  grad_norm     L2 norm of the last training step's gradients
  update_ratio  ||delta w|| / ||w|| of the optimizer step (healthy ~1e-3)
  norm_gain     mean |gain| of the block's L1LayerNorms (LIF inputs scale with it)
  attn_scale    SSA scale 2^s
  attn_empty    fraction of (head, query) rows whose Q.K^T overlap is all zero
  attn_entropy  attention spread: row entropy / log(#visible keys); 1 = uniform
  attn_local    share of attention mass on the current and previous 7 positions

With adaptive clipping (train_char.py --adaptive_clip), per block, over the
steps since the previous evaluation:
  clip_frac     fraction of steps whose gradients were clipped
  grad_spike    largest gradient norm / its running average (spikes stand out)

Scalars:
  context_gain   loss with attention limited to the current position minus the
                 normal loss (nats): how much the model gains from its context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import L1LayerNorm
from neurons import LIFNeuron, SurrogateSpike


def block_of(name: str) -> str:
    """Parameter / module name -> "input" (embeddings, input norm/LIF), "block i" or "head"."""
    if name.startswith("blocks."):
        return f"block {name.split('.')[1]}"
    return "head" if name.startswith("head.") else "input"


class LIFMonitor:
    """Forward hooks on every LIFNeuron, accumulating per-feature spike counts and
    the surrogate gradient while active (between start() and stop())."""

    def __init__(self, model: nn.Module):
        self.names = {m: n for n, m in model.named_modules() if isinstance(m, LIFNeuron)}
        self.handles = []

    def start(self):
        self.spikes, self.counts, self.surrogate = {}, {}, {}
        self.handles = [m.register_forward_hook(self._hook) for m in self.names]

    def _hook(self, module, inputs, spikes):
        n = self.names[module]
        s = spikes.detach().float()
        self.spikes[n] = self.spikes.get(n, 0) + s.reshape(-1, s.shape[-1]).sum(0)
        self.counts[n] = self.counts.get(n, 0) + s[..., 0].numel()
        # the hook runs after the soft reset, so the pre-spike potential is v + spike * threshold
        x = module.v.detach().float() + s * module.v_threshold - module.v_threshold
        a = SurrogateSpike.alpha
        sig = torch.sigmoid(a * x)
        self.surrogate[n] = self.surrogate.get(n, 0.0) + (a * sig * (1 - sig)).sum().item()

    def stop(self) -> dict:
        for h in self.handles:
            h.remove()
        self.handles = []
        out = {"fire_rate": {}, "fire_dead": {}, "fire_sat": {}, "surrogate": {}}
        for n, total in self.spikes.items():
            per_feature = total / self.counts[n]
            out["fire_rate"][n] = per_feature.mean().item()
            out["fire_dead"][n] = (total == 0).float().mean().item()
            out["fire_sat"][n] = (per_feature >= 0.9).float().mean().item()
            out["surrogate"][n] = self.surrogate[n] / (self.counts[n] * total.numel())
        return out


@torch.no_grad()
def param_stats(model: nn.Module) -> dict:
    """Per-block attention scale, mean |L1LayerNorm gain| and gradient norm."""
    scales = {f"block {i}": (blk.attn.scale if blk.attn.log2_scale is None
                             else 2.0 ** blk.attn.log2_scale.item())
              for i, blk in enumerate(model.blocks)}
    gains = {}
    for name, m in model.named_modules():
        if isinstance(m, L1LayerNorm):
            gains.setdefault(block_of(name), []).append(m.weight.abs().mean().item())
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[block_of(name)] = grads.get(block_of(name), 0.0) + p.grad.float().pow(2).sum().item()
    return {"attn_scale": scales,
            "norm_gain": {g: sum(v) / len(v) for g, v in gains.items()},
            "grad_norm": {g: v ** 0.5 for g, v in grads.items()}}


class UpdateTracker:
    """snapshot() before optimizer.step(), ratios() after: per-block ||delta w|| / ||w||."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.before = None

    @torch.no_grad()
    def snapshot(self):
        self.before = {n: p.detach().clone() for n, p in self.model.named_parameters()}

    @torch.no_grad()
    def ratios(self) -> dict:
        delta, norm = {}, {}
        for n, p in self.model.named_parameters():
            g = block_of(n)
            delta[g] = delta.get(g, 0.0) + (p - self.before[n]).float().pow(2).sum().item()
            norm[g] = norm.get(g, 0.0) + self.before[n].float().pow(2).sum().item()
        self.before = None
        return {g: (delta[g] / max(norm[g], 1e-12)) ** 0.5 for g in delta}


@torch.no_grad()
def attention_and_context(model: nn.Module, seqs: list, min_query: int = 16, local: int = 8) -> dict:
    """Attention statistics per block and the context gain, on a few val batches."""
    sums = {}
    loss_full = loss_self = 0.0
    for seq in seqs:
        tokens, targets = seq[:, :-1], seq[:, 1:]
        logits, attn_maps = model(tokens, return_attn=True)
        loss_full += F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        L = tokens.shape[1]
        self_only = ~torch.eye(L, dtype=torch.bool, device=tokens.device)
        logits_self = model(tokens, attn_mask=self_only)
        loss_self += F.cross_entropy(logits_self.reshape(-1, logits_self.size(-1)), targets.reshape(-1)).item()

        q = torch.arange(min_query, L, device=tokens.device)
        k = torch.arange(L, device=tokens.device)
        near = (k[None, :] <= q[:, None]) & (k[None, :] > q[:, None] - local)   # (Q, L)
        log_visible = torch.log(q.float() + 1)
        for i, a in enumerate(attn_maps):
            a = a[:, :, min_query:, :].float()                                  # (B, H, Q, L)
            row = a.sum(-1)
            p = a / row.clamp_min(1e-9)[..., None]
            entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / log_visible
            live = row > 0
            s = sums.setdefault(f"block {i}", [0.0, 0.0, 0.0, 0])
            s[0] += (~live).float().mean().item()
            s[1] += entropy[live].mean().item() if live.any() else 0.0
            s[2] += ((a * near).sum() / a.sum().clamp_min(1e-9)).item()
            s[3] += 1
    n = len(seqs)
    return {"attn_empty": {g: s[0] / s[3] for g, s in sums.items()},
            "attn_entropy": {g: s[1] / s[3] for g, s in sums.items()},
            "attn_local": {g: s[2] / s[3] for g, s in sums.items()},
            "context_gain": (loss_self - loss_full) / n}


class AdaptiveClipper:
    """Per-block gradient clipping relative to each block's own recent gradient norm.

    For every group from block_of() ("input", "block i", "head") it keeps an EMA of
    the gradient norm; when a step's norm exceeds factor x EMA, that group's
    gradients are scaled down to factor x EMA. Ordinary steps pass untouched, only
    spikes are cut, and a spike in one block does not shrink the others. The EMA
    is updated with the (clipped) norm, so a spike does not inflate it.

    stats() returns, since the previous call, the fraction of steps each group was
    clipped ("clip_frac") and the largest norm / EMA seen ("grad_spike").
    """

    def __init__(self, model: nn.Module, factor: float = 3.0, beta: float = 0.99):
        self.factor, self.beta = factor, beta
        self.groups = {}
        for name, p in model.named_parameters():
            self.groups.setdefault(block_of(name), []).append(p)
        self.ema = {}
        self._reset_stats()

    def _reset_stats(self):
        self.steps = 0
        self.clipped = {g: 0 for g in self.groups}
        self.max_ratio = {g: 0.0 for g in self.groups}

    @torch.no_grad()
    def clip(self):
        self.steps += 1
        for g, params in self.groups.items():
            grads = [p.grad for p in params if p.grad is not None]
            if not grads:
                continue
            norm = torch.sqrt(sum(gr.float().pow(2).sum() for gr in grads)).item()
            if g not in self.ema:
                self.ema[g] = norm
            ratio = norm / max(self.ema[g], 1e-30)
            self.max_ratio[g] = max(self.max_ratio[g], ratio)
            if ratio > self.factor:
                scale = self.factor / ratio
                for gr in grads:
                    gr.mul_(scale)
                norm *= scale
                self.clipped[g] += 1
            self.ema[g] = self.beta * self.ema[g] + (1 - self.beta) * norm

    def stats(self) -> dict:
        out = {"clip_frac": {g: c / max(self.steps, 1) for g, c in self.clipped.items()},
               "grad_spike": dict(self.max_ratio)}
        self._reset_stats()
        return out
