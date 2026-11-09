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
  gamma_min / gamma_max, beta_min / beta_max
                smallest / largest gain and bias over all L1LayerNorm features of
                the block (input = the input norm)
  attn_scale    SSA scale 2^s (geometric mean over heads with per-head scales)
  head_scale_spread  largest / smallest per-head scale (1 with a per-block scale)
  attn_dead_heads    fraction of heads whose attention LIF fires < 0.5% of the time
  attn_empty    fraction of (head, query) rows whose Q.K^T overlap is all zero
  attn_entropy  attention spread: row entropy / log(#visible keys); 1 = uniform
  attn_local    share of attention mass on the current and previous 7 positions

With adaptive clipping (train_char.py --adaptive_clip), per block, over the
steps since the previous evaluation:
  clip_frac     fraction of steps whose gradients were clipped
  grad_spike    largest gradient norm / its running average (spikes stand out)

With the homeostatic bias rule (train_char.py --homeostasis), per block:
  homeo_active  fraction of Q/K/V features firing below the target (being pushed up)
  homeo_high    with --homeo_upper: fraction firing above the upper target (pushed down)
  homeo_shift   mean cumulative homeostatic bias change of the Q/K/V L1 norms
  homeo_shift_up / homeo_shift_down  the same, counting only raised / only lowered
                features (mean over all features, so the two add up to homeo_shift)

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
        for m in self.names:  # multi-step LIFs keep every step's potential for the surrogate statistic
            m.record_v = True
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
        for m in self.names:
            m.record_v = False
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
    scales, spread = {}, {}
    for i, blk in enumerate(model.blocks):
        s = blk.attn.log2_scale
        scales[f"block {i}"] = blk.attn.scale if s is None else 2.0 ** s.float().mean().item()
        spread[f"block {i}"] = 1.0 if s is None or s.dim() == 0 else 2.0 ** (s.max() - s.min()).item()
    gains, gmin, gmax, bmin, bmax = {}, {}, {}, {}, {}
    for name, m in model.named_modules():
        if isinstance(m, L1LayerNorm):
            g = block_of(name)
            gains.setdefault(g, []).append(m.weight.abs().mean().item())
            gmin[g] = min(gmin.get(g, float("inf")), m.weight.min().item())
            gmax[g] = max(gmax.get(g, float("-inf")), m.weight.max().item())
            bmin[g] = min(bmin.get(g, float("inf")), m.bias.min().item())
            bmax[g] = max(bmax.get(g, float("-inf")), m.bias.max().item())
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[block_of(name)] = grads.get(block_of(name), 0.0) + p.grad.float().pow(2).sum().item()
    return {"attn_scale": scales, "head_scale_spread": spread,
            "norm_gain": {g: sum(v) / len(v) for g, v in gains.items()},
            "gamma_min": gmin, "gamma_max": gmax, "beta_min": bmin, "beta_max": bmax,
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
    head_sums, head_calls = {}, {}

    def head_hook(i, heads):  # per-head attention-LIF firing, summed over time steps
        def hook(module, inputs, spikes):
            per_head = spikes.detach().float().view(-1, heads, spikes.shape[-1] // heads).mean(dim=(0, 2))
            head_sums[i] = head_sums.get(i, 0) + per_head
            head_calls[i] = head_calls.get(i, 0) + 1
        return hook

    for seq in seqs:
        tokens, targets = seq[:, :-1], seq[:, 1:]
        hooks = [blk.attn.attn_lif.register_forward_hook(head_hook(i, blk.attn.num_heads))
                 for i, blk in enumerate(model.blocks) if blk.attn.attn_lif is not None]
        logits, attn_maps = model(tokens, return_attn=True)
        for h in hooks:  # count firing on the normal pass only, not the self-only one
            h.remove()
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
    dead_heads = {f"block {i}": ((head_sums[i] / head_calls[i]) < 0.005).float().mean().item()
                  for i in head_sums}
    return {"attn_dead_heads": dead_heads,
            "attn_empty": {g: s[0] / s[3] for g, s in sums.items()},
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

    def state_dict(self) -> dict:
        return {"ema": dict(self.ema)}

    def load_state_dict(self, state: dict):
        self.ema = dict(state["ema"])

    def stats(self) -> dict:
        out = {"clip_frac": {g: c / max(self.steps, 1) for g, c in self.clipped.items()},
               "grad_spike": dict(self.max_ratio)}
        self._reset_stats()
        return out


class Homeostasis:
    """Homeostatic bias rule for NL (L1LayerNorm -> LIF) units. `units` picks which:
    "qkv" (Q, K, V of every block), "attn" (attention NL), "proj" (projection NL),
    "mlp" (both MLP NLs), "input" (the input NL), or "all" for every NL unit.

    During training forward passes it keeps a running (EMA) firing rate for each
    feature of q_lif, k_lif and v_lif. After every optimizer step, each feature
    firing below `target` gets its L1LayerNorm bias raised by
    eta * (1 - rate / target), i.e. its LIF threshold is effectively lowered
    until it fires again. With `upper` > 0 the rule is two-sided: a feature
    firing above `upper` gets its bias lowered by
    eta * (rate - upper) / (1 - upper), pulling near-constant (saturated)
    features back into the informative range. Features inside the band are
    left alone. With `max_shift` > 0 each feature's cumulative shift is kept
    within [-max_shift, max_shift] (anti-windup): a feature that cannot be
    brought into the band within that range is left to the gradient instead of
    being pushed until its bias alone decides whether it fires.
    This is a direct update, not a gradient: silent neurons sit far below
    threshold where the surrogate gradient is ~0, so the loss cannot revive them.

    For the attention NL the push up is input-aware: a feature is only raised
    while its pre-norm input (attn · V) is nonzero on at least `target` of the
    tokens, so a feature whose attention input is empty is not made to fire
    from its bias alone (activity that carries no information).

    stats() returns per block the fraction of regulated features below `target`
    ("homeo_active"), above `upper` ("homeo_high"), and the mean bias shift so
    far ("homeo_shift").
    """

    def __init__(self, model: nn.Module, target: float = 0.01, eta: float = 1e-3, beta: float = 0.99,
                 upper: float = 0.0, units: tuple = ("qkv",), max_shift: float = 0.0):
        self.target, self.eta, self.beta, self.upper = target, eta, beta, upper
        self.max_shift = max_shift
        if "all" in units:
            units = ("input", "qkv", "attn", "proj", "mlp")
        self.units = []  # (group, lif, norm), group = "input" or "block i"
        gated = []       # norms whose push up needs a non-empty input (the attention NL)
        if "input" in units:
            self.units.append(("input", model.input_lif, model.input_norm))
        for i, blk in enumerate(model.blocks):
            a, g = blk.attn, f"block {i}"
            if "qkv" in units:
                self.units += [(g, a.q_lif, a.q_norm), (g, a.k_lif, a.k_norm), (g, a.v_lif, a.v_norm)]
            if "attn" in units and a.attn_norm is not None:
                self.units.append((g, a.attn_lif, a.attn_norm))
                gated.append(a.attn_norm)
            if "proj" in units and a.attn_proj:
                self.units.append((g, a.proj_lif, a.proj_norm))
            if "mlp" in units:
                self.units.append((g, blk.mlp.lif1, blk.mlp.norm1))
                if blk.mlp.layers == 2:
                    self.units.append((g, blk.mlp.lif2, blk.mlp.norm2))
        self.model = model
        # spike / nonzero-input counters are preallocated and updated in place, so the
        # hooks keep a fixed structure and torch.compile does not recompile them
        self.rate, self.in_rate = {}, {}
        self.sums = {id(norm): torch.zeros_like(norm.bias) for _, _, norm in self.units}
        self.counts = {id(norm): torch.zeros((), device=norm.bias.device) for _, _, norm in self.units}
        self.in_sums = {id(norm): torch.zeros_like(norm.bias) for norm in gated}
        self.in_counts = {id(norm): torch.zeros((), device=norm.bias.device) for norm in gated}
        self.shift = {id(norm): torch.zeros_like(norm.bias) for _, _, norm in self.units}
        self.handles = [lif.register_forward_hook(self._hook(id(norm))) for _, lif, norm in self.units]
        self.handles += [norm.register_forward_pre_hook(self._input_hook(id(norm))) for norm in gated]

    def _input_hook(self, key):
        def hook(module, inputs):
            if self.model.training:
                nz = (inputs[0].detach() != 0).float()
                self.in_sums[key].add_(nz.reshape(-1, nz.shape[-1]).sum(0))
                self.in_counts[key].add_(nz[..., 0].numel())
        return hook

    def _hook(self, key):
        def hook(module, inputs, spikes):
            if self.model.training:
                s = spikes.detach().float()
                self.sums[key].add_(s.reshape(-1, s.shape[-1]).sum(0))
                self.counts[key].add_(s[..., 0].numel())
        return hook

    @torch.no_grad()
    def step(self):
        """Call after optimizer.step()."""
        for _, _, norm in self.units:
            key = id(norm)
            r = self.sums[key] / self.counts[key].clamp_min(1)
            self.sums[key].zero_()
            self.counts[key].zero_()
            self.rate[key] = r if key not in self.rate else self.beta * self.rate[key] + (1 - self.beta) * r
            push = self.eta * (1 - self.rate[key] / self.target).clamp(0, 1)
            if key in self.in_sums:  # attention NL: only raise features that receive input
                r_in = self.in_sums[key] / self.in_counts[key].clamp_min(1)
                self.in_sums[key].zero_()
                self.in_counts[key].zero_()
                self.in_rate[key] = (r_in if key not in self.in_rate
                                     else self.beta * self.in_rate[key] + (1 - self.beta) * r_in)
                push = push * (self.in_rate[key] >= self.target).float()
            if self.upper > 0:
                push = push - self.eta * ((self.rate[key] - self.upper) / (1 - self.upper)).clamp(0, 1)
            if self.max_shift > 0:  # keep the cumulative shift bounded
                push = (self.shift[key] + push).clamp(-self.max_shift, self.max_shift) - self.shift[key]
            norm.bias.add_(push)
            self.shift[key] += push

    def state_dict(self) -> dict:
        """Running rates and cumulative shifts in unit order (ids differ between processes)."""
        keys = [id(norm) for _, _, norm in self.units]
        return {"rate": [self.rate.get(k) for k in keys], "in_rate": [self.in_rate.get(k) for k in keys],
                "shift": [self.shift[k] for k in keys]}

    def load_state_dict(self, state: dict):
        for (_, _, norm), r, ri, sh in zip(self.units, state["rate"], state["in_rate"], state["shift"]):
            k = id(norm)
            if r is not None:
                self.rate[k] = r.to(norm.bias.device)
            if ri is not None:
                self.in_rate[k] = ri.to(norm.bias.device)
            self.shift[k] = sh.to(norm.bias.device)
        self.enforce_bound()

    @torch.no_grad()
    def enforce_bound(self):
        """Pull shifts beyond max_shift (e.g. from a run resumed with a new bound) back to it,
        removing the excess from the norm bias too."""
        if self.max_shift <= 0:
            return
        for _, _, norm in self.units:
            k = id(norm)
            excess = self.shift[k] - self.shift[k].clamp(-self.max_shift, self.max_shift)
            norm.bias.sub_(excess)
            self.shift[k] -= excess

    def stats(self) -> dict:
        active, high, shift, up, down = {}, {}, {}, {}, {}
        for i, _, norm in self.units:
            key = id(norm)
            if key in self.rate:
                active.setdefault(i, []).append((self.rate[key] < self.target).float().mean().item())
                if self.upper > 0:
                    high.setdefault(i, []).append((self.rate[key] > self.upper).float().mean().item())
            sh = self.shift[key]
            shift.setdefault(i, []).append(sh.mean().item())
            up.setdefault(i, []).append(sh.clamp(min=0).mean().item())
            down.setdefault(i, []).append(sh.clamp(max=0).mean().item())
        out = {"homeo_shift_up": {g: sum(v) / len(v) for g, v in up.items()},
               "homeo_shift_down": {g: sum(v) / len(v) for g, v in down.items()},
               "homeo_active": {g: sum(v) / len(v) for g, v in active.items()},
               "homeo_shift": {g: sum(v) / len(v) for g, v in shift.items()}}
        if self.upper > 0:
            out["homeo_high"] = {g: sum(v) / len(v) for g, v in high.items()}
        return out
