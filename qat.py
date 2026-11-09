"""Quantization-aware training (QAT) of a spiking model's blocks: the hardware datapath of fixed_point_eval.py in
the forward pass, with straight-through gradients (every rounding is the identity in the backward pass).

enable_qat(model) patches, in place (parameter names and checkpoints are unchanged):
    linear layers   weights and biases fake-quantized to int8, one scale per tensor (bias in accumulator units)
    linear inputs   Q/K/V and fc1 inputs (the integer residual stream) -> leading bit, rounded, saturated at 8
    L1 norms        integer mean, MAD rounded to 12 bits (11.1: the fraction only for the attention head norm),
                    1/(MAD + eps) from the leading one and a 2^lut-entry table, n rounded to Q4.7 and saturated;
                    gamma and beta are not applied here ...
    LIF neurons     ... but folded into the LIF: u <- (u + n + beta/gamma) >> 1, spike when gamma (u - 1/gamma) >= 0,
                    u -= 1/gamma on a spike, all in Q4.7; neurons with |gamma| < gamma_min (at enable time) never fire
The attention integers (counts, count^2, sums) are already exact in the model. Gradients reach gamma and beta
through the folded constants and the surrogate spike, whose argument gamma (u - theta) equals v - 1 of the
ordinary LIF. Evaluate the result bit-faithfully (float32) with fixed_point_eval.py.
"""

import math
import types

import torch
import torch.nn.functional as F

from neurons import spike_fn

FRAC, BITS, LUT, GAMMA_MIN = 7, 12, 4, 0.05


def exp2i(e: torch.Tensor) -> torch.Tensor:
    """Exact 2^e for integer-valued e in [-80, 80] (torch.pow(2.0, e) is computed via exp/log on GPU: inexact)."""
    table = torch.tensor([2.0 ** k for k in range(-80, 81)], dtype=torch.float64, device=e.device)
    return table[e.long() + 80].to(e.dtype if e.is_floating_point() else torch.float64)


def ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Forward q, backward identity to x."""
    return x + (q - x).detach()


def fx(x: torch.Tensor) -> torch.Tensor:
    s, lim = 2.0 ** FRAC, 2.0 ** (BITS - 1)
    return ste(x, torch.clamp(torch.round(x * s), -lim, lim - 1) / s)


def lead1_round_sat8(x: torch.Tensor) -> torch.Tensor:
    e = (torch.frexp(x.clamp_min(1)).exponent - 1).to(x.dtype)     # exact (log2 is inexact on GPU)
    step = exp2i(e)
    q = torch.floor(x / step + 0.5) * step
    return torch.where(x > 0, q, x).clamp_max(8)


def lut_reciprocal(d: torch.Tensor) -> torch.Tensor:
    e = (torch.frexp(d).exponent - 1).to(d.dtype)          # exact leading-one position (log2 is inexact on GPU)
    m = d / exp2i(e)
    idx = torch.floor((m - 1) * 2 ** LUT)
    mid = 1 + (idx + 0.5) / 2 ** LUT
    r = torch.round((1 / mid) * 2 ** (LUT + 2)) / 2 ** (LUT + 2)
    return r * exp2i(-e)


def weight_scale(lin) -> torch.Tensor:
    return lin.weight.detach().abs().max() / 127


def _linear_forward(self, x):
    s = weight_scale(self)
    w = ste(self.weight, torch.round(self.weight / s) * s)
    b = None if self.bias is None else ste(self.bias, torch.round(self.bias / s) * s)
    return F.linear(x, w, b)


def _shift_pre_hook(mod, inp):
    x = inp[0]
    return (ste(x, lead1_round_sat8(x.detach().float()).to(x.dtype)),)


def _make_norm_forward(unit_fn, head: bool):
    def forward(self, x):
        u = unit_fn()
        xf = x.float() / u
        g = xf.view(*xf.shape[:-1], self.groups, -1)
        mean = ste(g.mean(-1, keepdim=True), torch.round(g.mean(-1, keepdim=True)).detach())
        c = g - mean
        mad = c.abs().mean(-1, keepdim=True)
        fb = 1 if head else 0
        mad = ste(mad, (torch.round(mad * 2 ** fb) / 2 ** fb).detach())
        d = mad + self.eps / u
        r = ste(1.0 / d, lut_reciprocal(d.detach()))
        return fx(c * r).reshape(x.shape)                     # n: no gamma / beta (folded into the LIF)
    return forward


def _make_lif_multistep(norm):
    def multistep(self, x):
        const = self.qat_const
        g, b = norm.weight.float(), norm.bias.float()
        gs = torch.where(const, torch.ones_like(g), g)
        theta, boff = fx(1.0 / gs), fx(b / gs)
        xs = x.view(self.steps, -1, *x.shape[1:]).float()
        u = torch.zeros_like(xs[0])
        out, vs = [], []
        for x_t in xs:
            u = fx(u + x_t + boff)
            u = ste(u / 2, (torch.floor(u.detach() * 2 ** FRAC / 2) / 2 ** FRAC))   # arithmetic >> 1
            s = spike_fn(gs * (u - theta)) * (~const)
            u = u - s * theta
            out.append(s.to(x.dtype))
            if self.record_v:
                vs.append(u.detach())
        self.v = torch.cat(vs) if self.record_v else None
        return torch.cat(out)
    return multistep


def enable_qat(model, gamma_min: float = GAMMA_MIN, verbose: bool = True):
    n_const = 0
    for blk in model.blocks:
        a, mlp = blk.attn, blk.mlp
        for lin in (a.q_linear, a.k_linear, a.v_linear, a.proj_linear, mlp.fc1, mlp.fc2):
            lin.forward = types.MethodType(_linear_forward, lin)
        for lin in (a.q_linear, a.k_linear, a.v_linear, mlp.fc1):
            lin.register_forward_pre_hook(_shift_pre_hook)
        pairs = (("q_norm", "q_lif", a, a.q_linear), ("k_norm", "k_lif", a, a.k_linear),
                 ("v_norm", "v_lif", a, a.v_linear), ("attn_norm", "attn_lif", a, None),
                 ("proj_norm", "proj_lif", a, a.proj_linear), ("norm1", "lif1", mlp, mlp.fc1),
                 ("norm2", "lif2", mlp, mlp.fc2))
        for nname, lname, owner, lin in pairs:
            norm, lif = getattr(owner, nname), getattr(owner, lname)
            assert lif.tau == 2.0 and lif.v_threshold == 1.0
            unit_fn = (lambda lin=lin: weight_scale(lin)) if lin is not None else (lambda a=a: a.scale)
            norm.forward = types.MethodType(_make_norm_forward(unit_fn, lin is None), norm)
            const = (norm.weight.detach().abs() < gamma_min)
            n_const += int(const.sum())
            lif.register_buffer("qat_const", const.clone(), persistent=False)   # moves with the model, not saved
            lif._forward_multistep = types.MethodType(_make_lif_multistep(norm), lif)
    if verbose:
        print(f"QAT: int8 per-tensor weights, 1-bit shifts (sat 8), integer norm stats, {2 ** LUT}-entry reciprocal, "
              f"Q{BITS - FRAC - 1}.{FRAC} norm/LIF, gamma/beta folded; {n_const} neurons with |gamma| < {gamma_min} never fire",
              flush=True)
    return model
