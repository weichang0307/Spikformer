"""Bit-level simulation of the hardware block datapath for a spiking run, and its accuracy.

Datapath simulated (inference, multi-step over T time steps):
    weights      int8, one scale per tensor (--weights tensor) or per output channel (--weights channel; the
                 norm then needs a per-element rescale) or float (--weights float)
    linear in    Q/K/V and fc1 inputs (the integer residual stream) -> leading bit, rounded, shift saturated at
                 <<3 (1, 2, 4, 8); binary inputs (block 0 Q/K/V, proj, fc2) unchanged
    norm         on integer accumulators: mean and mean absolute deviation (MAD) as exact sums and shifts,
                 1 / (MAD + eps) from a leading-one detector and a lookup table indexed by the top --lut bits of the
                 mantissa, then n = (x - mean) * reciprocal, rounded to --frac fractional bits and saturated to
                 --bits bits (signed)
    LIF          gamma and beta folded in: u <- (u + n + beta/gamma) >> 1, spike when u >= 1/gamma (u <= 1/gamma when
                 gamma < 0), u -= 1/gamma on a spike; u, 1/gamma, beta/gamma in the same fixed-point format.
                 Neurons with |gamma| < --gamma_min ignore their input: a fixed spike pattern from beta alone
    attention    exact integers (counts, count^2, sums), the 1/8 scale folded into the head norm's eps
The input norm and input LIF (outside the blocks) stay in float. --exact skips every rounding, to check that
the folding is an identity. Reports test bits/token.

Usage:
    python3 fixed_point_eval.py RUN [--weights tensor] [--frac 8] [--bits 16] [--lut 6] [--gamma_min 0.05]
    python3 fixed_point_eval.py RUN --sweep
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn

from char_data import CharDataset
from models import model_from_config
from modules import L1LayerNorm
from neurons import LIFNeuron
from shift_mult_eval import approx, test_bits


class Cfg:
    exact = False
    frac = 8
    bits = 16
    lut = 6
    gamma_min = 0.05
    recip = "lut"         # 1/(MAD+eps): "lut" (table + multiply), "shift_round" / "shift_cut" (MAD -> its leading bit,
                          # rounded on the first cut bit or cut: a shift), "shift2_round" (reciprocal -> 2 leading bits: shift-add)
    recip_norms = None    # None = every norm; else a set of norm names using `recip` (the others use the table)
    int_stats = False     # mean and MAD rounded to integers (head norm: MAD to 1 fractional bit), as a 12-bit MAD register
    dtype = torch.float32 # simulation precision (float64 makes the integer sums exact)
    round_mode = "even"   # "even": round half to even (torch.round); "up": round half up = floor(x + 1/2), one adder
    saturate = False      # clamp registers: 18-bit signed accumulators, 16-bit attention sums, 12-bit (11.1) MAD
    const = "pattern"     # neurons with |gamma| < gamma_min: "pattern" (spikes from beta alone) or "zero" (never fire)


def rnd(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + 0.5) if Cfg.round_mode == "up" else torch.round(x)


def exp2i(e: torch.Tensor) -> torch.Tensor:
    """Exact 2^e for integer-valued e in [-80, 80] (torch.pow(2.0, e) is computed via exp/log on GPU: inexact)."""
    table = torch.tensor([2.0 ** k for k in range(-80, 81)], dtype=torch.float64, device=e.device)
    return table[e.long() + 80].to(e.dtype if e.is_floating_point() else torch.float64)


def fx(x: torch.Tensor) -> torch.Tensor:
    """Round to Cfg.frac fractional bits and saturate to a signed Cfg.bits-bit integer (as a float)."""
    if Cfg.exact:
        return x
    s = 2.0 ** Cfg.frac
    lim = 2.0 ** (Cfg.bits - 1)
    return torch.clamp(rnd(x * s), -lim, lim - 1) / s


def lead_bits(d: torch.Tensor, bits: int, mode: str) -> torch.Tensor:
    """Keep the leading `bits` significant bits of positive reals d, rounded on the first cut bit or cut."""
    e = (torch.frexp(d).exponent - 1).to(d.dtype)
    step = exp2i(e - (bits - 1))
    return torch.floor(d / step + (0.5 if mode == "round" else 0.0)) * step


def reciprocal(d: torch.Tensor, mode: str = "lut") -> torch.Tensor:
    """1/d from the leading one and a 2^lut-entry table of 1/mantissa (entries rounded to lut + 2 bits)."""
    if Cfg.exact:
        return 1.0 / d
    if mode == "shift_round":
        return 1.0 / lead_bits(d, 1, "round")       # a power of two: the multiply is a shift
    if mode == "shift_cut":
        return 1.0 / lead_bits(d, 1, "cut")
    if mode == "shift2_round":
        return lead_bits(1.0 / d, 2, "round")      # two terms: one shift-add per element
    e = (torch.frexp(d).exponent - 1).to(d.dtype)              # exact leading-one position (log2 is inexact on GPU)
    m = d / exp2i(e)                                   # [1, 2)
    idx = torch.floor((m - 1) * 2 ** Cfg.lut)                   # top lut bits after the leading one
    mid = 1 + (idx + 0.5) / 2 ** Cfg.lut                        # table entry = 1 / centre of the interval
    r = rnd((1 / mid) * 2 ** (Cfg.lut + 2)) / 2 ** (Cfg.lut + 2)
    return r * exp2i(-e)


class FixedNorm(nn.Module):
    """L1 norm without gamma / beta on integer-unit inputs: x / unit (unit = weight scale, or the attention
    scale) is an integer accumulator plus a rounded integer bias."""

    def __init__(self, norm: L1LayerNorm, unit, bias_int: bool, name: str = ""):
        super().__init__()
        self.name = name
        self.groups, self.eps = norm.groups, norm.eps
        self.unit = unit                       # float, or a (dim,) tensor for per-channel weights
        self.bias_int = bias_int

    def forward(self, x):
        u = self.unit
        xi = x.to(Cfg.dtype) / u
        if not Cfg.exact and self.bias_int:
            xi = rnd(xi)                       # accumulator + bias rounded to the accumulator's unit
        if Cfg.saturate and not Cfg.exact:
            xi = xi.clamp(-2 ** 17, 2 ** 17 - 1) if self.bias_int else xi.clamp(0, 2 ** 16 - 1)
        if torch.is_tensor(u) and u.dim() == 1:
            xi = xi * u / u.max()              # per-channel: rescale to a common unit (a shift for pow2 scales)
            u = u.max()
        g = xi.view(*xi.shape[:-1], self.groups, -1)
        mean = g.mean(-1, keepdim=True)        # exact: sum then shift
        if Cfg.int_stats and not Cfg.exact:
            mean = rnd(mean)
        c = g - mean
        mad = c.abs().mean(-1, keepdim=True)
        if Cfg.int_stats and not Cfg.exact:
            fb = 1 if self.name == "attn_norm" else 0
            mad = rnd(mad * 2 ** fb) / 2 ** fb
            if Cfg.saturate:
                mad = mad.clamp_max(2 ** 11 - 2 ** -fb)
        mad = mad + self.eps / u
        if getattr(Cfg, "recip_map", None):
            mode = Cfg.recip_map.get(self.name, "lut")       # per-norm choice
        else:
            mode = Cfg.recip if Cfg.recip_norms is None or self.name in Cfg.recip_norms else "lut"
        n = fx(c * reciprocal(mad, mode))
        return n.reshape(x.shape)


class FoldedLIF(LIFNeuron):
    def __init__(self, lif: LIFNeuron, norm: L1LayerNorm):
        super().__init__(lif.tau, lif.v_threshold)
        assert lif.tau == 2.0 and lif.v_threshold == 1.0
        g, b = norm.weight.detach().to(Cfg.dtype), norm.bias.detach().to(Cfg.dtype)
        self.const = g.abs() < Cfg.gamma_min                   # input ignored: spikes from beta alone
        gs = torch.where(self.const, torch.ones_like(g), g)
        self.theta = fx(1.0 / gs)
        self.boff = fx(b / gs)
        self.neg = gs < 0
        self.beta = b

    def _forward_multistep(self, x):
        xs = x.view(self.steps, -1, *x.shape[1:]).to(Cfg.dtype)
        u = torch.zeros_like(xs[0])
        v_c = torch.zeros_like(xs[0])                           # constant neurons: ordinary LIF on beta
        out, vs = [], []
        for x_t in xs:
            u = fx(u + x_t + self.boff)
            u = torch.floor(u * 2 ** Cfg.frac / 2) / 2 ** Cfg.frac if not Cfg.exact else u / 2   # arithmetic >> 1
            fire = torch.where(self.neg, u <= self.theta, u >= self.theta)
            u = u - fire.to(Cfg.dtype) * self.theta
            v_c = (v_c + self.beta) / 2
            fire_c = (v_c >= 1) & (Cfg.const == "pattern")
            v_c = v_c - fire_c.to(Cfg.dtype)
            out.append(torch.where(self.const, fire_c, fire).to(x.dtype))
            if self.record_v:                                    # for monitors: the folded state
                vs.append(u.detach())
        self.v = torch.cat(vs) if self.record_v else None
        return torch.cat(out)


def build(run, weights, fold=True):
    config = json.load(open(os.path.join("runs", run, "config.json")))
    state = torch.load(os.path.join("runs", run, "best.pt"), map_location="cpu", weights_only=False)
    m = model_from_config(config, config["block_size"])
    m.load_state_dict(state.get("model", state))
    m.cuda().eval()
    for b in m.blocks:
        a, mlp = b.attn, b.mlp
        units = {}
        for name, lin in (("q", a.q_linear), ("k", a.k_linear), ("v", a.v_linear), ("proj", a.proj_linear),
                          ("fc1", mlp.fc1), ("fc2", mlp.fc2)):
            w = lin.weight.data
            if weights == "tensor":
                sc = w.abs().max() / 127
            elif weights == "channel":
                sc = w.abs().amax(1, keepdim=True) / 127
            elif weights == "channel_pow2":       # per-channel scale = a power of two: the rescale is a shift
                sc = torch.pow(2.0, torch.ceil(torch.log2(w.abs().amax(1, keepdim=True) / 127)))
            else:
                sc = None
            if sc is not None:
                lin.weight.data = rnd(w / sc) * sc
                units[name] = sc if weights == "tensor" else sc.flatten()
            else:
                units[name] = w.abs().max() / 127          # float weights: still an integer-ish unit for the norm
        if fold:
            pairs = (("q_norm", "q_lif", a, units["q"], True), ("k_norm", "k_lif", a, units["k"], True),
                     ("v_norm", "v_lif", a, units["v"], True), ("attn_norm", "attn_lif", a, a.scale, False),
                     ("proj_norm", "proj_lif", a, units["proj"], True), ("norm1", "lif1", mlp, units["fc1"], True),
                     ("norm2", "lif2", mlp, units["fc2"], True))
            for nn_, ln, owner, unit, bint in pairs:
                norm, lif = getattr(owner, nn_), getattr(owner, ln)
                setattr(owner, ln, FoldedLIF(lif, norm).cuda())
                setattr(owner, nn_, FixedNorm(norm, unit, bint and weights != "float", nn_))
        for lin in (a.q_linear, a.k_linear, a.v_linear, mlp.fc1):
            lin.register_forward_pre_hook(lambda mod, inp: (approx(inp[0].float(), 1, "round").clamp_max(8).to(inp[0].dtype),))
    for mod in m.modules():
        if isinstance(mod, FoldedLIF):
            for k in ("theta", "boff", "beta"):
                setattr(mod, k, getattr(mod, k).cuda())
            mod.neg, mod.const = mod.neg.cuda(), mod.const.cuda()
    return m, config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--weights", choices=("tensor", "channel", "channel_pow2", "float"), default="tensor")
    p.add_argument("--frac", type=int, default=8)
    p.add_argument("--bits", type=int, default=16)
    p.add_argument("--lut", type=int, default=6)
    p.add_argument("--gamma_min", type=float, default=0.05)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--const", choices=("pattern", "zero"), default="pattern",
                   help="neurons with |gamma| < --gamma_min: stored spike pattern, or never fire")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--extra", action="store_true", help="with --sweep: only the follow-up rows")
    args = p.parse_args()
    data = None

    def evaluate(weights, exact, frac, bits, lut, gmin, fold=True):
        nonlocal data
        Cfg.exact, Cfg.frac, Cfg.bits, Cfg.lut, Cfg.gamma_min = exact, frac, bits, lut, gmin
        m, config = build(args.run, weights, fold)
        if data is None:
            data = CharDataset(config["data_path"], 0.05, 0.05)
        return test_bits(m, data, config["T"], amp=False)      # float32: integer accumulators stay exact

    if not args.sweep:
        Cfg.const = args.const
        r = evaluate(args.weights, args.exact, args.frac, args.bits, args.lut, args.gamma_min)
        print(f"{args.weights} weights, {'exact' if args.exact else f'Q{args.bits - args.frac - 1}.{args.frac} ({args.bits}-bit), lut {args.lut}'}"
              f", gamma_min {args.gamma_min}, fixed neurons: {args.const}: {r:.4f} bits/token")
        return
    print("all scores in float32 (bits/token on the full test split)\n")
    if not args.extra:
        print(f"{'float weights + saturated 1-bit shifts, original norm and LIF (reference)':76s} {evaluate('float', True, 8, 16, 6, 0.0, fold=False):.4f}", flush=True)
        print(f"{'int8 per-channel weights + shifts, original norm and LIF':76s} {evaluate('channel', True, 8, 16, 6, 0.0, fold=False):.4f}", flush=True)
    rows = [("float", True, 8, 16, 6, 0.0, "float weights, shifts, folded LIF in exact arithmetic (identity check)"),
            ("float", True, 8, 16, 6, 0.05, "  + fixed spike patterns for |gamma| < 0.05"),
            ("tensor", True, 8, 16, 6, 0.05, "int8 per-tensor weights, exact norm/LIF"),
            ("channel", True, 8, 16, 6, 0.05, "int8 per-channel weights, exact norm/LIF")]
    for f in (4, 6, 8, 10):
        rows.append(("tensor", False, f, 16, 6, 0.05, f"int8 per-tensor, 16-bit with {f} fractional bits, lut 6"))
    for lut in (3, 4, 8):
        rows.append(("tensor", False, 8, 16, lut, 0.05, f"int8 per-tensor, 16-bit / 8 fractional, lut {lut}"))
    for bits in (12, 14):
        rows.append(("tensor", False, bits - 8, bits, 6, 0.05, f"int8 per-tensor, {bits}-bit / {bits - 8} fractional, lut 6"))
    for gm in (0.02, 0.1):
        rows.append(("tensor", False, 8, 16, 6, gm, f"int8 per-tensor, 16-bit / 8 fractional, lut 6, gamma_min {gm}"))
    rows = [r for r in rows if r[0] != "tensor" or not r[6].startswith("int8 per-tensor,")] if args.extra else rows
    if args.extra:
        rows = [("channel_pow2", True, 8, 16, 6, 0.05, "int8 per-channel power-of-two scales, exact norm/LIF"),
                ("channel_pow2", False, 8, 16, 4, 0.05, "int8 per-channel pow2, 16-bit / 8 fractional, lut 4"),
                ("channel", False, 8, 16, 4, 0.05, "int8 per-channel, 16-bit / 8 fractional, lut 4"),
                ("tensor", False, 8, 12, 6, 0.05, "int8 per-tensor, 12-bit / 8 fractional (range +-8), lut 6"),
                ("tensor", False, 7, 12, 6, 0.05, "int8 per-tensor, 12-bit / 7 fractional (range +-16), lut 6"),
                ("tensor", False, 8, 16, 2, 0.05, "int8 per-tensor, 16-bit / 8 fractional, lut 2")]
    for w, ex, f, b, lut, gm, label in rows:
        print(f"{label:76s} {evaluate(w, ex, f, b, lut, gm):.4f}", flush=True)


if __name__ == "__main__":
    main()
