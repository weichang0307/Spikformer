"""Shift-based multiplies for the integer-input linear layers of a spiking run (post-training, no retraining).

The residual stream feeding Q/K/V and fc1 holds small non-negative integers (0..2k+1 in block k). Each input x
is replaced by its leading 1 or 2 significant bits, so x * w becomes a shift (1 bit: w << e) or a shift and one
add (2 bits: (w << e) + (w << e-1)). The first dropped bit is either cut (truncate) or used to round
(half up; 3 -> 4 with 1 bit, 7 -> 8 with 2 bits: still a power of two or two terms). Layers with binary
inputs (block 0 Q/K/V, projections, fc2) are unchanged. Reports test bits/token per variant, per group of
layers, and the mean error of the approximated inputs.

Usage:
    python3 shift_mult_eval.py RUN [--T 8] [--groups]
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from char_data import CharDataset
from models import model_from_config


def exp2i(e: torch.Tensor) -> torch.Tensor:
    """Exact 2^e for integer-valued e in [-80, 80] (torch.pow(2.0, e) is computed via exp/log on GPU: inexact)."""
    table = torch.tensor([2.0 ** k for k in range(-80, 81)], dtype=torch.float64, device=e.device)
    return table[e.long() + 80].to(e.dtype if e.is_floating_point() else torch.float64)


def approx(x: torch.Tensor, bits: int, mode: str) -> torch.Tensor:
    """Keep the leading `bits` significant bits of non-negative integers x; mode 'cut' or 'round'."""
    e = (torch.frexp(x.clamp_min(1)).exponent - 1).to(x.dtype)     # exact (log2 is inexact on GPU)
    step = exp2i((e - (bits - 1)).clamp_min(0))
    q = torch.floor(x / step + (0.5 if mode == "round" else 0.0)) * step
    return torch.where(x > 0, q, x)


@torch.no_grad()
def test_bits(model, data, T, context=256, stride=128, batch=32, amp=True):
    model.T = T
    ids = data.splits["test"].long()
    starts = list(range(0, len(ids) - context, stride))
    tot, n = 0.0, 0
    for b in range(0, len(starts), batch):
        s = starts[b:b + batch]
        x = torch.stack([ids[i:i + context + 1] for i in s]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x[:, :-1]).float()
        nll = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none") / math.log(2)
        for k, i in enumerate(s):
            lo = 0 if i == 0 else context - stride
            tot += nll[k, lo:].sum().item(); n += context - lo
    return tot / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--T", type=int, default=0, help="time steps (default: the run's)")
    p.add_argument("--groups", action="store_true", help="also apply to Q/K/V only and to fc1 only")
    args = p.parse_args()
    config = json.load(open(os.path.join("runs", args.run, "config.json")))
    state = torch.load(os.path.join("runs", args.run, f"{args.ckpt}.pt"), map_location="cpu", weights_only=False)
    model = model_from_config(config, config["block_size"])
    model.load_state_dict(state.get("model", state))
    model.cuda().eval()
    data = CharDataset(config["data_path"], config.get("val_fraction", 0.05), config.get("test_fraction", 0.05))
    T = args.T or config["T"]

    layers = {"qkv": [m for b in model.blocks for m in (b.attn.q_linear, b.attn.k_linear, b.attn.v_linear)],
              "fc1": [b.mlp.fc1 for b in model.blocks]}
    errs = {}

    def run(bits, mode, groups):
        hooks = []
        def pre(mod, inp):
            x = inp[0]
            xi = x.float()
            q = approx(xi, bits, mode)
            nz = xi > 0
            if nz.any():
                e = errs.setdefault((bits, mode), [0.0, 0.0, 0])
                e[0] += ((q - xi)[nz] / xi[nz]).sum().item(); e[1] += ((q - xi)[nz] / xi[nz]).abs().sum().item()
                e[2] += nz.sum().item()
            return (q.to(x.dtype),)
        for g in groups:
            for m in layers[g]:
                hooks.append(m.register_forward_pre_hook(pre))
        try:
            return test_bits(model, data, T)
        finally:
            for h in hooks:
                h.remove()

    base = test_bits(model, data, T)
    print(f"{args.run} at T={T}: exact multiplies {base:.4f} bits/token\n")
    print(f"{'variant':28s} {'test bits/token':>15s} {'change':>8s} {'mean input error':>17s} {'mean |error|':>13s}")
    for bits in (1, 2):
        for mode in ("cut", "round"):
            errs.clear()
            b = run(bits, mode, ("qkv", "fc1"))
            e = errs[(bits, mode)]
            print(f"{f'{bits} bit, {mode}, Q/K/V + fc1':28s} {b:15.4f} {b - base:+8.4f} {100 * e[0] / e[2]:16.1f}% {100 * e[1] / e[2]:12.1f}%",
                  flush=True)
            if args.groups:
                for g in ("qkv", "fc1"):
                    bg = run(bits, mode, (g,))
                    print(f"{f'   only {g}':28s} {bg:15.4f} {bg - base:+8.4f}", flush=True)


if __name__ == "__main__":
    main()
