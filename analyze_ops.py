"""Operation and value-range analysis of a spiking run's blocks, for hardware design.

Runs the model on validation windows (multi-step, all T steps) and measures, per block and time step:

    linear layers    input sparsity and value range: a layer whose input is binary needs only additions
                     (one weight column added per spike); an integer input (the residual stream, 0..2k+1)
                     needs small multiplies or repeated adds. Reported as additions per token per time step
                     if every input unit's value v costs v column adds (accumulate-only), and as MACs if a
                     non-zero input costs one multiply-add.
    attention        Q.K overlap counts (AND + popcount over the 32-bit head), the share of causal
                     (query, key) pairs that overlap at all, count^2 range, the attention-times-V accumulators
                     (sum of count^2 over keys whose V spike is 1) and their bit width.
    norms            L1 LayerNorm inputs: range, MAD range (the divisor) and whether a power-of-two divisor
                     would fit; outputs feed the LIFs.
    LIF              input current and membrane range, firing rate.
Prints a table per block and writes runs/<run>/ops.json.

Usage:
    python3 analyze_ops.py RUN [--windows 4]
"""

import argparse
import json
import math
import os
from collections import defaultdict

import torch

from char_data import CharDataset
from models import model_from_config
from neurons import LIFNeuron


def qstats(x: torch.Tensor) -> dict:
    x = x.float().flatten()
    nz = x != 0
    return {"min": round(x.min().item(), 4), "max": round(x.max().item(), 4),
            "p99.9": round(x.abs().quantile(0.999).item() if x.numel() < 16_000_000 else
                           x.abs()[torch.randint(0, x.numel(), (4_000_000,), device=x.device)].quantile(0.999).item(), 4),
            "nonzero": round(nz.float().mean().item(), 4), "mean": round(x.mean().item(), 4)}


def bits_unsigned(m: float) -> int:
    return max(1, math.ceil(math.log2(m + 1)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--ckpt", default="best")
    p.add_argument("--windows", type=int, default=4)
    args = p.parse_args()
    config = json.load(open(os.path.join("runs", args.run, "config.json")))
    state = torch.load(os.path.join("runs", args.run, f"{args.ckpt}.pt"), map_location="cpu", weights_only=False)
    model = model_from_config(config, config["block_size"])
    model.load_state_dict(state.get("model", state))
    model.cuda().eval()
    data = CharDataset(config["data_path"], config.get("val_fraction", 0.05), config.get("test_fraction", 0.05))
    L, T, D, H = config["block_size"], config["T"], config["dim"], config["num_heads"]
    hd = D // H
    torch.manual_seed(0)
    v = data.splits["val"]
    x = torch.stack([v[i:i + L].long() for i in torch.randint(0, len(v) - L - 1, (args.windows,)).tolist()]).cuda()
    B = x.shape[0]

    rec = defaultdict(dict)        # block -> name -> tensor captured
    hooks = []
    for bi, blk in enumerate(model.blocks):
        a, m = blk.attn, blk.mlp
        def cap(name, bi=bi, what="in"):
            def f(mod, inp, out):
                rec[bi][name] = (inp[0] if what == "in" else out).detach()
            return f
        for name, mod in (("q_linear", a.q_linear), ("k_linear", a.k_linear), ("v_linear", a.v_linear),
                          ("proj_linear", a.proj_linear), ("fc1", m.fc1), ("fc2", m.fc2)):
            hooks.append(mod.register_forward_hook(cap(name)))
        for name, mod in (("q_norm", a.q_norm), ("k_norm", a.k_norm), ("v_norm", a.v_norm), ("attn_norm", a.attn_norm),
                          ("proj_norm", a.proj_norm), ("norm1", m.norm1), ("norm2", m.norm2)):
            hooks.append(mod.register_forward_hook(cap(name + ".in")))
            hooks.append(mod.register_forward_hook(cap(name + ".out", what="out")))
        for name, mod in (("q_lif", a.q_lif), ("k_lif", a.k_lif), ("v_lif", a.v_lif), ("attn_lif", a.attn_lif),
                          ("proj_lif", a.proj_lif), ("lif1", m.lif1), ("lif2", m.lif2)):
            hooks.append(mod.register_forward_hook(cap(name, what="out")))
        hooks.append(blk.register_forward_hook(cap("block_out", what="out")))
    lifs = [mod for mod in model.modules() if isinstance(mod, LIFNeuron)]
    for mod in lifs:
        mod.record_v = True
    vrec = {}
    def vcap(name):
        def f(mod, inp, out):
            vrec[name] = mod.v.detach()
        return f
    for n, mod in model.named_modules():
        if isinstance(mod, LIFNeuron):
            hooks.append(mod.register_forward_hook(vcap(n)))
    model(x)
    for h in hooks:
        h.remove()
    for mod in lifs:
        mod.record_v = False

    tok = B * L                     # tokens; every count below is per token per time step unless noted
    out = {"run": args.run, "T": T, "dim": D, "heads": H, "head_dim": hd, "context": L, "blocks": []}
    causal_pairs = L * (L + 1) / 2 / L   # mean keys per query over a full window
    for bi in range(len(model.blocks)):
        r = rec[bi]
        blk = {"block": bi, "linear": {}, "norm": {}, "lif": {}}
        for name in ("q_linear", "k_linear", "v_linear", "proj_linear", "fc1", "fc2"):
            inp = r[name].float()            # (T*B, L, fan_in)
            lin = getattr(model.blocks[bi].attn if name in ("q_linear", "k_linear", "v_linear", "proj_linear")
                          else model.blocks[bi].mlp, name)
            fan_in, fan_out = lin.in_features, lin.out_features
            s = qstats(inp)
            vals = sorted(set(inp.unique().tolist()))
            adds = inp.sum().item() / (T * tok) * fan_out              # value v costs v column adds
            macs = (inp != 0).sum().item() / (T * tok) * fan_out       # one MAC per non-zero input
            blk["linear"][name] = {"fan_in": fan_in, "fan_out": fan_out, "binary": vals in ([0.0, 1.0], [0.0], [1.0]),
                                   "input_values": vals[:16], "input_bits": bits_unsigned(max(vals)),
                                   "input_nonzero": s["nonzero"], "input_mean": s["mean"],
                                   "adds_per_token_step": round(adds), "macs_per_token_step": round(macs),
                                   "dense_macs": fan_in * fan_out}
        # attention: counts per time step and head
        q = r["q_lif"].view(T, B, L, H, hd).permute(0, 1, 3, 2, 4).float()
        k = r["k_lif"].view(T, B, L, H, hd).permute(0, 1, 3, 2, 4).float()
        vv = r["v_lif"].view(T, B, L, H, hd).permute(0, 1, 3, 2, 4).float()
        c = q @ k.transpose(-1, -2)                                       # (T, B, H, L, L)
        mask = torch.ones(L, L, device=c.device, dtype=torch.bool).tril()
        cm = c[..., mask]
        acc = (c * c * mask) @ vv                                         # sum_j c_ij^2 v_j  (before the 1/8 scale)
        pairs = mask.sum().item()
        nzp = (cm > 0).float().mean().item()
        hist = torch.bincount(cm.long().flatten(), minlength=hd + 1)[:hd + 1].float()
        hist = (hist / hist.sum()).tolist()
        blk["attention"] = {
            "q_rate": round(q.mean().item(), 4), "k_rate": round(k.mean().item(), 4), "v_rate": round(vv.mean().item(), 4),
            "pairs_per_query_full_window": round(pairs / L, 1),
            "overlapping_pairs": round(nzp, 4),
            "count_max": int(cm.max().item()), "count_mean_when_overlapping": round(cm[cm > 0].mean().item(), 2),
            "count_hist": [round(h, 4) for h in hist[:12]],
            "count_sq_max": int(cm.max().item() ** 2),
            "acc_max": int(acc.max().item()), "acc_p99.9": round(acc.flatten()[torch.randint(0, acc.numel(), (4_000_000,), device=acc.device)].quantile(0.999).item(), 1),
            "acc_bits": bits_unsigned(acc.max().item()),
            # per token per time step, full 256-token window, all heads:
            "and_popcounts": round(pairs / L * H),                         # one 32-bit AND + popcount per (query, key, head)
            "squares": round(nzp * pairs / L * H),                         # only overlapping pairs need c^2
            "v_accumulates": round(((c > 0) * mask).sum().item() / (T * B * L) * vv.mean().item() * hd),
        }
        for name in ("q_norm", "k_norm", "v_norm", "attn_norm", "proj_norm", "norm1", "norm2"):
            xin = r[name + ".in"].float()
            mod = getattr(model.blocks[bi].attn if name not in ("norm1", "norm2") else model.blocks[bi].mlp, name)
            g = mod.groups
            xg = xin.view(*xin.shape[:-1], g, -1)
            mad = (xg - xg.mean(-1, keepdim=True)).abs().mean(-1)
            blk["norm"][name] = {"width": xin.shape[-1] // g, "groups": g, "in": qstats(xin),
                                 "mad_min": round(mad.min().item(), 4), "mad_max": round(mad.max().item(), 4),
                                 "mad_log2_span": round(math.log2(mad.max().item() / max(mad.min().item(), 1e-9)), 1),
                                 "out": qstats(r[name + ".out"])}
        for name in ("q_lif", "k_lif", "v_lif", "attn_lif", "proj_lif", "lif1", "lif2"):
            full = f"blocks.{bi}.{'attn' if name not in ('lif1', 'lif2') else 'mlp'}.{name}"
            blk["lif"][name] = {"rate": round(r[name].float().mean().item(), 4), "v_after_reset": qstats(vrec[full])}
        res = r["block_out"].float()
        blk["residual_out"] = {"values": sorted(set(res.unique().tolist()))[:20], "max": res.max().item(),
                               "bits": bits_unsigned(res.max().item())}
        out["blocks"].append(blk)

    # totals per token (all T steps, all blocks), full window
    tot = defaultdict(float)
    for blk in out["blocks"]:
        for name, l in blk["linear"].items():
            tot["adds (linear, accumulate-only)"] += l["adds_per_token_step"] * T
            tot["macs (linear, one per non-zero input)"] += l["macs_per_token_step"] * T
            tot["dense macs (same layers, ANN-style)"] += l["dense_macs"]
        a = blk["attention"]
        tot["attention AND+popcount (32-bit)"] += a["and_popcounts"] * T
        tot["attention squares"] += a["squares"] * T
        tot["attention V accumulates"] += a["v_accumulates"] * T
        tot["norm elements"] += sum(n["width"] * n["groups"] for n in blk["norm"].values()) * T
        tot["LIF updates"] += (3 * D + D + D + 4 * D + D) * T
    out["per_token_totals"] = {k: round(v) for k, v in tot.items()}
    json.dump(out, open(os.path.join("runs", args.run, "ops.json"), "w"), indent=1)

    # print
    print(f"{args.run}: T={T}, {len(model.blocks)} blocks, dim {D}, {H} heads x {hd}, context {L}; "
          f"{B} validation windows\n")
    print("linear layers (per token per time step): input values | non-zero | adds if value v = v adds | MACs per non-zero")
    for blk in out["blocks"]:
        for name, l in blk["linear"].items():
            vals = l["input_values"]
            vs = "binary" if l["binary"] else f"0..{int(max(vals))} ({l['input_bits']} bit)"
            print(f"  b{blk['block']} {name:11s} {l['fan_in']:4d}->{l['fan_out']:<4d} {vs:15s} nz {100 * l['input_nonzero']:5.1f}%  "
                  f"adds {l['adds_per_token_step']:7,d}  macs {l['macs_per_token_step']:7,d}  (dense {l['dense_macs']:,})")
    print("\nattention (per query, per time step, all 8 heads, full 256 window):")
    for blk in out["blocks"]:
        a = blk["attention"]
        print(f"  b{blk['block']}: q/k/v rate {a['q_rate']:.3f}/{a['k_rate']:.3f}/{a['v_rate']:.3f} | overlapping pairs "
              f"{100 * a['overlapping_pairs']:4.1f}% | count max {a['count_max']} (mean {a['count_mean_when_overlapping']} when > 0) "
              f"| sum c^2 v max {a['acc_max']:,} ({a['acc_bits']} bit) | ops: {a['and_popcounts']:,} AND+pop, "
              f"{a['squares']:,} squares, {a['v_accumulates']:,} V adds")
        print(f"      count histogram 0..11: {[round(100 * h, 1) for h in a['count_hist']]}")
    print("\nL1 norms: input range | MAD range (log2 span) | output range")
    for blk in out["blocks"]:
        for name, n in blk["norm"].items():
            print(f"  b{blk['block']} {name:9s} in [{n['in']['min']:8.2f}, {n['in']['max']:8.2f}]  MAD [{n['mad_min']:.3f}, "
                  f"{n['mad_max']:.3f}] ({n['mad_log2_span']} octaves)  out [{n['out']['min']:6.2f}, {n['out']['max']:6.2f}] "
                  f"p99.9 |out| {n['out']['p99.9']:.2f}")
    print("\nLIF: firing rate | membrane after reset range")
    for blk in out["blocks"]:
        print(f"  b{blk['block']} " + "  ".join(f"{n} {100 * l['rate']:4.1f}% [{l['v_after_reset']['min']:.2f},{l['v_after_reset']['max']:.2f}]"
                                         for n, l in blk["lif"].items()))
    print("\nresidual stream after each block: " + ", ".join(f"b{b['block']} 0..{int(b['residual_out']['max'])} ({b['residual_out']['bits']} bit)"
                                                          for b in out["blocks"]))
    print("\nper token, all T steps and blocks, full 256-token window:")
    for k, v in out["per_token_totals"].items():
        print(f"  {k:42s} {v:14,d}")


if __name__ == "__main__":
    main()
