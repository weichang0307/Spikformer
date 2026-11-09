"""Full evaluation of a spiking run as the hardware computes it (fixed_point_eval.py's reference simulation, float32).

    test bits/token  at several time-step counts T
    token types      mean bits per token by token type, against a float reference run and the ANN teacher
    activity         spike rate of every LIF population, spikes per token, linear-layer add events per token and
                     weight-column fetches per token (each layer running all T steps together)
    tunes            n generated tunes (temperature 1): quality checks, why repeat checks fail, ABC syntax errors
Writes runs/<run>/eval_fixed.json and runs/<run>/generated_tunes_fixed.txt.

Usage:
    python3 eval_fixed.py RUN [--reference RUN] [--teacher RUN] [--n 500]
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from abc_quality import BARS, NOTE, VOCAB, check, decode, duration_factor
from analyze_folk import CATEGORIES, category, repeat_failures
from char_data import CharDataset
from fixed_point_eval import Cfg, build
from health import LIFMonitor
from models import model_from_config


def set_design():
    Cfg.exact, Cfg.frac, Cfg.bits, Cfg.lut, Cfg.gamma_min, Cfg.const = False, 7, 12, 4, 0.05, "zero"
    Cfg.recip, Cfg.recip_norms, Cfg.recip_map, Cfg.int_stats = "lut", None, None, True


@torch.no_grad()
def token_bits(model, data, T, amp, context=256, stride=128, batch=32):
    model.T = T
    ids = data.splits["test"].long()
    starts = list(range(0, len(ids) - context, stride))
    out = torch.full((len(ids),), float("nan"))
    for b in range(0, len(starts), batch):
        s = starts[b:b + batch]
        x = torch.stack([ids[i:i + context + 1] for i in s]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x[:, :-1]).float()
        nll = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none") / math.log(2)
        for k, i in enumerate(s):
            lo = 0 if i == 0 else context - stride
            out[i + 1 + lo:i + 1 + context] = nll[k, lo:].cpu()
    return out


def syntax_errors(t):
    e = []
    if len(t) < 2 or not t[0].startswith("M:") or not t[1].startswith("K:"):
        e.append("bad header")
    if any(x.startswith(("M:", "K:")) for x in t[2:]):
        e.append("M:/K: inside the body")
    in_chord, prev, tup = False, None, 0
    for x in t[2:]:
        note = bool(NOTE.match(x))
        dur = duration_factor(x) is not None and not note
        if x == "[":
            in_chord = True
        elif x == "]":
            if not in_chord:
                e.append("] without [")
            in_chord = False
        elif x in BARS:
            if in_chord:
                e.append("bar line inside chord")
            if tup:
                e.append("triplet cut by bar line")
            tup = 0
        elif x.startswith("("):
            tup = int(x[1:])
        elif x in (">", "<"):
            if not (prev and (NOTE.match(prev) or prev == "]" or duration_factor(prev) is not None)):
                e.append("> < not after a note")
        elif dur and not (prev and (NOTE.match(prev) or prev == "]")):
            e.append("duration not after a note")
        if note and not in_chord and tup:
            tup -= 1
        prev = x
    if in_chord:
        e.append("unclosed [")
    return e


@torch.no_grad()
def generate(model, chars, n, max_len=600, batch=100):
    nl = chars.index("\n")
    tunes = []
    torch.manual_seed(0)
    while len(tunes) < n:
        b = min(batch, n - len(tunes))
        ids = torch.full((b, 1), nl, device="cuda")
        done = torch.zeros(b, dtype=torch.bool, device="cuda")
        for _ in range(max_len):
            nxt = torch.multinomial(torch.softmax(model(ids[:, -256:])[:, -1].float(), -1), 1)
            nxt[done] = nl
            ids = torch.cat([ids, nxt], 1)
            done |= nxt[:, 0] == nl
            if done.all():
                break
        tunes += [decode("".join(chars[i] for i in row).split("\n")[0]) for row in ids[:, 1:].tolist()]
    return tunes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--reference", default="folkrnn_256d6L_T8_distill_homeo2s_sq8_spread8_50k_do05")
    p.add_argument("--teacher", default="folkrnn_256d6L_ann_base")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--Ts", default="4,6,8")
    p.add_argument("--skip_bits", action="store_true", help="skip the test-score and token-type parts")
    args = p.parse_args()
    set_design()
    model, config = build(args.run, "tensor")
    data = CharDataset(config["data_path"], 0.05, 0.05)
    T0 = config["T"]
    res = {"run": args.run, "design": "int8 per-tensor, 1-bit shifts sat 8, integer norm stats, 16-entry reciprocal, "
                                      "Q4.7 norm/LIF, gamma/beta folded, |gamma|<0.05 never fire"}

    res["bits_by_T"] = {}
    for T in ([] if args.skip_bits else [int(t) for t in args.Ts.split(",")]):
        b = token_bits(model, data, T, amp=False)
        res["bits_by_T"][T] = round(b.nanmean().item(), 4)
        print(f"fixed-point, T={T}: test {res['bits_by_T'][T]:.4f} bits/token", flush=True)
    model.T = T0
    if args.skip_bits:
        prev = json.load(open(os.path.join("runs", args.run, "eval_fixed.json"))) if os.path.exists(os.path.join("runs", args.run, "eval_fixed.json")) else {}
        res.update({k: v for k, v in prev.items() if k in ("bits_by_T", "by_category")})
    else:
        fixed = token_bits(model, data, T0, amp=False)

        def float_model(run):
            c = json.load(open(os.path.join("runs", run, "config.json")))
            s = torch.load(os.path.join("runs", run, "best.pt"), map_location="cpu", weights_only=False)
            m = model_from_config(c, c["block_size"]); m.load_state_dict(s.get("model", s))
            return m.cuda().eval(), c
        ref, rc = float_model(args.reference)
        refb = token_bits(ref, data, rc["T"], amp=True)
        ann, _ = float_model(args.teacher)
        annb = token_bits(ann, data, 1, amp=True)
        print(f"float reference {args.reference}: {refb.nanmean().item():.4f} | teacher: {annb.nanmean().item():.4f}", flush=True)
        toks = [VOCAB.get(data.chars[i], data.chars[i]) for i in data.splits["test"].tolist()]
        by = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        for cat, a, b, c in zip((category(t) for t in toks), fixed.tolist(), refb.tolist(), annb.tolist()):
            if not math.isnan(a):
                by[cat][0] += a; by[cat][1] += b; by[cat][2] += c; by[cat][3] += 1
        tot = sum(v[3] for v in by.values())
        res["by_category"] = {k: {"share": round(100 * by[k][3] / tot, 1), "fixed": round(by[k][0] / by[k][3], 3),
                                  "float_ref": round(by[k][1] / by[k][3], 3), "ann": round(by[k][2] / by[k][3], 3)}
                              for k in CATEGORIES if by[k][3]}
        print("\ntoken type        share   fixed-point  float ref  ANN")
        for k, v in res["by_category"].items():
            print(f"{k:16s} {v['share']:5.1f}%   {v['fixed']:9.3f}  {v['float_ref']:9.3f}  {v['ann']:6.3f}")

    # activity
    torch.manual_seed(0)
    vv = data.splits["val"]
    x = torch.stack([vv[i:i + 256].long() for i in torch.randint(0, len(vv) - 257, (4,)).tolist()]).cuda()
    rec, hooks = {}, []
    fan = {"q_linear": 768, "proj_linear": 256, "fc1": 1024, "fc2": 256}
    for bi, blk in enumerate(model.blocks):
        for name, mod in (("q_linear", blk.attn.q_linear), ("proj_linear", blk.attn.proj_linear), ("fc1", blk.mlp.fc1), ("fc2", blk.mlp.fc2)):
            hooks.append(mod.register_forward_hook(lambda mod, i, o, k=(bi, name): rec.__setitem__(k, i[0].detach())))
    mon = LIFMonitor(model); mon.start()
    model(x)
    for h in mon.handles + hooks:
        h.remove()
    for mm in mon.names:
        mm.record_v = False
    rates = defaultdict(dict)
    spikes = 0.0
    for name, s in mon.spikes.items():
        parts = name.split(".")
        r = (s.sum() / (mon.counts[name] * s.numel())).item()
        if parts[0] == "blocks":
            rates[parts[-1]][int(parts[1])] = round(r, 4)
        spikes += s.sum().item() / x.numel()
    adds = fetch = 0.0
    for (bi, name), t in rec.items():
        t = t.float().view(T0, -1, t.shape[-1])
        adds += (t != 0).float().sum(-1).mean().item() * fan[name] * T0
        fetch += (t != 0).any(0).float().sum(-1).mean().item() * fan[name]
    res["activity"] = {"spikes_per_token": round(spikes), "linear_add_events_per_token": round(adds),
                       "weight_fetches_per_token": round(fetch), "firing_rate": rates}
    print(f"\nactivity: {spikes:,.0f} spikes/token, {adds / 1e6:.2f}M linear add events/token, "
          f"{fetch / 1e6:.2f}M weight fetches/token (all {T0} steps per layer together)")
    for k, v in rates.items():
        print(f"  {k:9s} " + "  ".join(f"b{b}:{100 * r:4.1f}%" for b, r in sorted(v.items())))

    # tunes
    tunes = generate(model, config["chars"], args.n)
    open(os.path.join("runs", args.run, "generated_tunes_fixed.txt"), "w").write("\n".join(" ".join(t) for t in tunes) + "\n")
    cs = [check(t) for t in tunes if t]
    bars = sum(c["bars"] for c in cs)
    q = {"header": 100 * sum(c["header"] for c in cs) / len(cs), "bars": 100 * sum(c["valid_bars"] for c in cs) / max(bars, 1),
         "repeats": 100 * sum(c["repeats"] for c in cs) / len(cs), "tonic": 100 * sum(c["tonic"] for c in cs) / len(cs)}
    res["tunes"] = {k: round(v, 1) for k, v in q.items()} | {"median_length": sorted(len(t) for t in tunes)[len(tunes) // 2]}
    res["repeat_outcomes"] = repeat_failures(tunes, 600)
    errs = Counter(); bad = 0
    for t in tunes:
        es = set(syntax_errors(t)); bad += bool(es); errs.update(es)
    res["syntax"] = {"tunes_with_error_pct": round(100 * bad / len(tunes), 1), "errors": dict(errs.most_common())}
    print(f"\ntunes ({args.n}): header {q['header']:.1f}% | correct bars {q['bars']:.1f}% | repeats {q['repeats']:.1f}% | "
          f"ends on C {q['tonic']:.1f}% | median length {res['tunes']['median_length']}")
    print(f"repeat outcomes: {res['repeat_outcomes']}")
    print(f"syntax: {res['syntax']['tunes_with_error_pct']}% of tunes with an error {res['syntax']['errors']}")
    json.dump(res, open(os.path.join("runs", args.run, "eval_fixed.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
