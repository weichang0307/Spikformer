"""Is the current training run doing better than the best previous one? Compare them at matching steps.

The current run defaults to the run whose live.json changed most recently (within --minutes). The reference
defaults to the finished run of the same corpus and model type with the lowest best validation loss. The
corpus' ANN and LSTM runs (best by validation) are drawn as dotted context lines.

Prints validation and training loss (bits/token) with their gap every --every steps, the validation difference
at the latest common step (averaged over the last few evals, as single evals are noisy), and a verdict.
Writes runs/analysis/compare.png: validation (solid) and training (dashed) loss, full range and zoom.

Usage:
    python3 compare_runs.py                         # current run vs best finished run of its kind
    python3 compare_runs.py RUN                     # RUN vs best finished run of its kind
    python3 compare_runs.py RUN REFERENCE [MORE...] # explicit runs
"""

import argparse
import glob
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs"
COLORS = ["#9467bd", "#2ca02c", "#d62728", "#ff7f0e", "#8c564b", "#e377c2"]


def load(run: str):
    """(config, history) of a run; history has step / bpc (val, bits) and maybe train_loss (nats)."""
    d = os.path.join(RUNS, run)
    config = json.load(open(os.path.join(d, "config.json")))
    for name in ("live.json", "history.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            h = json.load(open(p))
            if h.get("step") and h.get("bpc"):
                return config, h
    return config, None


def describe(run: str) -> dict:
    config, h = load(run)
    p = os.path.join(RUNS, run, "live.json")
    if not os.path.exists(p):
        p = os.path.join(RUNS, run, "history.json")
    return {"run": run, "config": config, "h": h, "mtime": os.path.getmtime(p) if os.path.exists(p) else 0,
            "corpus": os.path.dirname(config.get("data_path", "")), "type": config.get("model_type", "spike")}


def all_runs() -> list:
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*", "config.json"))):
        try:
            r = describe(os.path.basename(os.path.dirname(d)))
        except (json.JSONDecodeError, OSError):
            continue
        if r["h"]:
            out.append(r)
    return out


def best_of(runs: list, **match):
    pool = [r for r in runs if all(r[k] == v for k, v in match.items())]
    return min(pool, key=lambda r: min(r["h"]["bpc"]), default=None)


def train_bits(h):
    return [v / math.log(2) for v in h["train_loss"]] if "train_loss" in h else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="current run, then reference run(s)")
    p.add_argument("--minutes", type=float, default=15, help="a run counts as 'current' if updated this recently")
    p.add_argument("--every", type=int, default=5000, help="table interval in steps")
    p.add_argument("--out", default=os.path.join(RUNS, "analysis", "compare.png"))
    args = p.parse_args()

    runs = all_runs()
    by_name = {r["run"]: r for r in runs}
    now = time.time()
    live = [r for r in runs if now - r["mtime"] < args.minutes * 60]
    if args.runs:
        missing = [n for n in args.runs if n not in by_name]
        if missing:
            raise SystemExit(f"no history for: {', '.join(missing)}")
        cur = by_name[args.runs[0]]
    elif live:
        cur = max(live, key=lambda r: r["mtime"])
    else:
        raise SystemExit(f"no run updated in the last {args.minutes:g} minutes; name one: compare_runs.py RUN")
    finished = [r for r in runs if r["run"] != cur["run"] and r not in live]
    refs = [by_name[n] for n in args.runs[1:]] if len(args.runs) > 1 else \
        [x for x in [best_of(finished, corpus=cur["corpus"], type=cur["type"])] if x]
    if not refs:
        raise SystemExit("no finished run of the same corpus and model type to compare with; name one")
    context = [x for t in ("ann", "lstm") if t != cur["type"]
               for x in [best_of(finished, corpus=cur["corpus"], type=t)] if x]

    ch = cur["h"]
    status = "running" if cur in live else "finished"
    total = cur["config"].get("steps") or 0
    print(f"current:   {cur['run']} ({status}, step {ch['step'][-1]:,}" + (f" / {total:,})" if total else ")"))
    for r in refs:
        rh = r["h"]
        i = min(range(len(rh["bpc"])), key=lambda i: rh["bpc"][i])
        print(f"reference: {r['run']} (best val {rh['bpc'][i]:.3f} at step {rh['step'][i]:,}, final step {rh['step'][-1]:,})")
    for r in context:
        print(f"context:   {r['run']} ({r['type']}, best val {min(r['h']['bpc']):.3f})")

    # table at matching steps: current vs the first reference
    ref = refs[0]["h"]
    rmap = {s: (v, t) for s, v, t in zip(ref["step"], ref["bpc"], train_bits(ref) or [None] * len(ref["step"]))}
    ctrain = train_bits(ch) or [None] * len(ch["step"])
    print(f"\n{'step':>7} | {'reference val':>13} {'train':>6} {'gap':>5} | {'current val':>11} {'train':>6} {'gap':>5} | {'val diff':>8}")
    common = []
    for s, v, t in zip(ch["step"], ch["bpc"], ctrain):
        if s in rmap:
            common.append((s, v, rmap[s][0]))
            if s % args.every == 0 or s == ch["step"][-1]:
                rv, rt = rmap[s]
                g = lambda val, tr: f"{val - tr:5.2f}" if tr is not None else "    -"
                print(f"{s:7,} | {rv:13.3f} {rt if rt is not None else float('nan'):6.3f} {g(rv, rt)} | "
                      f"{v:11.3f} {t if t is not None else float('nan'):6.3f} {g(v, t)} | {v - rv:+8.3f}")
    if not common:
        print("no common steps yet")
    else:
        last = common[-4:]   # average the last few evals: single evals are noisy
        diff = sum(v - r for _, v, r in last) / len(last)
        s_last = common[-1][0]
        cur_best, ref_best_so_far = min(v for _, v, _ in common), min(r for _, _, r in common)
        verdict = "better than" if diff < -0.005 else "worse than" if diff > 0.005 else "level with"
        print(f"\nat step {s_last:,} (mean of the last {len(last)} evals): current is {verdict} the reference by "
              f"{abs(diff):.3f} bits/token; best so far {cur_best:.3f} vs {ref_best_so_far:.3f} for the reference "
              f"up to the same step (its overall best: {min(ref['bpc']):.3f})")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(18, 6.5))
    series = [(cur, COLORS[0], "-")] + [(r, COLORS[1 + k % (len(COLORS) - 1)], "-") for k, r in enumerate(refs)] + \
             [(r, "grey" if r["type"] == "lstm" else "black", ":") for r in context]
    xmax = max(max(r["h"]["step"]) for r, _, _ in series)
    for r, c, ls in series:
        h = r["h"]
        label = ("CURRENT: " if r is cur else "" if ls == "-" else f"{r['type'].upper()}: ") + r["run"]
        for a in ax:
            a.plot(h["step"], h["bpc"], color=c, ls=ls, lw=2, label=label + (" - val" if ls == "-" else ""))
            tb = train_bits(h)
            if tb and ls == "-":
                a.plot(h["step"], tb, color=c, ls="--", lw=1.1, alpha=.8, label=label + " - train")
        i = min(range(len(h["bpc"])), key=lambda i: h["bpc"][i])
        ax[1].plot(h["step"][i], h["bpc"][i], "o", color=c, ms=7)
    lo = min(min(r["h"]["bpc"]) for r, _, _ in series)
    cur_last = ch["bpc"][-1]
    ax[0].set(xlabel="step", ylabel="bits/token", title="validation (solid) and training (dashed) loss",
              xlim=(0, xmax), ylim=(lo - 0.3, lo + 1.3))
    ax[1].set(xlabel="step", ylabel="bits/token", title="zoom (dots = best validation)",
              xlim=(xmax * 0.1, xmax), ylim=(lo - 0.25, max(cur_last, lo) + 0.3))
    ax[0].legend(fontsize=8); [a.grid(alpha=.3) for a in ax]
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
