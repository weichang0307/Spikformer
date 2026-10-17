"""Plot training curves from runs' live.json files into one figure.

First two panels: eval accuracy (recall or next char) at the largest evaluated
T (solid) and at T=4 (dotted); eval loss at the largest T (solid) and training
loss (faint), log scale.

For runs that log training-health metrics (train_char.py, see health.py), one
panel per metric follows, in a two-column grid: firing rate, never-fired and
saturated features, surrogate gradient, learning rate, gradient norm, update/weight ratio,
norm gain, attention scale and per-head scale spread, dead heads, empty attention rows, attention spread, attention
on the last 8 positions, context gain, and (with adaptive clipping) the share of
steps clipped and the largest gradient spike. Per-LIF metrics are averaged per
block; colour = block, line style = run.

Prints the latest values and a value every 1,000 steps for each run, plus the
latest health summary.

Usage:
    python3 plot_curves.py                       # runs updated in the last 10 minutes
    python3 plot_curves.py spike_8_32_32_T16 spike_12_32_32_T16
    python3 plot_curves.py --minutes 60 --out runs/analysis/curves.png
"""

import argparse
import json
import math
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (field, title, per-LIF, as percent, log y, y limits)
HEALTH = [
    ("fire_rate", "LIF firing rate (%)", True, True, False, (0, None)),
    ("fire_dead", "never-fired features (%)", True, True, False, (0, 100)),
    ("fire_sat", "saturated features, rate >= 90% (%)", True, True, False, (0, None)),
    ("surrogate", "mean surrogate gradient (peak 0.5)", True, False, False, (0, None)),
    ("lr", "learning rate", False, False, True, None),
    ("grad_norm", "gradient norm", False, False, True, None),
    ("update_ratio", "update / weight norm per step", False, False, True, None),
    ("norm_gain", "mean |L1 norm gain|", False, False, False, None),
    ("gamma_min", "min L1 norm gain (gamma) per block", False, False, False, None),
    ("gamma_max", "max L1 norm gain (gamma) per block", False, False, False, None),
    ("beta_min", "min L1 norm bias (beta) per block", False, False, False, None),
    ("beta_max", "max L1 norm bias (beta) per block", False, False, False, None),
    ("attn_scale", "SSA attention scale 2^s", False, False, True, None),
    ("head_scale_spread", "largest / smallest head scale", False, False, True, None),
    ("attn_dead_heads", "dead heads, attention LIF < 0.5% (%)", False, True, False, (0, 100)),
    ("attn_empty", "query rows with no Q.K overlap (%)", False, True, False, (0, 100)),
    ("attn_entropy", "attention spread (1 = uniform)", False, False, False, (0, 1)),
    ("attn_local", "attention on last 8 positions (%)", False, True, False, (0, None)),
    ("context_gain", "context gain: loss(self only) - loss (nats)", False, False, False, None),
    ("homeo_active", "Q/K/V features below target rate (%)", False, True, False, (0, 100)),
    ("homeo_high", "Q/K/V features above upper target (%)", False, True, False, (0, 100)),
    ("homeo_shift", "mean Q/K/V norm bias shift", False, False, False, None),
    ("homeo_shift_up", "Q/K/V bias raised (mean over features)", False, False, False, None),
    ("homeo_shift_down", "Q/K/V bias lowered (mean over features)", False, False, False, None),
    ("clip_frac", "steps clipped since last eval (%)", False, True, False, (0, None)),
    ("grad_spike", "largest gradient / running average", False, False, True, None),
]
GROUP_COLORS = {"input": "0.55", "head": "#8a6d3b", "block 0": "C1", "block 1": "C0", "block 2": "C2",
                "block 3": "C4", "block 4": "C6", "block 5": "C9"}


def recent_runs(runs_dir: str, minutes: float) -> list[str]:
    """Runs whose live.json changed within the last `minutes`, newest first."""
    now, found = time.time(), []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name, "live.json")
        if os.path.exists(path) and now - os.path.getmtime(path) < minutes * 60:
            found.append((os.path.getmtime(path), name))
    return [name for _, name in sorted(found, reverse=True)]


def group_of(name: str) -> str:
    m = re.match(r"blocks\.(\d+)\.", name)
    if m:
        return f"block {m.group(1)}"
    return name if name.startswith("block ") or name in ("input", "head") else "input"


def per_block(series) -> dict:
    """{name: [per eval]} -> {"input" / "block i" / "head": [mean over that group]};
    a plain list (scalar metric) becomes {"all": list}."""
    if isinstance(series, list):
        return {"all": series}
    groups = {}
    for name, values in series.items():
        groups.setdefault(group_of(name), []).append(values)
    rank = lambda g: -1 if g == "input" else 99 if g == "head" else int(g.split()[-1])
    out = {}
    for g in sorted(groups, key=rank):
        n = min(len(v) for v in groups[g])
        out[g] = [sum(v[len(v) - n + j] for v in groups[g]) / len(groups[g]) for j in range(n)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run names under --runs_dir (default: recently updated runs)")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--minutes", type=float, default=10, help="'recent' window when no runs are given")
    parser.add_argument("--out", default="runs/analysis/curves.png")
    args = parser.parse_args()

    runs = args.runs or recent_runs(args.runs_dir, args.minutes)
    if not runs:
        raise SystemExit(f"no run updated in the last {args.minutes:g} minutes; name runs explicitly")

    histories = {run: json.load(open(os.path.join(args.runs_dir, run, "live.json"))) for run in runs}
    for h in histories.values():  # derived: steps per second between evaluations (wall clock)
        if len(h.get("time", [])) == len(h["step"]):
            h["steps_per_s"] = [(s - (h["step"][i - 1] if i else 0)) / max(t - (h["time"][i - 1] if i else 0), 1e-9)
                                for i, (s, t) in enumerate(zip(h["step"], h["time"]))]
    def scale_fixed(field):  # attention-scale panels are skipped when the scale is not learnable
        if field not in ("attn_scale", "head_scale_spread"):
            return False
        values = [x for h in histories.values() if h.get(field)
                  for v in (per_block(h[field]).values()) for x in v]
        return len(set(values)) <= 1

    panels = [p for p in HEALTH if any(h.get(p[0]) for h in histories.values()) and not scale_fixed(p[0])]
    has_bpc = any(any(k.startswith("bpc_T") for k in h) for h in histories.values())
    rows = 1 + math.ceil((len(panels) + has_bpc) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.2 * rows), squeeze=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    acc_ax, loss_ax = axes[0]
    for color, run in zip(colors, runs):
        h = histories[run]
        Ts = sorted(int(k[5:]) for k in h if k.startswith("acc_T"))
        Tm = Ts[-1]
        acc_ax.plot(h["step"], [a * 100 for a in h[f"acc_T{Tm}"]], color=color, lw=1.3, label=f"{run}, T={Tm}")
        if 4 in Ts and Tm != 4:
            acc_ax.plot(h["step"], [a * 100 for a in h["acc_T4"]], color=color, lw=0.8, ls=":", label=f"{run}, T=4")
        loss_ax.plot(h["step"], h[f"loss_T{Tm}"], color=color, lw=1.3, label=f"{run}, eval")
        loss_ax.plot(h["step"], h["train_loss"], color=color, lw=0.8, alpha=0.4, label=f"{run}, train")

        i = -1
        latest = ", ".join(f"T={T} {h[f'acc_T{T}'][i] * 100:.1f}%" for T in Ts)
        speed = f", {h['steps_per_s'][i]:.2f} steps/s" if "steps_per_s" in h else ""
        print(f"{run}: step {h['step'][i]}, {h['time'][i] / 60:.0f} min{speed}, {latest}, eval loss {h[f'loss_T{Tm}'][i]:.3f}")
        print("   every 1,000 steps (step, accuracy %, loss):",
              [(s, round(a * 100, 1), round(l, 2)) for s, a, l in zip(h["step"], h[f"acc_T{Tm}"], h[f"loss_T{Tm}"])
               if s % 1000 == 0])
    acc_ax.set_title("accuracy (%) (recall / next char)", fontsize=9)  # y range follows the data
    loss_ax.set_title("loss (log scale)", fontsize=9); loss_ax.set_yscale("log")
    for ax in (acc_ax, loss_ax):
        ax.legend(fontsize=7)

    health_runs = [run for run in runs if any(histories[run].get(p[0]) for p in HEALTH)]
    styles = dict(zip(health_runs, ("-", "--", ":", "-.")))
    flat = [ax for row in axes[1:] for ax in row]
    if has_bpc:  # bits per character: val at each eval T (solid / dashed), train loss / ln 2 (faint)
        bpc_ax, flat = flat[0], flat[1:]
        for color, run in zip(colors, runs):
            h = histories[run]
            for T, ls in zip(sorted(int(k[5:]) for k in h if k.startswith("bpc_T")), ("-", "--", ":", "-.")):
                bpc_ax.plot(h["step"], h[f"bpc_T{T}"], color=color, lw=1.3, ls=ls, label=f"{run}, T={T}")
            if "bpc" in h:
                bpc_ax.plot(h["step"], [l / math.log(2) for l in h["train_loss"]], color=color, lw=0.8, alpha=0.4,
                            label=f"{run}, train")
        bpc_ax.set_title("bits per character (val; train faint)", fontsize=9)
        bpc_ax.legend(fontsize=6, ncol=2)
    for ax, (field, title, _, pct, log, _ylim) in zip(flat, panels):  # y range follows the data
        for run in health_runs:
            h = histories[run]
            if not h.get(field):
                continue
            k = 100 if pct else 1
            for g, values in per_block(h[field]).items():
                steps = h["step"][len(h["step"]) - len(values):]
                label = g if len(health_runs) == 1 else f"{run}, {g}"
                ax.plot(steps, [k * v for v in values], color=GROUP_COLORS.get(g, "C0"), ls=styles[run],
                        lw=1.2, label=label)
        ax.set_title(title, fontsize=9)
        if field == "fire_rate":
            ax.set_ylim(0, 100)  # firing rate always on the full 0-100% axis
        if field in ("fire_dead", "fire_sat") and ax.lines and max(max(l.get_ydata()) for l in ax.lines) < 10:
            ax.set_ylim(0, 10)  # 0-10% axis while every value is below 10%, adaptive above that
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=6, ncol=3)
    for ax in flat[len(panels):]:
        ax.axis("off")

    for run in health_runs:
        h = histories[run]
        summary = []
        for field, title, _, pct, log, _ in panels:
            if h.get(field):
                k = 100 if pct else 1
                vals = ", ".join(f"{g} {k * v[-1]:.3g}" for g, v in per_block(h[field]).items())
                summary.append(f"   {field}: {vals}")
        print(f"{run} health at step {h['step'][-1]}:\n" + "\n".join(summary))

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        if ax.axison:
            ax.set_xlabel("training step", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
