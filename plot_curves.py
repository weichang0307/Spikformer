"""Plot training curves from runs' live.json files into one figure.

Top: eval recall at the largest evaluated T (solid) and at T=4 (dotted).
Bottom: eval loss at the largest T (solid) and training loss (faint), log scale.
Prints the latest values and a value every 1,000 steps for each run.

Usage:
    python3 plot_curves.py                       # runs updated in the last 10 minutes
    python3 plot_curves.py spike_8_32_32_T16 spike_12_32_32_T16
    python3 plot_curves.py --minutes 60 --out runs/analysis/curves.png
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def recent_runs(runs_dir: str, minutes: float) -> list[str]:
    """Runs whose live.json changed within the last `minutes`, newest first."""
    now, found = time.time(), []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name, "live.json")
        if os.path.exists(path) and now - os.path.getmtime(path) < minutes * 60:
            found.append((os.path.getmtime(path), name))
    return [name for _, name in sorted(found, reverse=True)]


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

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, run in zip(colors, runs):
        h = json.load(open(os.path.join(args.runs_dir, run, "live.json")))
        Ts = sorted(int(k[5:]) for k in h if k.startswith("acc_T"))
        Tm = Ts[-1]
        axes[0].plot(h["step"], [a * 100 for a in h[f"acc_T{Tm}"]], color=color, lw=1.3, label=f"{run}, T={Tm}")
        if 4 in Ts and Tm != 4:
            axes[0].plot(h["step"], [a * 100 for a in h["acc_T4"]], color=color, lw=0.8, ls=":", label=f"{run}, T=4")
        axes[1].plot(h["step"], h[f"loss_T{Tm}"], color=color, lw=1.3, label=f"{run}, eval")
        axes[1].plot(h["step"], h["train_loss"], color=color, lw=0.8, alpha=0.4, label=f"{run}, train")

        i = -1
        latest = ", ".join(f"T={T} {h[f'acc_T{T}'][i] * 100:.1f}%" for T in Ts)
        print(f"{run}: step {h['step'][i]}, {h['time'][i] / 60:.0f} min, {latest}, eval loss {h[f'loss_T{Tm}'][i]:.3f}")
        print("   every 1,000 steps (step, recall %, loss):",
              [(s, round(a * 100, 1), round(l, 2)) for s, a, l in zip(h["step"], h[f"acc_T{Tm}"], h[f"loss_T{Tm}"])
               if s % 1000 == 0])

    axes[0].set_ylabel("recall accuracy (%)"); axes[0].set_ylim(0, 100); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("loss (log scale)"); axes[1].set_yscale("log"); axes[1].set_xlabel("training step")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
