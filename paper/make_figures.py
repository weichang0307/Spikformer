"""Figures for paper/main.tex, written to paper/figures/*.pdf (vector).

Numbers come from the project's evaluations (full folk-rnn test split, 256-token sliding window, stride 128):
eval_char.py / fixed_point_eval.py / eval_fixed.py outputs, and runs/<run>/live.json for training curves.

    python3 paper/make_figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
GREEN, RED, GREY, BLUE = "#4a9c5d", "#c0504d", "#b8b8b0", "#2f6db5"


def progression():
    """Test bits/token after each change on the main line of development (a bridge)."""
    rows = [("First spiking baseline (10k steps)", 2.402, True),
            ("+ distillation (output + attention KL)", 2.183, False),
            ("+ two-sided homeostasis", 2.087, False),
            ("+ learned count cut + spread penalty", 1.998, False),
            ("count$^2$ attention instead of the cut", 1.897, False),
            ("count$^2$ on all 8 heads", 1.850, False),
            ("30k steps, KL/penalty fade, $\\alpha$ 2$\\to$4", 1.540, False),
            ("spread penalty on all heads", 1.537, False),
            ("50k steps", 1.502, False),
            ("spike dropout 0.3", 1.439, False),
            ("8 time steps instead of 6", 1.425, False),
            ("dropout 0.5 (chosen for tune quality)", 1.457, False),
            ("integer hardware datapath (PTQ)", 1.469, False),
            ("+5k steps with the datapath (QAT)", 1.412, False),
            ("Final model, integer datapath", 1.412, True)]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    n = len(rows)
    for i, (name, v, total) in enumerate(rows):
        y = n - 1 - i
        prev = rows[i - 1][1] if i else v
        lo, hi = (0, v) if total else (min(prev, v), max(prev, v))
        col = GREY if total else (GREEN if v <= prev else RED)
        ax.barh(y, max(hi - lo, 0.004), left=lo, height=0.62, color=col, alpha=0.95 if total else 0.8, lw=0)
        lab = f"{v:.3f}" if total else f"{v - prev:+.3f}".replace("-", "−")
        ax.text(2.55, y, lab, va="center", fontsize=7, fontweight="bold" if total else "normal")
        if i and not total:
            ax.plot([prev, prev], [y + 0.31, y + 0.69], color="0.4", lw=0.5)
    for x, name in ((1.220, "ANN teacher"), (1.591, "LSTM")):
        ax.axvline(x, color="0.35", ls=(0, (3, 3)), lw=0.7)
        ax.text(x, n - 0.35, name, ha="right" if x < 1.4 else "left", va="bottom", fontsize=6.5, color="0.3")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7)
    ax.set_xlim(0, 2.5)
    ax.set_xlabel("test bits per token (lower is better)")
    ax.text(2.55, n - 0.35, "change", fontsize=6.5, color="0.3", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "progression.pdf"))


def energy():
    """Test bits/token against estimated energy per token, float and int8 versions."""
    pts = [("ANN fp32", 114.3, 1.220, "o", GREY), ("ANN int8", 23.5, 1.225, "o", "none"),
           ("LSTM fp32", 109.7, 1.591, "s", GREY), ("LSTM int8", 22.6, 1.596, "s", "none"),
           ("Spiking fp32", 109.0, 1.392, "D", GREY), ("Spiking PTQ", 16.1, 1.433, "D", "none"),
           ("Spiking QAT", 16.1, 1.412, "D", BLUE)]
    off = {"ANN fp32": (-4, 6, "right"), "ANN int8": (4, -9, "left"), "LSTM fp32": (-4, 6, "right"),
           "LSTM int8": (4, 4, "left"), "Spiking fp32": (-4, 6, "right"), "Spiking PTQ": (5, 4, "left"),
           "Spiking QAT": (5, -9, "left")}
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for name, e, b, m, fc in pts:
        ec = BLUE if fc == BLUE else "0.45"
        ax.scatter(e, b, marker=m, s=26, facecolor=fc, edgecolor=ec, lw=0.9, zorder=3)
        dx, dy, ha = off[name]
        ax.annotate(name, (e, b), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=6.5,
                    color=BLUE if fc == BLUE else "0.15")
    ax.set_xlim(0, 125)
    ax.set_ylim(1.15, 1.66)
    ax.set_xlabel("estimated energy per token ($\\mu$J)")
    ax.set_ylabel("test bits per token")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "energy.pdf"))


def theta():
    """Mean learned count threshold per block: cut alone vs cut + spread penalty."""
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True)
    runs = (("folkrnn_256d6L_T6_distill_homeo2s_thr", "learned cut"),
            ("folkrnn_256d6L_T6_distill_homeo2s_thr_spread", "learned cut + spread penalty"))
    for ax, (run, title) in zip(axs, runs):
        h = json.load(open(os.path.join(ROOT, "runs", run, "live.json")))
        for i in range(6):
            ax.plot(h["step"], h["attn_threshold"][f"block {i}"], lw=1, color=plt.cm.viridis(i / 5), label=f"block {i}")
        ax.axvspan(0, 2000, color="0.92", lw=0)
        for yv in (1, 2):
            ax.axhline(yv, color="0.4", ls=":", lw=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("step")
        ax.set_ylim(0.9, 3.1)
    axs[0].set_ylabel("mean $\\theta$ (heads 0-3)")
    axs[0].text(1000, 3.0, "blend-in", ha="center", va="top", fontsize=6.5, color="0.35")
    axs[1].legend(fontsize=6, ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "theta.pdf"))


if __name__ == "__main__":
    progression()
    energy()
    theta()
    print("wrote", sorted(os.listdir(OUT)))
