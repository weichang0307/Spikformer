"""Figures for thesis/main.tex, written to thesis/figures/ (vector PDF).

Reuses paper/make_figures.py for the progression, energy and threshold figures, and adds training curves,
loss by token type, the energy breakdown and the time-step trade-off. Numbers come from the project's
evaluations (full folk-rnn test split, 256-token sliding window, stride 128) and runs/<run>/live.json.

    python3 thesis/make_figures.py
"""

import json
import math
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "thesis", "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "paper"))
import make_figures as paper  # noqa: E402

paper.OUT = OUT
GREEN, RED, GREY, BLUE = paper.GREEN, paper.RED, paper.GREY, paper.BLUE

# every figure saved as PDF is also written as a 200-dpi PNG next to it (for the Docs report)
_savefig = matplotlib.figure.Figure.savefig


def _savefig_png_too(self, fname, *args, **kwargs):
    _savefig(self, fname, *args, **kwargs)
    if isinstance(fname, str) and fname.endswith(".pdf"):
        kwargs.pop("format", None)
        _savefig(self, fname[:-4] + ".png", *args, dpi=200, **kwargs)


matplotlib.figure.Figure.savefig = _savefig_png_too


def architecture():
    """Model and block diagram: every arrow carries spikes except the residual stream."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.set_xlim(0, 760)
    ax.set_ylim(560, 0)
    ax.axis("off")

    def box(x, y, w, h, title, sub=None, accent=False, fill=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                    fc=fill or ("#e7f1fb" if accent else "white"),
                                    ec=BLUE if accent else "#555555", lw=1.6 if accent else 1.0))
        cy = y + h / 2
        if sub:
            ax.text(x + w / 2, cy - 8, title, ha="center", va="center", fontsize=9, weight="bold")
            ax.text(x + w / 2, cy + 10, sub, ha="center", va="center", fontsize=7.5, color="#444444")
        else:
            ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=9, weight="bold")

    def arrow(x0, y0, x1, y1, dashed=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0,
                                    linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))

    # left column: whole model
    lx, lw, lh = 20, 210, 42
    ly = [60, 128, 196, 264, 332, 400]
    labels = ["Token + position embedding", "L1 norm → input LIF", "Spiking block × 6",
              "Mean over 8 time steps", "Linear head → 136 logits", "Sample the next token"]
    for i, (y, t) in enumerate(zip(ly, labels)):
        box(lx, y, lw, lh, t, fill="#f0efec" if i == 2 else None)
        if i:
            arrow(lx + lw / 2, ly[i - 1] + lh, lx + lw / 2, y)
    ax.text(lx, 470, "Each token runs for 8 time steps;\nLIF potentials carry across steps\n"
            "and only the read-out averages them.", fontsize=7.5, color="#444444", va="top")
    arrow(lx + lw, ly[2] + lh / 2, 270, ly[2] + lh / 2, dashed=True)

    # right: inside one block
    ax.add_patch(FancyBboxPatch((270, 20), 470, 525, boxstyle="round,pad=0,rounding_size=8",
                                fc="#f7f7f5", ec="#888888", lw=1.0))
    ax.text(286, 40, "Inside one spiking block (repeated 6 times)", fontsize=9.5, weight="bold", va="center")
    ax.text(300, 62, "block input: residual stream (integers 0–13)", fontsize=7.5, color="#444444",
            va="center")
    ix, iw = 300, 420
    qw = 132
    for j, name in enumerate(["Query", "Key", "Value"]):
        x = ix + j * (qw + 12)
        box(x, 76, qw, 50, name, "linear → L1 → LIF")
        arrow(x + qw / 2, 126, x + qw / 2, 150)
    stages = [(150, 54, "Count² attention, 8 heads",
               "(spikes shared by query and key)² weight the value spikes; no softmax", True),
              (228, 54, "Head norm → LIF → projection",
               "per-head L1 norm (ε = 1/16), then linear → L1 → LIF", False),
              (306, 38, "Spike dropout 0.5, add to residual stream", None, False),
              (368, 54, "Spiking MLP", "256 → 1,024 → 256, each layer linear → L1 → LIF", False),
              (446, 38, "Spike dropout 0.5, add to residual stream", None, False)]
    for k, (y, h, t, s, acc) in enumerate(stages):
        box(ix, y, iw, h, t, s, accent=acc)
        if k:
            py, ph = stages[k - 1][0], stages[k - 1][1]
            arrow(ix + iw / 2, py + ph, ix + iw / 2, y)
    # residual stream path on the left edge of the block
    ax.plot([284, 284, ix], [56, 325, 325], color="#555555", lw=1.0)
    arrow(284, 325, ix, 325)
    ax.plot([ix, 284, 284], [337, 337, 465], color="#555555", lw=1.0)
    arrow(284, 465, ix, 465)
    ax.text(380, 520, "Every arrow between layers carries binary spikes;\n"
            "dropout acts only during training.", fontsize=7.5, color="#444444", ha="left", va="center")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "architecture.pdf"))


def curves():
    """Validation bits/token over training for the reference models and key spiking runs."""
    runs = [("ANN teacher", "folkrnn_256d6L_ann_base", "live.json", "black", "-", 0),
            ("LSTM 3x512", "folkrnn_lstm3x512_30k", "history.json", "0.5", "--", 0),
            ("spiking baseline (plain counts)", "folkrnn_256d6L_T6_base", "live.json", RED, "-", 0),
            ("+ distillation, homeostasis, count$^2$", "folkrnn_256d6L_T6_distill_homeo2s_sq8_spread", "live.json", "#d8a24a", "-", 0),
            ("50k steps, no dropout", "folkrnn_256d6L_T6_distill_homeo2s_sq8_spread8_50k", "live.json", GREEN, "-", 0),
            ("50k steps, dropout 0.5, T=8", "folkrnn_256d6L_T8_distill_homeo2s_sq8_spread8_50k_do05", "live.json", BLUE, "-", 0),
            ("+ QAT (integer datapath)", "folkrnn_256d6L_T8_qat_from_do05", "live.json", BLUE, ":", 50000)]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    for name, run, f, c, ls, off in runs:
        h = json.load(open(os.path.join(ROOT, "runs", run, f)))
        ax.plot([s + off for s in h["step"]], h["bpc"], color=c, ls=ls, lw=1.1, label=name)
    ax.set_ylim(1.3, 2.8)
    ax.set_xlim(0, 56000)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation bits/token")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curves.pdf"))


def token_types():
    """Mean bits per test token by token type: final (QAT) spiking model vs ANN teacher."""
    rows = [("note", 69.0, 1.565, 1.343), ("duration", 12.7, 1.470, 1.276), ("bar line", 10.4, 0.244, 0.193),
            ("rhythm mark", 3.1, 1.732, 1.575), ("repeat sign", 1.8, 1.151, 1.001), ("header", 1.2, 1.645, 1.643),
            ("ending", 0.7, 1.402, 1.328), ("end of tune", 0.6, 0.840, 0.774), ("rest / chord", 0.5, 2.303, 2.109)]
    fig, ax = plt.subplots(figsize=(5.8, 2.5))
    xs = range(len(rows))
    ax.bar([x - 0.2 for x in xs], [r[2] for r in rows], 0.4, color=BLUE, label="spiking (QAT, integer)")
    ax.bar([x + 0.2 for x in xs], [r[3] for r in rows], 0.4, color=GREY, label="ANN teacher")
    for x, r in zip(xs, rows):
        ax.text(x, max(r[2], r[3]) + 0.06, f"{r[1]:.1f}%", ha="center", fontsize=6, color="0.3")
    ax.set_xticks(list(xs), [r[0] for r in rows], fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("bits per token")
    ax.set_ylim(0, 2.6)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "token_types.pdf"))


def energy_breakdown():
    """Estimated energy per token split into arithmetic, weight reads and cache reads (4 pJ/byte)."""
    rows = [("ANN fp32", 25.48, 76.28, 12.58), ("LSTM fp32", 24.44, 85.23, 0.0), ("Spiking fp32", 62.74, 44.48, 1.77),
            ("ANN int8", 1.27, 19.07, 3.15), ("LSTM int8", 1.28, 21.32, 0.0), ("Spiking int", 3.23, 11.12, 1.77)]
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    ys = range(len(rows))[::-1]
    for y, (name, c, w, k) in zip(ys, rows):
        ax.barh(y, c, color=BLUE, height=0.6)
        ax.barh(y, w, left=c, color=GREY, height=0.6)
        ax.barh(y, k, left=c + w, color="#d8a24a", height=0.6)
        ax.text(c + w + k + 1.5, y, f"{c + w + k:.1f} $\\mu$J", va="center", fontsize=6.5)
    ax.set_yticks(list(ys), [r[0] for r in rows], fontsize=7)
    ax.set_xlabel("estimated energy per token ($\\mu$J)")
    ax.set_xlim(0, 130)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="arithmetic"), Patch(color=GREY, label="weight reads"),
                       Patch(color="#d8a24a", label="cache reads")], fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "energy_breakdown.pdf"))


def time_steps():
    """Test bits/token of the final model at different numbers of time steps (trained at T=8)."""
    T = [4, 6, 8]
    b = [1.528, 1.4384, 1.4121]
    fig, ax = plt.subplots(figsize=(2.8, 2.0))
    ax.plot(T, b, "o-", color=BLUE, lw=1.2, ms=4)
    for t, v in zip(T, b):
        ax.annotate(f"{v:.3f}", (t, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_xticks(T)
    ax.set_xlabel("time steps $T$ at inference")
    ax.set_ylabel("test bits/token")
    ax.set_ylim(1.38, 1.56)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "time_steps.pdf"))


if __name__ == "__main__":
    architecture()
    paper.progression()
    paper.energy()
    paper.theta()
    curves()
    token_types()
    energy_breakdown()
    time_steps()
    shot = os.path.expanduser("~/.claude/projects/-home-th12-dv-Spikformer/344dcaaf-269c-45a6-accf-9896b000e689/"
                              "tool-results/mcp-Claude_Browser-blob-1793165894076-22tise.jpg")
    if os.path.exists(shot):
        shutil.copy(shot, os.path.join(OUT, "playground.jpg"))
    print("wrote", sorted(os.listdir(OUT)))
