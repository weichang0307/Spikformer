"""Quantitative attention analysis of character-level runs on the same validation windows.

For every block and head (rows of each map normalised to distributions, queries at
position >= 64 so they all have some context):
    spread      entropy / log(#visible keys)   1 = uniform, 0 = one key
    self        mass on the query's own position
    prev        mass on the previous character (a "previous-token" head)
    local       mass on the last 8 characters (distances 0-7)
    sink        mass on the window's first position
    induction   mass on positions j whose preceding character equals the query character
                (x[j-1] == x[i], j <= i): where an induction head looks to copy what came next
    peak        mean largest single weight in a row
plus each model's mass-by-distance profile per block. Prints a table and saves a figure
with the distance profile of every head.

Usage:
    python3 analyze_attention.py RUN [RUN ...] [--ckpt last|best ...] [--windows 8 --length 512]
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from char_data import CharDataset
from models import model_from_config

DIST_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]   # bins [edge_k, edge_k+1)


@torch.no_grad()
def maps_for(run: str, ckpt: str, seqs: torch.Tensor, device) -> tuple:
    config = json.load(open(os.path.join("runs", run, "config.json")))
    path = os.path.join("runs", run, f"{ckpt}.pt")
    if not os.path.exists(path):
        path = os.path.join("runs", run, "best.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    step = state.get("step") if "model" in state else None
    model = model_from_config(config, config["block_size"])
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    out = []
    for s in seqs:   # one window at a time keeps memory small
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, maps = model(s[None].to(device), return_attn=True)
        out.append(torch.stack([m[0].float().cpu() for m in maps]))   # (layers, H, L, L)
    del model
    torch.cuda.empty_cache()
    return torch.stack(out), step, config.get("model_type", "spike")   # (W, layers, H, L, L)


def head_stats(a: torch.Tensor, seqs: torch.Tensor, min_query: int = 64) -> dict:
    W, nL, H, L, _ = a.shape
    rows = a.sum(-1, keepdim=True)
    p = a / rows.clamp_min(1e-12)
    q = torch.arange(L)
    k = torch.arange(L)
    dist = (q[:, None] - k[None, :]).clamp_min(-1)                         # -1 = future
    sel = q >= min_query
    p = p[..., sel, :]                                                     # (W, nL, H, Q, L)
    d = dist[sel]
    visible = (q[sel] + 1).float()
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / visible.log()
    ind = torch.zeros(W, L, L, dtype=torch.bool)                           # induction targets
    for w in range(W):
        x = seqs[w, :L]
        prev_char = torch.cat([x[:1] * 0 - 1, x[:-1]])                     # x[j-1], -1 for j = 0
        ind[w] = (prev_char[None, :] == x[:, None]) & (k[None, :] <= q[:, None]) & (k[None, :] >= 1)
    ind = ind[:, sel][:, None, None]                                       # (W, 1, 1, Q, L)
    profile = torch.stack([(p * ((d >= lo) & (d < hi))).sum(-1)
                           for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])], -1)   # (W, nL, H, Q, bins)
    has_ind = ind.any(-1)                                                  # queries that have an induction target
    ind_mass = (p * ind).sum(-1)
    mean = lambda t: t.mean(dim=(0, 3))                                    # over windows and queries -> (nL, H)
    return {
        "spread": mean(ent), "self": mean((p * (d == 0)).sum(-1)),
        "prev": mean((p * (d == 1)).sum(-1)), "local": mean((p * ((d >= 0) & (d < 8))).sum(-1)),
        "sink": mean(p[..., 0]), "peak": mean(p.max(-1).values),
        "induction": (ind_mass * has_ind).sum(dim=(0, 3)) / has_ind.sum(dim=(0, 3)).clamp_min(1),
        # what a uniform head would put on induction targets, for reference
        "uniform_induction": ((has_ind * ind.sum(-1) / visible).sum(dim=(0, 3)) / has_ind.sum(dim=(0, 3)).clamp_min(1)).expand(nL, H),
        "profile": profile.mean(dim=(0, 3)),                               # (nL, H, bins)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ckpt", nargs="+", default=["last"], help="one per run, or one for all")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--out", default="runs/analysis/attention_analysis.png")
    args = parser.parse_args()
    ckpts = args.ckpt * len(args.runs) if len(args.ckpt) == 1 else args.ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = json.load(open(os.path.join("runs", args.runs[0], "config.json")))
    data = CharDataset(config["data_path"], config.get("val_fraction", 0.1), config.get("test_fraction", 0.0),
                       config.get("max_vocab", 0), val_size=config.get("val_chars", 0),
                       test_size=config.get("test_chars", 0))
    g = torch.Generator().manual_seed(0)
    val = data.splits["val"]
    starts = torch.randint(0, len(val) - args.length - 1, (args.windows,), generator=g)
    seqs = torch.stack([val[s:s + args.length] for s in starts.tolist()]).long()

    results = {}
    for run, ck in zip(args.runs, ckpts):
        a, step, kind = maps_for(run, ck, seqs, device)
        st = head_stats(a, seqs)
        results[run] = (st, step, kind)
        nL, H = st["spread"].shape
        print(f"\n=== {run} ({ck}, step {step}, {kind}) — {args.windows} windows x {args.length} chars, queries >= 64")
        print(f"{'block':>5s} {'spread':>7s} {'self%':>6s} {'prev%':>6s} {'local%':>7s} {'sink%':>6s} {'induct%':>8s} {'(unif%)':>8s} {'peak%':>6s}"
              f"   best induction head / best prev head")
        for l in range(nL):
            r = {k: st[k][l] for k in ("spread", "self", "prev", "local", "sink", "induction", "uniform_induction", "peak")}
            bi, bp = int(r["induction"].argmax()), int(r["prev"].argmax())
            print(f"{l:5d} {r['spread'].mean():7.3f} {100*r['self'].mean():6.1f} {100*r['prev'].mean():6.1f} "
                  f"{100*r['local'].mean():7.1f} {100*r['sink'].mean():6.1f} {100*r['induction'].mean():8.1f} "
                  f"{100*r['uniform_induction'].mean():8.1f} {100*r['peak'].mean():6.1f}"
                  f"   h{bi} {100*r['induction'][bi]:.0f}% / h{bp} {100*r['prev'][bp]:.0f}%")

    # figure: per-head mass-by-distance profile, one panel per run
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 7), squeeze=False)
    labels = [f"{lo}" if hi - lo == 1 else f"{lo}-{hi - 1}" for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]
    for ax, (run, (st, step, kind)) in zip(axes[0], results.items()):
        prof = st["profile"]                                               # (nL, H, bins)
        nL, H, nb = prof.shape
        img = prof.reshape(nL * H, nb).numpy()
        ax.imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(nb)); ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_yticks([l * H + H // 2 for l in range(nL)]); ax.set_yticklabels([f"b{l}" for l in range(nL)], fontsize=7)
        for l in range(1, nL):
            ax.axhline(l * H - 0.5, color="w", lw=0.4)
        ax.set_xlabel("distance from query (characters)", fontsize=8)
        ax.set_title(f"{run}\n({kind}, step {step})", fontsize=8)
    fig.suptitle("attention mass by distance, every head (rows), mean over queries >= 64", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
