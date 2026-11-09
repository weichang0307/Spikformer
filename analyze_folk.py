"""Analysis of a folk-rnn spiking run (and, for reference, its ANN teacher).

    cost vs time steps   test bits/token at several T (sliding window, as eval_char.py)
    loss by token type   mean bits per token on the test split for notes, durations, bar lines, repeat signs,
                         endings, headers, rhythm marks, rests / chords and the end of a tune - spiking model
                         vs the ANN, to see where the remaining gap is
    spikes               firing rate of every LIF population per block and spikes emitted per token
    repeats              500 generated tunes (and the real test tunes): why the repeat check fails -
                         unclosed |:, extra :|, |1 without |2 (or the reverse), or a tune cut off at max length
Writes runs/<run>/analysis.json and runs/analysis/folk_<run>.png, prints a summary.

Usage:
    python3 analyze_folk.py RUN [--teacher folkrnn_256d6L_ann_base] [--n 500]
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from abc_quality import BARS, NOTE, VOCAB, check, decode, generate
from char_data import CharDataset
from health import LIFMonitor
from models import model_from_config

CATEGORIES = ["note", "duration", "bar |", "repeat |: :|", "ending |1 |2", "rhythm > < (3", "rest / chord",
              "header M: K:", "end of tune"]


def category(tok: str) -> str:
    if tok == "\n":
        return "end of tune"
    if tok.startswith(("M:", "K:")):
        return "header M: K:"
    if tok == "|":
        return "bar |"
    if tok in ("|:", ":|"):
        return "repeat |: :|"
    if tok in ("|1", "|2"):
        return "ending |1 |2"
    if tok in (">", "<") or tok.startswith("("):
        return "rhythm > < (3"
    if NOTE.match(tok) and "z" not in tok:
        return "note"
    if tok in ("z", "[", "]") or tok.startswith("z"):
        return "rest / chord"
    return "duration"


def load(run: str, ckpt: str = "best"):
    config = json.load(open(os.path.join("runs", run, "config.json")))
    state = torch.load(os.path.join("runs", run, f"{ckpt}.pt"), map_location="cpu", weights_only=False)
    model = model_from_config(config, config["block_size"])
    model.load_state_dict(state.get("model", state))
    return model.cuda().eval(), config


@torch.no_grad()
def per_token_bits(model, data, T=None, context=256, stride=128, batch=32):
    """Bits for every test token (sliding window: each token scored once, with >= context - stride history)."""
    if T is not None:
        model.T = T
    ids = data.splits["test"].long()
    starts = list(range(0, len(ids) - context, stride))
    out = torch.full((len(ids),), float("nan"))
    for b in range(0, len(starts), batch):
        s = starts[b:b + batch]
        x = torch.stack([ids[i:i + context + 1] for i in s]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1]).float()
        nll = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none") / math.log(2)
        for k, i in enumerate(s):
            lo = 0 if i == 0 else context - stride
            out[i + 1 + lo:i + 1 + context] = nll[k, lo:].cpu()
    return out


@torch.no_grad()
def spike_stats(model, data, T, windows=8):
    model.T = T
    torch.manual_seed(0)
    x = torch.stack([data.splits["val"][i:i + 256].long() for i in
                     torch.randint(0, len(data.splits["val"]) - 257, (windows,)).tolist()]).cuda()
    mon = LIFMonitor(model)
    mon.start()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(x)
    for h in mon.handles:
        h.remove()
    for m in mon.names:
        m.record_v = False
    rates, spikes_per_token = {}, 0.0
    tokens = x.numel()
    for name, s in mon.spikes.items():
        steps = mon.counts[name]                  # = time steps x tokens seen by this population
        rates[name] = (s.sum() / (steps * s.numel())).item()
        spikes_per_token += s.sum().item() / tokens
    return rates, spikes_per_token


def repeat_failures(tunes, max_len=None):
    reasons = Counter()
    for t in tunes:
        opens, closes = t.count("|:"), t.count(":|")
        e1, e2 = t.count("|1"), t.count("|2")
        ok_rep = closes == opens or closes == opens + 1
        if ok_rep and e1 == e2:
            reasons["balanced"] += 1
            continue
        if max_len and len(t) >= max_len - 1:
            reasons["cut off at max length"] += 1
        elif closes < opens:
            reasons["unclosed |:"] += 1
        elif closes > opens + 1:
            reasons["extra :|"] += 1
        elif e1 > e2:
            reasons["|1 without |2"] += 1
        else:
            reasons["|2 without |1"] += 1
    n = max(len(tunes), 1)
    return {k: round(100 * v / n, 1) for k, v in reasons.most_common()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--teacher", default="folkrnn_256d6L_ann_base")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--Ts", default="1,2,4,6,8,12,16")
    args = p.parse_args()
    data = CharDataset(os.path.join("data", "folkrnn", "folkrnn.txt"), 0.05, 0.05)
    model, config = load(args.run)
    teacher, _ = load(args.teacher)
    train_T = config["T"]
    res = {"run": args.run, "teacher": args.teacher, "T": train_T}

    # 1. cost vs time steps
    res["bits_by_T"] = {}
    for T in [int(t) for t in args.Ts.split(",")]:
        b = per_token_bits(model, data, T)
        res["bits_by_T"][T] = round(b.nanmean().item(), 4)
        print(f"T={T:2d}: test {res['bits_by_T'][T]:.4f} bits/token", flush=True)
    model.T = train_T

    # 2. loss by token type
    ids = data.splits["test"].long()
    toks = [VOCAB.get(data.chars[i], data.chars[i]) for i in ids.tolist()]
    cats = [category(t) for t in toks]
    snn, ann = per_token_bits(model, data, train_T), per_token_bits(teacher, data)
    by = defaultdict(lambda: [0.0, 0.0, 0])
    for c, s, a in zip(cats, snn.tolist(), ann.tolist()):
        if not math.isnan(s):
            by[c][0] += s; by[c][1] += a; by[c][2] += 1
    total = sum(v[2] for v in by.values())
    res["by_category"] = {c: {"share": round(100 * by[c][2] / total, 1), "snn": round(by[c][0] / by[c][2], 3),
                              "ann": round(by[c][1] / by[c][2], 3),
                              "gap_share": round(100 * (by[c][0] - by[c][1]) / sum(v[0] - v[1] for v in by.values()), 1)}
                          for c in CATEGORIES if by[c][2]}
    print("\ntoken type        share   SNN bits  ANN bits  share of the gap")
    for c, v in res["by_category"].items():
        print(f"{c:16s} {v['share']:5.1f}%   {v['snn']:7.3f}   {v['ann']:7.3f}   {v['gap_share']:5.1f}%")

    # 3. spikes
    rates, spt = spike_stats(model, data, train_T)
    model.T = train_T
    groups = defaultdict(dict)
    for name, r in rates.items():
        parts = name.split(".")
        if parts[0] == "blocks":
            groups[parts[-1]][int(parts[1])] = round(r, 4)
        else:
            groups[name][0] = round(r, 4)
    res["firing_rate"] = {k: v for k, v in groups.items()}
    res["spikes_per_token"] = round(spt, 1)
    print(f"\nspikes per token (all LIF populations, {train_T} time steps): {spt:,.0f}")
    for k, v in groups.items():
        print(f"  {k:10s} " + "  ".join(f"b{b}:{100 * r:4.1f}%" for b, r in sorted(v.items())))

    # 4. repeats
    real = [decode(line) for line in data.decode(data.splits["test"].tolist()).split("\n")[1:-1]]
    gen = generate(args.run, "best", args.n, 1.0)
    res["repeats"] = {"real test tunes": repeat_failures(real), args.run: repeat_failures(gen, 600)}
    res["generated_quality"] = {k: round(100 * sum(check(t)[k] for t in gen if t) / len(gen), 1)
                                for k in ("header", "repeats", "tonic")}
    bars = [check(t) for t in gen if t]
    res["generated_quality"]["bars"] = round(100 * sum(b["valid_bars"] for b in bars) / max(sum(b["bars"] for b in bars), 1), 1)
    print("\nrepeat check outcome (% of tunes):")
    for k, v in res["repeats"].items():
        print(f"  {k}: {v}")
    open(os.path.join("runs", args.run, "generated_tunes.txt"), "w").write("\n".join(" ".join(t) for t in gen) + "\n")
    json.dump(res, open(os.path.join("runs", args.run, "analysis.json"), "w"), indent=1)

    # figure
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    Ts = list(res["bits_by_T"]); ax[0, 0].plot(Ts, list(res["bits_by_T"].values()), "o-", color="#9467bd", lw=2)
    ax[0, 0].axvline(train_T, color="k", ls=":", lw=1); ax[0, 0].set_xscale("log", base=2); ax[0, 0].set_xticks(Ts, Ts)
    ax[0, 0].set(xlabel="time steps T at inference", ylabel="test bits/token", title=f"cost vs accuracy (trained at T={train_T})")
    cs = list(res["by_category"]); xs = range(len(cs))
    ax[0, 1].bar([x - .2 for x in xs], [res["by_category"][c]["snn"] for c in cs], .4, color="#9467bd", label="spiking")
    ax[0, 1].bar([x + .2 for x in xs], [res["by_category"][c]["ann"] for c in cs], .4, color="#7f7f7f", label="ANN teacher")
    for x, c in zip(xs, cs):
        ax[0, 1].text(x, max(res["by_category"][c]["snn"], res["by_category"][c]["ann"]) + .05,
                      f"{res['by_category'][c]['share']}%", ha="center", fontsize=8)
    ax[0, 1].set_xticks(list(xs), cs, rotation=30, ha="right"); ax[0, 1].legend()
    ax[0, 1].set(ylabel="mean bits per token", title="loss by token type (label = share of test tokens)")
    names = list(groups); blocks = sorted({b for v in groups.values() for b in v})
    for k, name in enumerate(names):
        ax[1, 0].plot([b for b in blocks if b in groups[name]], [100 * groups[name][b] for b in blocks if b in groups[name]],
                      "o-", label=name)
    ax[1, 0].set(xlabel="block", ylabel="firing rate (% of neurons per time step)",
                 title=f"spike activity ({spt:,.0f} spikes per token)"); ax[1, 0].legend(fontsize=8)
    keys = sorted({k for v in res["repeats"].values() for k in v}, key=lambda k: k != "balanced")
    for j, (label, v) in enumerate(res["repeats"].items()):
        ax[1, 1].bar([x + (j - .5) * .4 for x in range(len(keys))], [v.get(k, 0) for k in keys], .4,
                     label=label.replace(args.run, "generated (spiking)"))
    ax[1, 1].set_xticks(range(len(keys)), keys, rotation=25, ha="right"); ax[1, 1].legend()
    ax[1, 1].set(ylabel="% of tunes", title="repeat check: balanced, or why not")
    for a in ax.flat:
        a.grid(alpha=.3)
    plt.tight_layout()
    out = os.path.join("runs", "analysis", f"folk_{args.run}.png")
    plt.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
