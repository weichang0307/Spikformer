"""Analyze saved runs (each a runs/<name>/ folder from train.py):

- overlays all runs' eval curves -> runs/analysis/eval_curves.png
- plots each run's QK attention heatmaps (best.pt, plus how they evolved
  over the saved checkpoints) -> runs/<name>/attention_*.png
- prints input / per-layer spike firing rates
- verifies the attention heatmaps (see `verify_attention`)

Usage:
    python3 analyze.py
    python3 analyze.py spike_3_32_32_T32 spike_16_32_32_T32
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import torch

from data import generate_batch, key_position_mask, vocab_size
from models import SpikingTransformer, model_from_config
from neurons import LIFNeuron
from spikformer_quant import QuantizedSpikformer, Trace
from train import checkpoint_attn_grid, pick_device, plot_histories

DEFAULT_RUNS = ["spike_3_32_32_T32", "spike_5_32_32_T32", "spike_8_32_32_T32", "spike_16_32_32_T32"]


def load_run(run_name: str, runs_dir: str = "runs") -> dict:
    run_dir = os.path.join(runs_dir, run_name)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(run_dir, "history.json")) as f:
        history = json.load(f)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    checkpoints = []
    for fname in sorted(os.listdir(ckpt_dir)):
        step = int(fname.removeprefix("checkpoint_step").removesuffix(".pt"))
        state = torch.load(os.path.join(ckpt_dir, fname), map_location="cpu", weights_only=True)
        checkpoints.append((step, state))
    checkpoints.sort(key=lambda x: x[0])

    best_state = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=True)
    return {"name": run_name, "dir": run_dir, "config": config, "history": history,
            "checkpoints": checkpoints, "best_state": best_state}


def build_model(run: dict) -> SpikingTransformer:
    config = run["config"]
    model = model_from_config(config, max_len=run["best_state"]["pos_embed"].shape[1])
    model.load_state_dict(run["best_state"])
    return model.eval()


def sample_sequences(config: dict, batch_size: int, device: str) -> torch.Tensor:
    return generate_batch(batch_size, config["num_kv_pairs"], config["num_rounds"],
                          config["num_keys"], config["num_values"], device=device)


def collect_spike_activity(model: SpikingTransformer, tokens: torch.Tensor) -> None:
    activity = {}
    handles = []

    def record(name):
        def hook(module, inputs, output):
            stats = activity.setdefault(name, {"spikes": 0.0, "elements": 0, "calls": 0})
            stats["spikes"] += output.detach().sum().item()
            stats["elements"] += output.numel()
            stats["calls"] += 1
        return hook

    for module_name, module in model.named_modules():
        if isinstance(module, LIFNeuron):
            handles.append(module.register_forward_hook(record(module_name)))
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()
        model.reset_state()

    rows = [("input", activity["input_lif"])]
    for layer_idx in range(len(model.blocks)):
        layer_stats = [s for n, s in activity.items() if n.startswith(f"blocks.{layer_idx}.")]
        rows.append((f"layer {layer_idx}", {
            "spikes": sum(s["spikes"] for s in layer_stats),
            "elements": sum(s["elements"] for s in layer_stats),
            "calls": max(s["calls"] for s in layer_stats),
        }))

    print(f"  {'site':<10} {'spikes / sequence':>18} {'spikes / timestep':>19} {'firing rate':>13}")
    for label, stats in rows:
        per_seq = stats["spikes"] / tokens.shape[0]
        per_step = per_seq / stats["calls"]  # calls = T time steps
        print(f"  {label:<10} {per_seq:>18.0f} {per_step:>19.1f} {stats['spikes'] / stats['elements']:>12.2%}")


def recompute_attention(model: SpikingTransformer, tokens: torch.Tensor) -> list[torch.Tensor]:
    """Independently rebuild the time-averaged QK^T maps from the raw Q/K
    spikes captured by forward hooks on each block's q_lif / k_lif."""
    captured = {}
    handles = []
    for i, blk in enumerate(model.blocks):
        for part in ("q", "k"):
            lif = getattr(blk.attn, f"{part}_lif")
            handles.append(lif.register_forward_hook(
                lambda m, inp, out, key=(i, part): captured.setdefault(key, []).append(out.detach())
            ))
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        for handle in handles:
            handle.remove()

    B, L = tokens.shape
    H = model.blocks[0].attn.num_heads
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1)
    maps = []
    for i in range(len(model.blocks)):
        total = torch.zeros(B, H, L, L, device=tokens.device)
        for q, k in zip(captured[(i, "q")], captured[(i, "k")]):
            q = q.view(B, L, H, -1).transpose(1, 2)
            k = k.view(B, L, H, -1).transpose(1, 2)
            total += torch.einsum("bhid,bhjd->bhij", q, k).masked_fill(causal, 0)
        maps.append(total / model.T)
    return maps


def induction_targets(tokens: torch.Tensor, num_kv_pairs: int) -> list[tuple[int, int, int, int]]:
    """For every round-2+ key position i in every sequence b, return
    (b, i, p, p + 1) where p is the round-1 position of the same key, so
    p + 1 holds the value that must be predicted at i."""
    targets = []
    round_len = 2 * num_kv_pairs
    for b in range(tokens.shape[0]):
        first_round = tokens[b, :round_len].tolist()
        for i in range(round_len, tokens.shape[1], 2):
            p = first_round.index(tokens[b, i].item())
            targets.append((b, i, p, p + 1))
    return targets


def attention_scores(attn: torch.Tensor, targets, num_kv_pairs: int) -> dict:
    """Score one layer's (B, H, L, L) map (max over heads, as plotted).

    For each kind of target column, report the mean share of the row's
    attention mass it receives, and how often it is the row's unique maximum.
    `other keys` is the combined share on every round-1 key except the
    queried one (an elimination strategy: look at the pairs that do not match).
    `uniform` is the share a flat row over the causal window would give.
    """
    attn = attn.amax(dim=1)  # (B, L, L)
    out = {}
    kinds = {
        "prev_token": [(b, i, i - 1) for b, i, _, _ in targets],
        "same_key (p)": [(b, i, p) for b, i, p, _ in targets],
        "value (p+1)": [(b, i, v) for b, i, _, v in targets],
    }
    for kind, cells in kinds.items():
        share, top1 = 0.0, 0
        for b, i, j in cells:
            row = attn[b, i, : i + 1]
            total = row.sum().item()
            share += row[j].item() / total if total > 0 else 0.0
            top1 += int(row[j] > torch.cat([row[:j], row[j + 1:]]).max())
        out[kind] = (share / len(cells), top1 / len(cells))
    share = 0.0
    for b, i, p, _ in targets:
        row = attn[b, i, : i + 1]
        others = [j for j in range(0, 2 * num_kv_pairs, 2) if j != p]
        total = row.sum().item()
        share += row[others].sum().item() / total if total > 0 else 0.0
    out["other keys"] = (share / len(targets), float("nan"))
    out["uniform"] = sum(1 / (i + 1) for _, i, _, _ in targets) / len(targets)
    return out


def verify_attention(run: dict, model: SpikingTransformer, device: str, batch_size: int) -> None:
    """Check that the plotted attention maps are what they claim to be:

    1. return_attn maps == QK^T rebuilt from hooked Q/K spikes (masked, /T).
    2. causal: every entry above the diagonal is zero.
    3. integer model (spikformer_quant) gives nearly the same maps.
    4. they show the induction mechanism the task needs (scores per layer).
    """
    config = run["config"]
    torch.manual_seed(config.get("seed", 0))
    tokens = sample_sequences(config, batch_size, device)
    with torch.no_grad():
        logits, maps = model(tokens, return_attn=True)
    rebuilt = recompute_attention(model, tokens)

    L = tokens.shape[1]
    above = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    for i, (m, r) in enumerate(zip(maps, rebuilt)):
        diff = (m - r).abs().max().item()
        leak = m[..., above].abs().max().item()
        status = "OK" if diff < 1e-5 and leak == 0 else "MISMATCH"
        print(f"  layer {i}: max |return_attn - rebuilt| = {diff:.2e}, max above diagonal = {leak:.2e}  [{status}]")

    # integer model on a subset
    n_int = min(batch_size, 16)
    qmodel = QuantizedSpikformer(config, run["best_state"]).to(device)
    trace = Trace(keep=True)
    qmodel(tokens[:n_int], trace)
    for i, m in enumerate(maps):
        qmap = torch.stack(trace.frames[f"blocks.{i}.attn.qk"]).cpu().double().mean(0)  # (B, H, L, L)
        fmap = m[:n_int].cpu().double()
        rel = (qmap - fmap).abs().sum() / fmap.abs().sum()
        same_argmax = (qmap.amax(1).argmax(-1) == fmap.amax(1).argmax(-1)).double().mean()
        print(f"  layer {i}: integer vs float map, relative L1 diff = {rel:.3f}, "
              f"same row argmax = {same_argmax:.1%}")

    mask = key_position_mask(config["num_kv_pairs"], config["num_rounds"], device=device)
    acc = (logits[:, :-1][:, mask].argmax(-1) == tokens[:, 1:][:, mask]).float().mean().item()
    targets = induction_targets(tokens, config["num_kv_pairs"])
    print(f"  induction scores on {len(targets)} round-2+ key positions (recall acc {acc:.1%}):")
    print(f"    {'layer':<6} {'target':<14} {'mass share':>11} {'unique top-1':>13}")
    for i, m in enumerate(maps):
        scores = attention_scores(m.cpu(), targets, config["num_kv_pairs"])
        for kind in ("prev_token", "same_key (p)", "value (p+1)", "other keys"):
            share, top1 = scores[kind]
            top1 = "-" if top1 != top1 else f"{top1:.1%}"
            print(f"    {i:<6} {kind:<14} {share:>11.1%} {top1:>13}")
        print(f"    {i:<6} {'uniform':<14} {scores['uniform']:>11.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS, help="run names under --runs_dir")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--verify_batch", type=int, default=64, help="sequences used for verification")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or pick_device()

    runs = {name: load_run(name, args.runs_dir) for name in args.runs}
    out_dir = os.path.join(args.runs_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    plot_histories({n: r["history"] for n, r in runs.items()}, title="Eval curve comparison",
                   save_path=os.path.join(out_dir, "eval_curves.png"))
    print(f"saved {out_dir}/eval_curves.png")

    for name, run in runs.items():
        config = run["config"]
        print(f"\n=== {name}: {len(run['checkpoints'])} checkpoints, "
              f"final eval acc {run['history']['acc'][-1] * 100:.1f}%")
        model = build_model(run).to(device)

        torch.manual_seed(config.get("seed", 0) + 1)
        example = sample_sequences(config, 1, device)
        for label, checkpoints in (("best", [("best.pt", run["best_state"])]),
                                   ("evolution", run["checkpoints"])):
            fig = checkpoint_attn_grid(model, checkpoints, example, f"{name} QK attention ({label})",
                                       num_kv_pairs=config["num_kv_pairs"])
            path = os.path.join(run["dir"], f"attention_{label}.png")
            fig.savefig(path, dpi=150)
            print(f"saved {path}")

        print("spike activity (best.pt, 64 sequences):")
        collect_spike_activity(model, sample_sequences(config, 64, device))
        print("attention verification:")
        verify_attention(run, model, device, args.verify_batch)


if __name__ == "__main__":
    main()
