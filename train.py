"""Train a Spiking Self-Attention Transformer (Spikformer-style) on a
synthetic causal Associative Recall task (induction-head style), and save
the run (config.json, history.json, best.pt, checkpoints/, eval_curve.png)
to runs/spike_<num_kv_pairs>_<num_keys>_<num_values>[_<suffix>].

Usage:
    python3 train.py
    python3 train.py --num_kv_pairs 16 --num_rounds 3 --steps 4000

Training runs until --steps unless --early_stop_acc is given. Evaluation
(every --eval_every steps, at each --eval_T) is written to RUN_DIR/live.json
for dashboard.html; to stop a run cleanly (saving best.pt), create
RUN_DIR/STOP, e.g. `touch runs/spike_8_32_32_T16/STOP`.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import generate_batch, key_position_mask, seq_len, vocab_size
from models import SpikingTransformer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_loss_and_acc(logits: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor):
    # next-token prediction: logits at position t predict seq[t+1]
    logits = logits[:, :-1, :]                     # (B, L-1, V)
    targets = seq[:, 1:]                            # (B, L-1)

    recall_logits = logits[:, mask, :]               # (B, M, V), M = number of masked (key, round>=2) positions
    recall_targets = targets[:, mask]                # (B, M)

    loss = nn.functional.cross_entropy(
        recall_logits.reshape(-1, recall_logits.size(-1)), recall_targets.reshape(-1)
    )
    preds = recall_logits.argmax(-1)
    acc = (preds == recall_targets).float().mean()
    return loss, acc


def train_one_model(name: str, model: nn.Module, mask: torch.Tensor, device: str, *,
                     num_kv_pairs: int, num_rounds: int, num_keys: int, num_values: int,
                     batch_size: int, eval_batch_size: int, steps: int, eval_every: int, lr: float,
                     checkpoint_every: int = None, early_stop_acc: float = None,
                     eval_Ts: list = None, run_dir: str = None):
    """Train `model`; returns a history dict.

    Every `eval_every` steps it evaluates on a fresh batch at each T in
    `eval_Ts` (default: just model.T) and records history["acc_T<t>"] /
    ["loss_T<t>"]; history["acc"] / ["loss"] are at model.T. It also records
    the mean training loss / accuracy since the previous evaluation.

    With `run_dir`, it writes run_dir/live.json at every evaluation (for the
    live dashboard), saves checkpoints straight to run_dir/checkpoints/, and
    stops cleanly when a file run_dir/STOP appears. Without it, checkpoints
    are kept in memory in history["checkpoints"] as (step, state_dict).

    If `early_stop_acc` is set, training also stops after 10 consecutive
    evaluations at or above it. history["final_state"] is the last step's
    state_dict (CPU), whichever way training ended.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_T = model.T
    eval_Ts = sorted(set(eval_Ts or []) | {train_T})

    history = {"step": [], "loss": [], "acc": [], "train_loss": [], "train_acc": [],
               "time": [], "checkpoints": []}
    for T in eval_Ts:
        history[f"acc_T{T}"] = []
        history[f"loss_T{T}"] = []
    stop_file = os.path.join(run_dir, "STOP") if run_dir else None
    ckpt_dir = os.path.join(run_dir, "checkpoints") if run_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    def checkpoint(step):
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ckpt_dir:
            torch.save(state, os.path.join(ckpt_dir, f"checkpoint_step{step:06d}.pt"))
        else:
            history["checkpoints"].append((step, state))

    start = time.time()
    over_threshold_count = 0
    train_loss_sum = train_acc_sum = 0.0
    train_count = 0
    for step in range(1, steps + 1):
        model.train()
        seq = generate_batch(
            batch_size, num_kv_pairs, num_rounds,
            num_keys, num_values, device=device,
        )
        logits = model(seq)
        loss, acc = compute_loss_and_acc(logits, seq, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        train_acc_sum += acc.item()
        train_count += 1

        user_stop = stop_file is not None and os.path.exists(stop_file)
        did_eval = step % eval_every == 0 or step == steps or user_stop
        due_checkpoint = checkpoint_every is not None and (step % checkpoint_every == 0 or step == steps)

        if did_eval:
            model.eval()
            with torch.no_grad():
                seq = generate_batch(
                    eval_batch_size, num_kv_pairs, num_rounds,
                    num_keys, num_values, device=device,
                )
                for T in eval_Ts:
                    model.T = T
                    eval_loss, eval_acc = compute_loss_and_acc(model(seq), seq, mask)
                    history[f"loss_T{T}"].append(eval_loss.item())
                    history[f"acc_T{T}"].append(eval_acc.item())
                model.T = train_T
            history["step"].append(step)
            history["loss"].append(history[f"loss_T{train_T}"][-1])
            history["acc"].append(history[f"acc_T{train_T}"][-1])
            history["train_loss"].append(train_loss_sum / train_count)
            history["train_acc"].append(train_acc_sum / train_count)
            history["time"].append(time.time() - start)
            train_loss_sum = train_acc_sum = 0.0
            train_count = 0

            accs = " ".join(f"T{T} {history[f'acc_T{T}'][-1]*100:5.1f}%" for T in eval_Ts)
            print(f"[{name}] step {step:5d} | train acc {history['train_acc'][-1]*100:5.1f}% "
                  f"| eval acc {accs} | {history['time'][-1]:.0f}s")
            if run_dir:
                write_live(run_dir, history)

            if early_stop_acc is not None:
                over_threshold_count = over_threshold_count + 1 if history["acc"][-1] >= early_stop_acc else 0

        should_stop = over_threshold_count >= 10 or user_stop
        if due_checkpoint or should_stop or step == steps:
            checkpoint(step)  # always capture the final step
        if should_stop:
            reason = "STOP file" if user_stop else f"eval acc >= {early_stop_acc*100:.1f}%"
            print(f"[{name}] stopped at step {step} ({reason})")
            if user_stop:
                os.remove(stop_file)
            break

    history["final_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return history


def plot_history(history: dict, title: str, save_path: str = None):
    """Loss and accuracy vs step: training, plus eval at every recorded T."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    eval_Ts = sorted(int(k[5:]) for k in history if k.startswith("acc_T"))
    for metric, ax in (("loss", axes[0]), ("acc", axes[1])):
        if f"train_{metric}" in history:
            ax.plot(history["step"], history[f"train_{metric}"], label="train", color="0.6")
        if eval_Ts:
            for T in eval_Ts:
                ax.plot(history["step"], history[f"{metric}_T{T}"], label=f"eval T={T}")
        else:
            ax.plot(history["step"], history[metric], label="eval")
        ax.set_xlabel("step")
        ax.set_ylabel(metric if metric == "loss" else "accuracy")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_histories(histories: dict, title: str = "Eval curve comparison", save_path: str = None):
    """Like plot_history, but overlays any number of named runs.

    `histories`: {run_name: history_dict} where each history_dict has
    "step"/"loss"/"acc" lists (as returned by train_one_model, or loaded back
    from a saved history.json).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, history in histories.items():
        axes[0].plot(history["step"], history["loss"], label=name)
        axes[1].plot(history["step"], history["acc"], label=name)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("eval loss")
    axes[0].legend()
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("eval accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def checkpoint_attn_grid(model: nn.Module, checkpoints, seq: torch.Tensor, title: str,
                          num_kv_pairs: int = None, max_cols: int = 8):
    """Plot a [layer x checkpoint] grid of head-max attention heatmaps.

    `checkpoints`: list of (step, state_dict), as stored in
    history["checkpoints"] by train_one_model (or loaded back from disk).
    Each cell is the max (not mean) over heads, so a position lights up if
    *any* head attends there. Rows are query positions, columns key
    positions; each heatmap has its own colour bar.

    If `num_kv_pairs` is given, tick labels show each token's role and id
    (k16 = key token 16, v60 = value token 60), dashed lines mark the start
    of round 2, and in the last layer a red outline marks the cell an
    induction head should read for every round-2 key query: the round-1
    value that followed the same key.

    Temporarily loads each checkpoint's weights into `model` to run the
    forward pass, then restores the model's original weights before
    returning.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    original_state = copy.deepcopy(model.state_dict())
    tokens = seq[0].cpu().tolist()
    L = len(tokens)
    if num_kv_pairs is not None:
        round_len = 2 * num_kv_pairs
        labels = [f"{'k' if p % 2 == 0 else 'v'}{t}" for p, t in enumerate(tokens)]
        expected = [(i, tokens[:round_len].index(tokens[i]) + 1) for i in range(round_len, L, 2)]
    else:
        labels, expected = tokens, []

    idxs = list(range(len(checkpoints)))
    if len(idxs) > max_cols:
        idxs = sorted(set(np.linspace(0, len(idxs) - 1, max_cols).round().astype(int).tolist()))

    fig, axes = None, None
    for col, i in enumerate(idxs):
        step, state = checkpoints[i]
        model_pos_embed = model.pos_embed
        checkpoint_pos_embed = state["pos_embed"]
        if checkpoint_pos_embed.shape[1] < L:
            raise ValueError(
                f"checkpoint at step {step} has positional embeddings for "
                f"{checkpoint_pos_embed.shape[1]} positions, but the input has {L}"
            )
        if checkpoint_pos_embed.shape != model_pos_embed.shape:
            state = dict(state)
            compatible_pos_embed = model_pos_embed.detach().clone()
            copied_len = min(checkpoint_pos_embed.shape[1], model_pos_embed.shape[1])
            compatible_pos_embed[:, :copied_len] = checkpoint_pos_embed[:, :copied_len]
            state["pos_embed"] = compatible_pos_embed
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            _, attn_maps = model(seq, return_attn=True)

        depth = len(attn_maps)
        if fig is None:
            fig, axes = plt.subplots(depth, len(idxs), figsize=(max(4.2 * len(idxs), 7), 4.0 * depth),
                                     squeeze=False)

        for layer in range(depth):
            mat = attn_maps[layer][0].amax(0).float().cpu().numpy()  # max over heads -> (L, L)
            ax = axes[layer][col]
            im = ax.imshow(mat, cmap="viridis", aspect="equal")
            bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            bar.ax.tick_params(labelsize=6)
            if col == len(idxs) - 1:
                bar.set_label("attention (overlap count, mean over T)", fontsize=7)
            if num_kv_pairs is not None:
                ax.axvline(round_len - 0.5, color="white", linewidth=0.9, linestyle="--")
                ax.axhline(round_len - 0.5, color="white", linewidth=0.9, linestyle="--")
                if layer == depth - 1:
                    for q, k in expected:
                        ax.add_patch(Rectangle((k - 0.5, q - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.1))

            ax.set_xticks(range(L))
            ax.set_yticks(range(L))
            ax.set_xticklabels(labels, fontsize=5, rotation=90)
            ax.set_yticklabels(labels if col == 0 else [], fontsize=5)
            ax.tick_params(length=2)
            ax.set_xlabel("key position (column) → attended token", fontsize=7)
            if layer == 0:
                ax.set_title(f"step {step}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"block {layer}\nquery position (row)", fontsize=8)

    if expected:
        axes[-1][0].add_patch(Rectangle((0, 0), 0, 0, fill=False, edgecolor="red", linewidth=1.1,
                                        label="expected: round-1 value of the query's key"))
        axes[-1][0].legend(loc="upper right", fontsize=6, framealpha=0.8)
    note = "rows = queries, columns = keys; k = key token, v = value token; dashed = start of round 2"
    fig.suptitle(f"{title}\n({note})" if num_kv_pairs is not None else title, fontsize=10)
    fig.tight_layout()
    model.load_state_dict(original_state)
    return fig


def history_curves(history: dict) -> dict:
    """The JSON-serializable per-evaluation lists of a history."""
    return {k: v for k, v in history.items() if k not in ("checkpoints", "final_state")}


def write_live(run_dir: str, history: dict) -> None:
    """Atomically rewrite run_dir/live.json (read by dashboard.html)."""
    path = os.path.join(run_dir, "live.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"run": os.path.basename(os.path.normpath(run_dir)), **history_curves(history)}, f)
    os.replace(path + ".tmp", path)


def save_run(run_dir: str, history: dict) -> None:
    """Save in-memory checkpoints, best.pt (the final step), history.json and eval_curve.png."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    for step, state in history["checkpoints"]:
        torch.save(state, os.path.join(ckpt_dir, f"checkpoint_step{step:06d}.pt"))
    torch.save(history["final_state"], os.path.join(run_dir, "best.pt"))
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history_curves(history), f, indent=2)
    run_name = os.path.basename(os.path.normpath(run_dir))
    plot_history(history, f"{run_name} eval curve", save_path=os.path.join(run_dir, "eval_curve.png"))
    print(f"final eval acc: {history['acc'][-1]*100:.1f}%, saved run to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_kv_pairs", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--num_keys", type=int, default=32)
    parser.add_argument("--num_values", type=int, default=32)

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--T", type=int, default=16,
                        help="spiking simulation time steps (power of two for the integer model)")
    parser.add_argument("--attn_lif", action=argparse.BooleanOptionalAction, default=True,
                        help="LIF on the attention output before proj (binary proj input)")
    parser.add_argument("--mask_round1", action=argparse.BooleanOptionalAction, default=False,
                        help="attention may only read round-1 positions (still causal)")
    parser.add_argument("--learnable_scale", action=argparse.BooleanOptionalAction, default=False,
                        help="learn the attention scale (2^s per block, init 1/8) instead of fixed 1/8")
    parser.add_argument("--pos_embed", choices=("learned", "sinusoid"), default="learned",
                        help="learned position embedding or fixed sinusoidal encoding")
    parser.add_argument("--attn_norm", action=argparse.BooleanOptionalAction, default=True,
                        help="L1LayerNorm on the attention output before the attention LIF")
    parser.add_argument("--attn_proj", action=argparse.BooleanOptionalAction, default=False,
                        help="keep the proj linear/norm/LIF after attention (default: removed)")
    parser.add_argument("--mlp_layers", type=int, choices=(1, 2), default=2,
                        help="2: fc1 128->512, fc2 512->128; 1: a single fc1 128->128")
    parser.add_argument("--pow2_gamma", action=argparse.BooleanOptionalAction, default=False,
                        help="train L1LayerNorm gamma as signed powers of two (shift in hardware)")
    parser.add_argument("--eval_T", type=int, nargs="+", default=[4, 8, 16],
                        help="time steps to evaluate at (the training T is always included)")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--early_stop_acc", type=float, default=None,
                        help="stop after 10 evals at or above this accuracy (default: off; "
                             "stop by creating RUN_DIR/STOP)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_suffix", type=str, default="",
                        help="appended to the run directory name, e.g. T32")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    print(f"device: {device}")

    V = vocab_size(args.num_keys, args.num_values)
    L = seq_len(args.num_kv_pairs, args.num_rounds)
    mask = key_position_mask(args.num_kv_pairs, args.num_rounds, device=device)
    print(f"vocab_size={V}, seq_len={L}, num_kv_pairs={args.num_kv_pairs}, num_rounds={args.num_rounds}")

    run_name = f"spike_{args.num_kv_pairs}_{args.num_keys}_{args.num_values}"
    if args.run_suffix:
        run_name += f"_{args.run_suffix}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run_dir={run_dir}")

    config = {"model_type": "spike"}
    config.update({k: v for k, v in vars(args).items() if k not in ("device", "run_suffix")})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model = SpikingTransformer(V, dim=args.dim, depth=args.depth, num_heads=args.num_heads,
                               T=args.T, max_len=L, attn_lif=args.attn_lif, pow2_gamma=args.pow2_gamma,
                               attn_proj=args.attn_proj, mlp_layers=args.mlp_layers,
                               attn_norm=args.attn_norm, pos_embed=args.pos_embed,
                               learnable_scale=args.learnable_scale,
                               mask_round_len=2 * args.num_kv_pairs if args.mask_round1 else 0)
    print(f"SpikingTransformer params: {sum(p.numel() for p in model.parameters()):,}")

    history = train_one_model(
        "spiking", model, mask, device,
        num_kv_pairs=args.num_kv_pairs, num_rounds=args.num_rounds,
        num_keys=args.num_keys, num_values=args.num_values,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        steps=args.steps, eval_every=args.eval_every, lr=args.lr,
        checkpoint_every=args.checkpoint_every, early_stop_acc=args.early_stop_acc,
        eval_Ts=args.eval_T, run_dir=run_dir,
    )

    save_run(run_dir, history)


if __name__ == "__main__":
    main()
