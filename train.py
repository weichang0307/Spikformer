"""Train and compare a Spiking Self-Attention Transformer (Spikformer-style)
against a standard softmax-attention Transformer on a synthetic causal
Associative Recall task (induction-head style).

Usage:
    python train.py
    python train.py --num_kv_pairs 16 --num_rounds 3 --steps 4000
"""

import argparse
import copy
import time

import torch
import torch.nn as nn

from data import generate_batch, key_position_mask, seq_len, vocab_size
from models import SpikingTransformer, VanillaTransformer


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
                     checkpoint_every: int = None, early_stop_acc: float = None):
    """Train `model`. If `checkpoint_every` is set, a CPU copy of the model's
    state_dict is stashed every that many steps into history["checkpoints"]
    (list of (step, state_dict)) -- e.g. for later attention analysis. If
    `early_stop_acc` is set, training stops as soon as eval accuracy reaches
    it (checkpointing that final step first, if due)."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"step": [], "loss": [], "acc": [], "checkpoints": []}
    start = time.time()
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

        did_eval = step % eval_every == 0 or step == steps
        due_checkpoint = checkpoint_every is not None and (step % checkpoint_every == 0 or step == steps)
        eval_acc = None

        if did_eval or due_checkpoint:
            model.eval()
            with torch.no_grad():
                seq = generate_batch(
                    eval_batch_size, num_kv_pairs, num_rounds,
                    num_keys, num_values, device=device,
                )
                eval_logits = model(seq)
                eval_loss, eval_acc = compute_loss_and_acc(eval_logits, seq, mask)

        should_stop = (early_stop_acc is not None and eval_acc is not None
                       and eval_acc.item() >= early_stop_acc)
        if should_stop and checkpoint_every is not None:
            due_checkpoint = True  # always capture the step that triggered early stopping

        if did_eval:
            history["step"].append(step)
            history["loss"].append(eval_loss.item())
            history["acc"].append(eval_acc.item())
            elapsed = time.time() - start
            print(f"[{name}] step {step:5d} | eval loss {eval_loss.item():.4f} "
                  f"| eval acc {eval_acc.item()*100:5.1f}% | {elapsed:.1f}s")

        if due_checkpoint:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["checkpoints"].append((step, state))

        if should_stop:
            print(f"[{name}] early stop at step {step}: eval acc {eval_acc.item()*100:.1f}% >= "
                  f"{early_stop_acc*100:.1f}%")
            break

    return history


def plot_comparison(hist_vanilla: dict, hist_spiking: dict, save_path: str = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist_vanilla["step"], hist_vanilla["loss"], label="vanilla")
    axes[0].plot(hist_spiking["step"], hist_spiking["loss"], label="spiking (SSA)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("eval loss")
    axes[0].legend()

    axes[1].plot(hist_vanilla["step"], hist_vanilla["acc"], label="vanilla")
    axes[1].plot(hist_spiking["step"], hist_spiking["acc"], label="spiking (SSA)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("eval accuracy")
    axes[1].legend()

    fig.suptitle("Associative Recall: Vanilla vs Spiking Self-Attention")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_history(history: dict, title: str, save_path: str = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["step"], history["loss"])
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("eval loss")

    axes[1].plot(history["step"], history["acc"])
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("eval accuracy")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_kv_pairs", type=int, default=8)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--num_keys", type=int, default=32)
    parser.add_argument("--num_values", type=int, default=32)

    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--T", type=int, default=4, help="spiking simulation time steps")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot", action="store_true", help="save a comparison plot to results.png")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    print(f"device: {device}")

    V = vocab_size(args.num_keys, args.num_values)
    L = seq_len(args.num_kv_pairs, args.num_rounds)
    mask = key_position_mask(args.num_kv_pairs, args.num_rounds, device=device)
    print(f"vocab_size={V}, seq_len={L}, num_kv_pairs={args.num_kv_pairs}, num_rounds={args.num_rounds}")

    vanilla = VanillaTransformer(V, dim=args.dim, depth=args.depth, num_heads=args.num_heads, max_len=L)
    spiking = SpikingTransformer(V, dim=args.dim, depth=args.depth, num_heads=args.num_heads, T=args.T, max_len=L)

    n_params_vanilla = sum(p.numel() for p in vanilla.parameters())
    n_params_spiking = sum(p.numel() for p in spiking.parameters())
    print(f"VanillaTransformer params: {n_params_vanilla:,}")
    print(f"SpikingTransformer params: {n_params_spiking:,}")

    train_kwargs = dict(
        num_kv_pairs=args.num_kv_pairs, num_rounds=args.num_rounds,
        num_keys=args.num_keys, num_values=args.num_values,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        steps=args.steps, eval_every=args.eval_every, lr=args.lr,
    )
    hist_vanilla = train_one_model("vanilla", vanilla, mask, device, **train_kwargs)
    hist_spiking = train_one_model("spiking", spiking, mask, device, **train_kwargs)

    print("\n=== Final comparison ===")
    print(f"VanillaTransformer  final eval acc: {hist_vanilla['acc'][-1]*100:.1f}%")
    print(f"SpikingTransformer  final eval acc: {hist_spiking['acc'][-1]*100:.1f}%")

    if args.plot:
        plot_comparison(hist_vanilla, hist_spiking, save_path="results.png")
        print("saved plot to results.png")


if __name__ == "__main__":
    main()
