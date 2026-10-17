"""Train the Spiking Self-Attention Transformer as a character-level GPT on
Tiny Shakespeare (or any text file via --data_path), and save the run
(config.json, history.json, best.pt, checkpoints/, eval_curve.png,
samples.txt) to runs/char_<dim>d<depth>L_T<T>[_<suffix>].

Usage:
    python3 train_char.py
    python3 train_char.py --block_size 256 --depth 6 --steps 20000

Every --eval_every steps it reports, at each --eval_T, the validation loss
(nats/char), bits per character and next-character accuracy, averaged over
--eval_iters batches, and writes them to RUN_DIR/live.json for
dashboard.html / plot_curves.py ("acc" there is next-character accuracy).
Every --sample_every steps it prints a generated sample and appends it to
RUN_DIR/samples.txt. To stop a run cleanly (saving best.pt), create
RUN_DIR/STOP. best.pt is the checkpoint with the lowest val loss; it is
rewritten whenever the val loss improves, so it survives a crash or kill.

--init_from RUN_DIR/best.pt continues from saved weights (model flags must
match; the optimizer starts fresh), e.g. to fine-tune with a decaying lr.

With --steps 0 the run has no end: after warmup the learning rate stays
constant and training continues until RUN_DIR/STOP appears.
"""

import argparse
import itertools
import json
import math
import os
import time

import torch
import torch.nn as nn

from char_data import DEFAULT_PATH, CharDataset
from health import AdaptiveClipper, LIFMonitor, UpdateTracker, attention_and_context, param_stats
from models import SpikingTransformer
from train import pick_device, save_run, write_live


def loss_and_acc(model: nn.Module, seq: torch.Tensor):
    logits = model(seq[:, :-1])
    targets = seq[:, 1:]
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    acc = (logits.argmax(-1) == targets).float().mean()
    return loss, acc


@torch.no_grad()
def generate(model: nn.Module, data: CharDataset, prompt: str, num_chars: int, block_size: int,
             temperature: float = 1.0, device="cpu") -> str:
    """Autoregressive sampling; the context is cropped to the last block_size chars."""
    model.eval()
    ids = torch.tensor([data.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(num_chars):
        logits = model(ids[:, -block_size:])[:, -1] / temperature
        next_id = torch.multinomial(torch.softmax(logits, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
    return data.decode(ids[0].tolist())


def train(model: nn.Module, data: CharDataset, args, device: str, run_dir: str) -> dict:
    model.to(device)
    # weight decay only on Linear weight matrices: decaying the L1LayerNorm gains
    # shrinks every LIF's input below threshold over time (neurons go silent), and
    # embeddings / biases / attention scales should not be pulled toward zero either
    decay = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    groups = [{"params": [p for p in model.parameters() if id(p) in decay], "weight_decay": args.weight_decay},
              {"params": [p for p in model.parameters() if id(p) not in decay], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    open_ended = args.steps <= 0
    warmup = args.warmup_steps if open_ended else min(args.warmup_steps, args.steps)

    def lr_at(step):  # linear warmup, then cosine decay to lr / 10 (constant if open-ended)
        if step <= warmup:
            return args.lr * step / max(warmup, 1)
        if open_ended:
            return args.lr
        progress = (step - warmup) / max(args.steps - warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    train_T = model.T
    eval_Ts = sorted(set(args.eval_T) | {train_T})
    history = {"step": [], "loss": [], "acc": [], "bpc": [], "train_loss": [], "train_acc": [],
               "lr": [], "time": [], "checkpoints": []}
    for T in eval_Ts:
        history[f"acc_T{T}"], history[f"loss_T{T}"], history[f"bpc_T{T}"] = [], [], []
    # training-health metrics (see health.py): {name: [one value per evaluation]}
    health_keys = ("fire_rate", "fire_dead", "fire_sat", "surrogate", "grad_norm", "update_ratio",
                   "norm_gain", "attn_scale", "attn_empty", "attn_entropy", "attn_local")
    if args.adaptive_clip > 0:
        health_keys += ("clip_frac", "grad_spike")
    for key in health_keys:
        history[key] = {}
    history["context_gain"] = []
    lif_monitor, updates = LIFMonitor(model), UpdateTracker(model)
    clipper = AdaptiveClipper(model, args.adaptive_clip) if args.adaptive_clip > 0 else None
    update_ratios = {}

    stop_file = os.path.join(run_dir, "STOP")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    samples_path = os.path.join(run_dir, "samples.txt")

    def state_dict():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # fixed validation batches, so evaluations are comparable across steps
    gen = torch.Generator().manual_seed(1234)
    val_data = data.splits["val"]
    val_batches = []
    for _ in range(args.eval_iters):
        starts = torch.randint(len(val_data) - args.block_size - 1, (args.eval_batch_size,), generator=gen)
        val_batches.append(torch.stack([val_data[s: s + args.block_size + 1] for s in starts.tolist()]))

    best_loss, best_state = float("inf"), None
    start = time.time()
    train_loss_sum = train_acc_sum = 0.0
    train_count = 0
    for step in itertools.count(1) if open_ended else range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = acc = 0.0
        for _ in range(args.grad_accum):  # micro-batches summing to one batch of batch_size
            seq = data.get_batch("train", args.batch_size // args.grad_accum, args.block_size, device=device)
            micro_loss, micro_acc = loss_and_acc(model, seq)
            (micro_loss / args.grad_accum).backward()
            loss += micro_loss.item() / args.grad_accum
            acc += micro_acc.item() / args.grad_accum
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if clipper is not None:
            clipper.clip()
        if step % args.eval_every == 0:
            updates.snapshot()
        optimizer.step()
        if step % args.eval_every == 0:
            update_ratios = updates.ratios()
        train_loss_sum += loss
        train_acc_sum += acc
        train_count += 1

        user_stop = os.path.exists(stop_file)
        if step % args.eval_every == 0 or step == args.steps or user_stop:
            health = dict(param_stats(model), update_ratio=update_ratios)
            if clipper is not None:
                health.update(clipper.stats())
            model.eval()
            with torch.no_grad():
                for T in eval_Ts:
                    model.T = T
                    if T == train_T:
                        lif_monitor.start()
                    losses, accs = zip(*(loss_and_acc(model, b.to(device)) for b in val_batches))
                    if T == train_T:
                        health.update(lif_monitor.stop())
                        health.update(attention_and_context(model, [b.to(device) for b in val_batches[:2]]))
                    eval_loss = torch.stack(losses).mean().item()
                    history[f"loss_T{T}"].append(eval_loss)
                    history[f"bpc_T{T}"].append(eval_loss / math.log(2))
                    history[f"acc_T{T}"].append(torch.stack(accs).mean().item())
                model.T = train_T
            history["context_gain"].append(health.pop("context_gain"))
            for key, values in health.items():
                for name, value in values.items():
                    history[key].setdefault(name, []).append(value)
            history["step"].append(step)
            for key in ("loss", "acc", "bpc"):
                history[key].append(history[f"{key}_T{train_T}"][-1])
            history["train_loss"].append(train_loss_sum / train_count)
            history["train_acc"].append(train_acc_sum / train_count)
            history["lr"].append(lr_at(step))
            history["time"].append(time.time() - start)
            train_loss_sum = train_acc_sum = 0.0
            train_count = 0

            per_T = " ".join(f"T{T} {history[f'bpc_T{T}'][-1]:.3f}" for T in eval_Ts)
            print(f"step {step:6d} | train loss {history['train_loss'][-1]:.3f} "
                  f"| val bpc {per_T} | val acc {history['acc'][-1]*100:5.1f}% "
                  f"| lr {history['lr'][-1]:.2e} | {history['time'][-1]:.0f}s", flush=True)
            dead = {}
            for name, values in history["fire_dead"].items():
                group = "b" + name.split(".")[1] if name.startswith("blocks.") else "in"
                dead.setdefault(group, []).append(values[-1])
            print("  health | never-fired " + " ".join(f"{g} {100 * sum(v) / len(v):.0f}%" for g, v in dead.items())
                  + f" | context gain {history['context_gain'][-1]:.3f} nats"
                  + " | grad " + " ".join(f"{g.replace('block ', 'b')} {v[-1]:.0e}"
                                          for g, v in history["grad_norm"].items())
                  + " | empty attn " + " ".join(f"{100 * v[-1]:.0f}%" for v in history["attn_empty"].values())
                  + ("" if clipper is None else " | clipped " + " ".join(
                      f"{g.replace('block ', 'b')} {100 * v[-1]:.0f}%" for g, v in history["clip_frac"].items())),
                  flush=True)
            write_live(run_dir, history)
            if history["loss"][-1] < best_loss:
                best_loss, best_state = history["loss"][-1], state_dict()
                torch.save(best_state, os.path.join(run_dir, "best.pt"))

        if step % args.sample_every == 0 or step == args.steps or user_stop:
            text = generate(model, data, args.prompt, args.sample_chars, args.block_size,
                            args.temperature, device)
            print(f"--- sample @ step {step} ---\n{text}\n---", flush=True)
            with open(samples_path, "a") as f:
                f.write(f"=== step {step} ===\n{text}\n\n")

        if step % args.checkpoint_every == 0 or step == args.steps or user_stop:
            torch.save(state_dict(), os.path.join(ckpt_dir, f"checkpoint_step{step:06d}.pt"))
        if user_stop:
            print(f"stopped at step {step} (STOP file)")
            os.remove(stop_file)
            break

    history["final_state"] = best_state  # save_run writes this as best.pt
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--block_size", type=int, default=128, help="context length in characters")

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--T", type=int, default=8,
                        help="spiking simulation time steps (power of two for the integer model)")
    parser.add_argument("--attn_lif", action=argparse.BooleanOptionalAction, default=True,
                        help="LIF on the attention output before proj (binary proj input)")
    parser.add_argument("--learnable_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="learn the attention scale (2^s per block, init 1/8) instead of fixed 1/8")
    parser.add_argument("--pos_embed", choices=("learned", "sinusoid"), default="learned",
                        help="learned position embedding or fixed sinusoidal encoding")
    parser.add_argument("--attn_norm", action=argparse.BooleanOptionalAction, default=True,
                        help="L1LayerNorm on the attention output before the attention LIF")
    parser.add_argument("--attn_proj", action=argparse.BooleanOptionalAction, default=False,
                        help="keep the proj linear/norm/LIF after attention (default: removed)")
    parser.add_argument("--mlp_layers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--pow2_gamma", action=argparse.BooleanOptionalAction, default=False,
                        help="train L1LayerNorm gamma as signed powers of two (shift in hardware)")
    parser.add_argument("--eval_T", type=int, nargs="+", default=[4, 8],
                        help="time steps to evaluate at (the training T is always included)")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="split each batch into this many micro-batches (same gradient, less memory)")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_iters", type=int, default=8, help="val batches per evaluation")
    parser.add_argument("--steps", type=int, default=10000,
                        help="0 = train until RUN_DIR/STOP (constant lr after warmup)")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adaptive_clip", type=float, default=0.0,
                        help="per-block clipping: cut a block's gradient norm to this many times its running "
                             "average (e.g. 3); 0 = off. See health.AdaptiveClipper")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="clip the global grad norm (default 0 = off: surrogate-gradient norms reach "
                             "~1e10 at init, so a small clip freezes the head; Adam copes with the scale)")

    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--sample_chars", type=int, default=300)
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_suffix", type=str, default="", help="appended to the run directory name")
    parser.add_argument("--init_from", type=str, default="",
                        help="state_dict (.pt, e.g. a run's best.pt) to start from; the model flags must match it")
    args = parser.parse_args()
    assert args.batch_size % args.grad_accum == 0, "batch_size must be divisible by grad_accum"

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    data = CharDataset(args.data_path)
    print(f"device: {device}, vocab_size={data.vocab_size}, "
          f"train chars={len(data.splits['train']):,}, val chars={len(data.splits['val']):,}")

    run_name = f"char_{args.dim}d{args.depth}L_T{args.T}"
    if args.run_suffix:
        run_name += f"_{args.run_suffix}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run_dir={run_dir}")

    config = {"model_type": "spike", "task": "char_lm", "vocab_size": data.vocab_size, "chars": data.chars}
    config.update({k: v for k, v in vars(args).items() if k not in ("device", "run_suffix")})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model = SpikingTransformer(data.vocab_size, dim=args.dim, depth=args.depth, num_heads=args.num_heads,
                               T=args.T, max_len=args.block_size, attn_lif=args.attn_lif,
                               pow2_gamma=args.pow2_gamma, attn_proj=args.attn_proj,
                               mlp_layers=args.mlp_layers, attn_norm=args.attn_norm,
                               pos_embed=args.pos_embed, learnable_scale=args.learnable_scale)
    print(f"SpikingTransformer params: {sum(p.numel() for p in model.parameters()):,}")
    if args.init_from:
        # weights only: the optimizer state is not saved, so Adam restarts its
        # statistics (use a short --warmup_steps when resuming)
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"initialised weights from {args.init_from}")

    history = train(model, data, args, device, run_dir)
    save_run(run_dir, history)
    print(f"best val bpc: {min(history['bpc']):.3f}")


if __name__ == "__main__":
    main()
