"""folk-rnn-style LSTM baseline on a character-level corpus, with the same data, split and evaluation as
train_char.py, for comparisons with its Transformer / spiking runs.

Architecture as folk-rnn v2 (Sturm et al. 2016): one-hot inputs, 3 LSTM layers x 512 units, dropout 0.5 after
each layer, linear softmax head (~5.6M parameters with a 136-token vocabulary). Trained on
random block_size-token windows like train_char.py (rather than whole bucketed tunes), with AdamW, cosine
decay and gradient clipping. The best checkpoint by val loss is saved to runs/<name>/best.pt with a
config.json, so eval_char.py-style sliding-window scoring and generation work the same way.

Usage:
    python3 lstm_baseline.py --data_path data/folkrnn/folkrnn.txt --steps 10000
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_data import CharDataset


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 512, layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size   # one-hot inputs, as folk-rnn (5.62M parameters at 3 x 512, 137 tokens)
        self.lstm = nn.LSTM(vocab_size, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)
        self.T = 1   # eval_char.py compatibility (ignored)

    def forward(self, tokens: torch.Tensor, state=None, return_state: bool = False):
        x = F.one_hot(tokens, self.vocab_size).to(self.head.weight.dtype)
        out, state = self.lstm(x, state)
        logits = self.head(self.drop(out))
        return (logits, state) if return_state else logits


@torch.no_grad()
def val_loss(model, data, block, batches, bs, device):
    model.eval()
    g = torch.Generator().manual_seed(0)
    v = data.splits["val"]
    tot = 0.0
    for _ in range(batches):
        idx = torch.randint(0, len(v) - block - 1, (bs,), generator=g)
        seq = torch.stack([v[i:i + block + 1] for i in idx.tolist()]).long().to(device)
        lg = model(seq[:, :-1])
        tot += F.cross_entropy(lg.reshape(-1, lg.size(-1)), seq[:, 1:].reshape(-1)).item()
    model.train()
    return tot / batches


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--test_fraction", type=float, default=0.05)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--run_name", default="")
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    data = CharDataset(args.data_path, args.val_fraction, args.test_fraction)
    corpus = os.path.basename(os.path.dirname(os.path.abspath(args.data_path)))
    name = args.run_name or f"{corpus}_lstm{args.layers}x{args.hidden}"
    run_dir = os.path.join("runs", name)
    os.makedirs(run_dir, exist_ok=True)
    config = {"model_type": "lstm", "task": "char_lm", "vocab_size": data.vocab_size, "chars": data.chars,
              "T": 1, "dim": args.hidden, "depth": args.layers, "num_heads": 0, **vars(args)}
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    model = LSTMLM(data.vocab_size, args.hidden, args.layers, args.dropout).to(device)
    print(f"LSTM params: {sum(q.numel() for q in model.parameters()):,}; run_dir={run_dir}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    lr_at = lambda s: args.lr * s / args.warmup_steps if s <= args.warmup_steps else \
        args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - args.warmup_steps) / (args.steps - args.warmup_steps))))
    best, t0, train_sum, n = float("inf"), time.time(), 0.0, 0
    history = {"step": [], "train_loss": [], "loss": [], "bpc": []}
    for step in range(1, args.steps + 1):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        seq = data.get_batch("train", args.batch_size, args.block_size, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(seq[:, :-1])
        loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), seq[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        train_sum += loss.item(); n += 1
        if step % args.eval_every == 0 or step == args.steps:
            vl = val_loss(model, data, args.block_size, 16, 16, device)
            for k, v in (("step", step), ("train_loss", train_sum / n), ("loss", vl), ("bpc", vl / math.log(2))):
                history[k].append(v)
            print(f"step {step:6d} | train loss {train_sum / n:.3f} | val bpc {vl / math.log(2):.3f} | lr {lr_at(step):.2e} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            train_sum, n = 0.0, 0
            if vl < best:
                best = vl
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(run_dir, "best.pt"))
            json.dump(history, open(os.path.join(run_dir, "history.json"), "w"))
    print(f"best val bpc {best / math.log(2):.3f}", flush=True)


if __name__ == "__main__":
    main()
