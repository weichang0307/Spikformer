"""Full-split, sliding-window evaluation of a character-level run (train_char.py).

Every character of the split (except the very first) is scored exactly once, with
as much context as the model allows: windows of --context characters move forward
by --stride; the first window scores all its positions, every later window only
the positions not scored yet (its last `stride` ones). With stride = context / 2
each scored character sees at least context / 2 previous characters, as in the
usual enwik8 / text8 test protocol, rather than the random training-time windows
where characters near a window's start have little context.

Reports bits per character and bits per byte (total bits / UTF-8 bytes of the
split; equal for ASCII corpora), for spiking runs at each --T. Results are saved
to runs/<run>/eval_<split>.json.

Usage:
    python3 eval_char.py enwik8_512d12L_ann
    python3 eval_char.py enwik8_512d12L_ann --ckpt last --split val
    python3 eval_char.py ptt_body_hanzi_512d6L_T6 --T 6 12 --stride 32
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from char_data import CharDataset
from models import model_from_config


@torch.no_grad()
def sliding_window_bits(model, ids: torch.Tensor, context: int, stride: int, batch: int, device, amp) -> float:
    """Total bits over ids[1:], each predicted once from up to `context` previous characters."""
    n = len(ids)
    if n - 1 <= context:
        begins, first_new = [0], [0]
    else:
        begins = list(range(0, n - 1 - context, stride)) + [n - 1 - context]
        first_new = [0] + [b_prev + context - b for b_prev, b in zip(begins[:-1], begins[1:])]
    window = min(context, n - 1)
    total = 0.0
    for i in range(0, len(begins), batch):
        b = torch.tensor(begins[i:i + batch])
        idx = b[:, None] + torch.arange(window + 1)[None, :]                      # (B, window + 1)
        seq = ids[idx].to(device=device, dtype=torch.long)
        with amp():
            logits = model(seq[:, :-1])
        logp = torch.log_softmax(logits.float(), -1).gather(-1, seq[:, 1:, None])[..., 0]   # (B, window)
        keep = torch.arange(window, device=device)[None, :] >= torch.tensor(first_new[i:i + batch], device=device)[:, None]
        total += -(logp * keep).sum().item() / math.log(2)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run name under runs/ (or a run directory)")
    parser.add_argument("--ckpt", default="best", help="best, last, or a path to a checkpoint")
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--context", type=int, default=0, help="window length (default: the run's block_size)")
    parser.add_argument("--stride", type=int, default=0, help="window step (default: context / 2)")
    parser.add_argument("--T", type=int, nargs="+", default=None, help="time steps for spiking runs (default: training T)")
    parser.add_argument("--batch", type=int, default=16, help="windows per forward pass")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast (default: fp32 with TF32 matmuls)")
    args = parser.parse_args()

    run_dir = args.run if os.path.isdir(args.run) else os.path.join("runs", args.run)
    config = json.load(open(os.path.join(run_dir, "config.json")))
    context = args.context or config["block_size"]
    stride = args.stride or max(context // 2, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.bf16 else torch.no_grad

    data = CharDataset(config["data_path"], config.get("val_fraction", 0.1), config.get("test_fraction", 0.0),
                       config.get("max_vocab", 0), val_size=config.get("val_chars", 0),
                       test_size=config.get("test_chars", 0))
    assert data.chars == config["chars"], "the corpus' vocabulary differs from the one the run was trained with"
    ids = data.splits[args.split]
    n_bytes = len(data.decode(ids.tolist()).encode("utf-8"))

    path = args.ckpt if os.path.isfile(args.ckpt) else os.path.join(run_dir, f"{args.ckpt}.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    step = state.get("step") if "model" in state else None
    model = model_from_config(config, config["block_size"])
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    spiking = config.get("model_type", "spike") == "spike"
    Ts = (args.T or [config["T"]]) if spiking else [None]
    print(f"{run_dir} ({os.path.basename(path)}{f', step {step}' if step else ''}): {args.split} split "
          f"{len(ids):,} chars / {n_bytes:,} bytes, context {context}, stride {stride}", flush=True)
    results = []
    for T in Ts:
        if T is not None:
            model.T = T
        t0 = time.time()
        bits = sliding_window_bits(model, ids, context, stride, args.batch, device, amp)
        scored = len(ids) - 1
        r = {"T": T, "bpc": bits / scored, "bits_per_byte": bits / n_bytes, "chars": scored, "bytes": n_bytes,
             "seconds": round(time.time() - t0, 1)}
        results.append(r)
        print(f"  {'' if T is None else f'T={T}: '}{r['bpc']:.4f} bits/char, {r['bits_per_byte']:.4f} bits/byte "
              f"({r['seconds']:.0f} s)", flush=True)
    out = {"checkpoint": path, "step": step, "split": args.split, "context": context, "stride": stride,
           "bf16": args.bf16, "results": results}
    with open(os.path.join(run_dir, f"eval_{args.split}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
