"""Evaluate trained runs as float and INT32 models at several inference T.

For every run and every T, on the same fixed test set:
    float        the float model (exact per-token L1 divide)
    int32/divide INT32 model, exact divide in L1LayerNorm
    int32/shift  INT32 model, divide by mad rounded to the nearest power of two
    int32/floor  INT32 model, divide by 2^floor(log2 mad)
(see spikformer_quant.py for the integer arithmetic). Results are printed
and saved to runs/analysis/evaluate.json.

Usage:
    python3 evaluate.py
    python3 evaluate.py spike_8_32_32_T16 --T 2 4 8 16 --sequences 4096
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from data import generate_batch, vocab_size
from models import model_from_config
from spikformer_quant import QuantizedSpikformer, evaluate

DEFAULT_RUNS = ["spike_8_32_32_T16", "spike_12_32_32_T16", "spike_16_32_32_T16"]
VARIANTS = {"float": None, "int32/divide": "divide", "int32/shift": "shift", "int32/floor": "shift_floor"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--T", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--sequences", type=int, default=2048, help="test sequences per run")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345, help="test-set seed (differs from training)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    header = f"{'run':<20} {'T':>3} " + " ".join(f"{v:>13}" for v in VARIANTS)
    print(f"device={device}, {args.sequences} test sequences per run\n{header}")
    for name in args.runs:
        run_dir = Path(args.runs_dir) / name
        config = json.loads((run_dir / "config.json").read_text())
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        task = (config["num_kv_pairs"], config["num_rounds"], config["num_keys"], config["num_values"])
        torch.manual_seed(args.seed)
        sequences = [generate_batch(args.batch_size, *task, device="cpu")
                     for _ in range(args.sequences // args.batch_size)]

        float_model = model_from_config(config, max_len=state["pos_embed"].shape[1])
        float_model.load_state_dict(state)
        float_model.to(device).eval()
        int_model = QuantizedSpikformer(config, state).to(device)

        results[name] = {}
        for T in args.T:
            start = time.time()
            row = {}
            for variant, mode in VARIANTS.items():
                if mode is None:
                    float_model.T = T
                    row[variant] = evaluate(float_model, [s.to(device) for s in sequences], config)
                else:
                    int_model.set_T(T).set_norm_mode(mode)
                    row[variant] = evaluate(int_model, sequences, config)
            results[name][T] = row
            print(f"{name:<20} {T:>3} " + " ".join(f"{row[v]:>13.2%}" for v in VARIANTS)
                  + f"   ({time.time() - start:.0f}s)", flush=True)

    out = os.path.join(args.runs_dir, "analysis", "evaluate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"sequences": args.sequences, "seed": args.seed, "results": results}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
