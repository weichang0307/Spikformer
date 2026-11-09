"""Post-training quantization tests of a trained character-level spiking run (no retraining).

Every variant is applied to a fresh copy of the checkpoint and scored on the same fixed validation
windows (default 256 x 1,024 characters) at one T. Variants:

    weights        Linear / embedding / head weights fake-quantized to N-bit signed integers,
                   symmetric, one scale per output row (per channel); biases kept in float
    pow2 gamma     every L1-norm gain gamma rounded to sign * 2^k (a shift in hardware)
    beta grid      every L1-norm bias beta rounded to a fixed-point grid (e.g. 1/16)
    residual cap   the integer residual stream entering Q/K/V and MLP layer 1 saturated at 2^b - 1
                   (the residual itself keeps its value; only the copy those layers read is capped)
    combined       several of the above at once

Prints val bpc, the change against the float model, and the weight storage at that bit width.

Usage:
    python3 quant_eval.py enwik8_512d12L_T6_distill_lr3e4 --T 6
    python3 quant_eval.py RUN --windows 512 --test       # also score the best combined variant on the full test split
"""

import argparse
import copy
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_data import CharDataset
from models import model_from_config
from modules import L1LayerNorm


def fake_quant_rows(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric N-bit quantize-dequantize with one scale per row (output channel)."""
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(flat / scale).clamp(-qmax, qmax) * scale).reshape(w.shape)


@torch.no_grad()
def quantize_weights(model: nn.Module, bits: int) -> int:
    """Quantize every Linear weight, the embedding and the position table; returns weight bytes at that width."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.weight.copy_(fake_quant_rows(mod.weight, bits)); n += mod.weight.numel()
        elif isinstance(mod, nn.Embedding):
            mod.weight.copy_(fake_quant_rows(mod.weight, bits)); n += mod.weight.numel()
    if getattr(model, "pos_embed", None) is not None and isinstance(model.pos_embed, nn.Parameter):
        model.pos_embed.copy_(fake_quant_rows(model.pos_embed[0], bits)[None]); n += model.pos_embed.numel()
    return n * bits // 8


@torch.no_grad()
def quantize_norms(model: nn.Module, pow2_gamma: bool, beta_step: float):
    for mod in model.modules():
        if isinstance(mod, L1LayerNorm):
            if pow2_gamma:
                mod.pow2_gamma = True
            if beta_step > 0:
                mod.bias.copy_(torch.round(mod.bias / beta_step) * beta_step)


def cap_residual(model: nn.Module, bits: int) -> list:
    """Saturate the inputs of the integer-residual-fed Linears (Q, K, V, MLP layer 1) at 2^bits - 1."""
    cap = 2 ** bits - 1
    hook = lambda mod, inp: (inp[0].clamp(max=cap),)
    handles = []
    for blk in model.blocks:
        for lin in (blk.attn.q_linear, blk.attn.k_linear, blk.attn.v_linear, blk.mlp.fc1):
            handles.append(lin.register_forward_pre_hook(hook))
    return handles


@torch.no_grad()
def evaluate(model: nn.Module, seqs: torch.Tensor, batch: int = 4) -> float:
    bits = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for b in seqs.split(batch):
            lg = model(b[:, :-1]).float()
            bits += F.cross_entropy(lg.reshape(-1, lg.size(-1)), b[:, 1:].reshape(-1), reduction="sum").item()
    return bits / (seqs.shape[0] * (seqs.shape[1] - 1)) / math.log(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--ckpt", default="last")
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--test", action="store_true", help="also run the full-test sliding-window eval of the best combined variant")
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run)
    config = json.load(open(os.path.join(run_dir, "config.json")))
    state = torch.load(os.path.join(run_dir, f"{args.ckpt}.pt"), map_location="cpu", weights_only=False)
    weights = state["model"] if "model" in state else state
    data = CharDataset(config["data_path"], config.get("val_fraction", 0.1), config.get("test_fraction", 0.0),
                       config.get("max_vocab", 0), val_size=config.get("val_chars", 0), test_size=config.get("test_chars", 0))
    val = data.splits["val"]
    L = config["block_size"]
    g = torch.Generator().manual_seed(0)
    seqs = torch.stack([val[s:s + L + 1] for s in torch.randint(0, len(val) - L - 1, (args.windows,), generator=g).tolist()]).long().cuda()

    base_model = model_from_config(config, L)
    base_model.load_state_dict(weights)
    base_model.cuda().eval()
    base_model.T = args.T
    float_bytes = sum(p.numel() for p in base_model.parameters()) * 4

    def variant(weight_bits=0, pow2_gamma=False, beta_step=0.0, residual_bits=0):
        m = copy.deepcopy(base_model)
        wbytes = quantize_weights(m, weight_bits) if weight_bits else None
        quantize_norms(m, pow2_gamma, beta_step)
        handles = cap_residual(m, residual_bits) if residual_bits else []
        bpc = evaluate(m, seqs)
        for h in handles:
            h.remove()
        return m, bpc, wbytes

    print(f"{args.run} ({args.ckpt}, step {state.get('step')}), T={args.T}, {args.windows} val windows x {L} chars; "
          f"float32 parameters {float_bytes / 2**20:.0f} MiB", flush=True)
    _, base, _ = variant()
    print(f"  {'float (bf16 compute)':40s} val bpc {base:.4f}", flush=True)
    rows = [("weights int8", dict(weight_bits=8)), ("weights int6", dict(weight_bits=6)),
            ("weights int4", dict(weight_bits=4)), ("weights int3", dict(weight_bits=3)),
            ("pow2 gamma", dict(pow2_gamma=True)), ("beta grid 1/16", dict(beta_step=1 / 16)),
            ("beta grid 1/4", dict(beta_step=1 / 4)), ("residual cap 3 bits (<= 7)", dict(residual_bits=3)),
            ("residual cap 2 bits (<= 3)", dict(residual_bits=2)),
            ("combined: int8 + beta/16", dict(weight_bits=8, beta_step=1 / 16)),
            ("combined: int8 + beta/16 + res 3b", dict(weight_bits=8, beta_step=1 / 16, residual_bits=3)),
            ("combined: int6 + beta/16 + res 3b", dict(weight_bits=6, beta_step=1 / 16, residual_bits=3))]
    # (pow2 gamma is left out of the combinations: post-training it costs ~1.8 bpc on its own; it needs training)
    results = {}
    for name, kw in rows:
        m, bpc, wbytes = variant(**kw)
        results[name] = (bpc, kw)
        size = f"   weights {wbytes / 2**20:5.1f} MiB" if wbytes else ""
        print(f"  {name:40s} val bpc {bpc:.4f} ({bpc - base:+.4f}){size}", flush=True)
        del m
        torch.cuda.empty_cache()

    if args.test:
        from eval_char import sliding_window_bits
        combos = {k: v for k, v in results.items() if k.startswith("combined")}
        best = min(combos, key=lambda k: combos[k][0] if combos[k][0] - base < 0.02 else 1e9)
        kw = combos[best][1]
        m = copy.deepcopy(base_model)
        if kw.get("weight_bits"):
            quantize_weights(m, kw["weight_bits"])
        quantize_norms(m, kw.get("pow2_gamma", False), kw.get("beta_step", 0.0))
        if kw.get("residual_bits"):
            cap_residual(m, kw["residual_bits"])
        test = data.splits["test"]
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        bits = sliding_window_bits(m, test, L, L // 2, 16, "cuda", amp)
        print(f"  full test, {best}: {bits / (len(test) - 1):.4f} bits/char", flush=True)


if __name__ == "__main__":
    main()
