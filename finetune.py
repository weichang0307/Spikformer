"""Fine-tune a trained run so its L1 norms use a per-position denominator
lookup table instead of dividing by each token's mean absolute deviation.

1. Calibrate: for each L1 norm in network order, measure the mean mad at
   every sequence position over fresh sequences (with every earlier norm
   already switched to its table), and freeze it as that norm's mad_table.
2. Fine-tune all weights with the tables frozen, so the network adapts to
   the fixed denominators.
3. Save the result as a new run, runs/<source run>_lut, whose best.pt holds
   the tables (spikformer_quant.py turns them into integer reciprocals).

Usage:
    python3 finetune.py runs/spike_8_32_32_T32
"""

import argparse
import json
import os

import torch

from data import generate_batch, key_position_mask, vocab_size
from models import model_from_config
from train import compute_loss_and_acc, pick_device, save_run, train_one_model

MIN_MAD = 1 / 16  # one Q4.4 LSB, the integer model's mad floor


@torch.no_grad()
def calibrate_mad_tables(model: SpikingTransformer, sequences: list[torch.Tensor]) -> None:
    """Fill every norm's mad_table with its mean mad per position, in network order."""
    model.eval()
    for norm in model.norms():
        norm.use_table = False
    for norm in model.norms():
        sums = torch.zeros_like(norm.mad_table)
        count = 0

        def measure(module, inputs):
            nonlocal sums, count
            (x,) = inputs
            centered = x - x.mean(dim=-1, keepdim=True)
            mad = centered.abs().mean(dim=-1)  # (B, L)
            sums[: mad.shape[1]] += mad.sum(dim=0)
            count += mad.shape[0]

        handle = norm.register_forward_pre_hook(measure)
        try:
            for sequence in sequences:
                model(sequence)
        finally:
            handle.remove()
        norm.mad_table.copy_((sums / count).clamp_min(MIN_MAD))
        norm.use_table = True


@torch.no_grad()
def evaluate(model: SpikingTransformer, sequences: list[torch.Tensor], mask: torch.Tensor) -> float:
    model.eval()
    accs = [compute_loss_and_acc(model(s), s, mask)[1].item() for s in sequences]
    return sum(accs) / len(accs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="trained run (config.json + best.pt)")
    parser.add_argument("--calib_batches", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--early_stop_acc", type=float, default=0.997)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or pick_device()
    torch.manual_seed(args.seed)
    with open(os.path.join(args.run_dir, "config.json")) as f:
        config = json.load(f)
    state = torch.load(os.path.join(args.run_dir, "best.pt"), map_location="cpu", weights_only=True)

    model = model_from_config(dict(config, norm_table=True), max_len=state["pos_embed"].shape[1])
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.endswith("mad_table") for k in missing):
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)

    task = (config["num_kv_pairs"], config["num_rounds"], config["num_keys"], config["num_values"])
    mask = key_position_mask(config["num_kv_pairs"], config["num_rounds"], device=device)
    batch = config.get("eval_batch_size", 512)
    eval_sequences = [generate_batch(batch, *task, device=device) for _ in range(2)]

    for norm in model.norms():
        norm.use_table = False
    acc_exact = evaluate(model, eval_sequences, mask)
    calibrate_mad_tables(model, [generate_batch(batch, *task, device=device)
                                 for _ in range(args.calib_batches)])
    acc_table = evaluate(model, eval_sequences, mask)
    print(f"before fine-tuning: per-token mad {acc_exact:.2%}, lookup table {acc_table:.2%}")

    history = train_one_model(
        "finetune", model, mask, device,
        num_kv_pairs=config["num_kv_pairs"], num_rounds=config["num_rounds"],
        num_keys=config["num_keys"], num_values=config["num_values"],
        batch_size=config["batch_size"], eval_batch_size=batch,
        steps=args.steps, eval_every=args.eval_every, lr=args.lr,
        checkpoint_every=args.checkpoint_every, early_stop_acc=args.early_stop_acc,
    )
    acc_tuned = evaluate(model, eval_sequences, mask)
    print(f"after fine-tuning: lookup table {acc_tuned:.2%}")

    out_dir = os.path.normpath(args.run_dir) + "_lut"
    os.makedirs(out_dir, exist_ok=True)
    out_config = dict(config, norm_table=True, finetuned_from=os.path.basename(os.path.normpath(args.run_dir)),
                      finetune_lr=args.lr, finetune_calib_batches=args.calib_batches,
                      acc_before_finetune=acc_table, acc_after_finetune=acc_tuned)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(out_config, f, indent=2)
    save_run(out_dir, history)


if __name__ == "__main__":
    main()
