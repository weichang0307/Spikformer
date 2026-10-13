"""Apply dynamic INT8 quantization to a saved Spikformer checkpoint.

This quantizes nn.Linear weights (including Q/K/V, projections, MLP, and
classifier layers). Embeddings, normalization, and spiking neuron state remain
floating point. Dynamic quantized Linear modules run on CPU.

Example:
    python quantize.py runs/spike_3_32_32
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

from data import generate_batch, key_position_mask
from models import SpikingTransformer


def build_model(config: dict, state_dict: dict) -> SpikingTransformer:
    """Recreate the float architecture using dimensions stored with the run."""
    if config.get("model_type") != "spike":
        raise ValueError(
            f"Expected a spiking run, got model_type={config.get('model_type')!r}."
        )

    vocab_size = config["num_keys"] + config["num_values"]
    return SpikingTransformer(
        vocab_size,
        dim=config["dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        T=config["T"],
        max_len=state_dict["pos_embed"].shape[1],
        norm_type=config.get("norm_type", "l1"),
        encoder_type=config.get("encoder_type", "identity"),
    )


def evaluate_model(
    model: SpikingTransformer,
    config: dict,
    eval_batches: int,
    seed: int,
) -> tuple[float, float]:
    """Return recall loss and accuracy on reproducible generated task batches."""
    batch_size = config.get("eval_batch_size", 512)
    batch_args = (
        config["num_kv_pairs"],
        config["num_rounds"],
        config["num_keys"],
        config["num_values"],
    )
    sequences = [
        generate_batch(batch_size, *batch_args, device="cpu")
        for _ in range(eval_batches)
    ]
    mask = key_position_mask(config["num_kv_pairs"], config["num_rounds"])

    total_loss = 0.0
    total_correct = 0
    total_targets = 0
    model.eval()
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        for batch_idx, sequence in enumerate(sequences):
            # Use the same random encoder samples for float and quantized models.
            torch.manual_seed(seed + batch_idx)
            logits = model(sequence)[:, :-1, :][:, mask, :]
            targets = sequence[:, 1:][:, mask]
            total_loss += nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            ).item()
            total_correct += (logits.argmax(-1) == targets).sum().item()
            total_targets += targets.numel()

    return total_loss / total_targets, total_correct / total_targets


def quantize_run(
    run_dir: Path,
    output_path: Path,
    eval_batches: int = 10,
    seed: int | None = None,
) -> Path:
    """Quantize the run's best.pt, evaluate accuracy, and save the result."""
    run_dir = run_dir.resolve()
    checkpoint_path = run_dir / "best.pt"
    output_path = output_path.resolve()
    if checkpoint_path == output_path:
        raise ValueError("Output path must not overwrite the source checkpoint.")

    with (run_dir / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model = build_model(config, state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    evaluation_seed = config.get("seed", 0) if seed is None else seed
    float_loss, float_accuracy = evaluate_model(
        model, config, eval_batches, evaluation_seed
    )

    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    ).eval()

    sequence_length = min(
        2 * config["num_kv_pairs"] * config["num_rounds"],
        quantized_model.pos_embed.shape[1],
    )
    sample_tokens = torch.zeros((1, sequence_length), dtype=torch.long)
    with torch.inference_mode():
        sample_logits = quantized_model(sample_tokens)
    if not torch.isfinite(sample_logits).all():
        raise RuntimeError("Quantized model smoke test produced non-finite logits.")

    quantized_loss, quantized_accuracy = evaluate_model(
        quantized_model, config, eval_batches, evaluation_seed
    )

    quantized_layers = sum(
        isinstance(module, nn.quantized.dynamic.Linear)
        for module in quantized_model.modules()
    )
    if quantized_layers == 0:
        raise RuntimeError("No Linear layers were dynamically quantized.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "spikformer_dynamic_int8",
            "config": config,
            "max_len": quantized_model.pos_embed.shape[1],
            "quantized_linear_layers": quantized_layers,
            "state_dict": quantized_model.state_dict(),
        },
        output_path,
    )

    source_bytes = checkpoint_path.stat().st_size
    quantized_bytes = output_path.stat().st_size
    print(f"Quantized {quantized_layers} Linear layers on CPU.")
    print(f"Checkpoint: {checkpoint_path} ({source_bytes / 1024:.1f} KiB)")
    print(f"Output:     {output_path} ({quantized_bytes / 1024:.1f} KiB)")
    print(f"Smoke test: logits shape {tuple(sample_logits.shape)}")
    print(
        f"Recall accuracy ({eval_batches} x {config.get('eval_batch_size', 512)} samples): "
        f"FP32 {float_accuracy:.2%} (loss {float_loss:.4f}), "
        f"INT8 {quantized_accuracy:.2%} (loss {quantized_loss:.4f}), "
        f"delta {(quantized_accuracy - float_accuracy) * 100:+.2f} pp"
    )
    return output_path


def load_quantized_checkpoint(checkpoint_path: Path) -> SpikingTransformer:
    """Load a checkpoint produced by this script for CPU inference."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(payload["config"], payload["state_dict"])
    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="run folder containing config.json and best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output file (default: RUN_DIR/quantized_int8.pt)",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=10,
        help="number of generated evaluation batches (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="evaluation seed (default: run config seed)",
    )
    args = parser.parse_args()

    if args.eval_batches <= 0:
        parser.error("--eval-batches must be positive")
    output_path = args.output or args.run_dir / "quantized_int8.pt"
    quantize_run(args.run_dir, output_path, args.eval_batches, args.seed)


if __name__ == "__main__":
    main()
