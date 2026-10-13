"""Integer-only, fixed-point Spikformer inference from a saved best.pt.

All activations and weights are signed INT8 with 4 fractional bits. Linear,
attention, normalization, residual, encoder, and LIF state calculations use
integer arithmetic (INT32/INT64 intermediates). Spikes are represented as
0 or 16. This reference implementation runs on CPU; it is not a hardware-
optimized kernel.

Usage:
    python spikformer_quant.py runs/spike_3_32_32
"""

import argparse
import json
import math
from pathlib import Path

import torch
from data import generate_batch, key_position_mask
from models import SpikingTransformer


Q = 16
INT8_MIN = -128
INT8_MAX = 127
MULTIPLIER_BITS = 24
MULTIPLIER = 1 << MULTIPLIER_BITS


def saturate_int8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(INT8_MIN, INT8_MAX).to(torch.int32)


def round_divide(
    numerator: torch.Tensor, denominator: torch.Tensor | int
) -> torch.Tensor:
    if not isinstance(denominator, torch.Tensor):
        denominator = torch.tensor(
            denominator, dtype=torch.int64, device=numerator.device
        )
    denominator = denominator.clamp_min(1)
    magnitude = (numerator.abs() + denominator // 2) // denominator
    return torch.where(numerator < 0, -magnitude, magnitude)


def integer_sqrt(value: torch.Tensor) -> torch.Tensor:
    """Floor square root using an integer Newton iteration."""
    value = value.clamp_min(0).to(torch.int64)
    root = value.clamp_min(1)
    for _ in range(32):
        next_root = torch.minimum(
            root, (root + value // root.clamp_min(1)) // 2
        )
        if torch.equal(next_root, root):
            break
        root = next_root
    return root


class QuantizedLinear:
    """Per-output-channel symmetric INT8 weights with integer requantization."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        saturate_output: bool = True,
    ):
        self.saturate_output = saturate_output
        weight = weight.to(torch.float64)
        row_scale = weight.abs().amax(dim=1).clamp_min(1e-12) / INT8_MAX
        self.weight = torch.round(weight / row_scale[:, None]).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        self.multiplier = torch.round(row_scale * MULTIPLIER).to(torch.int64)
        if bias is None:
            self.bias = torch.zeros(weight.shape[0], dtype=torch.int64)
        else:
            self.bias = torch.round(bias.to(torch.float64) * Q).to(torch.int64)

    @classmethod
    def from_state(cls, state: dict, prefix: str) -> "QuantizedLinear":
        bias = state.get(f"{prefix}.bias")
        return cls(state[f"{prefix}.weight"], bias, saturate_output=prefix != "head")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        accum = x.to(torch.int64) @ self.weight.to(torch.int64).t()
        requantized = round_divide(accum * self.multiplier, MULTIPLIER)
        output = requantized + self.bias
        return saturate_int8(output) if self.saturate_output else output.to(torch.int32)

    def export(self, prefix: str, output: dict) -> None:
        output[f"{prefix}.weight"] = self.weight
        output[f"{prefix}.multiplier"] = self.multiplier
        output[f"{prefix}.bias"] = self.bias

    @classmethod
    def from_packed(cls, packed: dict, prefix: str) -> "QuantizedLinear":
        layer = cls.__new__(cls)
        layer.weight = packed[f"{prefix}.weight"]
        layer.multiplier = packed[f"{prefix}.multiplier"]
        layer.bias = packed[f"{prefix}.bias"]
        layer.saturate_output = prefix != "head"
        return layer


class QuantizedNorm:
    def __init__(self, state: dict, prefix: str, norm_type: str):
        self.norm_type = norm_type
        self.weight = torch.round(
            state[f"{prefix}.weight"].to(torch.float64) * Q
        ).clamp(INT8_MIN, INT8_MAX).to(torch.int8)
        self.bias = state.get(f"{prefix}.bias")
        if self.bias is not None:
            self.bias = torch.round(self.bias.to(torch.float64) * Q).clamp(
                INT8_MIN, INT8_MAX
            ).to(torch.int8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int64)
        if self.norm_type == "l1" or self.norm_type == "l2":
            centered = x - round_divide(x.sum(dim=-1, keepdim=True), x.shape[-1])
        else:
            centered = x

        if self.norm_type == "l1":
            denominator = round_divide(
                centered.abs().sum(dim=-1, keepdim=True), centered.shape[-1]
            )
        elif self.norm_type == "l2":
            variance = round_divide(
                centered.square().sum(dim=-1, keepdim=True), centered.shape[-1]
            )
            denominator = integer_sqrt(variance)
        elif self.norm_type == "rms":
            mean_square = round_divide(x.square().sum(dim=-1, keepdim=True), x.shape[-1])
            denominator = integer_sqrt(mean_square)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type!r}")

        normalized = round_divide(centered * Q, denominator)
        gamma = self.weight.to(torch.int64)
        output = round_divide(normalized * gamma, Q)
        if self.bias is not None:
            output = output + self.bias.to(torch.int64)
        return saturate_int8(output)

    def export(self, prefix: str, output: dict) -> None:
        output[f"{prefix}.weight"] = self.weight
        if self.bias is not None:
            output[f"{prefix}.bias"] = self.bias

    @classmethod
    def from_packed(
        cls, packed: dict, prefix: str, norm_type: str
    ) -> "QuantizedNorm":
        layer = cls.__new__(cls)
        layer.norm_type = norm_type
        layer.weight = packed[f"{prefix}.weight"]
        layer.bias = packed.get(f"{prefix}.bias")
        return layer


class QuantizedLIF:
    def __init__(self, tau: int = 2):
        self.tau = tau
        self.state = None

    def reset_state(self) -> None:
        self.state = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            self.state = torch.zeros_like(x, dtype=torch.int32)
        self.state = self.state + round_divide((x - self.state).to(torch.int64), self.tau)
        spikes = (self.state >= Q).to(torch.int32) * Q
        self.state = self.state - spikes
        return spikes


class QuantizedRateEncoder:
    def __init__(self):
        # LUT indexed by signed INT8 activation after shifting into [0, 255].
        codes = torch.arange(INT8_MIN, INT8_MAX + 1, dtype=torch.float64)
        self.probability = torch.round(torch.sigmoid(codes / Q) * 256).clamp(0, 256).to(torch.int32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        probability = self.probability[(x.to(torch.int32) - INT8_MIN).to(torch.long)]
        random_codes = torch.randint(0, 256, x.shape, dtype=torch.int32)
        return (random_codes < probability).to(torch.int32) * Q


class QuantizedAttention:
    def __init__(self, state: dict, prefix: str, dim: int, heads: int, norm_type: str):
        self.heads = heads
        self.head_dim = dim // heads
        self.linears = {
            name: QuantizedLinear.from_state(state, f"{prefix}.{name}_linear")
            for name in ("q", "k", "v", "proj")
        }
        self.norms = {
            name: QuantizedNorm(state, f"{prefix}.{name}_norm", norm_type)
            for name in ("q", "k", "v", "proj")
        }
        self.neurons = {name: QuantizedLIF() for name in ("q", "k", "v", "proj")}

    def reset_state(self) -> None:
        for neuron in self.neurons.values():
            neuron.reset_state()

    def __call__(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        spikes = {}
        for name in ("q", "k", "v"):
            current = self.norms[name](self.linears[name](x))
            spikes[name] = self.neurons[name](current).view(
                batch, length, self.heads, self.head_dim
            ).transpose(1, 2) // Q

        attention = spikes["q"].to(torch.int32) @ spikes["k"].to(torch.int32).transpose(-2, -1)
        attention = attention.masked_fill(causal_mask, 0)
        attended = attention @ spikes["v"].to(torch.int32)
        attended = round_divide(attended * Q, 8)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        projected = self.linears["proj"](saturate_int8(attended))
        return self.neurons["proj"](self.norms["proj"](projected))

    def export(self, prefix: str, output: dict) -> None:
        for name in ("q", "k", "v", "proj"):
            self.linears[name].export(f"{prefix}.{name}_linear", output)
            self.norms[name].export(f"{prefix}.{name}_norm", output)

    @classmethod
    def from_packed(
        cls,
        packed: dict,
        prefix: str,
        dim: int,
        heads: int,
        norm_type: str,
    ) -> "QuantizedAttention":
        attention = cls.__new__(cls)
        attention.heads = heads
        attention.head_dim = dim // heads
        attention.linears = {
            name: QuantizedLinear.from_packed(packed, f"{prefix}.{name}_linear")
            for name in ("q", "k", "v", "proj")
        }
        attention.norms = {
            name: QuantizedNorm.from_packed(
                packed, f"{prefix}.{name}_norm", norm_type
            )
            for name in ("q", "k", "v", "proj")
        }
        attention.neurons = {name: QuantizedLIF() for name in ("q", "k", "v", "proj")}
        return attention


class QuantizedMLP:
    def __init__(self, state: dict, prefix: str, norm_type: str):
        self.fc1 = QuantizedLinear.from_state(state, f"{prefix}.fc1")
        self.norm1 = QuantizedNorm(state, f"{prefix}.norm1", norm_type)
        self.lif1 = QuantizedLIF()
        self.fc2 = QuantizedLinear.from_state(state, f"{prefix}.fc2")
        self.norm2 = QuantizedNorm(state, f"{prefix}.norm2", norm_type)
        self.lif2 = QuantizedLIF()

    def reset_state(self) -> None:
        self.lif1.reset_state()
        self.lif2.reset_state()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lif1(self.norm1(self.fc1(x)))
        return self.lif2(self.norm2(self.fc2(x)))

    def export(self, prefix: str, output: dict) -> None:
        self.fc1.export(f"{prefix}.fc1", output)
        self.norm1.export(f"{prefix}.norm1", output)
        self.fc2.export(f"{prefix}.fc2", output)
        self.norm2.export(f"{prefix}.norm2", output)

    @classmethod
    def from_packed(
        cls, packed: dict, prefix: str, norm_type: str
    ) -> "QuantizedMLP":
        mlp = cls.__new__(cls)
        mlp.fc1 = QuantizedLinear.from_packed(packed, f"{prefix}.fc1")
        mlp.norm1 = QuantizedNorm.from_packed(packed, f"{prefix}.norm1", norm_type)
        mlp.lif1 = QuantizedLIF()
        mlp.fc2 = QuantizedLinear.from_packed(packed, f"{prefix}.fc2")
        mlp.norm2 = QuantizedNorm.from_packed(packed, f"{prefix}.norm2", norm_type)
        mlp.lif2 = QuantizedLIF()
        return mlp


class QuantizedBlock:
    def __init__(
        self,
        state: dict,
        prefix: str,
        dim: int,
        heads: int,
        norm_type: str,
    ):
        self.attn = QuantizedAttention(state, f"{prefix}.attn", dim, heads, norm_type)
        self.mlp = QuantizedMLP(state, f"{prefix}.mlp", norm_type)

    def reset_state(self) -> None:
        self.attn.reset_state()
        self.mlp.reset_state()

    def __call__(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = saturate_int8(x.to(torch.int64) + self.attn(x, causal_mask))
        return saturate_int8(x.to(torch.int64) + self.mlp(x))

    def export(self, prefix: str, output: dict) -> None:
        self.attn.export(f"{prefix}.attn", output)
        self.mlp.export(f"{prefix}.mlp", output)

    @classmethod
    def from_packed(
        cls,
        packed: dict,
        prefix: str,
        dim: int,
        heads: int,
        norm_type: str,
    ) -> "QuantizedBlock":
        block = cls.__new__(cls)
        block.attn = QuantizedAttention.from_packed(
            packed, f"{prefix}.attn", dim, heads, norm_type
        )
        block.mlp = QuantizedMLP.from_packed(
            packed, f"{prefix}.mlp", norm_type
        )
        return block


class QuantizedSpikformer:
    """CPU fixed-point Spikformer; logits are INT32 values in 1/Q units."""

    def __init__(self, config: dict, state: dict):
        if config.get("model_type") != "spike":
            raise ValueError("QuantizedSpikformer only supports model_type='spike'.")
        self.config = config
        self.T = int(config["T"])
        self.encoder_type = config.get("encoder_type", "identity")
        self.norm_type = config.get("norm_type", "l1")
        self.max_len = int(state["pos_embed"].shape[1])
        self.embed = torch.round(state["embed.weight"].to(torch.float64) * Q).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        self.pos_embed = torch.round(state["pos_embed"].to(torch.float64) * Q).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        self.input_norm = QuantizedNorm(state, "input_norm", self.norm_type)
        self.input_lif = QuantizedLIF()
        self.encoder = QuantizedRateEncoder() if self.encoder_type == "rate" else None
        dim = int(config["dim"])
        self.blocks = [
            QuantizedBlock(
                state, f"blocks.{index}", dim, int(config["num_heads"]), self.norm_type
            )
            for index in range(int(config["depth"]))
        ]
        self.head = QuantizedLinear.from_state(state, "head")

    @classmethod
    def from_checkpoint(cls, run_dir: Path) -> "QuantizedSpikformer":
        with (run_dir / "config.json").open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        return cls(config, state)

    def reset_state(self) -> None:
        self.input_lif.reset_state()
        for block in self.blocks:
            block.reset_state()

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.device.type != "cpu":
            raise ValueError("QuantizedSpikformer currently supports CPU tensors only.")
        batch, length = tokens.shape
        if length > self.max_len:
            raise ValueError(f"Input length {length} exceeds max_len={self.max_len}.")

        x = self.embed[tokens.long()].to(torch.int32)
        x = x.to(torch.int64) + self.pos_embed[:, :length].to(torch.int64)
        x = self.input_norm(x)
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)

        self.reset_state()
        accumulated = torch.zeros((batch, length, int(self.config["dim"])), dtype=torch.int64)
        for _ in range(self.T):
            if self.encoder_type == "identity":
                current = self.input_lif(x)
            elif self.encoder_type == "rate":
                current = self.encoder(x)
            else:
                raise ValueError(f"Unsupported encoder: {self.encoder_type!r}")
            for block in self.blocks:
                current = block(current, causal_mask)
            accumulated += current

        readout = round_divide(accumulated, self.T)
        return self.head(saturate_int8(readout))

    def load_state_dict(self, state: dict) -> None:
        replacement = type(self)(self.config, state)
        self.__dict__.update(replacement.__dict__)

    def export_state(self) -> dict:
        output = {
            "embed": self.embed,
            "pos_embed": self.pos_embed,
        }
        self.input_norm.export("input_norm", output)
        for index, block in enumerate(self.blocks):
            block.export(f"blocks.{index}", output)
        self.head.export("head", output)
        return output

    @classmethod
    def from_packed(cls, config: dict, packed: dict) -> "QuantizedSpikformer":
        model = cls.__new__(cls)
        model.config = config
        model.T = int(config["T"])
        model.encoder_type = config.get("encoder_type", "identity")
        model.norm_type = config.get("norm_type", "l1")
        model.max_len = int(packed["pos_embed"].shape[1])
        model.embed = packed["embed"]
        model.pos_embed = packed["pos_embed"]
        model.input_norm = QuantizedNorm.from_packed(
            packed, "input_norm", model.norm_type
        )
        model.input_lif = QuantizedLIF()
        model.encoder = QuantizedRateEncoder() if model.encoder_type == "rate" else None
        dim = int(config["dim"])
        model.blocks = [
            QuantizedBlock.from_packed(
                packed,
                f"blocks.{index}",
                dim,
                int(config["num_heads"]),
                model.norm_type,
            )
            for index in range(int(config["depth"]))
        ]
        model.head = QuantizedLinear.from_packed(packed, "head")
        return model

    @classmethod
    def from_quantized_checkpoint(cls, checkpoint_path: Path) -> "QuantizedSpikformer":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("format") != "spikformer_fixed_point_int8":
            raise ValueError(f"Unsupported quantized checkpoint: {checkpoint_path}")
        if payload.get("fractional_bits") != int(math.log2(Q)):
            raise ValueError("Checkpoint fixed-point scale does not match this model.")
        return cls.from_packed(payload["config"], payload["state"])


def evaluate(model, sequences: list[torch.Tensor], config: dict, seed: int) -> float:
    mask = key_position_mask(config["num_kv_pairs"], config["num_rounds"])
    correct = 0
    count = 0
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        for batch_index, sequence in enumerate(sequences):
            torch.manual_seed(seed + batch_index)
            logits = model(sequence)[:, :-1, :][:, mask, :]
            targets = sequence[:, 1:][:, mask]
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            count += targets.numel()
    return correct / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory containing config.json and best.pt")
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="quantized checkpoint path (default: RUN_DIR/best_int8.pt)",
    )
    args = parser.parse_args()
    if args.eval_batches <= 0 or args.batch_size <= 0:
        parser.error("--eval-batches and --batch-size must be positive")

    run_dir = args.run_dir.resolve()
    quantized_model = QuantizedSpikformer.from_checkpoint(run_dir)
    config = quantized_model.config
    seed = config.get("seed", 0) if args.seed is None else args.seed
    sequences = [
        generate_batch(
            args.batch_size,
            config["num_kv_pairs"],
            config["num_rounds"],
            config["num_keys"],
            config["num_values"],
            device="cpu",
        )
        for _ in range(args.eval_batches)
    ]

    float_state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    float_model = SpikingTransformer(
        config["num_keys"] + config["num_values"],
        dim=config["dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        T=config["T"],
        max_len=float_state["pos_embed"].shape[1],
        norm_type=config.get("norm_type", "l1"),
        encoder_type=config.get("encoder_type", "identity"),
    )
    float_model.load_state_dict(float_state)
    float_model.eval()
    float_accuracy = evaluate(float_model, sequences, config, seed)
    quantized_accuracy = evaluate(quantized_model, sequences, config, seed)

    output_path = args.output or run_dir / "best_int8.pt"
    torch.save(
        {
            "format": "spikformer_fixed_point_int8",
            "fractional_bits": int(math.log2(Q)),
            "config": config,
            "state": quantized_model.export_state(),
        },
        output_path,
    )
    reloaded_model = QuantizedSpikformer.from_quantized_checkpoint(output_path)
    smoke_logits = reloaded_model(sequences[0][:1])
    if smoke_logits.dtype != torch.int32 or not torch.isfinite(smoke_logits).all():
        raise RuntimeError("Reloaded integer model failed its inference smoke test.")
    print(f"FP32 recall accuracy: {float_accuracy:.2%}")
    print(f"INT8 recall accuracy: {quantized_accuracy:.2%}")
    print(f"Accuracy change:      {(quantized_accuracy - float_accuracy) * 100:+.2f} pp")
    print(f"Saved integer model metadata/state to {output_path}")


if __name__ == "__main__":
    main()
