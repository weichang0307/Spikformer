"""Hardware-exact integer Spikformer inference: the golden model for the RTL.

Every operation here is one the hardware performs bit-for-bit, and every
intermediate value -- accumulators, products, requantization -- fits a signed
32-bit integer (INT32). The products that could overflow are checked at run
time (`check_int32`) and raise OverflowError instead of wrapping.

- Activations and weights are signed INT8. Activations are fixed point with
  Q_BITS = 4 fractional bits (Q4.4); spikes enter the datapath as 0 or Q.
- Linear: INT8 x INT8 -> INT32 accumulator, then a per-output-row
  requantization `(acc * multiplier_r + 2^(shift_r - 1)) >>> shift_r`, plus a
  bias in Q4.4 units. Each row's (multiplier, shift) is chosen so that
  `acc * multiplier` stays below 2^30 for ANY INT8 input: the worst-case
  |acc| of row r is sum_k |W[r,k]| * 128.
- Every divide by a power of two is a round-half-up arithmetic shift,
  `shift_round(x, n) = (x + 2^(n-1)) >>> n`. This covers the L1 norm's mean
  and mean-absolute-deviation over D (D must be a power of two), the gamma
  scale, the requantization, and the rate readout over T (T must be a power
  of two).
- L1LayerNorm computes `centered = x - mean` and `mad = mean(|centered|)`
  (both shifts over D, mad clamped to >= 1), then `normalized` = centered / mad
  in Q4.4, in one of these modes (`QuantizedSpikformer.set_norm_mode`):
    divide       exact divide `(centered * Q) / mad`, truncating toward zero
                 like Verilog's signed `/`. The default.
    shift        divide by mad rounded to the nearest power of two:
                 e = leading-one position of mad, plus 1 if the bit below the
                 leading one is set; `(centered * Q + 2^(e-1)) >>> e`.
    shift_floor  the same with e = leading-one position only (2^e <= mad).
  Checkpoints fine-tuned with per-position denominator tables (finetune.py,
  config norm_table=True) instead use mode lut: a reciprocal table indexed
  by position, `recip[pos] = round(2^16 / mad_table[pos])`, and
  `normalized = (centered * recip[pos] + 2^15) >>> 16`.
- Multiplier-free blocks (checkpoints trained with attn_lif and pow2_gamma,
  see modules.py): every linear layer's per-row scale is a power of two, so
  requantization is `(acc + 2^(s-1)) >>> s` with no multiplier; each norm's
  gamma is +-2^e, so the gamma scale is a shift and a sign; and an LIF on the
  attention output makes proj's input binary (or, with attn_proj=False,
  proj is removed and the attention spikes go straight to the residual). Inside the blocks every linear
  input is then a sum of spikes (a non-negative multiple of Q, checked at run
  time), so those layers need only adds. Only the head, outside the blocks,
  multiplies general INT8 activations.
- LIF (tau = 2): `v += (x - v) >>> 1` (floor), spike when v >= Q, soft reset
  `v -= Q`.
- Attention scale 1/8: `(Q K^T V) * Q / 8 = (Q K^T V) << 1`.
- Every INT8 result saturates to [-128, 127].

Matrix products are computed in float64, which is exact here: every operand
is an integer and `int_matmul` checks that every partial sum stays below
2^31 (far inside float64's exact 2^53). That lets the model use BLAS / the
GPU instead of PyTorch's slow integer matmul; results are bit-identical to
integer arithmetic. Tensors are held as int64 in PyTorch, but no value
exceeds the INT32 range.

The simulated number of time steps can be changed after loading
(`set_T`, any power of two) to evaluate a model at fewer steps than it was
trained with.

Running this script also writes golden vectors for RTL verification: every
weight and every intermediate tensor of one sample, as $readmemh-compatible
hex files plus a manifest.json describing shapes and bit widths (see
`dump_golden`). It also reports the min/max of every intermediate over the
evaluation set, to size the RTL registers.

Usage:
    python3 spikformer_quant.py runs/spike_8_32_32_T16
    python3 spikformer_quant.py runs/spike_8_32_32_T16 --norm shift --T 4
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from data import generate_batch, key_position_mask
from models import attention_mask, model_from_config


Q_BITS = 4
Q = 1 << Q_BITS
INT8_MIN = -128
INT8_MAX = 127
INT32_LIMIT = 1 << 31
REQUANT_PRODUCT_LIMIT = 1 << 30  # worst-case |acc * multiplier|, leaving room for rounding
LIF_TAU_SHIFT = 1  # tau = 2
ATTN_SCALE_SHIFT = 3  # attention scale = 1/8
ATTN_SCALE_BITS = 12  # fractional bits of a learned attention scale (Q4.4 multiplier)
NORM_RECIP_BITS = 16  # fractional bits of the per-position 1/mad lookup table


def log2_exact(value: int, what: str) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{what} must be a power of two for the integer model, got {value}")
    return value.bit_length() - 1


def saturate_int8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(INT8_MIN, INT8_MAX).to(torch.int64)


def shift_round(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Divide by 2^shift, rounding half up: (x + 2^(shift-1)) >>> shift."""
    x = x.to(torch.int64)
    if shift == 0:
        return x
    return (x + (1 << (shift - 1))) >> shift


def check_int32(x: torch.Tensor, what: str) -> torch.Tensor:
    """Raise if any value of x falls outside the signed 32-bit range."""
    if x.numel() and (int(x.max()) >= INT32_LIMIT or int(x.min()) < -INT32_LIMIT):
        raise OverflowError(f"{what} exceeds INT32: range [{int(x.min())}, {int(x.max())}]")
    return x


def int_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact integer a @ b via float64 (BLAS / GPU), returned as int64.

    Every partial sum is bounded by K * max|a| * max|b|; this checks that the
    bound is inside INT32 (so an INT32 accumulator never overflows), which
    also keeps float64 exact.
    """
    bound = a.shape[-1] * int(a.abs().max()) * int(b.abs().max())
    if bound >= INT32_LIMIT:
        raise OverflowError(f"int_matmul partial sums can reach {bound} >= 2^31")
    return (a.to(torch.float64) @ b.to(torch.float64)).to(torch.int64)


def leading_one(x: torch.Tensor) -> torch.Tensor:
    """floor(log2 x) for integer x >= 1: the position of the leading 1 bit."""
    e = torch.floor(torch.log2(x.to(torch.float64))).to(torch.int64)
    e = e - ((torch.ones_like(e) << e) > x).to(torch.int64)       # guard float rounding
    e = e + ((torch.ones_like(e) << (e + 1)) <= x).to(torch.int64)
    return e


def to_device(value, device):
    """Move every tensor held by the model's objects, lists and dicts to `device`."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if type(value).__module__ == __name__ and hasattr(value, "__dict__"):
        for key, v in vars(value).items():
            setattr(value, key, to_device(v, device))
    return value


def trunc_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Signed integer divide truncating toward zero (Verilog `/`)."""
    return torch.div(numerator, denominator, rounding_mode="trunc")


class Trace:
    """Records intermediate tensors by name, one entry per call.

    Always tracks the running min/max of each site (for bit-width sizing);
    with keep=True it also stores the tensors themselves (for golden dumps).
    """

    def __init__(self, keep: bool = False):
        self.keep = keep
        self.frames: dict[str, list[torch.Tensor]] = {}
        self.ranges: dict[str, list[int]] = {}

    def record(self, name: str, x: torch.Tensor) -> None:
        lo, hi = int(x.min()), int(x.max())
        if name in self.ranges:
            r = self.ranges[name]
            r[0], r[1] = min(r[0], lo), max(r[1], hi)
        else:
            self.ranges[name] = [lo, hi]
        if self.keep:
            self.frames.setdefault(name, []).append(x.detach().clone())


class NullTrace(Trace):
    def record(self, name: str, x: torch.Tensor) -> None:
        pass


NULL_TRACE = NullTrace()


class QuantizedLinear:
    """Per-output-row symmetric INT8 weights with integer requantization.

    requant="multiplier": (acc * m_r + 2^(s_r-1)) >>> s_r, m_r sized for INT32.
    requant="shift": the row scale is rounded up to a power of two 2^-s_r
    before the weights are quantized, so requantization is a rounded shift.
    spike_input=True: the input must be a sum of spikes (non-negative multiples
    of Q), so the layer is adds only; checked on every call.
    """

    def __init__(self, state: dict, name: str, requant: str = "multiplier", spike_input: bool = False):
        self.name = name
        self.requant = requant
        self.spike_input = spike_input
        self.saturate_output = name != "head"
        weight = state[f"{name}.weight"].to(torch.float64)
        row_scale = weight.abs().amax(dim=1).clamp_min(1e-12) / INT8_MAX
        if requant == "shift":
            row_scale = torch.pow(2.0, torch.ceil(torch.log2(row_scale)))
        elif requant != "multiplier":
            raise ValueError(f"unknown requant mode {requant!r}")
        self.weight = torch.round(weight / row_scale[:, None]).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        # worst-case |acc| per row for any INT8 input (|x| <= 128)
        self.acc_bound = self.weight.to(torch.int64).abs().sum(dim=1) * 128
        bias = state.get(f"{name}.bias")
        if bias is None:
            self.bias = torch.zeros(weight.shape[0], dtype=torch.int64)
        else:
            self.bias = torch.round(bias.to(torch.float64) * Q).to(torch.int64)
        if requant == "shift":  # row_scale = 2^-shift exactly
            self.shift = (-torch.log2(row_scale)).round().to(torch.int64)
            self.multiplier = torch.ones_like(self.shift)
            if bool((self.acc_bound >= INT32_LIMIT).any()):
                raise ValueError(f"{name}: accumulator could exceed INT32")
            return
        shift = torch.floor(torch.log2(
            REQUANT_PRODUCT_LIMIT / (row_scale * self.acc_bound.clamp_min(1))
        )).clamp(0, 30).to(torch.int64)
        multiplier = torch.round(row_scale * torch.pow(2.0, shift.double())).to(torch.int64)
        while bool(((too_big := multiplier * self.acc_bound >= REQUANT_PRODUCT_LIMIT) & (shift > 0)).any()):
            shift = shift - (too_big & (shift > 0)).to(torch.int64)
            multiplier = torch.round(row_scale * torch.pow(2.0, shift.double())).to(torch.int64)
        if bool((multiplier == 0).any()) or bool((multiplier * self.acc_bound >= REQUANT_PRODUCT_LIMIT).any()):
            raise ValueError(f"{name}: cannot fit the requantization multiplier in INT32")
        self.multiplier = multiplier
        self.shift = shift

    def __call__(self, x: torch.Tensor, trace: Trace) -> torch.Tensor:
        if self.spike_input:
            if bool((x < 0).any()) or bool((x % Q != 0).any()):
                raise ValueError(f"{self.name}: input is not a sum of spikes; the layer would need multipliers")
            trace.record(f"{self.name}.in_count", x // Q)
        acc = int_matmul(x, self.weight.t())
        trace.record(f"{self.name}.acc", acc)
        if self.requant == "shift":
            right = self.shift.clamp_min(0)
            half = torch.where(right > 0, torch.ones_like(right) << (right - 1).clamp_min(0), 0)
            scaled = check_int32((acc << (-self.shift).clamp_min(0)) + half, f"{self.name} requantization")
            output = check_int32((scaled >> right) + self.bias, f"{self.name} output")
        else:
            half = torch.where(self.shift > 0, torch.ones_like(self.shift) << (self.shift - 1).clamp_min(0), 0)
            product = check_int32(acc * self.multiplier + half, f"{self.name} requantization")
            output = check_int32((product >> self.shift) + self.bias, f"{self.name} output")
        if self.saturate_output:
            output = saturate_int8(output)
        trace.record(self.name, output)
        return output

    def export(self, output: dict) -> None:
        output[f"{self.name}.weight"] = self.weight
        if self.requant == "multiplier":
            output[f"{self.name}.multiplier"] = self.multiplier
        output[f"{self.name}.shift"] = self.shift
        output[f"{self.name}.bias"] = self.bias


class QuantizedL1Norm:
    """L1LayerNorm; with pow2_gamma, gamma = sign * 2^e is applied as a shift."""

    def __init__(self, state: dict, name: str, pow2_gamma: bool = False):
        self.name = name
        self.pow2_gamma = pow2_gamma
        gamma = state[f"{name}.weight"].to(torch.float64)
        self.gamma_sign = torch.sign(gamma).to(torch.int64)
        self.gamma_exp = torch.round(torch.log2(gamma.abs().clamp_min(1e-8))).to(torch.int64)
        table = state.get(f"{name}.mad_table")
        self.recip = None if table is None else torch.round(
            (1 << NORM_RECIP_BITS) / table.to(torch.float64)
        ).to(torch.int64)
        self.weight = torch.round(state[f"{name}.weight"].to(torch.float64) * Q).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        self.bias = torch.round(state[f"{name}.bias"].to(torch.float64) * Q).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        self.dim_shift = log2_exact(self.weight.numel(), f"{name} width")
        self.mode = "divide" if self.recip is None else "lut"

    def __call__(self, x: torch.Tensor, trace: Trace) -> torch.Tensor:
        x = x.to(torch.int64)
        mean = shift_round(x.sum(dim=-1, keepdim=True), self.dim_shift)
        centered = x - mean
        trace.record(f"{self.name}.mean", mean)
        if self.mode == "lut":
            recip = self.recip[: x.shape[-2], None]  # (L, 1): one reciprocal per position
            normalized = shift_round(check_int32(centered * recip, f"{self.name} lut product"),
                                     NORM_RECIP_BITS)
        else:
            mad = shift_round(centered.abs().sum(dim=-1, keepdim=True), self.dim_shift).clamp_min(1)
            trace.record(f"{self.name}.mad", mad)
            if self.mode == "divide":
                normalized = trunc_div(centered * Q, mad)
            elif self.mode in ("shift", "shift_floor"):
                e = leading_one(mad)
                if self.mode == "shift":  # round to nearest power of two: mad >= 1.5 * 2^e -> e + 1
                    e = e + torch.where(e > 0, (mad >> (e - 1).clamp_min(0)) & 1, 0)
                trace.record(f"{self.name}.shift", e)
                half = torch.where(e > 0, torch.ones_like(e) << (e - 1).clamp_min(0), 0)
                normalized = (centered * Q + half) >> e
            else:
                raise ValueError(f"unknown norm mode {self.mode!r}")
        trace.record(f"{self.name}.normalized", normalized)
        if self.pow2_gamma:  # normalized * (+-2^e): a shift and a sign, no multiplier
            left = check_int32(normalized << self.gamma_exp.clamp_min(0), f"{self.name} gamma shift")
            right = (-self.gamma_exp).clamp_min(0)
            half = torch.where(right > 0, torch.ones_like(right) << (right - 1).clamp_min(0), 0)
            shifted = (left + half) >> right
            scaled_q = torch.where(self.gamma_sign > 0, shifted, torch.where(self.gamma_sign < 0, -shifted, 0))
        else:
            scaled = check_int32(normalized * self.weight.to(torch.int64), f"{self.name} gamma product")
            scaled_q = shift_round(scaled, Q_BITS)
        output = saturate_int8(scaled_q + self.bias.to(torch.int64))
        trace.record(self.name, output)
        return output

    def export(self, output: dict) -> None:
        if self.pow2_gamma:
            output[f"{self.name}.gamma_sign"] = self.gamma_sign
            output[f"{self.name}.gamma_exp"] = self.gamma_exp
        else:
            output[f"{self.name}.weight"] = self.weight
        output[f"{self.name}.bias"] = self.bias
        if self.recip is not None:
            output[f"{self.name}.recip"] = self.recip


class QuantizedLIF:
    def __init__(self, name: str):
        self.name = name
        self.v = None

    def reset_state(self) -> None:
        self.v = None

    def __call__(self, x: torch.Tensor, trace: Trace) -> torch.Tensor:
        """Return spikes as 0/Q; the trace records them as 0/1 bits."""
        if self.v is None:
            self.v = torch.zeros_like(x, dtype=torch.int64)
        self.v = self.v + ((x.to(torch.int64) - self.v) >> LIF_TAU_SHIFT)
        fired = (self.v >= Q).to(torch.int64)
        self.v = self.v - fired * Q
        trace.record(f"{self.name}.v", self.v)
        trace.record(f"{self.name}.spike", fired)
        return fired * Q


class QuantizedAttention:
    def __init__(self, state: dict, name: str, dim: int, heads: int, hw: dict):
        self.name = name
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_lif = hw["attn_lif"]
        self.attn_proj = hw["attn_proj"]
        parts = ("q", "k", "v", "proj") if self.attn_proj else ("q", "k", "v")
        self.linears = {p: QuantizedLinear(state, f"{name}.{p}_linear", hw["requant"],
                                           spike_input=hw["attn_lif"]) for p in parts}
        self.norms = {p: QuantizedL1Norm(state, f"{name}.{p}_norm", hw["pow2_gamma"]) for p in parts}
        self.neurons = {p: QuantizedLIF(f"{name}.{p}_lif") for p in parts}
        if self.attn_lif:
            self.neurons["attn"] = QuantizedLIF(f"{name}.attn_lif")
        log2_scale = state.get(f"{name}.log2_scale")
        # learned scale -> Q4.4 multiplier: attended * scale * Q = (attended * m + 2^11) >>> 12
        self.scale_multiplier = None if log2_scale is None else int(round(
            2.0 ** float(log2_scale) * Q * (1 << ATTN_SCALE_BITS)))
        self.attn_norm = hw["attn_norm"]
        if self.attn_norm:
            self.norms["attn"] = QuantizedL1Norm(state, f"{name}.attn_norm", hw["pow2_gamma"])

    def reset_state(self) -> None:
        for neuron in self.neurons.values():
            neuron.reset_state()

    def __call__(self, x: torch.Tensor, causal_mask: torch.Tensor, trace: Trace) -> torch.Tensor:
        batch, length, dim = x.shape
        spikes = {}
        for p in ("q", "k", "v"):
            current = self.norms[p](self.linears[p](x, trace), trace)
            spikes[p] = (self.neurons[p](current, trace) // Q).view(
                batch, length, self.heads, self.head_dim
            ).transpose(1, 2)

        attention = int_matmul(spikes["q"], spikes["k"].transpose(-2, -1)).masked_fill(causal_mask, 0)
        trace.record(f"{self.name}.qk", attention)
        attended = int_matmul(attention, spikes["v"])
        trace.record(f"{self.name}.qkv", attended)
        # (QK^T V) / 8 in Q4.4 units = (QK^T V) * Q / 8
        if self.scale_multiplier is None:
            attended = attended << (Q_BITS - ATTN_SCALE_SHIFT)
        else:
            attended = shift_round(check_int32(attended * self.scale_multiplier, f"{self.name} scale"),
                                   ATTN_SCALE_BITS)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        # the norm takes the full counts; without it the LIF input saturates to INT8
        attended = self.norms["attn"](attended, trace) if self.attn_norm else saturate_int8(attended)
        if self.attn_lif:
            attended = self.neurons["attn"](attended, trace)  # binary spikes (0 / Q) into proj
        if not self.attn_proj:
            return attended  # the attention LIF's spikes go straight to the residual
        projected = self.linears["proj"](attended, trace)
        return self.neurons["proj"](self.norms["proj"](projected, trace), trace)

    def export(self, output: dict) -> None:
        if self.scale_multiplier is not None:
            output[f"{self.name}.scale_multiplier"] = torch.tensor([self.scale_multiplier])
        for p in self.linears:
            self.linears[p].export(output)
        for norm in self.norms.values():
            norm.export(output)


class QuantizedMLP:
    def __init__(self, state: dict, name: str, hw: dict):
        spike = hw["attn_lif"]
        self.layers = hw["mlp_layers"]
        self.fc1 = QuantizedLinear(state, f"{name}.fc1", hw["requant"], spike_input=spike)
        self.norm1 = QuantizedL1Norm(state, f"{name}.norm1", hw["pow2_gamma"])
        self.lif1 = QuantizedLIF(f"{name}.lif1")
        if self.layers == 2:
            self.fc2 = QuantizedLinear(state, f"{name}.fc2", hw["requant"], spike_input=spike)
            self.norm2 = QuantizedL1Norm(state, f"{name}.norm2", hw["pow2_gamma"])
            self.lif2 = QuantizedLIF(f"{name}.lif2")

    def reset_state(self) -> None:
        self.lif1.reset_state()
        if self.layers == 2:
            self.lif2.reset_state()

    def __call__(self, x: torch.Tensor, trace: Trace) -> torch.Tensor:
        x = self.lif1(self.norm1(self.fc1(x, trace), trace), trace)
        if self.layers == 2:
            x = self.lif2(self.norm2(self.fc2(x, trace), trace), trace)
        return x

    def norms(self) -> list:
        return [self.norm1] + ([self.norm2] if self.layers == 2 else [])

    def export(self, output: dict) -> None:
        self.fc1.export(output)
        self.norm1.export(output)
        if self.layers == 2:
            self.fc2.export(output)
            self.norm2.export(output)


class QuantizedBlock:
    def __init__(self, state: dict, name: str, dim: int, heads: int, hw: dict):
        self.name = name
        self.attn = QuantizedAttention(state, f"{name}.attn", dim, heads, hw)
        self.mlp = QuantizedMLP(state, f"{name}.mlp", hw)

    def reset_state(self) -> None:
        self.attn.reset_state()
        self.mlp.reset_state()

    def __call__(self, x: torch.Tensor, causal_mask: torch.Tensor, trace: Trace) -> torch.Tensor:
        x = saturate_int8(x + self.attn(x, causal_mask, trace))
        trace.record(f"{self.name}.res1", x)
        x = saturate_int8(x + self.mlp(x, trace))
        trace.record(f"{self.name}.out", x)
        return x

    def export(self, output: dict) -> None:
        self.attn.export(output)
        self.mlp.export(output)


class QuantizedSpikformer:
    """Integer Spikformer; logits are INT32 values in 1/Q units."""

    def __init__(self, config: dict, state: dict):
        if config.get("model_type", "spike") != "spike":
            raise ValueError("QuantizedSpikformer only supports model_type='spike'.")
        if config.get("norm_type", "l1") != "l1" or config.get("encoder_type", "identity") != "identity":
            raise ValueError("Only L1LayerNorm with identity encoding is supported.")
        self.config = config
        self.T = int(config["T"])
        self.T_shift = log2_exact(self.T, "T")
        self.dim = int(config["dim"])
        self.max_len = int(state["pos_embed"].shape[1])
        self.embed = torch.round(state["embed.weight"].to(torch.float64) * Q).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        self.pos_embed = torch.round(state["pos_embed"][0].to(torch.float64) * Q).clamp(
            INT8_MIN, INT8_MAX
        ).to(torch.int8)
        # multiplier-free blocks when the model was trained for it (see the module docstring)
        multiplier_free = bool(config.get("attn_lif")) and bool(config.get("pow2_gamma"))
        self.hw = {
            "attn_lif": bool(config.get("attn_lif", False)),
            "pow2_gamma": bool(config.get("pow2_gamma", False)),
            "requant": "shift" if multiplier_free else "multiplier",
            "attn_proj": bool(config.get("attn_proj", True)),
            "mlp_layers": int(config.get("mlp_layers", 2)),
            "attn_norm": bool(config.get("attn_norm", False)),
        }
        self.input_norm = QuantizedL1Norm(state, "input_norm", self.hw["pow2_gamma"])
        self.input_lif = QuantizedLIF("input_lif")
        self.blocks = [
            QuantizedBlock(state, f"blocks.{i}", self.dim, int(config["num_heads"]), self.hw)
            for i in range(int(config["depth"]))
        ]
        self.head = QuantizedLinear(state, "head", self.hw["requant"])

        self.device = torch.device("cpu")

    def set_T(self, T: int) -> "QuantizedSpikformer":
        """Simulate T time steps (a power of two) instead of the trained T."""
        self.T_shift = log2_exact(T, "T")
        self.T = T
        return self

    def set_norm_mode(self, mode: str) -> "QuantizedSpikformer":
        """Choose how every L1 norm divides by mad: divide, shift or shift_floor."""
        if mode not in ("divide", "shift", "shift_floor"):
            raise ValueError(f"unknown norm mode {mode!r}")
        for norm in self.norms():
            if norm.mode == "lut":
                raise ValueError("this checkpoint uses per-position lookup tables (mode lut)")
            norm.mode = mode
        return self

    def to(self, device) -> "QuantizedSpikformer":
        to_device(self, torch.device(device))
        self.device = torch.device(device)
        return self

    @classmethod
    def from_run(cls, run_dir: Path, device="cpu") -> "QuantizedSpikformer":
        with (run_dir / "config.json").open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        return cls(config, state).to(device)

    def reset_state(self) -> None:
        self.input_lif.reset_state()
        for block in self.blocks:
            block.reset_state()

    def __call__(self, tokens: torch.Tensor, trace: Trace = NULL_TRACE) -> torch.Tensor:
        batch, length = tokens.shape
        if length > self.max_len:
            raise ValueError(f"Input length {length} exceeds max_len={self.max_len}.")

        tokens = tokens.to(self.device)
        x = self.embed[tokens.long()].to(torch.int64) + self.pos_embed[:length].to(torch.int64)
        trace.record("embed_sum", x)
        x = self.input_norm(x, trace)
        round_len = 2 * int(self.config["num_kv_pairs"]) if self.config.get("mask_round1") else 0
        causal_mask = attention_mask(length, round_len, self.device)

        self.reset_state()
        accumulated = torch.zeros((batch, length, self.dim), dtype=torch.int64, device=self.device)
        for _ in range(self.T):
            current = self.input_lif(x, trace)
            for block in self.blocks:
                current = block(current, causal_mask, trace)
            accumulated += current
        trace.record("readout_acc", accumulated)

        readout = saturate_int8(shift_round(accumulated, self.T_shift))
        trace.record("readout", readout)
        return self.head(readout, trace)

    def norms(self) -> list[QuantizedL1Norm]:
        norms = [self.input_norm]
        for block in self.blocks:
            norms += list(block.attn.norms.values())
            norms += block.mlp.norms()
        return norms

    def export_weights(self) -> dict:
        output = {"embed": self.embed, "pos_embed": self.pos_embed}
        self.input_norm.export(output)
        for block in self.blocks:
            block.export(output)
        self.head.export(output)
        return output


def signed_bits(lo: int, hi: int) -> int:
    """Two's-complement width that holds [lo, hi] (unsigned if lo >= 0)."""
    if lo >= 0:
        return max(hi.bit_length(), 1)
    return max(hi.bit_length(), (-lo - 1).bit_length()) + 1


def write_hex(out_dir: Path, relative: str, tensor: torch.Tensor) -> dict:
    """Write a tensor row-major, one two's-complement value per line."""
    path = out_dir / relative
    values = tensor.to(torch.int64).flatten().cpu()
    lo, hi = int(values.min()), int(values.max())
    bits = 8 if tensor.dtype == torch.int8 else signed_bits(lo, hi)
    digits = (bits + 3) // 4
    mask = (1 << (4 * digits)) - 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{v & mask:0{digits}x}\n" for v in values.tolist()))
    return {
        "file": relative,
        "shape": list(tensor.shape),
        "bits": bits,
        "signed": lo < 0 or tensor.dtype == torch.int8,
        "min": lo,
        "max": hi,
    }


def read_hex(path: Path, bits: int, signed: bool) -> torch.Tensor:
    """Inverse of write_hex (flat); for testbenches and self-checks."""
    values = [int(line, 16) for line in path.read_text().split()]
    digits_mask = (1 << (4 * ((bits + 3) // 4))) - 1
    if signed:
        values = [v - (digits_mask + 1) if v >> (bits - 1) & 1 else v for v in values]
    return torch.tensor(values, dtype=torch.int64)


def dump_golden(model: QuantizedSpikformer, tokens: torch.Tensor, out_dir: Path) -> dict:
    """Write weights and every intermediate tensor for one sequence.

    Layout of out_dir:
        tokens.hex, logits.hex
        weights/<name>.hex        e.g. blocks.0.attn.q_linear.weight (out, in)
        trace/<site>.hex          shape (calls, ...) with the batch dim removed;
                                  calls = T for per-time-step sites, 1 otherwise
        manifest.json             shape / bits / signedness / range per file
    All files are row-major, one value per line, two's complement.
    """
    if tokens.shape[0] != 1:
        raise ValueError("dump_golden expects a single sequence (batch size 1).")
    for stale in ("weights", "trace"):  # drop files from earlier dumps
        shutil.rmtree(out_dir / stale, ignore_errors=True)
    trace = Trace(keep=True)
    logits = model(tokens, trace).cpu()

    manifest = {
        "config": model.config,
        "fixed_point": {
            "Q_BITS": Q_BITS,
            "REQUANT": ("per-row (acc + 2^(shift-1)) >>> shift (left shift if shift < 0)"
                        if model.hw["requant"] == "shift"
                        else "per-row (acc * multiplier + 2^(shift-1)) >>> shift"),
            "ATTN_LIF": model.hw["attn_lif"],
            "POW2_GAMMA": model.hw["pow2_gamma"],
            "ATTN_PROJ": model.hw["attn_proj"],
            "MLP_LAYERS": model.hw["mlp_layers"],
            "ATTN_NORM": model.hw["attn_norm"],
            "NORM_MODE": model.norms()[0].mode,
            "T": model.T,
            "LIF_TAU_SHIFT": LIF_TAU_SHIFT,
            "ATTN_SCALE_SHIFT": ATTN_SCALE_SHIFT,
            "T_SHIFT": model.T_shift,
            "NORM_RECIP_BITS": NORM_RECIP_BITS,
        },
        "tokens": write_hex(out_dir, "tokens.hex", tokens[0]),
        "logits": write_hex(out_dir, "logits.hex", logits[0]),
        "prediction": logits[0].argmax(-1).tolist(),
        "weights": {},
        "trace": {},
        "trace_order": list(trace.frames),
    }
    for name, tensor in model.export_weights().items():
        manifest["weights"][name] = write_hex(out_dir, f"weights/{name}.hex", tensor)
    for name, frames in trace.frames.items():
        stacked = torch.stack([frame[0] for frame in frames]).cpu()  # drop batch dim
        info = write_hex(out_dir, f"trace/{name}.hex", stacked)
        reread = read_hex(out_dir / info["file"], info["bits"], info["signed"])
        if not torch.equal(reread, stacked.flatten()):
            raise RuntimeError(f"hex round trip failed for {name}")
        manifest["trace"][name] = info

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def evaluate(model, sequences: list[torch.Tensor], config: dict, trace: Trace = None) -> float:
    mask = key_position_mask(config["num_kv_pairs"], config["num_rounds"])
    correct = 0
    count = 0
    with torch.inference_mode():
        for sequence in sequences:
            logits = (model(sequence) if trace is None else model(sequence, trace)).cpu()
            logits = logits[:, :-1, :][:, mask, :]
            targets = sequence.cpu()[:, 1:][:, mask]
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            count += targets.numel()
    return correct / count


def site_kind(name: str) -> str:
    """Collapse per-block site names, e.g. blocks.1.attn.q_lif.v -> attn.q_lif.v."""
    parts = name.split(".")
    return ".".join(parts[2:]) if parts[0] == "blocks" else name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run directory containing config.json and best.pt")
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="default: cuda if available, else cpu")
    parser.add_argument("--norm", choices=("divide", "shift", "shift_floor"), default="divide",
                        help="how L1 norms divide by mad (ignored for lookup-table checkpoints)")
    parser.add_argument("--T", type=int, help="time steps to simulate (default: the trained T)")
    parser.add_argument("--dump-dir", type=Path,
                        help="golden vector directory (default: RUN_DIR/golden_<norm>_T<T>)")
    args = parser.parse_args()
    if args.eval_batches <= 0 or args.batch_size <= 0:
        parser.error("--eval-batches and --batch-size must be positive")

    run_dir = args.run_dir.resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantized_model = QuantizedSpikformer.from_run(run_dir, device)
    config = quantized_model.config
    T = args.T or int(config["T"])
    quantized_model.set_T(T)
    norm_mode = "lut" if config.get("norm_table") else args.norm
    if norm_mode != "lut":
        quantized_model.set_norm_mode(norm_mode)
    tag = f"{norm_mode}_T{T}"
    torch.manual_seed(config.get("seed", 0) if args.seed is None else args.seed)
    sequences = [
        generate_batch(
            args.batch_size, config["num_kv_pairs"], config["num_rounds"],
            config["num_keys"], config["num_values"], device="cpu",
        )
        for _ in range(args.eval_batches)
    ]

    float_state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    float_model = model_from_config(dict(config, T=T), max_len=float_state["pos_embed"].shape[1])
    float_model.load_state_dict(float_state)
    float_model.eval()
    float_model.T = T
    float_accuracy = evaluate(float_model, sequences, config)
    ranges = Trace()
    quantized_accuracy = evaluate(quantized_model, sequences, config, ranges)

    print(f"L1 norm mode: {norm_mode}, T = {T}")
    print(f"FP32 recall accuracy: {float_accuracy:.2%}")
    print(f"INT  recall accuracy: {quantized_accuracy:.2%}")
    print(f"Accuracy change:      {(quantized_accuracy - float_accuracy) * 100:+.2f} pp")

    kinds: dict[str, list[int]] = {}
    for name, (lo, hi) in ranges.ranges.items():
        r = kinds.setdefault(site_kind(name), [lo, hi])
        r[0], r[1] = min(r[0], lo), max(r[1], hi)
    print(f"\nValue ranges over {args.eval_batches * args.batch_size} sequences (all blocks merged):")
    print(f"{'site':<28} {'min':>10} {'max':>10} {'bits':>5}")
    for kind, (lo, hi) in kinds.items():
        print(f"{kind:<28} {lo:>10} {hi:>10} {signed_bits(lo, hi):>5}")
    with (run_dir / f"value_ranges_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump({name: {"min": lo, "max": hi, "bits": signed_bits(lo, hi)}
                   for name, (lo, hi) in ranges.ranges.items()}, f, indent=2)

    dump_dir = (args.dump_dir or run_dir / f"golden_{tag}").resolve()
    manifest = dump_golden(quantized_model, sequences[0][:1], dump_dir)
    print(f"\nWrote golden vectors for one sequence ({len(manifest['weights'])} weight tensors, "
          f"{len(manifest['trace'])} trace sites) to {dump_dir}")


if __name__ == "__main__":
    main()
