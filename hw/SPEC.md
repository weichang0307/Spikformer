# Spiking block accelerator: datapath specification

The bit-exact reference is `hw/reference.py`. If this document and the reference disagree, the reference wins:
fix the document. Parameters come from the QAT model `runs/folkrnn_256d6L_T8_qat_from_do05`; the integer tables
are exported to `hw/params_folkrnn_256d6L_T8_qat_from_do05.npz`.

**Accuracy.** Test score 1.4121 bits/token on the folk-rnn test split. This is the score of the datapath described
here, and the reference model reproduces it exactly. For comparison: the float model before QAT scores 1.457, the
folk-rnn LSTM 1.591 and the ANN teacher 1.220. Generated tunes: 85.1% bars of the right length, 99.2% valid
headers, 72.0% balanced repeats.

## 1. Scope and interface

The chip runs the 6 spiking blocks. The host runs the input stage and the read-out.

| | per token | format |
|---|---|---|
| **in** | input-LIF spikes `S[t][i]`, t = 0..7, i = 0..255 | 1 bit (2,048 bits per token) |
| **out** | residual stream after block 5, `R[t][i]` | 4-bit unsigned, 0..13 (8,192 bits = 1 KB per token) |

- **T = 8 time steps** per token. The steps of one token form one unit of work. LIF state does not carry from one
  token to the next.
- **Positions 0..255.** Generation stops at 256 tokens (the cap). There is no sliding window.
- **Per tune:** clear the K/V cache, then send tokens in order, position 0 first.
- **Host side** (float, outside this spec): embedding plus position, input L1 norm and input LIF, which produce S.
  The host's read-out is `logits = head(mean over t of R)`, followed by softmax and sampling.

## 2. Number formats

| Signal | Format |
|---|---|
| Weights `W` | int8, one implicit scale per tensor. The scale is never applied on chip; the norm removes it |
| Biases `B` | signed integers in accumulator units (range −76..342 here). Q, K and V have no bias |
| Residual stream `R` | 4-bit unsigned. Block k outputs at most 2k+3, at most 13, so it never overflows |
| Linear-layer input value | 0, 1, 2, 4 or 8: a shift amount for residual inputs, 0/1 for spike inputs |
| Accumulators `x` | 18-bit signed, saturating at −131072..131071 |
| Attention sums `A` | 16-bit unsigned, saturating at 65535 |
| Norm mean | integer, the width of the input |
| Norm MAD | 12-bit unsigned: integer 0..2047 for the linear-layer norms, half units 0..4095 for the head norm |
| Reciprocal table entry | 6-bit unsigned (33..62) |
| Norm output `n`, LIF state `u`, θ, β′ | **Q4.7**: 12-bit signed, value = int / 128, saturating at −2048..2047 |
| Spikes | 1 bit |

**Rounding** is always half up: `round(v) = floor(v + 1/2)`. For an integer shifted right by k bits:
`(x + 2^(k−1)) >> k`, using an arithmetic shift (−2.5 → −2).

**Saturation:** every register listed above clamps to its range. Nothing wraps around.

## 3. Parameters

All parameters are in the `.npz` file, named `b{block}.{name}.*`. The linear layers are q, k, v, proj, fc1 and fc2.
The norm and LIF pairs are q, k, v, attn, proj, fc1 and fc2.

| Array | Shape | Meaning |
|---|---|---|
| `b{b}.{lin}.w` | (out, in) int8 | weights: q, k, v and proj 256×256, fc1 1024×256, fc2 256×1024 |
| `b{b}.{lin}.b` | (out,) int | bias in accumulator units (zeros for q, k, v) |
| `b{b}.{n}.theta` | (width,) Q4.7 | θ = round(128 / γ), saturated |
| `b{b}.{n}.boff` | (width,) Q4.7 | β′ = round(128 β / γ), saturated |
| `b{b}.{n}.neg` | (width,) bool | γ < 0: the spike comparison is reversed |
| `b{b}.{n}.fixed` | (width,) bool | \|γ\| < 0.05: the neuron never fires. 40 neurons here: 29 in fc1, 10 in K, 1 in Q |
| `lut` | (16,) | reciprocal table: `round(64 / (1 + (i + 1/2) / 16))` = 62 59 55 53 50 48 46 44 42 40 39 37 36 35 34 33 |
| `shift_code` | (16,) | residual value → input value: `0 1 2 4 4 4 8 8 8 8 8 8 8 8 8 8` |

The widths are 256 for every norm and LIF except fc1, which is 1024. One θ and one β′ saturate.

## 4. Block datapath (per token; repeated for blocks 0..5)

`R_in[t][i]` is the block's input: S for block 0, otherwise the previous block's output. All 8 steps are computed.

### 4.1 Sparse linear layer `x = LIN(input, W, B)`
```
for every output o and step t:
    x[t][o] = sat18( B[o] + Σ_i W[o][i] · val[t][i] )
val = shift_code[R[t][i]]   for Q/K/V and fc1 (residual inputs): a product is W << {0,1,2,3}
val = spike[t][i]           for proj and fc2 (spike inputs): a product is W or 0
```
A weight column `W[:, i]` is needed only if `val[t][i] ≠ 0` for some t. Fetch it once and use it for all 8 steps.

**Accumulate wide, saturate once.** The sum may be computed in any order, but only if partial sums are not
saturated. Use a 21-bit signed accumulator, which cannot overflow: the largest possible |Σ| is
1024 × 127 × 8 = 1,040,384 < 2^20. Clamp to 18 bits once, after the bias is added. Saturating partial sums would
make the result depend on the summation order.

### 4.2 L1 norm `n = NORM(x)` (per step t, per vector of N = 256 or 1024, or per head group of N = 32)
```
mean = (Σ x + N/2) >> log2 N
c    = x − mean                                   (signed integer)
a    = Σ |c|
linear norms:  mad = min((a + N/2) >> log2 N, 2047);       d = max(mad, 1)
head norm:     mad = min((2a + N/2) >> log2 N, 4095);      d = mad + 1     (half units; +1 = ε of 0.5)
e    = position of the leading one of d           (d = 2^e … 2^(e+1) − 1)
idx  = the 4 bits after the leading one = ((d << 4) >> e) − 16
k    = e − 1   (linear norms)    or    e − 2   (head norm)
n    = sat12( round( c · lut[idx] · 2^−k ) )      (k ≤ 0: left shift; k > 0: (P + 2^(k−1)) >> k)
```
The only multiplier in the block is `c × lut[idx]`: c is up to 19 bits signed, the table entry is 6 bits.

### 4.3 Folded LIF `s = LIF(n)` (per neuron; steps t = 0..7 in order, u = 0 at t = 0 for every token)
```
u = sat12(u + n[t] + β′) >> 1                     (arithmetic shift: floor)
s[t] = fixed ? 0 : (neg ? u ≤ θ : u ≥ θ)
if s[t]: u = u − θ
```
This is exactly the trained LIF neuron (time constant 2, threshold 1, soft reset) applied to `γ·n + β`, in units
of γ. That is why γ and β never appear on chip.

### 4.4 The block
```
Q = LIF_q(NORM(LIN(R_in, Wq, 0)))       K = LIF_k(NORM(LIN(R_in, Wk, 0)))       V = LIF_v(NORM(LIN(R_in, Wv, 0)))
write K[t][·], V[t][·] of this position to the K/V cache (block b, step t)
for every step t and head h (8 heads × 32 channels, channel = 32h + j):
    for every cached position p = 0 .. current position (the current position included):
        cnt = popcount(Q[t][h] AND Kcache[p][t][h])                 0..32
        A[t][h][j] += cnt² · Vcache[p][t][h][j]   for j = 0..31     cnt² from a 33-entry table; skip cnt = 0
    A = sat16(A)
ATT  = LIF_attn(NORM_head(A))           (head norm: per head group of 32, half-unit MAD, ε = 1 half unit)
P    = LIF_proj(NORM(LIN(ATT, Wproj, Bproj)))
R_mid = R_in + P
Hs   = LIF_fc1(NORM(LIN(R_mid, Wfc1, Bfc1)))        (1,024 neurons)
M    = LIF_fc2(NORM(LIN(Hs, Wfc2, Bfc2)))
R_out = R_mid + M
```
Attention at step t uses only step-t spikes of Q, K and V. The 1/8 attention scale of the trained model is
absorbed into the head norm's ε.

## 5. Memory per block

| | Size |
|---|---|
| Weights | 786,432 × 8 bits; 4.72 MB for all 6 blocks |
| Biases | 1,792 integers per block (18 bits) |
| LIF constants | 2,560 neurons per block × (12 + 12 + 1 + 1) bits |
| K/V cache | 256 positions × 8 steps × 512 bits per block = 128 KB per block, 768 KB in total |
| Working state per token | R 8×256×4 bits; accumulators up to 8×1024×18 bits; u up to 1024×12 bits |

## 6. Verification

```
python3 hw/reference.py export folkrnn_256d6L_T8_qat_from_do05      # writes the .npz above
python3 hw/reference.py check  folkrnn_256d6L_T8_qat_from_do05      # bit-exactness check
python3 hw/reference.py vectors folkrnn_256d6L_T8_qat_from_do05 --tokens 32
```

The `check` command verifies three things:
1. The reference matches the PyTorch datapath simulation (`fixed_point_eval.py`, run in float64) on 100% of block
   outputs (8 test windows × 256 positions × 6 blocks × 8 steps).
2. Both give the same score.
3. Running token by token with the K/V cache gives exactly the same results as running a whole window at once.

`hw/vectors/` holds one file per token for the first 32 tokens of a tune, run in the chip's order (one token at a
time, with the cache). Each file has the position, the token, the input spikes, and per block b:

| array | meaning |
|---|---|
| `b.in` | residual in (8×256) |
| `b.q`, `b.k`, `b.v` | Q, K and V spikes |
| `b.attn_sum` | attention sums before the head norm |
| `b.attn_spk` | attention LIF spikes |
| `b.proj_spk` | projection LIF spikes |
| `b.mid` | residual after attention |
| `b.h` | fc1 spikes (8×1024) |
| `b.m` | fc2 spikes |
| `b.out` | residual out |

An RTL test bench feeds `input_spikes` token by token and compares every stage.

## 7. Not specified here (architecture choices)

- The number of parallel shift-add lanes, the memory banking, where the accumulators live, and the clock and
  target speed in tokens per second.
- The loop order inside a layer (input-stationary, output-stationary or step by step). Every order gives
  identical results, within these dependencies:
  1. A norm at step t needs all accumulators of step t, because it uses the whole vector's mean and MAD.
  2. Each LIF neuron processes steps 0..7 in order, because u carries from one step to the next. Neurons can run
     in parallel.
  3. Attention at step t needs Q at step t, plus K and V at step t for every cached position, including the
     current one.
  4. Fetching each weight column once for all 8 steps needs the layer's input for all 8 steps first. That is a
     choice: a step-by-step order removes the wait but reads the weights 8 times.
- The host-side formats of the input stage and the read-out.
