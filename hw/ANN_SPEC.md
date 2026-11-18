# int8 ANN block accelerator: datapath specification

This is the baseline to compare against the spiking chip (`hw/SPEC.md`). The bit-exact reference is
`hw/ann_reference.py`; if this document and the reference disagree, the reference wins. Parameters come from the
ANN teacher `runs/folkrnn_256d6L_ann_base` (`last.pt`, softmax attention, LayerNorm, GELU, 4.87M parameters),
quantized after training without fine-tuning (PTQ, calibrated on 64 training windows). The integer tables are in
`hw/ann_params_folkrnn_256d6L_ann_base.npz`.

**Accuracy.** Test score 1.2252 bits/token on the folk-rnn test split (256-token windows, stride 128), against
1.2196 for the float model (+0.006, with no fine-tuning). For comparison: the spiking chip scores 1.4121 and the folk-rnn LSTM 1.591.

## 1. Scope and interface

The chip runs the 6 transformer blocks, which is the same boundary as the spiking chip.

| | per token | format |
|---|---|---|
| **in** | residual stream after embedding and position, `x0[i]`, i = 0..255 | int16 at scale 2^−9 (4,096 bits) |
| **out** | residual stream after block 5, `x[i]` | int16 at scale 2^−9 (4,096 bits) |

- One pass per token. There are no time steps.
- **Positions 0..255.** Generation stops at 256 tokens. There is no sliding window.
- **Per tune:** clear the K/V cache, then send tokens in order, position 0 first.
- **Host side** (float, outside this spec):
  - Input: `x0 = round((embed[token] + pos[position]) · 2^9)`, saturated to int16.
  - Read-out: `logits = head(ln_f(x · 2^−9))`, followed by softmax and sampling.

## 2. Number formats

| Signal | Format |
|---|---|
| Weights `W` | int8 −127..127, one scale per output channel. LN γ is folded into the weights of the next layer |
| Biases `B` | int32 in accumulator units (LN β is folded in). In this model they fit in 16 bits |
| Residual stream `x` | int16, scale 2^−9 (range ±64; the largest value seen is 29) |
| LayerNorm output `n` | int8, scale 2^−frac (`frac` = 4 in block 0, 3 in blocks 1–5) |
| Activations q, k, v, attention output, fc1 output, GELU output | int8 −127..127, symmetric, one scale per tensor |
| Accumulators | int32 (26 bits are enough: 1024 × 127 × 127 plus the bias) |
| Requantization | per output channel: `M` 15-bit unsigned in [2^14, 2^15), shift `sh` (17..27 here) |
| Attention scores | 21-bit signed (\|q·k\| ≤ 32 × 127²) |
| Exponentials `e` | 8-bit unsigned 0..255. The row maximum always gets 255 |
| Exponential sums | 16-bit unsigned (≤ 256 × 255) |
| K/V cache | int8 |

**Rounding** is always half up: `(v + 2^(k−1)) >> k`, with an arithmetic shift.

**Saturation:** int8 values clamp to −127..127, the residual to −32768..32767, and accumulators to int32. Nothing
wraps. On the test windows no int8 value reached ±127, and the residual stayed below 15,000.

## 3. Parameters

All arrays are in the `.npz` file, named `b{block}.{name}`. The linear layers are q, k and v (256×256), proj
(256×256), fc1 (1024×256) and fc2 (256×1024).

| Array | Shape | Meaning |
|---|---|---|
| `b{b}.{lin}.w` | (out, in) int8 | weights; for q, k, v and fc1, already multiplied by the LN γ |
| `b{b}.{lin}.b` | (out,) int | bias in accumulator units; for q, k, v and fc1 it includes `W · β` |
| `b{b}.{lin}.M`, `.sh` | (out,) | requantization: `y = round(acc · M · 2^−sh)` |
| `b{b}.ln1.frac`, `.ln2.frac` | scalar | the LN output scale is 2^−frac |
| `b{b}.att.C`, `.sh` | scalar | softmax exponent constant: `t = round(d · C · 2^−sh)` |
| `b{b}.gelu` | (255,) int8 | GELU table, indexed by `h + 127` |
| `rsqrt` | (64,) | 1/√ table (see 4.2) |
| `exp` | (32,) | `round(255 · 2^(−f/32))`, f = 0..31: 255 250 244 … 133 130 |
| `res_frac` | scalar | 9: the residual scale is 2^−9 |
| `b{b}.scales` | (5,) float | documentation only: the real scales of q, k, v, fc1 and GELU |

## 4. Block datapath (per token; repeated for blocks 0..5)

### 4.1 Linear layer `y = LIN(a, W, B, M, sh)`
```
acc[o] = sat32( B[o] + Σ_i W[o][i] · a[i] )             int8 × int8 multiply-accumulate
y[o]   = sat( (acc[o] · M[o] + 2^(sh[o]−1)) >> sh[o] )   to int8, or to int16 for proj and fc2
```
For proj and fc2, `y` is already in residual units (scale 2^−9) and is added to the residual stream with
saturation. Every weight is read once per token.

### 4.2 LayerNorm `n = LN(x, frac)` (per vector of 256; γ and β are in the next layer)
```
mean = (Σ x + 128) >> 8
c    = x − mean                                   17-bit signed
var  = max((Σ c² + 128) >> 8, 1)                  up to 33 bits
e    = position of the leading one of var
odd  = e & 1;   ee = e − odd                      var = m · 2^ee with m in [1, 4)
idx  = 32 · odd + (the 5 bits after the leading one)
r    = rsqrt[idx]                                 9-bit unsigned (129..254), ≈ 2^8 / √m
k    = 8 + ee/2 − frac
n    = sat8( round(c · r · 2^−k) )                (a left shift if k < 0)
```
`rsqrt[i] = round(256 / √(1 + (i + 1/2)/32))` for i = 0..31, and `round(256 / √(2 (1 + (i − 32 + 1/2)/32)))` for
i = 32..63. The LN needs one 17 × 9 multiply per element, plus c² for the variance.

### 4.3 Softmax attention (per head h of 32 channels, for the query at the current position)
```
s[p] = Σ_j q[32h+j] · k_cache[p][32h+j]          for every cached position p ≤ current, the current one included
d[p] = max_p s − s[p]                             ≥ 0
t[p] = (d[p] · C + 2^(sh−1)) >> sh                exponent in 1/32 units of log2 (C already holds the scales,
                                                  1/√32 and log2 e)
e[p] = t >> 5 ≥ 9 ? 0 : round(exp[t & 31] · 2^−(t >> 5))       8-bit
S    = Σ_p e[p]                                   ≥ 255
acc[j] = Σ_p e[p] · v_cache[p][32h+j]             8-bit unsigned × int8 multiply-accumulate
r    = (2^24 + S/2) div S                         one integer division per head and token
o[32h+j] = sat8( round(acc[j] · r · 2^−24) )
```
Because the softmax weights sum to one, `o` is a convex mix of the values, so it keeps V's scale and cannot
overflow. That is why no extra rescaling multiply is needed here.

### 4.4 The block
```
n1 = LN(x_in, ln1.frac)
q, k, v = LIN(n1, Wq ...), LIN(n1, Wk ...), LIN(n1, Wv ...)          int8
write k, v of this position to the K/V cache (block b)
o  = ATTENTION(q, k_cache, v_cache)                                   int8, V's scale
x_mid = sat16(x_in + LIN(o, Wproj ...))                               proj requantizes to residual units
n2 = LN(x_mid, ln2.frac)
h  = LIN(n2, Wfc1 ...)                                                int8, 1,024 values
g  = gelu[h + 127]                                                    int8 table lookup
x_out = sat16(x_mid + LIN(g, Wfc2 ...))
```

## 5. Arithmetic and memory per block

| | per token at position p |
|---|---|
| int8 × int8 MACs in the linear layers | 786,432 |
| Attention MACs (q·k, then e·v) | 2 × 256 × (p + 1) |
| Requantization multiplies (26 × 15 bits) | 2,304 |
| LayerNorm multiplies (17 × 9 bits) and squares | 512 each |
| Softmax constant multiplies (21 × 15 bits) | 8 × (p + 1) |
| Attention normalization multiplies (24 × 17 bits) and divisions | 256 multiplies, 8 divisions |

| Memory | Size |
|---|---|
| Weights | 786,432 bytes; 4.72 MB for all 6 blocks (the same as the spiking chip) |
| Biases and requantization constants | 2,304 output channels per block × (32 + 15 + 5) bits |
| Tables | 64 × 9 bits (1/√), 32 × 8 bits (exp), 255 × 8 bits GELU per block |
| K/V cache | 256 positions × 512 bytes = 128 KB per block, 768 KB in total (the spiking chip also needs 768 KB: 8 steps × 1 bit instead of 8 bits) |

Unlike the spiking chip, every multiplier here is a real multiplier. The int8 MACs are the bulk of them (99% of
the multiplies), and the energy model in the report counts them at 0.2 pJ each.

## 6. Verification

```
python3 hw/ann_reference.py export  folkrnn_256d6L_ann_base     # calibrate and write the .npz
python3 hw/ann_reference.py check   folkrnn_256d6L_ann_base     # exactness checks
python3 hw/ann_reference.py score   folkrnn_256d6L_ann_base     # full test split, about 5 minutes on the CPU
python3 hw/ann_reference.py vectors folkrnn_256d6L_ann_base --tokens 32
```

The `check` command verifies three things:
1. The fast path (float64 BLAS matrix products, exact for integers below 2^53) gives exactly the same results as
   pure int64 arithmetic, at every stage of every block.
2. Running token by token with the K/V cache gives exactly the same results as running a whole window at once.
3. It reports how often each int8 tensor saturates (0% here).

Unlike the spiking chip, there is no second, independent implementation to compare against: the PyTorch
simulation came first for the spiking chip, while this model was written directly in integers. The test score is
the check that the datapath is right.

`hw/ann_vectors/` holds one file per token for the first 32 tokens of a test tune, in the chip's order (one token
at a time, with the cache). Each file has `position`, `token` and `x0`, and per block b: `b.in`, `b.ln1`,
`b.q`/`b.k`/`b.v`, `b.attn` (attention output), `b.proj`, `b.mid`, `b.ln2`, `b.fc1`, `b.gelu`, `b.fc2` and `b.out`.

## 7. Not specified here (architecture choices)

- The size of the MAC array, the memory banking, and the clock.
- The loop order inside a layer. A LayerNorm needs its whole 256-value vector first (for the mean and variance).
  The softmax needs every score of a head row first (for the maximum); a design that wants to stream can use the
  online-softmax rescaling instead, but that is not bit-exact with this reference.
- The host-side formats.
