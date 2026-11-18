"""Bit-exact integer reference model of the int8 ANN transformer blocks (W8A8), for RTL verification.
See hw/ANN_SPEC.md. The model is the ANN teacher runs/folkrnn_256d6L_ann_base (softmax attention, LayerNorm, GELU).

Everything after export is NumPy integer arithmetic. The chip boundary is the same as for the spiking chip:
    in   per token, the residual stream after embedding + position, x0[256], int16 (host quantizes)
    out  per token, the residual stream after the last block, x[256], int16 (host runs ln_f and the head)
Inside, per token and block: int8 x int8 linear layers with int32 accumulators and per-channel requantization
(15-bit multiplier + shift), an integer LayerNorm (mean, variance, 64-entry 1/sqrt table, one multiply), LN gamma and
beta folded into the next linear layer, softmax with an exp2 table on 8-bit exponentials and one division per head
row, a 256-entry GELU table, an int16 residual stream and an int8 K/V cache. Rounding is half up; all registers
saturate.

    python3 hw/ann_reference.py export RUN            # calibrate + quantize -> hw/ann_params_<RUN>.npz
    python3 hw/ann_reference.py check RUN             # int64 vs fast path, cache mode, score on a few windows
    python3 hw/ann_reference.py score RUN             # full test split (256-token windows, stride 128)
    python3 hw/ann_reference.py vectors RUN [--tokens 32]   # -> hw/ann_vectors/
"""

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

D, H, HD, DH, NB, CTX = 256, 8, 32, 1024, 6, 256
A_MAX = 127                                  # int8 activations and weights, symmetric (-127..127)
R_MIN, R_MAX = -(1 << 15), (1 << 15) - 1     # int16 residual
ACC_MIN, ACC_MAX = -(1 << 31), (1 << 31) - 1  # int32 accumulators
M_BITS = 15                                  # requantization multiplier: 15-bit unsigned, in [2^14, 2^15)
RSQRT_IDX = 5                                # 1/sqrt table: parity bit + 5 mantissa bits = 64 entries
RSQRT_FRAC = 8                               # table entries: round(2^8 / sqrt(m)), m in [1, 4) -> 128..256
EXP_FRAC = 5                                 # softmax exponent t in units of 2^-5 (log2 units)
EXP_MAX = 255                                # 8-bit unsigned exponentials
RECIP_FRAC = 24                              # attention normaliser: r = round(2^24 / sum e)
LINEARS = ("q", "k", "v", "proj", "fc1", "fc2")
CALIB_WINDOWS = 64


def rhu(x):
    """Round half up for float export values."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


def shr_round(x, k):
    """x * 2^-k rounded half up (arithmetic shift), integer x, integer or per-element k >= 0."""
    k = np.asarray(k, dtype=np.int64)
    half = np.where(k > 0, np.left_shift(np.int64(1), np.maximum(k - 1, 0)), 0)
    return (x + half) >> k


def leading_one(d):
    """floor(log2 d) for integer d >= 1, exact."""
    return (np.frexp(np.asarray(d, dtype=np.float64))[1] - 1).astype(np.int64)


def multiplier(m):
    """Real m > 0 -> (M, sh) with M in [2^14, 2^15) and M * 2^-sh ~ m."""
    m = np.asarray(m, dtype=np.float64)
    sh = (M_BITS - 1) - np.floor(np.log2(m)).astype(np.int64)
    M = rhu(m * 2.0 ** sh)
    over = M >= (1 << M_BITS)
    M, sh = np.where(over, M >> 1, M), np.where(over, sh - 1, sh)
    assert (sh >= 0).all() and (sh <= 62).all()
    return M, sh


def rsqrt_lut():
    """64 entries: index = (exponent parity, 5 bits after the leading one); m = mantissa in [1,2) or [2,4)."""
    i = np.arange(1 << RSQRT_IDX)
    m_even = 1 + (i + 0.5) / (1 << RSQRT_IDX)
    return np.concatenate([rhu(2 ** RSQRT_FRAC / np.sqrt(m_even)), rhu(2 ** RSQRT_FRAC / np.sqrt(2 * m_even))])


def exp_lut():
    """32 entries: round(255 * 2^(-f / 32)), f = fractional part of the exponent."""
    return rhu(EXP_MAX * 2.0 ** (-np.arange(1 << EXP_FRAC) / (1 << EXP_FRAC)))


RSQRT = rsqrt_lut()
EXP = exp_lut()


# ---------------------------------------------------------------------------------------------- export

def float_model(run):
    import torch
    from models import model_from_config
    cfg = json.load(open(os.path.join(ROOT, "runs", run, "config.json")))
    sd = torch.load(os.path.join(ROOT, "runs", run, "last.pt"), map_location="cpu", weights_only=False)
    m = model_from_config(cfg, cfg["block_size"])
    m.load_state_dict(sd.get("model", sd))
    return m.double().eval(), cfg


def dataset(cfg):
    from char_data import CharDataset
    return CharDataset(os.path.join(ROOT, cfg["data_path"]), cfg["val_fraction"], cfg["test_fraction"])


def calibrate(m, data):
    """Max |value| of every chip tensor of the float model on CALIB_WINDOWS random training windows."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(1)
    ids = data.splits["train"].long()
    starts = torch.randint(0, len(ids) - CTX, (CALIB_WINDOWS,))
    x = torch.stack([ids[i:i + CTX] for i in starts])
    st = {}

    def upd(k, v):
        st[k] = max(st.get(k, 0.0), float(v.abs().max()))

    with torch.no_grad():
        h = m.embed(x) + m.pos_embed[:, :CTX]
        upd("res", h)
        for b, blk in enumerate(m.blocks):
            n1 = F.layer_norm(h, (D,), eps=blk.ln1.eps)
            upd(f"b{b}.ln1", n1)
            y = blk.ln1(h)
            a = blk.attn
            for l, lin in (("q", a.q_linear), ("k", a.k_linear), ("v", a.v_linear)):
                upd(f"b{b}.{l}", lin(y))
            h = h + a(y)
            upd("res", h)
            upd(f"b{b}.ln2", F.layer_norm(h, (D,), eps=blk.ln2.eps))
            f1 = blk.mlp.fc1(blk.ln2(h))
            upd(f"b{b}.fc1", f1)
            upd(f"b{b}.gelu", F.gelu(f1))
            h = h + blk.mlp.fc2(F.gelu(f1))
            upd("res", h)
    return st


def export(run, out=None):
    import torch.nn.functional as F
    import torch
    m, cfg = float_model(run)
    data = dataset(cfg)
    st = calibrate(m, data)
    p = {}
    # residual: power-of-two scale with 2x headroom over the calibration maximum
    fr = int(math.floor(math.log2(R_MAX / (2 * st["res"]))))
    p["res_frac"] = np.array(fr)
    s_r = 2.0 ** -fr
    for b, blk in enumerate(m.blocks):
        a = blk.attn
        g1, be1 = blk.ln1.weight.detach().numpy(), blk.ln1.bias.detach().numpy()
        g2, be2 = blk.ln2.weight.detach().numpy(), blk.ln2.bias.detach().numpy()
        # LayerNorm outputs: power-of-two scales (the LN produces n = c / sigma directly at 2^-frac)
        fl = {}
        for n in ("ln1", "ln2"):
            fl[n] = int(math.floor(math.log2(A_MAX / st[f"b{b}.{n}"])))
            p[f"b{b}.{n}.frac"] = np.array(fl[n])
        s_in = {"q": 2.0 ** -fl["ln1"], "k": 2.0 ** -fl["ln1"], "v": 2.0 ** -fl["ln1"], "fc1": 2.0 ** -fl["ln2"]}
        s_out = {l: st[f"b{b}.{l}"] / A_MAX for l in ("q", "k", "v", "fc1")}
        s_g = st[f"b{b}.gelu"] / A_MAX
        s_in["proj"], s_in["fc2"] = s_out["v"], s_g               # attention output has V's scale (convex mix)
        s_out["proj"] = s_out["fc2"] = s_r
        lins = {"q": a.q_linear, "k": a.k_linear, "v": a.v_linear, "proj": a.proj_linear,
                "fc1": blk.mlp.fc1, "fc2": blk.mlp.fc2}
        for l, lin in lins.items():
            w = lin.weight.detach().numpy().copy()
            bias = lin.bias.detach().numpy().copy()
            if l in ("q", "k", "v"):                                  # fold LN gamma / beta
                bias, w = bias + w @ be1, w * g1[None, :]
            elif l == "fc1":
                bias, w = bias + w @ be2, w * g2[None, :]
            sw = np.maximum(np.abs(w).max(1), 1e-12) / A_MAX            # per output channel
            p[f"b{b}.{l}.w"] = rhu(w / sw[:, None]).astype(np.int8)
            p[f"b{b}.{l}.b"] = np.clip(rhu(bias / (s_in[l] * sw)), ACC_MIN, ACC_MAX)
            M, sh = multiplier(s_in[l] * sw / s_out[l])
            p[f"b{b}.{l}.M"], p[f"b{b}.{l}.sh"] = M, sh
        # softmax exponent: t = d * s_q * s_k / sqrt(32) * log2(e), in units of 2^-EXP_FRAC
        C, csh = multiplier(s_out["q"] * s_out["k"] / math.sqrt(HD) * math.log2(math.e) * 2 ** EXP_FRAC)
        p[f"b{b}.att.C"], p[f"b{b}.att.sh"] = np.array(C), np.array(csh)
        # GELU table: int8 fc1 output (scale s_fc1) -> int8 (scale s_g)
        hq = np.arange(-A_MAX, A_MAX + 1) * s_out["fc1"]
        gl = F.gelu(torch.tensor(hq)).numpy()
        p[f"b{b}.gelu"] = np.clip(rhu(gl / s_g), -A_MAX, A_MAX).astype(np.int8)   # index = h + 127
        p[f"b{b}.scales"] = np.array([s_out["q"], s_out["k"], s_out["v"], s_out["fc1"], s_g])  # documentation
    p["rsqrt"], p["exp"] = RSQRT, EXP
    out = out or os.path.join(ROOT, "hw", f"ann_params_{run}.npz")
    np.savez_compressed(out, **p)
    print(f"exported {run} -> {out}; residual Q.{fr} (max |x| {st['res']:.2f}); LN fracs "
          + " ".join(f"{int(p[f'b{b}.ln1.frac'])}/{int(p[f'b{b}.ln2.frac'])}" for b in range(NB)))
    return out


# ---------------------------------------------------------------------------------------------- the datapath

EXACT = False      # True: int64 matmuls (slow); False: float64 BLAS matmuls, exact while |sums| < 2^53


def imatmul(a, b):
    if EXACT:
        return np.matmul(a.astype(np.int64), b.astype(np.int64))
    return np.rint(np.matmul(a.astype(np.float64), b.astype(np.float64))).astype(np.int64)


def requant(acc, M, sh, lo, hi):
    return np.clip(shr_round(acc * M, sh), lo, hi)


def linear(x, p, b, l, lo=-A_MAX, hi=A_MAX):
    """x: (..., in) int8 values; returns the requantized output (int8, or int16 for proj / fc2)."""
    acc = np.clip(imatmul(x, p[f"b{b}.{l}.w"].T) + p[f"b{b}.{l}.b"], ACC_MIN, ACC_MAX)
    return requant(acc, p[f"b{b}.{l}.M"], p[f"b{b}.{l}.sh"], lo, hi)


def layernorm(x, frac):
    """x: (..., 256) int16 residual -> n = (x - mean) / sigma as int8 at scale 2^-frac (gamma, beta folded away)."""
    mean = (x.sum(-1, keepdims=True) + D // 2) >> 8
    c = x - mean                                                    # 17-bit signed
    var = np.maximum((np.square(c).sum(-1, keepdims=True) + D // 2) >> 8, 1)   # up to 34 bits
    e = leading_one(var)
    odd = e & 1
    ee = e - odd                                                    # even exponent; mantissa var / 2^ee in [1, 4)
    idx = (((var << RSQRT_IDX) >> e) - (1 << RSQRT_IDX)) + (odd << RSQRT_IDX)   # parity, 5 bits after leading 1
    r = RSQRT[idx]
    k = RSQRT_FRAC + (ee >> 1) - frac                               # n = c * r * 2^-(8 + ee/2) * 2^frac
    n = np.where(k >= 0, shr_round(c * r, np.maximum(k, 0)), (c * r) << np.maximum(-k, 0))
    return np.clip(n, -A_MAX, A_MAX)


def attention(q, k, v, C, csh, causal_from=0):
    """q: (Lq, 256) int8; k, v: (Lk, 256) int8 (the cache, all positions so far). Returns (Lq, 256) int8 at V's
    scale, and the per-row exponent sums."""
    Lq, Lk = q.shape[0], k.shape[0]
    qh, kh, vh = (z.reshape(-1, H, HD).transpose(1, 0, 2) for z in (q, k, v))
    s = imatmul(qh, kh.transpose(0, 2, 1))                          # (H, Lq, Lk), 21-bit signed
    visible = np.arange(Lk)[None, :] <= (np.arange(Lq)[:, None] + causal_from)
    mx = np.where(visible, s, np.iinfo(np.int64).min).max(-1, keepdims=True)
    d = np.where(visible, mx - s, 0)                                # >= 0
    t = shr_round(d * C, csh)                                       # exponent in 2^-5 units of log2
    ti, tf = t >> EXP_FRAC, t & ((1 << EXP_FRAC) - 1)
    e = np.where(ti < 9, shr_round(EXP[tf], np.minimum(ti, 8)), 0) * visible   # 8-bit unsigned
    se = e.sum(-1, keepdims=True)                                   # 16 bits; >= 255 (the max element)
    acc = imatmul(e, vh)                                            # (H, Lq, 32), 24-bit signed
    r = (np.int64(1 << RECIP_FRAC) + se // 2) // se                 # one division per head row
    o = np.clip(shr_round(acc * r, RECIP_FRAC), -A_MAX, A_MAX)
    return o.transpose(1, 0, 2).reshape(Lq, D)


def block(x, p, b, cache=None, trace=None):
    """x: (L, 256) int16 residual for L new positions; cache: dict 'k', 'v' (Lpast, 256) int8 or None."""
    n1 = layernorm(x, int(p[f"b{b}.ln1.frac"]))
    q, k, v = (linear(n1, p, b, l) for l in ("q", "k", "v"))
    past = 0
    if cache is not None and cache.get("k") is not None:
        past = cache["k"].shape[0]
        k_all, v_all = np.concatenate([cache["k"], k]), np.concatenate([cache["v"], v])
    else:
        k_all, v_all = k, v
    o = attention(q, k_all, v_all, int(p[f"b{b}.att.C"]), int(p[f"b{b}.att.sh"]), past)
    pr = linear(o, p, b, "proj", R_MIN, R_MAX)
    xm = np.clip(x + pr, R_MIN, R_MAX)
    n2 = layernorm(xm, int(p[f"b{b}.ln2.frac"]))
    h = linear(n2, p, b, "fc1")
    g = p[f"b{b}.gelu"].astype(np.int64)[h + A_MAX]
    f2 = linear(g, p, b, "fc2", R_MIN, R_MAX)
    xo = np.clip(xm + f2, R_MIN, R_MAX)
    if trace is not None:
        trace.update({f"b{b}.in": x, f"b{b}.ln1": n1, f"b{b}.q": q, f"b{b}.k": k, f"b{b}.v": v,
                      f"b{b}.attn": o, f"b{b}.proj": pr, f"b{b}.mid": xm, f"b{b}.ln2": n2, f"b{b}.fc1": h,
                      f"b{b}.gelu": g, f"b{b}.fc2": f2, f"b{b}.out": xo})
    return xo, {"k": k_all, "v": v_all}


def chip(x0, p, caches=None, trace=None):
    caches = caches if caches is not None else [None] * NB
    x, new = x0.astype(np.int64), []
    for b in range(NB):
        x, c = block(x, p, b, caches[b], trace)
        new.append(c)
    return x, new


def load(path):
    return dict(np.load(path))


# ---------------------------------------------------------------------------------------------- host side (float)

def host_input(m, tokens, p):
    import torch
    with torch.no_grad():
        x0 = (m.embed(torch.tensor(tokens)) + m.pos_embed[0, :len(tokens)]).numpy()
    return np.clip(rhu(x0 * 2.0 ** int(p["res_frac"])), R_MIN, R_MAX)


def host_logprobs(m, x, p):
    import torch
    with torch.no_grad():
        h = torch.tensor(x * 2.0 ** -int(p["res_frac"]))
        return torch.log_softmax(m.head(m.ln_f(h)), -1).numpy()


def params(run):
    path = os.path.join(ROOT, "hw", f"ann_params_{run}.npz")
    if not os.path.exists(path):
        export(run)
    return load(path)


def score(run, max_windows=None, verbose=True):
    """Test bits/token, 256-token windows with stride 128 (as eval_char.py / eval_fixed.py)."""
    p = params(run)
    m, cfg = float_model(run)
    ids = dataset(cfg).splits["test"].long().numpy()
    starts = list(range(0, len(ids) - CTX, 128))[:max_windows]
    tot, n = 0.0, 0
    for w, i in enumerate(starts):
        tok = ids[i:i + CTX + 1]
        x, _ = chip(host_input(m, tok[:-1], p), p)
        lp = host_logprobs(m, x, p)
        lo = 0 if i == 0 else CTX - 128
        tot += -lp[np.arange(lo, CTX), tok[1 + lo:]].sum()
        n += CTX - lo
        if verbose and (w + 1) % 200 == 0:
            print(f"  {w + 1}/{len(starts)} windows: {tot / n / math.log(2):.4f}", flush=True)
    bits = tot / n / math.log(2)
    print(f"test bits/token (int8 datapath, {len(starts)} windows, {n} tokens): {bits:.4f}")
    return bits


def check(run, windows=2):
    global EXACT
    p = params(run)
    m, cfg = float_model(run)
    ids = dataset(cfg).splits["test"].long().numpy()
    # 1. int64 matmuls == float64 matmuls (the fast path is exact)
    same = True
    for i in (0, 128 * 37)[:windows]:
        x0 = host_input(m, ids[i:i + CTX], p)
        EXACT = True; t1 = {}; chip(x0, p, trace=t1)
        EXACT = False; t2 = {}; chip(x0, p, trace=t2)
        same &= all(np.array_equal(t1[k], t2[k]) for k in t1)
    print(f"int64 vs float64 matmuls, {windows} windows, every stage of every block: "
          f"{'identical' if same else 'DIFFERENT'}")
    # 2. token by token with the K/V cache == whole window
    tok = ids[:64]
    x0 = host_input(m, tok, p)
    whole, _ = chip(x0, p)
    caches, steps = None, []
    for t in range(len(tok)):
        r, caches = chip(x0[t:t + 1], p, caches)
        steps.append(r)
    print(f"token-by-token with the K/V cache vs whole window (64 tokens): "
          f"{'identical' if np.array_equal(np.concatenate(steps), whole) else 'DIFFERENT'}")
    # 3. saturation counts on a window
    tr = {}
    chip(host_input(m, ids[:CTX], p), p, trace=tr)
    for key in ("q", "k", "v", "attn", "fc1", "gelu"):
        sat = np.mean([np.mean(np.abs(tr[f"b{b}.{key}"]) >= A_MAX) for b in range(NB)])
        print(f"  {key:5s} values at +-127: {100 * sat:.3f}%")
    print(f"  residual max |x|: {max(np.abs(tr[f'b{b}.out']).max() for b in range(NB))} of {R_MAX}")


def vectors(run, tokens):
    p = params(run)
    m, cfg = float_model(run)
    data = dataset(cfg)
    text = data.splits["test"].long().numpy()
    nl = data.chars.index("\n")
    start = int(np.flatnonzero(text == nl)[0])
    tok = text[start:start + tokens].tolist()
    x0 = host_input(m, tok, p)
    outdir = os.path.join(ROOT, "hw", "ann_vectors")
    os.makedirs(outdir, exist_ok=True)
    caches = None
    for t in range(len(tok)):
        tr = {}
        _, caches = chip(x0[t:t + 1], p, caches, tr)
        np.savez_compressed(os.path.join(outdir, f"token_{t:03d}.npz"), position=t, token=tok[t], x0=x0[t],
                            **{k: v[0] for k, v in tr.items()})
    json.dump({"run": run, "tokens": tok, "params": f"ann_params_{run}.npz",
               "arrays": "per block b: b.in int16 residual in, b.ln1 int8, b.q/k/v int8, b.attn int8 attention "
                         "output, b.proj int16, b.mid int16 residual, b.ln2 int8, b.fc1 int8, b.gelu int8, "
                         "b.fc2 int16, b.out int16 residual out"},
              open(os.path.join(outdir, "index.json"), "w"), indent=1)
    print(f"wrote {len(tok)} token vectors to {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("export", "check", "score", "vectors"))
    ap.add_argument("run")
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.run)
    elif a.cmd == "check":
        check(a.run)
    elif a.cmd == "score":
        score(a.run, a.windows)
    else:
        vectors(a.run, a.tokens)


if __name__ == "__main__":
    main()
