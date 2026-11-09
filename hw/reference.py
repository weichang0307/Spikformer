"""Bit-exact integer reference model of the spiking blocks (the chip), for RTL verification. See hw/SPEC.md.

Everything here is NumPy int64 arithmetic: no floating point after export. The chip boundary:
    in   per token, the input-LIF spikes S[T][256] (0/1), computed by the host
    out  per token, the residual stream after the last block R[T][256] (0..13), read out by the host
Inside, per token and block (T = 8 time steps): sparse shift-add linear layers, integer L1 norms with a
16-entry reciprocal table and one multiply, gamma / beta folded into Q4.7 LIF neurons, integer spiking attention
over a K/V spike cache. Rounding is half up (floor(x + 1/2)); every register saturates.

    python3 hw/reference.py export RUN              # -> hw/params_<RUN>.npz (all integer tables)
    python3 hw/reference.py check RUN [--windows 4]  # bit-exactness vs fixed_point_eval.py, cache mode, test bits
    python3 hw/reference.py vectors RUN [--tokens 32] # -> hw/vectors/ (per-stage dumps for RTL test benches)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

T, D, H, HD, DH, NB, CTX = 8, 256, 8, 32, 1024, 6, 256
FRAC = 7                                   # Q4.7: 12-bit signed, value = int / 128
Q_MIN, Q_MAX = -2048, 2047
ACC_MIN, ACC_MAX = -(1 << 17), (1 << 17) - 1   # 18-bit signed accumulators
ATT_MAX = (1 << 16) - 1                    # 16-bit unsigned attention sums
GAMMA_MIN = 0.05                           # |gamma| below this: the neuron never fires
SHIFT_CODE = np.array([0, 1, 2, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=np.int64)  # residual 0..15 -> input value
LINEARS = ("q", "k", "v", "proj", "fc1", "fc2")
NORMS = ("q", "k", "v", "attn", "proj", "fc1", "fc2")   # each followed by its LIF population


def rhu(x):
    """Round half up for float export values (floor(x + 1/2))."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)


def shr_round(x, k):
    """x * 2^-k rounded half up, for integer arrays and an integer (or per-element) k; k <= 0 shifts left."""
    k = np.asarray(k, dtype=np.int64)
    pos = np.maximum(k, 0)
    up = (x + np.where(pos > 0, np.left_shift(np.int64(1), np.maximum(pos - 1, 0)), 0)) >> pos
    return np.where(k > 0, up, x << np.maximum(-k, 0))


def reciprocal_lut():
    """16 entries: round_half_up(64 / (1 + (i + 1/2) / 16)), i = the 4 bits after the leading one (values 33..62)."""
    i = np.arange(16)
    return rhu(64.0 / (1 + (i + 0.5) / 16))


LUT = reciprocal_lut()


# ---------------------------------------------------------------------------------------------- export

def export(run: str, out: str = None) -> str:
    import torch
    cfg = json.load(open(os.path.join(ROOT, "runs", run, "config.json")))
    sd = torch.load(os.path.join(ROOT, "runs", run, "best.pt"), map_location="cpu", weights_only=False)
    sd = {k: v.double().numpy() for k, v in sd.get("model", sd).items()}
    assert cfg["T"] == T and cfg["dim"] == D and cfg["depth"] == NB and cfg["num_heads"] == H
    p, fixed_total = {}, 0
    names = {"q": "attn.q", "k": "attn.k", "v": "attn.v", "proj": "attn.proj", "fc1": "mlp.fc1", "fc2": "mlp.fc2"}
    norms = {"q": "attn.q_norm", "k": "attn.k_norm", "v": "attn.v_norm", "attn": "attn.attn_norm",
             "proj": "attn.proj_norm", "fc1": "mlp.norm1", "fc2": "mlp.norm2"}
    for b in range(NB):
        for l in LINEARS:
            key = f"blocks.{b}.{names[l]}" + ("_linear" if l in ("q", "k", "v", "proj") else "")
            w = sd[key + ".weight"]
            s = np.abs(w).max() / 127
            p[f"b{b}.{l}.w"] = rhu(w / s).astype(np.int8)                      # int8, one scale per tensor
            bias = sd.get(key + ".bias")
            p[f"b{b}.{l}.b"] = rhu(bias / s) if bias is not None else np.zeros(w.shape[0], np.int64)
            p[f"b{b}.{l}.scale"] = np.array(s)                                   # host only (not used on chip)
        for n in NORMS:
            g, be = sd[f"blocks.{b}.{norms[n]}.weight"], sd[f"blocks.{b}.{norms[n]}.bias"]
            fixed = np.abs(g) < GAMMA_MIN
            gs = np.where(fixed, 1.0, g)
            p[f"b{b}.{n}.theta"] = np.clip(rhu(2 ** FRAC / gs), Q_MIN, Q_MAX)    # 1/gamma in Q4.7
            p[f"b{b}.{n}.boff"] = np.clip(rhu(2 ** FRAC * be / gs), Q_MIN, Q_MAX)  # beta/gamma in Q4.7
            p[f"b{b}.{n}.neg"] = (gs < 0)
            p[f"b{b}.{n}.fixed"] = fixed
            fixed_total += int(fixed.sum())
    p["lut"] = LUT
    p["shift_code"] = SHIFT_CODE
    out = out or os.path.join(ROOT, "hw", f"params_{run}.npz")
    np.savez_compressed(out, **p)
    print(f"exported {run} -> {out}: {NB} blocks, {fixed_total} fixed neurons, LUT {LUT.tolist()}")
    return out


# ---------------------------------------------------------------------------------------------- the datapath

def linear(code, w, b):
    """code: (..., fan_in) input values (shift codes 0/1/2/4/8 or spikes 0/1); w: (fan_out, fan_in) int8."""
    acc = code @ w.astype(np.int64).T + b
    return np.clip(acc, ACC_MIN, ACC_MAX)


def l1norm(x, head: bool):
    """x: (..., N) integers (accumulators, or attention sums for the head norm with N = 32). Returns n in Q4.7."""
    N = x.shape[-1]
    lg = N.bit_length() - 1
    mean = (x.sum(-1, keepdims=True) + N // 2) >> lg
    c = x - mean
    a = np.abs(c).sum(-1, keepdims=True)
    if head:                                    # MAD in half units (11.1), plus eps = 0.5 = 1 half unit
        mad = np.minimum((2 * a + N // 2) >> lg, 4095)
        d = mad + 1
    else:                                       # MAD integer (11.0); eps is far below one unit here
        mad = np.minimum((a + N // 2) >> lg, 2047)
        d = np.maximum(mad, 1)
    e = (np.frexp(d.astype(np.float64))[1] - 1).astype(np.int64)   # leading-one position, exact
    idx = ((d << 4) >> e) - 16                  # the 4 bits after the leading one
    r = LUT[idx]                                # 6-bit table entry, 1/mantissa in 1/64 units
    k = e - (2 if head else 1)                  # n * 128 = c * r * 2^-(e-1)   (head: one more, half units)
    n = shr_round(c * r, k)
    return np.clip(n, Q_MIN, Q_MAX)


def lif(n_steps, theta, boff, neg, fixed):
    """n_steps: (T, ..., width) Q4.7 inputs. Folded LIF: u <- sat(u + n + boff) >> 1; spike if u >= theta
    (u <= theta when gamma < 0); u -= theta on a spike. Fixed neurons never fire."""
    u = np.zeros(n_steps.shape[1:], np.int64)
    out = np.zeros(n_steps.shape, np.int64)
    for t in range(n_steps.shape[0]):
        u = np.clip(u + n_steps[t] + boff, Q_MIN, Q_MAX) >> 1
        s = np.where(neg, u <= theta, u >= theta) & ~fixed
        u = u - s * theta
        out[t] = s
    return out


def norm_lif(acc, p, b, name, head=False):
    """acc: (T, L, width) accumulators -> spikes (T, L, width)."""
    if head:
        n = l1norm(acc.reshape(*acc.shape[:-1], H, HD), True).reshape(acc.shape)
    else:
        n = l1norm(acc, False)
    return lif(n, p[f"b{b}.{name}.theta"], p[f"b{b}.{name}.boff"], p[f"b{b}.{name}.neg"], p[f"b{b}.{name}.fixed"])


def attention(q, k, v, causal_from=0):
    """q: (T, Lq, D) spikes of the queries at absolute positions causal_from.. ; k, v: (T, Lk, D) spikes of every
    position 0..Lk-1 (the cache). Per step and head: count = popcount(q & k), sum over visible keys of
    count^2 * v, saturated to 16 bits."""
    Tq, Lq, _ = q.shape
    Lk = k.shape[1]
    qh = q.reshape(Tq, Lq, H, HD).transpose(0, 2, 1, 3)
    kh = k.reshape(Tq, Lk, H, HD).transpose(0, 2, 1, 3)
    vh = v.reshape(Tq, Lk, H, HD).transpose(0, 2, 1, 3)
    cnt = qh @ kh.transpose(0, 1, 3, 2)                                  # (T, H, Lq, Lk)
    pos_q = np.arange(Lq)[:, None] + causal_from
    visible = np.arange(Lk)[None, :] <= pos_q
    sq = cnt * cnt * visible
    a = np.minimum(sq @ vh, ATT_MAX)                                     # (T, H, Lq, HD)
    return a.transpose(0, 2, 1, 3).reshape(Tq, Lq, D)


def block(r_in, p, b, cache=None, trace=None):
    """One spiking block. r_in: (T, L, 256) residual (0..13) for L new positions; cache: dict with 'k', 'v'
    (T, Lpast, 256) spikes of earlier positions (None = these L positions are the whole sequence).
    Returns the new residual (T, L, 256) and the updated cache."""
    code = SHIFT_CODE[r_in]
    q = norm_lif(linear(code, p[f"b{b}.q.w"], p[f"b{b}.q.b"]), p, b, "q")
    k = norm_lif(linear(code, p[f"b{b}.k.w"], p[f"b{b}.k.b"]), p, b, "k")
    v = norm_lif(linear(code, p[f"b{b}.v.w"], p[f"b{b}.v.b"]), p, b, "v")
    past = 0
    if cache is not None and cache.get("k") is not None:
        past = cache["k"].shape[1]
        k_all, v_all = np.concatenate([cache["k"], k], 1), np.concatenate([cache["v"], v], 1)
    else:
        k_all, v_all = k, v
    a = attention(q, k_all, v_all, causal_from=past)
    s = norm_lif(a, p, b, "attn", head=True)
    pr = norm_lif(linear(s, p[f"b{b}.proj.w"], p[f"b{b}.proj.b"]), p, b, "proj")
    r = r_in + pr
    h = norm_lif(linear(SHIFT_CODE[r], p[f"b{b}.fc1.w"], p[f"b{b}.fc1.b"]), p, b, "fc1")
    m = norm_lif(linear(h, p[f"b{b}.fc2.w"], p[f"b{b}.fc2.b"]), p, b, "fc2")
    r_out = r + m
    if trace is not None:
        trace.update({f"b{b}.in": r_in, f"b{b}.q": q, f"b{b}.k": k, f"b{b}.v": v, f"b{b}.attn_sum": a,
                      f"b{b}.attn_spk": s, f"b{b}.proj_spk": pr, f"b{b}.mid": r, f"b{b}.h": h, f"b{b}.m": m,
                      f"b{b}.out": r_out})
    return r_out, {"k": k_all, "v": v_all}


def chip(spikes, p, caches=None, trace=None):
    """All blocks. spikes: (T, L, 256) input-LIF spikes for L new positions; caches: list per block (None: fresh)."""
    caches = caches if caches is not None else [None] * NB
    r = spikes.astype(np.int64)
    new = []
    for b in range(NB):
        r, c = block(r, p, b, caches[b], trace)
        new.append(c)
    return r, new


def load(path):
    return dict(np.load(path))


# ---------------------------------------------------------------------------------------------- host side (float)

def host_model(run):
    """The float host parts: embedding + position -> input norm -> input LIF (spikes), and the read-out head."""
    import torch
    from models import model_from_config
    cfg = json.load(open(os.path.join(ROOT, "runs", run, "config.json")))
    sd = torch.load(os.path.join(ROOT, "runs", run, "best.pt"), map_location="cpu", weights_only=False)
    m = model_from_config(cfg, cfg["block_size"])
    m.load_state_dict(sd.get("model", sd))
    return m.eval(), cfg


def host_input_spikes(m, tokens):
    import torch
    with torch.no_grad():
        x0 = m.input_norm(m.embed(torch.tensor(tokens)[None]) + m.pos_embed[:, :len(tokens)])
        v = torch.zeros_like(x0)
        out = []
        for _ in range(T):
            v = v + (x0 - v) / 2
            s = (v >= 1).float()
            v = v - s
            out.append(s[0])
    return torch.stack(out).numpy().astype(np.int64)                     # (T, L, 256)


def host_logits(m, r):
    import torch
    with torch.no_grad():
        return m.head(torch.tensor(r, dtype=torch.float32).mean(0)).numpy()   # (L, vocab)


# ---------------------------------------------------------------------------------------------- checks

def check(run, windows, double=True):
    import torch
    from char_data import CharDataset
    from fixed_point_eval import Cfg, build
    path = os.path.join(ROOT, "hw", f"params_{run}.npz")
    if not os.path.exists(path):
        export(run)
    p = load(path)
    Cfg.exact, Cfg.frac, Cfg.bits, Cfg.lut, Cfg.gamma_min, Cfg.const = False, 7, 12, 4, 0.05, "zero"
    Cfg.recip, Cfg.recip_norms, Cfg.recip_map, Cfg.int_stats = "lut", None, None, True
    Cfg.round_mode, Cfg.saturate = "up", True
    Cfg.dtype = torch.float64 if double else torch.float32
    sim, cfg = build(run, "tensor")
    if double:
        sim = sim.double()
        for b, blk in enumerate(sim.blocks):            # exactly W_int * scale in float64, as exported
            for l, lin in (("q", blk.attn.q_linear), ("k", blk.attn.k_linear), ("v", blk.attn.v_linear),
                           ("proj", blk.attn.proj_linear), ("fc1", blk.mlp.fc1), ("fc2", blk.mlp.fc2)):
                sc = float(p[f"b{b}.{l}.scale"])
                lin.weight.data = torch.tensor(p[f"b{b}.{l}.w"].astype(np.float64) * sc, device=lin.weight.device)
                if lin.bias is not None:
                    lin.bias.data = torch.tensor(p[f"b{b}.{l}.b"].astype(np.float64) * sc, device=lin.weight.device)
    host, _ = host_model(run)
    data = CharDataset(os.path.join(ROOT, cfg["data_path"]), 0.05, 0.05)
    torch.manual_seed(0)
    ids = data.splits["test"].long()
    starts = torch.randint(0, len(ids) - CTX - 1, (windows,)).tolist()

    # 1. block outputs, reference vs float32 simulation, on whole windows
    agree = total = 0
    nll_ref = nll_sim = 0.0
    for i in starts:
        tok = ids[i:i + CTX + 1].tolist()
        spikes = host_input_spikes(host, tok[:-1])
        outs = {}
        hooks = [blk.register_forward_hook(lambda mod, inp, out, b=b: outs.__setitem__(b, out.detach()))
                 for b, blk in enumerate(sim.blocks)]
        hooks.append(sim.input_lif.register_forward_hook(lambda mod, inp, out: outs.__setitem__("in", out.detach())))
        with torch.no_grad():
            sim_logits = sim(torch.tensor(tok[:-1])[None].cuda()).float().cpu()
        first = None
        for h in hooks:
            h.remove()
        sim_in = outs["in"].cpu().numpy().reshape(T, CTX, D).astype(np.int64)
        assert (sim_in == spikes).mean() > 0.999, "host input spikes differ from the simulation's input LIF"
        trace = {}
        r, _ = chip(sim_in, p, trace=trace)
        for b in range(NB):
            so = outs[b].cpu().numpy().reshape(T, CTX, D)
            agree += (so == trace[f"b{b}.out"]).sum(); total += so.size
            if first is None and not np.array_equal(so, trace[f"b{b}.out"]):
                first = b
        print(f"  window at {i}: first block whose output differs: {first}")
        y = np.array(tok[1:])
        ref_lp = torch.log_softmax(torch.tensor(host_logits(host, r)), -1).numpy()
        sim_lp = torch.log_softmax(sim_logits[0], -1).numpy()
        nll_ref += -ref_lp[np.arange(CTX), y].sum(); nll_sim += -sim_lp[np.arange(CTX), y].sum()
    n = windows * CTX
    print(f"block outputs (residual, all {NB} blocks, {T} steps): {100 * agree / total:.4f}% of values identical "
          f"to the {'float64' if double else 'float32'} fixed-point simulation")
    print(f"bits/token on these {windows} windows: reference {nll_ref / n / math.log(2):.4f} | "
          f"simulation {nll_sim / n / math.log(2):.4f}")

    # 2. token-by-token with the K/V cache == whole window
    tok = ids[starts[0]:starts[0] + 64].tolist()
    spikes = host_input_spikes(host, tok)
    whole, _ = chip(spikes, p)
    caches, steps = None, []
    for t in range(len(tok)):
        r, caches = chip(spikes[:, t:t + 1], p, caches)
        steps.append(r)
    inc = np.concatenate(steps, 1)
    print(f"token-by-token with the K/V cache vs whole window (64 tokens): "
          f"{'identical' if np.array_equal(inc, whole) else 'DIFFERENT'}")


# ---------------------------------------------------------------------------------------------- test vectors

def vectors(run, tokens):
    import torch
    from char_data import CharDataset
    path = os.path.join(ROOT, "hw", f"params_{run}.npz")
    if not os.path.exists(path):
        export(run)
    p = load(path)
    host, cfg = host_model(run)
    data = CharDataset(os.path.join(ROOT, cfg["data_path"]), 0.05, 0.05)
    text = data.splits["test"].long()
    nl = data.chars.index("\n")
    start = (text == nl).nonzero().flatten()[0].item()                    # a tune boundary in the test split
    tok = text[start:start + tokens].tolist()
    spikes = host_input_spikes(host, tok)
    outdir = os.path.join(ROOT, "hw", "vectors")
    os.makedirs(outdir, exist_ok=True)
    caches = None
    for t in range(len(tok)):                                              # the chip's order: one token at a time
        trace = {}
        r, caches = chip(spikes[:, t:t + 1], p, caches, trace)
        np.savez_compressed(os.path.join(outdir, f"token_{t:03d}.npz"), position=t, token=tok[t],
                            input_spikes=spikes[:, t], **{k: v[:, 0] for k, v in trace.items()})
    json.dump({"run": run, "tokens": tok, "params": os.path.basename(path),
               "arrays": "each token_XXX.npz holds, per block b: b.in residual in (T,256), b.q/k/v spikes, "
                         "b.attn_sum (T,256) attention sums before the head norm, b.attn_spk, b.proj_spk, "
                         "b.mid residual after attention, b.h (T,1024) MLP hidden spikes, b.m, b.out residual out"},
              open(os.path.join(outdir, "index.json"), "w"), indent=1)
    print(f"wrote {len(tok)} token vectors to {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("export", "check", "vectors"))
    ap.add_argument("run")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.run)
    elif a.cmd == "check":
        check(a.run, a.windows)
    else:
        vectors(a.run, a.tokens)


if __name__ == "__main__":
    main()
