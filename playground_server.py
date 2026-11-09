"""Playground server: serves the repository's static files (dashboard.html,
playground.html, runs/*) like `python3 -m http.server`, plus a JSON API that
runs a character-level run's latest checkpoint:

    GET  /api/runs        char-LM runs with a checkpoint (newest first)
    POST /api/generate    {run, ckpt, prompt, num_chars, temperature, top_k, T}
                          -> sampled text
    GET  /api/folk/info   folk-rnn ABC vocabulary: meters, modes, the default folk run
    POST /api/folk/generate {run, ckpt, meter, mode, opening, temperature, top_k, T, seed, n}
                          -> n tunes (folk-rnn token runs, e.g. a spiking folkrnn_* run): tokens with the
                             probability the model gave each and its entropy, the tune as ABC text for
                             rendering, and the abc_quality checks (bar lengths per bar, repeats, tonic)
    POST /api/attention   {run, ckpt, text, T, max_len}
                          -> the last min(block_size, max_len) characters, per layer / head
                             the attention map (L x L, causal: time-averaged spike Q.K overlap
                             counts, or softmax weights for an ANN run) as row sums plus
                             uint8-quantised row distributions, and the top-5 next-character
                             predictions per position

ckpt is "last" (last.pt, refreshed every --save_every steps) or "best"
(best.pt). A checkpoint is reloaded whenever its file changes. The model runs
on the CPU by default so it does not slow down a training run on the GPU.

Usage:
    python3 playground_server.py               # http://127.0.0.1:8770/playground.html
    python3 playground_server.py --port 8770 --device cuda
"""

import argparse
import base64
import functools
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import torch

from models import model_from_config

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
UNK = "\ufffd"
INFER_LOCK = threading.Lock()  # the LIF neurons keep state, so one forward pass at a time


class ModelCache:
    """One loaded model per (run, ckpt), reloaded when the checkpoint file changes."""

    def __init__(self, device: str):
        self.device = device
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, run: str, ckpt: str):
        if os.sep in run or run.startswith(".") or ckpt not in ("last", "best"):
            raise ValueError("bad run or checkpoint name")
        run_dir = os.path.join(RUNS_DIR, run)
        path = os.path.join(run_dir, f"{ckpt}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{run}/{ckpt}.pt does not exist yet")
        mtime = os.path.getmtime(path)
        with self.lock:
            hit = self.cache.get((run, ckpt))
            if hit and hit["mtime"] == mtime:
                return hit
            config = json.load(open(os.path.join(run_dir, "config.json")))
            state = torch.load(path, map_location="cpu", weights_only=False)
            step = state.get("step") if "model" in state else None
            weights = state["model"] if "model" in state else state
            model = model_from_config(config, config["block_size"])
            model.load_state_dict(weights)
            model.to(self.device).eval()
            chars = config["chars"]
            hit = {"mtime": mtime, "model": model, "config": config, "chars": chars, "step": step,
                   "stoi": {c: i for i, c in enumerate(chars)}}
            self.cache[(run, ckpt)] = hit
            return hit


def encode(entry, text: str) -> list[int]:
    stoi = entry["stoi"]
    unk = stoi.get(UNK, stoi.get(" ", 0))
    return [stoi.get(c, unk) for c in text]


def with_T(model, T):
    """Set the number of simulated time steps (if T is given); returns the previous T."""
    old = model.T
    if T:
        model.T = max(1, min(int(T), 64))
    return old


@torch.no_grad()
def generate(entry, prompt: str, num_chars: int, temperature: float, top_k: int, T: int) -> dict:
    model, block = entry["model"], entry["config"]["block_size"]
    ids = encode(entry, prompt) or encode(entry, " ")
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    old_T = with_T(model, T)
    try:
        for _ in range(num_chars):
            logits = model(x[:, -block:])[:, -1].float() / max(temperature, 1e-4)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            x = torch.cat([x, nxt], dim=1)
    finally:
        model.T = old_T
    chars = entry["chars"]
    return {"prompt": "".join(chars[i] for i in ids), "text": "".join(chars[i] for i in x[0, len(ids):].tolist())}


@torch.no_grad()
def attention(entry, text: str, T: int, max_len: int = 0) -> dict:
    model, block, chars = entry["model"], entry["config"]["block_size"], entry["chars"]
    keep = min(block, max_len) if max_len > 0 else block
    ids = (encode(entry, text) or encode(entry, " "))[-keep:]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    old_T = with_T(model, T)
    try:
        logits, maps = model(x, return_attn=True)
    finally:
        model.T = old_T
    probs = torch.softmax(logits[0].float(), -1)
    top = torch.topk(probs, 5)
    nxt = ids[1:]
    # (layers, H, L, L): spike overlap counts averaged over T (spiking) or softmax weights (ANN).
    # Sent compactly: each row's sum, and the row's distribution (row / sum) quantised to 0-255, base64.
    a = torch.stack([m[0].float().cpu() for m in maps])
    rows = a.sum(-1)
    q = (a / rows.clamp_min(1e-12)[..., None] * 255).round().clamp(0, 255).to(torch.uint8)
    return {
        "tokens": [chars[i] for i in ids],
        "shape": list(a.shape),
        "row_sums": [[[round(v, 4) for v in r] for r in layer] for layer in rows.tolist()],
        "attn_q": base64.b64encode(q.numpy().tobytes()).decode("ascii"),
        "kind": "softmax" if entry["config"].get("model_type") == "ann" else "spike counts",
        "top": [[[chars[i], round(p, 4)] for i, p in zip(ti, tp)]
                for ti, tp in zip(top.indices.tolist(), top.values.tolist())],
        # probability the model gave to the character that actually follows (None at the last position)
        "p_next": [round(probs[t, n].item(), 4) for t, n in enumerate(nxt)] + [None],
    }


FOLK_VOCAB_PATH = os.path.join(ROOT, "data", "folkrnn", "vocab.json")
FOLK_DEFAULT_RUN = "folkrnn_256d6L_T8_distill_homeo2s_sq8_spread8_50k_do05"


def folk_vocab() -> dict:
    """char -> ABC token for the folk-rnn corpus (one private-use character per token)."""
    return json.load(open(FOLK_VOCAB_PATH))


def folk_info() -> dict:
    tokens = list(folk_vocab().values())
    runs = [r["name"] for r in list_runs() if r["name"].startswith("folkrnn_") and "lstm" not in r["name"]]
    return {"meters": sorted([t for t in tokens if t.startswith("M:")], key=lambda t: (len(t), t)),
            "modes": [t for t in ("K:Cmaj", "K:Cmin", "K:Cdor", "K:Cmix") if t in tokens],
            "runs": runs, "default_run": FOLK_DEFAULT_RUN if FOLK_DEFAULT_RUN in runs else (runs[0] if runs else None)}


def folk_tokenize(text: str, token_to_char: dict) -> list[str]:
    """Split typed ABC (spaces optional) into vocabulary tokens, longest match first."""
    out, i, longest = [], 0, max(len(t) for t in token_to_char)
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        for n in range(min(longest, len(text) - i), 0, -1):
            if text[i:i + n] in token_to_char:
                out.append(text[i:i + n])
                i += n
                break
        else:
            raise ValueError(f"'{text[i:i + 8]}' is not in the folk-rnn vocabulary (at character {i})")
    return out


def folk_abc(tokens: list[str], title: str) -> str:
    """ABC text abcjs can render: X/T/L header, the tune's M: and K:, then the body (no spaces inside bars,
    so durations stay attached to their notes; a space after every bar line)."""
    meter = next((t for t in tokens if t.startswith("M:")), "M:4/4")
    key = next((t for t in tokens if t.startswith("K:")), "K:Cmaj")
    body, bars = "", 0
    for t in tokens:
        if t.startswith(("M:", "K:")):
            continue
        body += t
        if t in ("|", "|:", ":|", "|1", "|2"):
            # a line break after every 4th completed bar (|1 / |2 open a bar, they do not close one)
            bars += t not in ("|1", "|2") and body.rstrip()[:-len(t)].strip() != ""
            body += "\n" if t not in ("|1", "|2") and bars and bars % 4 == 0 else " "
    return f"X:1\nT:{title}\nL:1/8\n{meter}\n{key}\n{body.strip()}\n"


@torch.no_grad()
def folk_generate(entry, meter: str, mode: str, opening: str, temperature: float, top_k: int, T, seed, n: int,
                  max_tokens: int = 600) -> dict:
    from abc_quality import bar_lengths, check, METER
    vocab = folk_vocab()
    token_to_char = {t: c for c, t in vocab.items()}
    model, block, stoi, chars = entry["model"], entry["config"]["block_size"], entry["stoi"], entry["chars"]
    head = [t for t in (meter, mode) if t]
    if mode and not meter:
        raise ValueError("choose a meter to go with the mode (tunes start with M: then K:)")
    prefix = head + folk_tokenize(opening, token_to_char) if head or opening else []
    if opening and not head:
        raise ValueError("choose a meter and mode before giving an opening")
    nl = stoi["\n"]
    ids = [nl] + [stoi[token_to_char[t]] for t in prefix]
    device = next(model.parameters()).device
    if seed not in (None, ""):
        torch.manual_seed(int(seed))
    x = torch.tensor([ids] * n, dtype=torch.long, device=device)
    probs_out = [[] for _ in range(n)]
    done = torch.zeros(n, dtype=torch.bool, device=device)
    old_T = with_T(model, T)
    try:
        for _ in range(max_tokens - len(prefix)):
            logits = model(x[:, -block:])[:, -1].float()
            p_full = torch.softmax(logits, -1)
            ent = -(p_full * p_full.clamp_min(1e-12).log2()).sum(-1)
            logits = logits / max(temperature, 1e-4)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            nxt[done] = nl
            for k in range(n):
                if not done[k]:
                    probs_out[k].append((round(p_full[k, nxt[k, 0]].item(), 4), round(ent[k].item(), 3)))
            x = torch.cat([x, nxt], 1)
            done |= nxt[:, 0] == nl
            if done.all():
                break
    finally:
        model.T = old_T
    tunes = []
    for k in range(n):
        gen_ids = x[k, len(ids):].tolist()
        gen = []
        for i in gen_ids:
            if i == nl:
                break
            gen.append(vocab.get(chars[i], "?"))
        toks = prefix + gen
        pe = probs_out[k][:len(gen)]
        q = check(toks)
        target = METER.get(toks[0]) if toks else None
        bars = [None if b is None else float(b) for b in bar_lengths(toks)]
        tunes.append({"tokens": toks, "n_prefix": len(prefix), "prob": [None] * len(prefix) + [a for a, _ in pe],
                      "entropy": [None] * len(prefix) + [b for _, b in pe], "finished": nl in gen_ids,
                      "abc": folk_abc(toks, f"Spiking tune {k + 1}"), "bar_lengths": bars, "bar_target": target,
                      "checks": {"header": q["header"], "repeats": q["repeats"], "tonic": q["tonic"],
                                 "bars": q["bars"], "valid_bars": q["valid_bars"]}})
    return {"tunes": tunes, "T": T or entry["config"]["T"]}


def list_runs() -> list[dict]:
    out = []
    for name in os.listdir(RUNS_DIR):
        run_dir = os.path.join(RUNS_DIR, name)
        cfg = os.path.join(run_dir, "config.json")
        ckpts = [c for c in ("last", "best") if os.path.exists(os.path.join(run_dir, f"{c}.pt"))]
        if not ckpts or not os.path.exists(cfg):
            continue
        try:
            config = json.load(open(cfg))
        except (OSError, ValueError):
            continue
        if config.get("task") != "char_lm":
            continue
        mtime = max(os.path.getmtime(os.path.join(run_dir, f"{c}.pt")) for c in ckpts)
        out.append({"name": name, "ckpts": ckpts, "mtime": mtime, "T": config["T"],
                    "block_size": config["block_size"], "dim": config["dim"], "depth": config["depth"],
                    "num_heads": config["num_heads"], "vocab_size": config["vocab_size"],
                    "prompt": config.get("prompt", " ")})
    return sorted(out, key=lambda r: -r["mtime"])


class Handler(SimpleHTTPRequestHandler):
    cache: ModelCache = None

    def log_message(self, fmt, *args):  # only API calls, not every static file
        if self.path.startswith("/api/"):
            super().log_message(fmt, *args)

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/runs":
            return self.send_json(list_runs())
        if self.path.split("?")[0] == "/api/folk/info":
            return self.send_json(folk_info())
        return super().do_GET()

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            entry = self.cache.get(req.get("run", ""), req.get("ckpt", "last"))
            info = {"step": entry["step"]}
            with INFER_LOCK:
                res = self.run_api(entry, req)
            if res is None:
                return self.send_json({"error": "unknown endpoint"}, 404)
            self.send_json({**info, **res})
        except (ValueError, KeyError, FileNotFoundError) as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:  # keep the server alive, report to the page
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def run_api(self, entry, req):
        if self.path == "/api/generate":
            return generate(entry, str(req.get("prompt", "")), max(1, min(int(req.get("num_chars", 200)), 2000)),
                            float(req.get("temperature", 0.8)), int(req.get("top_k", 0)), req.get("T"))
        if self.path == "/api/folk/generate":
            return folk_generate(entry, str(req.get("meter", "")), str(req.get("mode", "")), str(req.get("opening", "")),
                                 float(req.get("temperature", 1.0)), int(req.get("top_k", 0)), req.get("T"),
                                 req.get("seed"), max(1, min(int(req.get("n", 1)), 4)))
        if self.path == "/api/attention":
            return attention(entry, str(req.get("text", "")), req.get("T"), int(req.get("max_len", 0) or 0))
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    Handler.cache = ModelCache(args.device)
    server = ThreadingHTTPServer((args.bind, args.port), functools.partial(Handler, directory=ROOT))
    print(f"playground: http://{args.bind}:{args.port}/playground.html (model on {args.device})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
