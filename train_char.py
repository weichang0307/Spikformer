"""Train the Spiking Self-Attention Transformer as a character-level GPT on
Tiny Shakespeare (or any text file via --data_path), and save the run
(config.json, history.json, best.pt, checkpoints/, eval_curve.png,
samples.txt) to runs/char_<dim>d<depth>L_T<T>[_<suffix>].

Usage:
    python3 train_char.py
    python3 train_char.py --block_size 256 --depth 6 --steps 20000

Every --eval_every steps it reports, at each --eval_T, the validation loss
(nats/char), bits per character and next-character accuracy, averaged over
--eval_iters batches, and writes them to RUN_DIR/live.json for
dashboard.html / plot_curves.py ("acc" there is next-character accuracy).
Every --sample_every steps it prints a generated sample and appends it to
RUN_DIR/samples.txt. To stop a run cleanly (saving best.pt), create
RUN_DIR/STOP. best.pt is the checkpoint with the lowest val loss; it is
rewritten whenever the val loss improves, so it survives a crash or kill.

--init_from RUN_DIR/best.pt continues from saved weights (model flags must
match; the optimizer starts fresh), e.g. to fine-tune with a decaying lr.
--resume continues a stopped run losslessly from RUN_DIR/last.pt (weights,
optimizer, step, clipper / homeostasis state, history, RNG): rerun the same
command with --resume. last.pt is written every --save_every steps and when
the run is stopped by RUN_DIR/STOP, Ctrl-C or kill (SIGINT / SIGTERM).
--bf16 and --compile make training about 2x faster.

With --steps 0 the run has no end: after warmup the learning rate stays
constant and training continues until RUN_DIR/STOP appears.
"""

import argparse
import contextlib
import itertools
import random
import signal
import json
import math
import os
import time

import torch
import torch.nn as nn

from char_data import DEFAULT_PATH, CharDataset
from health import AdaptiveClipper, Homeostasis, LIFMonitor, UpdateTracker, attention_and_context, param_stats
from models import ANNTransformer, SpikingTransformer, model_from_config
from neurons import set_surrogate_alpha, use_alpha_tensor
from train import pick_device, save_run, write_live


def loss_and_acc(model: nn.Module, seq: torch.Tensor):
    logits = model(seq[:, :-1])
    targets = seq[:, 1:]
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    acc = (logits.argmax(-1) == targets).float().mean()
    return loss, acc


def attention_kl(s_maps: list, t_maps: list, rows: torch.Tensor, blocks: list, reverse: bool = False) -> torch.Tensor:
    """Mean over the given blocks, heads, batch and sampled query rows of KL(teacher row || student row), or with
    reverse=True KL(student row || teacher row). The forward KL is mass-covering: it heavily penalises any key
    the teacher weights but the student leaves near zero, so count-based attention learns to flatten its rows.
    The reverse KL is mode-seeking: it penalises student weight where the teacher has little, so the student
    concentrates on the teacher's high-weight keys and may leave the rest at zero.
    Student rows are spike overlap counts (time-averaged), normalised to distributions over the visible keys
    (with a tiny floor, so a row with no overlap at all is uniform rather than undefined); teacher rows are
    the ANN's softmax weights. Same head index = same head: the student's Q/K started as the teacher's."""
    L = s_maps[0].shape[-1]
    visible = (torch.arange(L, device=rows.device)[None, :] <= rows[:, None]).float()   # (R, L)
    total = 0.0
    for b in blocks:
        sp = s_maps[b].float() + 1e-4 * visible
        sp = sp / sp.sum(-1, keepdim=True)
        tp = t_maps[b].float()
        log_ratio = tp.clamp_min(1e-9).log() - sp.clamp_min(1e-9).log()
        total = total + ((-sp * log_ratio) if reverse else (tp * log_ratio)).sum(-1).mean()
    return total / len(blocks)


def attention_spread(s_maps: list, rows: torch.Tensor, heads: int) -> torch.Tensor:
    """Mean over every block, heads [0, heads), batch and the sampled query rows of the attention spread: the
    entropy of the row's (time-averaged) overlap counts normalised to a distribution over the visible keys, divided
    by log(#visible keys) (0 = one key, 1 = uniform; the attn_entropy health measure). Penalising it pushes weak
    keys to zero, through the learned count threshold (theta rises) and through sharper Q/K spikes."""
    L = s_maps[0].shape[-1]
    visible = (torch.arange(L, device=rows.device)[None, :] <= rows[:, None]).float()   # (R, L)
    log_n = (rows.float() + 1).log()                                                    # (R,)
    total = 0.0
    for m in s_maps:
        sp = m[:, :heads].float() + 1e-4 * visible
        sp = sp / sp.sum(-1, keepdim=True)
        total = total + ((-sp * sp.clamp_min(1e-9).log()).sum(-1) / log_n).mean()
    return total / len(s_maps)


def distill_loss_and_acc(model: nn.Module, teacher: nn.Module, seq: torch.Tensor, alpha: float, temp: float,
                         attn_weight: float = 0.0, attn_blocks: list = (), attn_rows: int = 64, attn_min_pos: int = 64,
                         attn_reverse: bool = False, spread_weight: float = 0.0, spread_heads: int = 0):
    """Knowledge distillation: alpha * CE(targets) + (1 - alpha) * temp^2 * KL(teacher || student) on the
    temperature-softened next-character distributions, plus (attn_weight > 0) attn_weight * the attention-map
    KL of attention_kl() on attn_rows query positions sampled from [attn_min_pos, L).
    spread_weight > 0 adds spread_weight * attention_spread() of heads [0, spread_heads) on the same rows.
    Returns (objective, CE, KL, attention KL, spread, accuracy)."""
    x = seq[:, :-1]
    rows = None
    if attn_weight > 0 or spread_weight > 0:
        L = x.shape[1]
        rows = torch.randperm(L - attn_min_pos, device=x.device)[:attn_rows].sort().values + attn_min_pos
        logits, s_maps = model(x, return_attn=True, attn_rows=rows)
    else:
        logits = model(x)
    targets = seq[:, 1:]
    ce = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    with torch.no_grad():
        if attn_weight > 0:
            t_logits, t_maps = teacher(x, return_attn=True, attn_rows=rows)
        else:
            t_logits = teacher(x)
        t_logp = torch.log_softmax(t_logits.float().flatten(0, 1) / temp, -1)
    s_logp = torch.log_softmax(logits.float().flatten(0, 1) / temp, -1)
    kl = nn.functional.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") * temp ** 2
    acc = (logits.argmax(-1) == targets).float().mean()
    objective = alpha * ce + (1 - alpha) * kl
    attn_kl = torch.zeros((), device=x.device)
    if attn_weight > 0:
        attn_kl = attention_kl(s_maps, t_maps, rows, attn_blocks, attn_reverse)
        objective = objective + attn_weight * attn_kl
    spread = torch.zeros((), device=x.device)
    if spread_weight > 0:
        spread = attention_spread(s_maps, rows, spread_heads)
        objective = objective + spread_weight * spread
    return objective, ce, kl, attn_kl, spread, acc


@torch.no_grad()
def generate(model: nn.Module, data: CharDataset, prompt: str, num_chars: int, block_size: int,
             temperature: float = 1.0, device="cpu", amp=contextlib.nullcontext) -> str:
    """Autoregressive sampling; the context is cropped to the last block_size chars."""
    model.eval()
    ids = torch.tensor([data.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(num_chars):
        with amp():
            logits = model(ids[:, -block_size:])[:, -1].float() / temperature
        next_id = torch.multinomial(torch.softmax(logits, -1), 1)
        ids = torch.cat([ids, next_id], dim=1)
    return data.decode(ids[0].tolist())


def train(model: nn.Module, data: CharDataset, args, device: str, run_dir: str, teacher: nn.Module = None) -> dict:
    model.to(device)
    # weight decay only on Linear weight matrices: decaying the L1LayerNorm gains
    # shrinks every LIF's input below threshold over time (neurons go silent), and
    # embeddings / biases / attention scales should not be pulled toward zero either
    decay = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    groups = [{"params": [p for p in model.parameters() if id(p) in decay], "weight_decay": args.weight_decay},
              {"params": [p for p in model.parameters() if id(p) not in decay], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    open_ended = args.steps <= 0
    warmup = args.warmup_steps if open_ended else min(args.warmup_steps, args.steps)

    cooldown = {"start": None}  # set below when --cooldown is given (or restored from last.pt)
    rewarm = {"start": None, "from": None}  # set below when --rewarm is given (or restored)

    def lr_at(step):  # linear warmup, then cosine decay to lr / 10 (constant if open-ended)
        if rewarm["start"] is not None and step - rewarm["start"] < args.rewarm:
            # --rewarm: linear ramp from the resumed lr up to --lr, then the normal schedule
            return rewarm["from"] + (args.lr - rewarm["from"]) * (step - rewarm["start"]) / args.rewarm
        if cooldown["start"] is not None and step > cooldown["start"]:  # --cooldown: cosine to lr_end over N steps
            progress = min((step - cooldown["start"]) / args.cooldown, 1.0)
            lr_end = args.cooldown_to if args.cooldown_to > 0 else args.lr / 10
            return lr_end + (args.lr - lr_end) * 0.5 * (1 + math.cos(math.pi * progress))
        if rewarm["start"] is not None and open_ended:  # after a rewarm: constant until a cooldown starts
            return args.lr
        if step <= warmup:
            return args.lr * step / max(warmup, 1)
        if open_ended:
            return args.lr
        progress = (step - warmup) / max(args.steps - warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    train_T = model.T
    eval_Ts = sorted(set(args.eval_T) | {train_T})
    history = {"step": [], "loss": [], "acc": [], "bpc": [], "train_loss": [], "train_acc": [],
               "lr": [], "time": [], "checkpoints": []}
    for T in eval_Ts:
        history[f"acc_T{T}"], history[f"loss_T{T}"], history[f"bpc_T{T}"] = [], [], []
    # training-health metrics (see health.py): {name: [one value per evaluation]}
    health_keys = ("fire_rate", "fire_dead", "fire_sat", "surrogate", "grad_norm", "update_ratio",
                   "norm_gain", "gamma_min", "gamma_max", "beta_min", "beta_max",
                   "attn_scale", "head_scale_spread", "attn_empty", "attn_entropy", "attn_local",
                   "attn_dead_heads")
    if args.adaptive_clip > 0:
        health_keys += ("clip_frac", "grad_spike")
    if args.homeostasis > 0:
        health_keys += (("homeo_active", "homeo_shift", "homeo_shift_up", "homeo_shift_down")
                        + (("homeo_high",) if args.homeo_upper > 0 else ()))
    for key in health_keys:
        history[key] = {}
    history["context_gain"] = []
    lif_monitor, updates = LIFMonitor(model), UpdateTracker(model)
    clipper = AdaptiveClipper(model, args.adaptive_clip) if args.adaptive_clip > 0 else None
    homeo = (Homeostasis(model, args.homeostasis, args.homeo_eta, upper=args.homeo_upper,
                         units=tuple(args.homeo_units.split(",")), max_shift=args.homeo_max_shift)
             if args.homeostasis > 0 else None)
    update_ratios = {}

    # bf16 autocast for forward passes; torch.compile for the training forward only
    # (evaluation and sampling use the plain module: they run with monitoring hooks
    # and other sequence lengths, which would force recompiles)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.bf16 else contextlib.nullcontext
    if args.compile:
        # each homeostasis / monitoring hook is compiled once per LIF it is attached to
        # (20+ in a 4-block model), more than dynamo's default of 8 versions per function
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 64
    train_model = torch.compile(model) if args.compile else model

    stop_file = os.path.join(run_dir, "STOP")
    last_path = os.path.join(run_dir, "last.pt")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    samples_path = os.path.join(run_dir, "samples.txt")

    def state_dict():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # fixed validation batches, so evaluations are comparable across steps
    gen = torch.Generator().manual_seed(1234)
    val_data = data.splits["val"]
    val_batches = []
    for _ in range(args.eval_iters):
        starts = torch.randint(len(val_data) - args.block_size - 1, (args.eval_batch_size,), generator=gen)
        val_batches.append(torch.stack([val_data[s: s + args.block_size + 1] for s in starts.tolist()]).long())

    best_loss, best_state = float("inf"), None
    train_loss_sum = train_acc_sum = 0.0
    train_count = 0
    first_step = 1

    def save_last(step):
        """Full training state, so --resume continues exactly where this run stopped."""
        torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cooldown_start": cooldown["start"], "rewarm": dict(rewarm),
                    "clipper": clipper.state_dict() if clipper else None,
                    "homeo": homeo.state_dict() if homeo else None,
                    "history": {k: v for k, v in history.items() if k not in ("checkpoints", "final_state")},
                    "best_loss": best_loss, "train_sums": (train_loss_sum, train_acc_sum, train_count),
                    "kl_sums": (kl_sum, akl_sum),
                    "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                            "python": random.getstate()}},
                   last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

    if args.resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if clipper and state["clipper"]:
            clipper.load_state_dict(state["clipper"])
        if homeo and state["homeo"]:
            homeo.load_state_dict(state["homeo"])
        history.update(state["history"])
        best_loss = state["best_loss"]
        train_loss_sum, train_acc_sum, train_count = state["train_sums"]
        resumed_kl_sums = state.get("kl_sums", (0.0, 0.0))   # older checkpoints did not save them
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        random.setstate(state["rng"]["python"])
        first_step = state["step"] + 1
        if args.cooldown > 0 and not args.cooldown_from and state.get("cooldown_start") is not None:
            cooldown["start"] = state["cooldown_start"]  # resuming in the middle of a cooldown
        if args.rewarm > 0:
            saved = state.get("rewarm") or {}
            if saved.get("start") is not None:  # resuming in the middle of a rewarm
                rewarm.update(saved)
            else:  # start the ramp from the lr the run was using when it stopped
                rewarm.update(start=state["step"], **{"from": optimizer.param_groups[0]["lr"]})
            print(f"rewarm: lr {rewarm['from']:.2e} -> {args.lr:.2e} (linear) over steps "
                  f"{rewarm['start'] + 1}-{rewarm['start'] + args.rewarm}"
                  + (", then constant" if open_ended else ""), flush=True)
        if os.path.exists(os.path.join(run_dir, "best.pt")):
            best_state = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
        print(f"resumed from {last_path} at step {state['step']}", flush=True)

    # Ctrl-C / kill: finish the current step, save last.pt and exit
    signal_stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: signal_stop.append(signum))

    start = time.time() - (history["time"][-1] if history["time"] else 0.0)
    if args.cooldown > 0:
        if cooldown["start"] is None:  # from --cooldown_from (a fixed step) or from where this run resumed
            cooldown["start"] = args.cooldown_from if args.cooldown_from > 0 else first_step - 1
        lr_end = args.cooldown_to if args.cooldown_to > 0 else args.lr / 10
        print(f"cooldown: lr {args.lr:.1e} -> {lr_end:.1e} (cosine) over steps "
              f"{cooldown['start'] + 1}-{cooldown['start'] + args.cooldown}, "
              + ("then constant (no end step)" if args.cooldown_hold else "then stop"), flush=True)
    cooldown_end = cooldown["start"] + args.cooldown if args.cooldown > 0 and not args.cooldown_hold else None
    # distillation KLs since the last evaluation, saved with train_sums so the first average after a resume is right
    kl_sum, akl_sum = resumed_kl_sums if args.resume else (0.0, 0.0)
    spread_sum = 0.0
    if args.attn_spread > 0 and args.attn_spread_heads <= 0:
        args.attn_spread_heads = (getattr(args, "attn_threshold_heads", 0) or getattr(args, "attn_square_heads", 0)
                                  or model.blocks[0].attn.num_heads)

    def spread_weight_at(step):  # --attn_spread, ramped in linearly over --attn_spread_ramp steps
        w = args.attn_spread * (min(step / args.attn_spread_ramp, 1.0) if args.attn_spread_ramp > 0 else 1.0)
        end = args.attn_spread_decay_to or args.steps
        if args.attn_spread_decay_from > 0 and end > args.attn_spread_decay_from:   # then linearly to 0 at `end`
            w *= 1.0 - min(max((step - args.attn_spread_decay_from) / (end - args.attn_spread_decay_from), 0.0), 1.0)
        return w
    attn_from = args.attn_distill_ramp_from or first_step - 1

    def attn_weight_at(step):  # --attn_distill, ramped linearly from --attn_distill_start over --attn_distill_ramp steps
        if args.attn_distill_ramp <= 0:
            return args.attn_distill
        progress = min(max((step - attn_from) / args.attn_distill_ramp, 0.0), 1.0)
        return args.attn_distill_start + (args.attn_distill - args.attn_distill_start) * progress
    attn_on = args.attn_distill > 0 or args.attn_distill_start > 0   # the attention term is in use at some point
    ce_from = args.distill_alpha_ramp_from or first_step - 1

    def ce_weight_at(step):  # --distill_alpha (CE weight of the output loss), ramped to --distill_alpha_to
        if args.distill_alpha_ramp <= 0:
            return args.distill_alpha
        progress = min(max((step - ce_from) / args.distill_alpha_ramp, 0.0), 1.0)
        return args.distill_alpha + (args.distill_alpha_to - args.distill_alpha) * progress
    attn_blocks = []
    for part in args.attn_distill_blocks.split(","):   # e.g. "2-10" or "2,3,5"
        lo, _, hi = part.partition("-")
        attn_blocks += list(range(int(lo), int(hi or lo) + 1))
    ramp_from = args.surrogate_alpha_ramp_from or first_step - 1

    def alpha_at(step):  # --surrogate_alpha_ramp: linear from --surrogate_alpha to --surrogate_alpha_to
        progress = min(max((step - ramp_from) / args.surrogate_alpha_ramp, 0.0), 1.0)
        return args.surrogate_alpha + (args.surrogate_alpha_to - args.surrogate_alpha) * progress
    for step in itertools.count(first_step) if open_ended else range(first_step, args.steps + 1):
        final = step == args.steps or (cooldown_end is not None and step >= cooldown_end)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        if args.surrogate_alpha_ramp > 0:
            set_surrogate_alpha(alpha_at(step))
        if getattr(args, "attn_square_heads", 0) > 0:   # blend count -> count^2 in
            blend = min(step / args.attn_square_ramp, 1.0) if args.attn_square_ramp > 0 else 1.0
            for blk in model.blocks:
                blk.attn.sq_blend.fill_(blend)
        if getattr(args, "attn_threshold_heads", 0) > 0:   # blend the learned count threshold in
            blend = min(step / args.attn_threshold_ramp, 1.0) if args.attn_threshold_ramp > 0 else 1.0
            for blk in model.blocks:
                blk.attn.thr_blend.fill_(blend)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = acc = 0.0
        for _ in range(args.grad_accum):  # micro-batches summing to one batch of batch_size
            seq = data.get_batch("train", args.batch_size // args.grad_accum, args.block_size, device=device)
            with amp():
                if teacher is None:
                    objective, micro_acc = loss_and_acc(train_model, seq)
                    micro_loss = objective
                else:  # train loss stays the plain cross-entropy; the KL term is logged separately
                    objective, micro_loss, micro_kl, micro_akl, micro_spread, micro_acc = distill_loss_and_acc(
                        train_model, teacher, seq, ce_weight_at(step), args.distill_temp, attn_weight_at(step),
                        attn_blocks, args.attn_distill_rows, args.attn_distill_min_pos,
                        args.attn_distill_kl == "reverse", spread_weight_at(step), args.attn_spread_heads)
                    spread_sum += micro_spread.item() / args.grad_accum
                    kl_sum += micro_kl.item() / args.grad_accum
                    akl_sum += micro_akl.item() / args.grad_accum
            (objective / args.grad_accum).backward()
            loss += micro_loss.item() / args.grad_accum
            acc += micro_acc.item() / args.grad_accum
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if clipper is not None:
            clipper.clip()
        if step % args.eval_every == 0:
            updates.snapshot()
        optimizer.step()
        if homeo is not None:
            homeo.step()
        if step % args.eval_every == 0:
            update_ratios = updates.ratios()
        train_loss_sum += loss
        train_acc_sum += acc
        train_count += 1

        user_stop = os.path.exists(stop_file)
        if step % args.eval_every == 0 or final or user_stop:
            health = dict(param_stats(model), update_ratio=update_ratios)
            if clipper is not None:
                health.update(clipper.stats())
            if homeo is not None:
                health.update(homeo.stats())
            model.eval()
            with torch.no_grad():
                for T in eval_Ts:
                    model.T = T
                    if T == train_T:
                        lif_monitor.start()
                    with amp():
                        losses, accs = zip(*(loss_and_acc(model, b.to(device)) for b in val_batches))
                    if T == train_T:
                        health.update(lif_monitor.stop())
                        with amp():
                            health.update(attention_and_context(model, [b.to(device) for b in val_batches[:2]]))
                    eval_loss = torch.stack(losses).mean().item()
                    history[f"loss_T{T}"].append(eval_loss)
                    history[f"bpc_T{T}"].append(eval_loss / math.log(2))
                    history[f"acc_T{T}"].append(torch.stack(accs).mean().item())
                model.T = train_T
            history["context_gain"].append(health.pop("context_gain"))
            for key, values in health.items():
                for name, value in values.items():
                    history[key].setdefault(name, []).append(value)
            history["step"].append(step)
            for key in ("loss", "acc", "bpc"):
                history[key].append(history[f"{key}_T{train_T}"][-1])
            history["train_loss"].append(train_loss_sum / train_count)
            history["train_acc"].append(train_acc_sum / train_count)
            if args.surrogate_alpha_ramp > 0:
                history.setdefault("surrogate_alpha", []).append(alpha_at(step))
            if getattr(args, "attn_threshold_heads", 0) > 0:   # mean learned threshold of the thresholded heads
                thr = history.setdefault("attn_threshold", {})
                for i, blk in enumerate(model.blocks):
                    thr.setdefault(f"block {i}", []).append(
                        blk.attn.threshold()[:args.attn_threshold_heads].mean().item())
            if teacher is not None and attn_on:
                history.setdefault("attn_weight", []).append(attn_weight_at(step))
            if teacher is not None and args.distill_alpha_ramp > 0:
                history.setdefault("ce_weight", []).append(ce_weight_at(step))
            if teacher is not None:
                history.setdefault("train_kl", []).append(kl_sum / train_count)
                kl_sum = 0.0
                if attn_on:
                    history.setdefault("train_attn_kl", []).append(akl_sum / train_count)
                akl_sum = 0.0
                if args.attn_spread > 0:
                    history.setdefault("train_spread", []).append(spread_sum / train_count)
                    history.setdefault("spread_weight", []).append(spread_weight_at(step))
                spread_sum = 0.0
            history["lr"].append(lr_at(step))
            history["time"].append(time.time() - start)
            train_loss_sum = train_acc_sum = 0.0
            train_count = 0

            per_T = " ".join(f"T{T} {history[f'bpc_T{T}'][-1]:.3f}" for T in eval_Ts)
            print(f"step {step:6d} | train loss {history['train_loss'][-1]:.3f} "
                  + (f"(KL to teacher {history['train_kl'][-1]:.3f}) " if teacher is not None else "")
                  + (f"(attention KL {history['train_attn_kl'][-1]:.3f}) " if teacher is not None and attn_on
                     else "")
                  + 
                  f"| val bpc {per_T} | val acc {history['acc'][-1]*100:5.1f}% "
                  f"| lr {history['lr'][-1]:.2e} "
                  + (f"| alpha {history['surrogate_alpha'][-1]:.2f} " if args.surrogate_alpha_ramp > 0 else "")
                  + f"| {history['time'][-1]:.0f}s", flush=True)
            dead = {}
            for name, values in history["fire_dead"].items():
                group = "b" + name.split(".")[1] if name.startswith("blocks.") else "in"
                dead.setdefault(group, []).append(values[-1])
            print("  health | never-fired " + " ".join(f"{g} {100 * sum(v) / len(v):.0f}%" for g, v in dead.items())
                  + f" | context gain {history['context_gain'][-1]:.3f} nats"
                  + " | grad " + " ".join(f"{g.replace('block ', 'b')} {v[-1]:.0e}"
                                          for g, v in history["grad_norm"].items())
                  + " | empty attn " + " ".join(f"{100 * v[-1]:.0f}%" for v in history["attn_empty"].values())
                  + ("" if clipper is None else " | clipped " + " ".join(
                      f"{g.replace('block ', 'b')} {100 * v[-1]:.0f}%" for g, v in history["clip_frac"].items())),
                  flush=True)
            write_live(run_dir, history)
            if history["loss"][-1] < best_loss:
                best_loss, best_state = history["loss"][-1], state_dict()
                torch.save(best_state, os.path.join(run_dir, "best.pt"))

        if step % args.sample_every == 0 or final or user_stop:
            text = generate(model, data, args.prompt, args.sample_chars, args.block_size,
                            args.temperature, device, amp)
            print(f"--- sample @ step {step} ---\n{text}\n---", flush=True)
            with open(samples_path, "a") as f:
                f.write(f"=== step {step} ===\n{text}\n\n")

        if step % args.checkpoint_every == 0 or final or user_stop:
            torch.save(state_dict(), os.path.join(ckpt_dir, f"checkpoint_step{step:06d}.pt"))
        if step % args.save_every == 0 or final or user_stop or signal_stop:
            save_last(step)
        if cooldown_end is not None and step >= cooldown_end:
            print(f"cooldown finished at step {step}", flush=True)
            break
        if user_stop:
            print(f"stopped at step {step} (STOP file); resume with --resume", flush=True)
            os.remove(stop_file)
            break
        if signal_stop:
            print(f"stopped at step {step} (signal {signal_stop[0]}); saved {last_path}, resume with --resume",
                  flush=True)
            break

    history["final_state"] = best_state  # save_run writes this as best.pt
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="share of the text used for validation (text8 standard: 0.05)")
    parser.add_argument("--test_fraction", type=float, default=0.0,
                        help="share held out at the end as a test split (text8 standard: 0.05)")
    parser.add_argument("--max_vocab", type=int, default=0,
                        help="keep only the most frequent characters (the rest map to one UNK symbol); 0 = all")
    parser.add_argument("--val_chars", type=int, default=0,
                        help="exact val split length in characters (overrides --val_fraction)")
    parser.add_argument("--test_chars", type=int, default=0,
                        help="exact test split length in characters (overrides --test_fraction)")
    parser.add_argument("--block_size", type=int, default=128, help="context length in characters")

    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--T", type=int, default=8,
                        help="spiking simulation time steps (power of two for the integer model)")
    parser.add_argument("--attn_lif", action=argparse.BooleanOptionalAction, default=True,
                        help="LIF on the attention output before proj (binary proj input)")
    parser.add_argument("--head_scale", action=argparse.BooleanOptionalAction, default=False,
                        help="one learned attention scale per head instead of per block (needs --learnable_scale)")
    parser.add_argument("--head_norm", action=argparse.BooleanOptionalAction, default=False,
                        help="attention L1 norm per head (mean/MAD over each head's features) instead of shared")
    parser.add_argument("--attn_norm_eps", type=float, default=1e-6,
                        help="constant added to the attention norm's MAD (e.g. 0.0625 with --head_norm)")
    parser.add_argument("--learnable_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="learn the attention scale (2^s per block, init 1/8) instead of fixed 1/8")
    parser.add_argument("--pos_embed", choices=("learned", "sinusoid"), default="learned",
                        help="learned position embedding or fixed sinusoidal encoding")
    parser.add_argument("--attn_norm", action=argparse.BooleanOptionalAction, default=True,
                        help="L1LayerNorm on the attention output before the attention LIF")
    parser.add_argument("--attn_proj", action=argparse.BooleanOptionalAction, default=False,
                        help="keep the proj linear/norm/LIF after attention (default: removed)")
    parser.add_argument("--mlp_layers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--model", choices=("spike", "ann"), default="spike",
                        help="ann: non-spiking GPT baseline with the same dimensions (softmax attention, LayerNorm, "
                             "GELU; spiking-only options are ignored, T is fixed to 1)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout: --model ann as usual; spiking model: spike dropout on each block's attention "
                             "and MLP outputs before the residual add (inverted, training only)")
    parser.add_argument("--surrogate_alpha", type=float, default=2.0,
                        help="steepness of the sigmoid surrogate gradient (peak alpha / 4, width ~ 1 / alpha); "
                             "affects the backward pass only")
    parser.add_argument("--surrogate_alpha_to", type=float, default=0.0,
                        help="with --surrogate_alpha_ramp: alpha to move to (linearly), e.g. 1")
    parser.add_argument("--surrogate_alpha_ramp", type=int, default=0,
                        help="number of steps over which alpha moves from --surrogate_alpha to --surrogate_alpha_to "
                             "(gradually, so Adam's statistics can follow the changing gradient scale)")
    parser.add_argument("--surrogate_alpha_ramp_from", type=int, default=0,
                        help="step at which the alpha ramp starts (default: the step this run starts or resumes at); "
                             "give it explicitly when resuming in the middle of a ramp")
    parser.add_argument("--attn_threshold_heads", type=int, default=0,
                        help="spiking model: the first N heads of every block keep only keys whose Q.K overlap count "
                             "reaches a learned per-head threshold (0 = off)")
    parser.add_argument("--attn_threshold_init", type=float, default=2.5, help="initial learned count threshold")
    parser.add_argument("--attn_square_heads", type=int, default=0,
                        help="spiking model: the first N heads of every block weight keys by their squared Q.K "
                             "overlap count instead of the count (0 = off)")
    parser.add_argument("--attn_square_ramp", type=int, default=0,
                        help="steps over which count^2 blends in, from the plain count")
    parser.add_argument("--attn_threshold_fixed", action="store_true",
                        help="keep the count threshold fixed at --attn_threshold_init (not learned)")
    parser.add_argument("--attn_threshold_ramp", type=int, default=0,
                        help="steps over which the threshold blends in, from no cut to the hard cut")
    parser.add_argument("--attn_spread", type=float, default=0.0,
                        help="with --distill_from: weight of a penalty on the attention spread (normalised row "
                             "entropy) of heads [0, --attn_spread_heads), on the sampled attention rows (0 = off)")
    parser.add_argument("--attn_spread_heads", type=int, default=0,
                        help="heads the spread penalty applies to (default: --attn_threshold_heads, else all)")
    parser.add_argument("--attn_spread_ramp", type=int, default=0, help="steps over which the spread weight ramps in")
    parser.add_argument("--attn_spread_decay_from", type=int, default=0,
                        help="step from which the spread weight decays linearly to 0 at --attn_spread_decay_to "
                             "(0 = no decay)")
    parser.add_argument("--attn_spread_decay_to", type=int, default=0,
                        help="step at which the decaying spread weight reaches 0 (default: --steps)")
    parser.add_argument("--multistep", action=argparse.BooleanOptionalAction, default=True,
                        help="spiking model: run each block on all T time steps at once (same result as the "
                             "step-by-step loop, faster); --no-multistep for the original loop")
    parser.add_argument("--checkpoint_blocks", action="store_true",
                        help="spiking model (multi-step): recompute block activations in backward, trading about "
                             "a third more compute for much less activation memory (allows larger micro-batches)")
    parser.add_argument("--mlp_hidden", type=int, default=0, help="MLP hidden width (default 0 = 4 x dim)")
    parser.add_argument("--pow2_gamma", action=argparse.BooleanOptionalAction, default=False,
                        help="train L1LayerNorm gamma as signed powers of two (shift in hardware)")
    parser.add_argument("--eval_T", type=int, nargs="+", default=[4, 8],
                        help="time steps to evaluate at (the training T is always included)")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="split each batch into this many micro-batches (same gradient, less memory)")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_iters", type=int, default=8, help="val batches per evaluation")
    parser.add_argument("--steps", type=int, default=10000,
                        help="0 = train until RUN_DIR/STOP (constant lr after warmup)")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adaptive_clip", type=float, default=0.0,
                        help="per-block clipping: cut a block's gradient norm to this many times its running "
                             "average (e.g. 3); 0 = off. See health.AdaptiveClipper")
    parser.add_argument("--homeostasis", type=float, default=0.0,
                        help="target firing rate for the Q/K/V homeostatic bias rule (e.g. 0.01); 0 = off. "
                             "See health.Homeostasis")
    parser.add_argument("--homeo_upper", type=float, default=0.0,
                        help="with --homeostasis: also push down Q/K/V features firing above this rate "
                             "(e.g. 0.5); 0 = push up only")
    parser.add_argument("--homeo_units", type=str, default="qkv",
                        help="NL units the homeostatic rule regulates: comma list of qkv, attn, proj, mlp, "
                             "input, or all")
    parser.add_argument("--homeo_max_shift", type=float, default=0.0,
                        help="bound each feature's cumulative homeostatic bias shift to +/- this (anti-windup); "
                             "0 = unbounded")
    parser.add_argument("--homeo_eta", type=float, default=1e-3,
                        help="largest bias increase per step for a silent Q/K/V feature")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="clip the global grad norm (default 0 = off: surrogate-gradient norms reach "
                             "~1e10 at init, so a small clip freezes the head; Adam copes with the scale)")

    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--sample_chars", type=int, default=300)
    parser.add_argument("--prompt", type=str, default=None,
                        help="sampling prompt (default: newline, or a space if the corpus has no newline)")
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_suffix", type=str, default="", help="appended to the run directory name")
    parser.add_argument("--rewarm", type=int, default=0,
                        help="with --resume: raise the lr linearly from the value the run stopped at to --lr "
                             "over this many steps (then constant with --steps 0)")
    parser.add_argument("--cooldown", type=int, default=0,
                        help="with --resume (e.g. of a constant-lr run): decay the lr from --lr to lr/10 with a "
                             "cosine over this many steps from the resume point, then stop")
    parser.add_argument("--cooldown_to", type=float, default=0.0,
                        help="with --cooldown: the lr to decay to (default 0 = lr / 10)")
    parser.add_argument("--cooldown_hold", action="store_true",
                        help="with --cooldown: keep training at the decayed lr instead of stopping after the decay")
    parser.add_argument("--cooldown_from", type=int, default=0,
                        help="with --cooldown: start the cooldown at this step (warmup and constant lr before it) "
                             "instead of at the step the run starts or resumes from")
    parser.add_argument("--resume", action="store_true",
                        help="continue this run (same flags) from RUN_DIR/last.pt: weights, optimizer, step, "
                             "clipper / homeostasis state, history and RNG, as if it had never stopped")
    parser.add_argument("--save_every", type=int, default=500,
                        help="write the full training state to RUN_DIR/last.pt every this many steps")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False,
                        help="bf16 autocast for forward passes (about 1.3x faster, half the activation memory)")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile the training forward (about 1.8x faster; ~10 s compile at start)")
    parser.add_argument("--init_from_ann", type=str, default="",
                        help="checkpoint of an ANN run: copy every tensor whose name and shape match (embedding, "
                             "positions, Q/K/V/proj/MLP weights, head) before training")
    parser.add_argument("--distill_from", type=str, default="",
                        help="checkpoint of a teacher run (its config.json alongside): add a distillation loss")
    parser.add_argument("--distill_alpha", type=float, default=0.5,
                        help="weight of the cross-entropy term; 1 - alpha weighs the KL to the teacher")
    parser.add_argument("--distill_temp", type=float, default=1.0, help="softmax temperature of the KL term")
    parser.add_argument("--distill_alpha_to", type=float, default=1.0,
                        help="with --distill_alpha_ramp: CE weight to move to (1 = pure cross-entropy, no output KL)")
    parser.add_argument("--distill_alpha_ramp", type=int, default=0,
                        help="steps over which the CE weight moves linearly from --distill_alpha to --distill_alpha_to")
    parser.add_argument("--distill_alpha_ramp_from", type=int, default=0,
                        help="step the CE-weight ramp starts at (default: where this run starts or resumes)")
    parser.add_argument("--attn_distill", type=float, default=0.0,
                        help="with --distill_from: weight of an attention-map distillation term, the KL between the "
                             "teacher's and the student's attention rows, head by head (0 = off)")
    parser.add_argument("--attn_distill_ramp", type=int, default=0,
                        help="ramp the attention weight up from 0 over this many steps (lets Adam adapt when the term "
                             "is added to a running training)")
    parser.add_argument("--attn_distill_start", type=float, default=0.0,
                        help="weight the attention-term ramp starts from (e.g. the previous --attn_distill)")
    parser.add_argument("--attn_distill_ramp_from", type=int, default=0,
                        help="step the attention-weight ramp starts at (default: where this run starts or resumes)")
    parser.add_argument("--attn_distill_kl", choices=("forward", "reverse"), default="forward",
                        help="forward: KL(teacher || student), mass-covering (flattens count attention); reverse: "
                             "KL(student || teacher), mode-seeking (concentrates on the teacher's peaks)")
    parser.add_argument("--attn_distill_blocks", type=str, default="0-11",
                        help="blocks the attention term covers, e.g. 2-10 or 2,4,6")
    parser.add_argument("--attn_distill_rows", type=int, default=64,
                        help="query rows sampled per micro-batch for the attention term (keeps memory small)")
    parser.add_argument("--attn_distill_min_pos", type=int, default=64,
                        help="sampled query rows come from positions >= this (enough context to look back on)")
    parser.add_argument("--qat", action="store_true",
                        help="spiking model: quantization-aware training with the hardware datapath of qat.py "
                             "(int8 weights, shift multiplies, fixed-point norms, gamma/beta folded into the LIF)")
    parser.add_argument("--init_from", type=str, default="",
                        help="state_dict (.pt, e.g. a run's best.pt) to start from; the model flags must match it")
    args = parser.parse_args()
    assert args.batch_size % args.grad_accum == 0, "batch_size must be divisible by grad_accum"

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmuls on Ampere+: faster, loss unchanged
    torch.backends.cudnn.allow_tf32 = True
    device = args.device or pick_device()
    data = CharDataset(args.data_path, args.val_fraction, args.test_fraction, args.max_vocab,
                       val_size=args.val_chars, test_size=args.test_chars)
    if getattr(data, "unk_fraction", None) is not None:
        print(f"vocabulary capped at {data.vocab_size}: {100 * data.unk_fraction:.3f}% of characters map to UNK")
    if args.prompt is None:
        args.prompt = "\n" if "\n" in data.stoi else " "
    print(f"device: {device}, vocab_size={data.vocab_size}, "
          f"train chars={len(data.splits['train']):,}, val chars={len(data.splits['val']):,}")

    corpus = os.path.basename(os.path.dirname(os.path.abspath(args.data_path)))
    prefix = "char" if corpus == "tinyshakespeare" else corpus  # e.g. text8_512d4L_T8
    if args.model == "ann":
        args.T, args.eval_T, args.homeostasis = 1, [1], 0.0
    run_name = f"{prefix}_{args.dim}d{args.depth}L_" + ("ann" if args.model == "ann" else f"T{args.T}")
    if args.run_suffix:
        run_name += f"_{args.run_suffix}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run_dir={run_dir}")

    config = {"model_type": args.model, "task": "char_lm", "vocab_size": data.vocab_size, "chars": data.chars}
    config.update({k: v for k, v in vars(args).items() if k not in ("device", "run_suffix")})
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if args.model == "ann":
        model = ANNTransformer(data.vocab_size, dim=args.dim, depth=args.depth, num_heads=args.num_heads,
                               max_len=args.block_size, mlp_hidden=args.mlp_hidden, dropout=args.dropout)
    else:
        model = SpikingTransformer(data.vocab_size, dim=args.dim, depth=args.depth, num_heads=args.num_heads,
                                   T=args.T, max_len=args.block_size, attn_lif=args.attn_lif,
                                   pow2_gamma=args.pow2_gamma, attn_proj=args.attn_proj,
                                   mlp_layers=args.mlp_layers, mlp_hidden=args.mlp_hidden, attn_norm=args.attn_norm,
                                   pos_embed=args.pos_embed, learnable_scale=args.learnable_scale,
                                   head_scale=args.head_scale, head_norm=args.head_norm,
                                   attn_norm_eps=args.attn_norm_eps)
    if args.model == "spike":
        model.multistep, model.checkpoint_blocks = args.multistep, args.checkpoint_blocks
        if args.surrogate_alpha_ramp > 0:
            use_alpha_tensor(device)   # alpha changes every step without recompiling the graph
        if args.dropout > 0:
            for blk in model.blocks:
                blk.drop.p = args.dropout
            print(f"spike dropout {args.dropout} on every block's attention and MLP outputs", flush=True)
        if args.attn_square_heads > 0:
            for blk in model.blocks:
                blk.attn.enable_square(args.attn_square_heads)
            print(f"squared Q.K counts in heads 0-{args.attn_square_heads - 1} of every block, blended in over "
                  f"{args.attn_square_ramp} steps", flush=True)
        if args.attn_threshold_heads > 0:
            for blk in model.blocks:
                blk.attn.enable_learned_threshold(args.attn_threshold_heads, args.attn_threshold_init)
                if args.attn_threshold_fixed:   # same cut and surrogate, theta frozen at its initial value
                    blk.attn.thr_raw.requires_grad_(False)
            print(f"learned count threshold in heads 0-{args.attn_threshold_heads - 1} of every block, init "
                  f"{args.attn_threshold_init}{' (fixed)' if args.attn_threshold_fixed else ''}, blended in over {args.attn_threshold_ramp} steps", flush=True)
        set_surrogate_alpha(args.surrogate_alpha)
        print(f"surrogate gradient alpha {args.surrogate_alpha:g}"
              + (f" -> {args.surrogate_alpha_to:g} (linear) over {args.surrogate_alpha_ramp} steps"
                 if args.surrogate_alpha_ramp > 0 else ""), flush=True)
    print(f"{type(model).__name__} params: {sum(p.numel() for p in model.parameters()):,}")
    if args.init_from and not args.resume:
        # weights only: the optimizer state is not saved, so Adam restarts its
        # statistics (use a short --warmup_steps when resuming)
        state = torch.load(args.init_from, map_location="cpu")
        own = model.state_dict()
        for k, v in state.items():  # a per-block scale seeds every head's scale (--head_scale)
            if k.endswith("log2_scale") and v.dim() == 0 and own[k].dim() == 1:
                state[k] = v.expand_as(own[k]).clone()
        model.load_state_dict(state)
        print(f"initialised weights from {args.init_from}")

    if args.qat:   # after the weights are loaded: the fixed-gamma neurons are chosen from the loaded gammas
        from qat import enable_qat
        enable_qat(model)

    if args.init_from_ann and not args.resume:
        # copy every ANN tensor whose name and shape match (embedding, positions, Q/K/V/proj and MLP
        # weights, head); the spiking model's L1 norms and the ANN's LayerNorms / Q/K/V biases have no match
        state = torch.load(args.init_from_ann, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        own = model.state_dict()
        copied = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(copied, strict=False)
        print(f"copied {len(copied)} tensors ({sum(v.numel() for v in copied.values()):,} of "
              f"{sum(v.numel() for v in own.values()):,} values) from {args.init_from_ann}; "
              f"not in the ANN: {len(own) - len(copied)} tensors; ANN-only: {len(state) - len(copied)}", flush=True)
    teacher = None
    if args.distill_from:
        t_config = json.load(open(os.path.join(os.path.dirname(args.distill_from), "config.json")))
        assert t_config["chars"] == data.chars, "teacher vocabulary differs"
        teacher = model_from_config(t_config, t_config["block_size"])
        t_state = torch.load(args.distill_from, map_location="cpu", weights_only=False)
        teacher.load_state_dict(t_state.get("model", t_state))
        teacher.to(device).eval().requires_grad_(False)
        print(f"distilling from {args.distill_from} ({type(teacher).__name__}): "
              f"loss = {args.distill_alpha} * CE + {1 - args.distill_alpha:g} * T^2 * KL, T = {args.distill_temp}",
              flush=True)
    history = train(model, data, args, device, run_dir, teacher)
    save_run(run_dir, history)
    print(f"best val bpc: {min(history['bpc']):.3f}")


if __name__ == "__main__":
    main()
