"""Matched end-to-end CPU benchmark for LUMEN hot paths."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from torch.nn.utils import clip_grad_norm_
from model.config import CONFIG
from model.lumen import RoninLumen
from train.dataset import load_jsonl, load_token_cache, pack_token_cache, collate_cached_indices
from train.loss import perdida_total

def configure(p):
    torch.set_num_threads(int(p["threads"]))
    torch.set_num_interop_threads(int(p["interop_threads"]))
    torch.backends.mkldnn.enabled = bool(p["mkldnn_enabled"])
    affinity = None
    if p["cpu_affinity"] and hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        affinity = available[:int(p["threads"])]
        os.sched_setaffinity(0, set(affinity))
    return affinity

def load_batch(a):
    rows, data_hash = load_jsonl(a.data)
    tokenizer_hash = hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    encoded = load_token_cache(a.token_cache, data_hash, tokenizer_hash, a.max_len, len(rows))
    packed, lengths = pack_token_cache(encoded, a.max_len)
    import random
    rng = random.Random(42)
    valid = [i for i, n in enumerate(lengths.tolist()) if n >= 2]
    return collate_cached_indices(packed, lengths, rng.choices(valid, k=a.batch))

def make_model(ckpt, p):
    model = RoninLumen().cpu()
    model.load_state_dict(ckpt["model"], strict=True)
    for layer in model.delta_layers:
        layer.pre_normalize_sequence = bool(p["delta_pre_normalize"])
    for organ in model.homeo:
        organ.fused_scatter_add = bool(p["homeostasis_fused_scatter"])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=CONFIG.weight_decay, foreach=bool(p["foreach"]))
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        for group in opt.param_groups:
            group["foreach"] = bool(p["foreach"])
    return model, opt

def update(model, opt, tokens, collect_diagnostics=False):
    opt.zero_grad(set_to_none=True)
    output = model(tokens, reset_state=True, collect_diagnostics=collect_diagnostics)
    loss, _ = perdida_total(output, tokens, model, use_jepa=True, use_rep=True)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite benchmark loss")
    loss.backward()
    clip_grad_norm_(model.parameters(), 0.5, foreach=opt.param_groups[0]["foreach"])
    opt.step()
    return float(loss.detach())

def trial(ckpt, tokens, p, warmup, steps, repeats, collect_diagnostics=False):
    samples = []
    for repeat in range(repeats):
        torch.manual_seed(4200 + repeat)
        model, opt = make_model(ckpt, p)
        for _ in range(warmup):
            update(model, opt, tokens, collect_diagnostics)
        started = time.perf_counter()
        losses = [update(model, opt, tokens, collect_diagnostics) for _ in range(steps)]
        elapsed = time.perf_counter() - started
        samples.append({"elapsed_seconds": elapsed, "steps_per_second": steps / elapsed,
            "tokens_per_second": steps * int(tokens.ne(0).sum()) / elapsed,
            "mean_loss": statistics.mean(losses), "last_loss": losses[-1]})
        del model, opt
    return {"runs": samples,
        "median_steps_per_second": statistics.median(x["steps_per_second"] for x in samples),
        "median_tokens_per_second": statistics.median(x["tokens_per_second"] for x in samples),
        "median_mean_loss": statistics.median(x["mean_loss"] for x in samples)}

def worker(a):
    p = json.loads(a.worker_profile)
    affinity = configure(p)
    torch.manual_seed(4200)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if int(ckpt.get("step", -1)) != 3000:
        raise ValueError(f"Expected step-3000 checkpoint, got {ckpt.get('step')}")
    tokens = load_batch(a)
    result = {"name": a.name, "profile": p, "affinity_cpus": affinity, "checkpoint_step": 3000,
        "full_objective": trial(ckpt, tokens, p, a.warmup, a.steps, a.repeats)}
    result["diagnostic_collection_cost"] = trial(ckpt, tokens, p, 2, 8, 1, True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "data", "tokenizer", "token_cache", "output"):
        p.add_argument("--" + name.replace("_", "-"))
    p.add_argument("--name", default="candidate")
    p.add_argument("--worker-profile", required=True, help=argparse.SUPPRESS)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    for field in ("checkpoint", "data", "tokenizer", "token_cache", "output"):
        if not getattr(a, field):
            p.error("--" + field.replace("_", "-") + " is required")
    worker(a)

if __name__ == "__main__":
    main()
