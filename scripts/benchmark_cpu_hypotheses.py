"""Seven follow-up CPU throughput hypotheses, anchored to the best combined profile."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from model.config import CONFIG
from model.lumen import RoninLumen
from train.loss import perdida_total
from train.transformer_baseline import SmallTransformerLM

# Every candidate inherits the winning profile from benchmark #2:
# intra-op=2, inter-op=2, MKLDNN off, foreach off, CPU affinity enabled.
PROFILES = {
    "control_combined": {},
    "compile_default": {"compile_mode": "default"},
    "compile_reduce_overhead": {"compile_mode": "reduce-overhead"},
    "cpu_bfloat16": {"autocast_bf16": True},
    "fused_adamw": {"fused_adamw": True},
    "zero_grad_reuse": {"set_to_none": False},
    "flush_denormal": {"flush_denormal": True},
}


def make_model(name, seq_len):
    if name == "lumen":
        return RoninLumen().cpu()
    return SmallTransformerLM(CONFIG.vocab_size, seq_len).cpu()


def run_one(name, profile, tokens, steps, warmup, seed):
    torch.manual_seed(seed)
    model = make_model(name, tokens.shape[1])
    if profile.get("compile_mode"):
        model = torch.compile(model, mode=profile["compile_mode"])
    kwargs = {"lr": 1e-4, "weight_decay": CONFIG.weight_decay, "foreach": False}
    if profile.get("fused_adamw"):
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    use_autocast = profile.get("autocast_bf16", False)
    set_to_none = profile.get("set_to_none", True)

    def update():
        optimizer.zero_grad(set_to_none=set_to_none)
        amp = torch.autocast("cpu", dtype=torch.bfloat16) if use_autocast else nullcontext()
        with amp:
            if name == "lumen":
                output = model(tokens, reset_state=True, collect_diagnostics=False)
                loss, _ = perdida_total(output, tokens, model)
            else:
                logits = model(tokens)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    ignore_index=0,
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        optimizer.step()
        return float(loss.detach())

    started_warmup = time.perf_counter()
    last_loss = None
    for _ in range(warmup):
        last_loss = update()
    warmup_seconds = time.perf_counter() - started_warmup

    started = time.perf_counter()
    for _ in range(steps):
        last_loss = update()
    elapsed = time.perf_counter() - started
    return {
        "status": "ok",
        "elapsed_seconds": round(elapsed, 4),
        "warmup_and_compile_seconds": round(warmup_seconds, 4),
        "steps_per_second": steps / elapsed,
        "tokens_per_second": steps * tokens.numel() / elapsed,
        "last_loss": last_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Seven matched CPU throughput hypotheses.")
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.threads, args.batch, args.seq_len, args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("threads, batch, seq-len, steps and repeats must be positive; warmup cannot be negative")

    profile = PROFILES[args.profile]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = False
    affinity_cpus = None
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        affinity_cpus = available[:args.threads]
        os.sched_setaffinity(0, set(affinity_cpus))
    if profile.get("flush_denormal"):
        torch.set_flush_denormal(True)

    results = {}
    for model_name in ("lumen", "transformer"):
        samples = []
        errors = []
        for repeat in range(args.repeats):
            torch.manual_seed(args.seed)
            tokens = torch.randint(4, CONFIG.vocab_size, (args.batch, args.seq_len), dtype=torch.long)
            try:
                samples.append(run_one(
                    model_name, profile, tokens, args.steps, args.warmup, args.seed + repeat
                ))
            except Exception as exc:  # An unsupported optimization is data, not a failed matrix.
                errors.append({"repeat": repeat, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                del tokens
        successful = [sample for sample in samples if sample["status"] == "ok"]
        results[model_name] = {
            "runs": samples,
            "errors": errors,
            "successful_repeats": len(successful),
            "median_steps_per_second": (
                round(statistics.median(x["steps_per_second"] for x in successful), 4)
                if successful else None
            ),
            "median_tokens_per_second": (
                round(statistics.median(x["tokens_per_second"] for x in successful), 2)
                if successful else None
            ),
        }

    report = {
        "profile": args.profile,
        "hypothesis": profile,
        "base_profile": {
            "intraop_threads": args.threads, "interop_threads": 2,
            "mkldnn": False, "foreach": False, "cpu_affinity": affinity_cpus,
        },
        "device": "cpu",
        "batch": args.batch,
        "sequence_length": args.seq_len,
        "warmup_steps": args.warmup,
        "measured_steps_per_repeat": args.steps,
        "repeats": args.repeats,
        "includes_data_loading": False,
        "environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_DYNAMIC",
            "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY"
        )},
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
