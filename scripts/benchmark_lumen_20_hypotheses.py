"""Matched 20-way CPU throughput tournament for Lumen only.

Each workflow matrix job runs the same 100-step control and one candidate profile
on the same GitHub-hosted runner. This is a compute-path microbenchmark:
synthetic fixed tokens, no data loading, no checkpoint or repository mutation.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

from model.config import CONFIG
from model.lumen import RoninLumen
from train.loss import perdida_total

PROFILES = {
    "control": {},
    "threads_1": {"threads": 1},
    "threads_4": {"threads": 4},
    "interop_2": {"interop": 2},
    "mkldnn_off": {"mkldnn": False},
    "foreach_off": {"foreach": False},
    "affinity_2": {"affinity": True},
    "fused_adamw": {"fused": True},
    "cpu_bfloat16": {"bf16": True},
    "flush_denormal": {"flush_denormal": True},
    "zero_grad_reuse": {"set_to_none": False},
    "compile_default": {"compile": "default"},
    "compile_reduce_overhead": {"compile": "reduce-overhead"},
    "matmul_high": {"matmul_precision": "high"},
    "threads_1_mkldnn_off": {"threads": 1, "mkldnn": False},
    "threads_4_mkldnn_on": {"threads": 4, "mkldnn": True},
    "threads_2_interop_2_affinity": {"interop": 2, "affinity": True},
    "fused_adamw_bf16": {"fused": True, "bf16": True},
    "compile_default_bf16": {"compile": "default", "bf16": True},
    "compile_reduce_overhead_threads_1": {"compile": "reduce-overhead", "threads": 1},
}

def configure(profile):
    threads = int(profile.get("threads", 2))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(int(profile.get("interop", 1)))
    torch.backends.mkldnn.enabled = bool(profile.get("mkldnn", True))
    if "matmul_precision" in profile:
        torch.set_float32_matmul_precision(profile["matmul_precision"])
    if profile.get("flush_denormal"):
        torch.set_flush_denormal(True)
    affinity = None
    if profile.get("affinity") and hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        affinity = available[:threads]
        os.sched_setaffinity(0, set(affinity))
    return threads, affinity

def execute(name, steps, warmup, seed, batch, seq_len, output_path):
    profile = PROFILES[name]
    result = {
        "profile": name, "settings": profile, "device": "cpu",
        "batch": batch, "sequence_length": seq_len,
        "warmup_steps": warmup, "measured_steps": steps,
        "includes_data_loading": False,
    }
    try:
        threads, affinity = configure(profile)
        result["intraop_threads"] = threads
        result["interop_threads"] = int(profile.get("interop", 1))
        result["affinity_cpus"] = affinity

        torch.manual_seed(seed)
        tokens = torch.randint(4, CONFIG.vocab_size, (batch, seq_len), dtype=torch.long)
        model = RoninLumen().cpu()
        if profile.get("compile"):
            model = torch.compile(model, mode=profile["compile"])
        optimizer_kwargs = {
            "lr": 1e-4, "weight_decay": CONFIG.weight_decay,
            "foreach": bool(profile.get("foreach", True)),
        }
        if profile.get("fused"):
            optimizer_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        set_to_none = bool(profile.get("set_to_none", True))
        use_bf16 = bool(profile.get("bf16", False))

        def update():
            optimizer.zero_grad(set_to_none=set_to_none)
            amp = torch.autocast("cpu", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
            with amp:
                out = model(tokens, reset_state=True, collect_diagnostics=False)
                loss, _ = perdida_total(out, tokens, model)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            optimizer.step()
            return float(loss.detach())

        warmup_started = time.perf_counter()
        for _ in range(warmup):
            last_loss = update()
        warmup_elapsed = time.perf_counter() - warmup_started
        if hasattr(torch, "_dynamo") and profile.get("compile"):
            torch._dynamo.reset()
        started = time.perf_counter()
        losses = []
        for _ in range(steps):
            last_loss = update()
            losses.append(last_loss)
        elapsed = time.perf_counter() - started
        result.update({
            "status": "ok",
            "elapsed_seconds": round(elapsed, 5),
            "warmup_compile_seconds": round(warmup_elapsed, 5),
            "steps_per_second": round(steps / elapsed, 5),
            "tokens_per_second": round(steps * batch * seq_len / elapsed, 2),
            "first_loss": losses[0], "last_loss": last_loss,
            "loss_mean": statistics.mean(losses),
            "loss_stdev": statistics.pstdev(losses),
            "loss_finite": all(torch.isfinite(torch.tensor(x)) for x in losses),
        })
    except Exception as exc:
        result.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-5000:],
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    execute(args.profile, args.steps, args.warmup, args.seed, args.batch, args.seq_len, args.output)

if __name__ == "__main__":
    main()
