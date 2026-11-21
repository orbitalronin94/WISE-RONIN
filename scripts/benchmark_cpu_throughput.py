"""Repeated matched CPU throughput benchmark for seven 2-thread runtime profiles."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from model.config import CONFIG
from model.lumen import RoninLumen
from train.loss import perdida_total
from train.transformer_baseline import SmallTransformerLM

PROFILES = {
    "baseline": {"interop": 1, "mkldnn": True, "foreach": True, "affinity": False},
    "interop2": {"interop": 2, "mkldnn": True, "foreach": True, "affinity": False},
    "mkldnn_off": {"interop": 1, "mkldnn": False, "foreach": True, "affinity": False},
    "foreach_off": {"interop": 1, "mkldnn": True, "foreach": False, "affinity": False},
    "omp_affinity": {"interop": 1, "mkldnn": True, "foreach": True, "affinity": False},
    "cpu_affinity": {"interop": 1, "mkldnn": True, "foreach": True, "affinity": True},
    "combined": {"interop": 2, "mkldnn": False, "foreach": False, "affinity": True},
}


def make_model(name, seq_len):
    if name == "lumen":
        return RoninLumen().cpu()
    return SmallTransformerLM(CONFIG.vocab_size, seq_len).cpu()


def run_one(name, model, tokens, steps, warmup, grad_clip, foreach):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=CONFIG.weight_decay,
        foreach=foreach,
    )

    def update():
        optimizer.zero_grad(set_to_none=True)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=foreach)
        optimizer.step()

    for _ in range(warmup):
        update()
    started = time.perf_counter()
    for _ in range(steps):
        update()
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": round(elapsed, 4),
        "steps_per_second": steps / elapsed,
        "tokens_per_second": steps * tokens.numel() / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Repeated CPU compute-only throughput benchmark.")
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.threads, args.batch, args.seq_len, args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("threads, batch, seq-len, steps and repeats must be positive; warmup cannot be negative")

    profile = PROFILES[args.profile]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(profile["interop"])
    torch.backends.mkldnn.enabled = profile["mkldnn"]
    affinity_cpus = None
    if profile["affinity"] and hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        affinity_cpus = available[:args.threads]
        os.sched_setaffinity(0, set(affinity_cpus))

    results = {}
    for model_name in ("lumen", "transformer"):
        samples = []
        for repeat in range(args.repeats):
            torch.manual_seed(args.seed)
            tokens = torch.randint(4, CONFIG.vocab_size, (args.batch, args.seq_len), dtype=torch.long)
            model = make_model(model_name, args.seq_len)
            samples.append(run_one(
                model_name, model, tokens, args.steps, args.warmup, 1.0, profile["foreach"]
            ))
            del model, tokens
        results[model_name] = {
            "runs": samples,
            "median_steps_per_second": round(statistics.median(x["steps_per_second"] for x in samples), 4),
            "median_tokens_per_second": round(statistics.median(x["tokens_per_second"] for x in samples), 2),
        }

    report = {
        "profile": args.profile,
        "settings": profile,
        "device": "cpu",
        "intraop_threads": args.threads,
        "interop_threads": profile["interop"],
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
        "affinity_cpus": affinity_cpus,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
