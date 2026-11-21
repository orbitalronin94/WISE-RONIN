"""Profile the real LUMEN training pipeline against the frozen step-2000 pilot inputs.

This is a diagnostic benchmark, not a training run: every profile reloads the same
parent checkpoint and samples only from the frozen training cache. No holdout is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.config import CONFIG
from model.lumen import RoninLumen
from train.dataset import load_jsonl, load_token_cache, pack_token_cache, collate_cached_indices
from train.loss import perdida_total


def configure_runtime(threads: int = 2):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = False
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        affinity = available[:threads]
        os.sched_setaffinity(0, set(affinity))
    return affinity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=(
        "eager_reference", "no_health_snapshots", "compile_default",
        "no_aux_losses", "stage_profile",
    ), required=True)
    parser.add_argument("--parent", default="checkpoints/continuation/lumen_step_2000_continuation.pt")
    parser.add_argument("--data", default="data/pilot_train_19k.jsonl")
    parser.add_argument("--tokenizer", default="data/pilot_tokenizer/tokenizer.json")
    parser.add_argument("--token-cache", default="data/pilot_tokenized_128.pt")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.threads) < 1:
        parser.error("warmup, steps and threads must be positive")

    affinity = configure_runtime(args.threads)
    rows, data_hash = load_jsonl(args.data)
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    encoded = load_token_cache(
        args.token_cache, data_hash, tokenizer_hash, 128, len(rows)
    )
    packed, lengths = pack_token_cache(encoded, 128)
    row_indices = {id(row): i for i, row in enumerate(rows)}

    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    if parent.get("architecture_id") != "ronin_lumen_v3_1_residual_homeostasis":
        raise ValueError("Benchmark requires the frozen V3.1 LUMEN parent")
    if parent.get("data_sha256") != data_hash:
        raise ValueError("Parent/data hash mismatch")
    if parent.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Parent/tokenizer hash mismatch")
    if int(parent.get("step", -1)) != 2000:
        raise ValueError("Expected exact step-2000 parent")

    model = RoninLumen().cpu()
    model.load_state_dict(parent["model"])
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=CONFIG.weight_decay, foreach=False
    )
    optimizer.load_state_dict(parent["optimizer"])
    for group in optimizer.param_groups:
        group["foreach"] = False

    sampler = random.Random(42 + 2000)
    rng_state = parent.get("rng", {}).get("local_sampler")
    if rng_state is not None:
        sampler.setstate(rng_state)

    compiled = torch.compile(model, mode="default") if args.profile == "compile_default" else None
    compile_seconds = 0.0
    compile_pending = compiled is not None

    stages = {key: [] for key in ("batch", "forward", "loss", "backward", "optimizer", "step_total")}
    diagnostic_steps = 0
    total_tokens = 0
    measured_tokens = 0
    last_loss = None
    total_iterations = args.warmup + args.steps

    for iteration in range(total_iterations):
        step_start = time.perf_counter()
        collect = (iteration == 0 or iteration % 100 == 0)
        t = time.perf_counter()
        chosen = sampler.choices(rows, k=8)
        tokens = collate_cached_indices(
            packed, lengths, [row_indices[id(row)] for row in chosen]
        )
        batch_seconds = time.perf_counter() - t
        total_tokens += int(tokens.ne(0).sum())

        t = time.perf_counter()
        forward_model = model
        if compiled is not None and not collect:
            forward_model = compiled
        output = forward_model(tokens, reset_state=True, collect_diagnostics=collect)
        forward_seconds = time.perf_counter() - t
        if collect:
            diagnostic_steps += 1

        # The production trainer retains per-unit health windows every step.
        if args.profile != "no_health_snapshots":
            for organ in model.homeo:
                if organ.last_gate_activity_per_unit is not None:
                    _snapshot = organ.last_gate_activity_per_unit.detach().cpu()

        t = time.perf_counter()
        loss, _ = perdida_total(
            output, tokens, model,
            use_jepa=args.profile != "no_aux_losses",
            use_rep=args.profile != "no_aux_losses",
            use_free_energy=False, beta=0.01, pad_id=0,
        )
        # Production trainer checks target-loss thresholds each update.
        last_loss = float(loss.detach())
        loss_seconds = time.perf_counter() - t

        optimizer.zero_grad(set_to_none=True)
        t = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - t
        if compile_pending:
            # torch.compile builds the training graph lazily; include both forward
            # and backward compilation in startup cost, not steady-state throughput.
            compile_seconds = forward_seconds + backward_seconds
            compile_pending = False

        t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, foreach=False)
        optimizer.step()
        optimizer_seconds = time.perf_counter() - t

        elapsed = time.perf_counter() - step_start
        if iteration >= args.warmup:
            measured_tokens += int(tokens.ne(0).sum())
            stages["batch"].append(batch_seconds)
            stages["forward"].append(forward_seconds)
            stages["loss"].append(loss_seconds)
            stages["backward"].append(backward_seconds)
            stages["optimizer"].append(optimizer_seconds)
            stages["step_total"].append(elapsed)

    # Measure one representative checkpoint serialization separately.
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "profile.pt"
        checkpoint_started = time.perf_counter()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        checkpoint_bytes = checkpoint_path.stat().st_size

    medians = {key: round(statistics.median(values), 5) for key, values in stages.items()}
    total_median = medians["step_total"]
    report = {
        "profile": args.profile,
        "parent_step": int(parent["step"]),
        "parent_sha256": hashlib.sha256(Path(args.parent).read_bytes()).hexdigest(),
        "data_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "settings": {
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "mkldnn": torch.backends.mkldnn.enabled,
            "adamw_foreach": False,
            "batch": 8, "max_len": 128,
            "warmup": args.warmup, "measured_steps": args.steps,
            "compile_seconds": round(compile_seconds, 3),
            "diagnostic_forward_calls": diagnostic_steps,
            "health_snapshots_enabled": args.profile != "no_health_snapshots",
            "auxiliary_losses_enabled": args.profile != "no_aux_losses",
            "affinity_cpus": affinity,
        },
        "median_stage_seconds_per_step": medians,
        "median_steps_per_second": round(1.0 / total_median, 4),
        "median_nonpad_tokens_per_second": round(
            measured_tokens / args.steps / total_median, 2
        ),
        "checkpoint_serialization_seconds": round(checkpoint_seconds, 4),
        "checkpoint_bytes": checkpoint_bytes,
        "last_loss": round(last_loss, 6),
        "stage_samples": stages,
        "includes_real_batch_sampling": True,
        "includes_checkpoint_in_step_rate": False,
        "reads_holdout": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
