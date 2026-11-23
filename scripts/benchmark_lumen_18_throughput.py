"""Matched 18-way real-training CPU throughput tournament for LUMEN.

Each candidate is compared with the committed main CPU profile on the same
runner, from the same frozen step-170700 checkpoint and exact data/cache.
Candidates may change execution/runtime settings only: no organ, objective,
plasticity, memory, or model-architecture switch is permitted.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Only execution-level knobs are allowed in this tournament.
CANDIDATES = {
    "01_threads_2": {"profile": {"threads": 2}},
    "02_threads_6": {"profile": {"threads": 6}},
    "03_threads_8": {"profile": {"threads": 8}},
    "04_interop_1": {"profile": {"interop_threads": 1}},
    "05_interop_4": {"profile": {"interop_threads": 4}},
    "06_mkldnn_on": {"profile": {"mkldnn_enabled": True}},
    "07_foreach_on": {"profile": {"foreach": True}},
    "08_affinity_off": {"profile": {"cpu_affinity": False}},
    "09_threads_2_interop_2": {"profile": {"threads": 2, "interop_threads": 2}},
    "10_threads_2_mkldnn_on": {"profile": {"threads": 2, "mkldnn_enabled": True}},
    "11_threads_2_foreach_on": {"profile": {"threads": 2, "foreach": True}},
    "12_threads_2_affinity": {"profile": {"threads": 2, "cpu_affinity": True}},
    "13_threads_6_interop_2": {"profile": {"threads": 6, "interop_threads": 2}},
    "14_interop_1_affinity_off": {"profile": {"interop_threads": 1, "cpu_affinity": False}},
    "15_mkldnn_foreach_on": {"profile": {"mkldnn_enabled": True, "foreach": True}},
    "16_compile_default": {"profile": {}, "compile": "default"},
    "17_compile_reduce_overhead": {"profile": {}, "compile": "reduce-overhead"},
    "18_omp_active_compact": {
        "profile": {},
        "env": {
            "OMP_WAIT_POLICY": "ACTIVE",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "KMP_BLOCKTIME": "0",
        },
    },
}
ALLOWED_PROFILE_KEYS = {
    "threads", "interop_threads", "mkldnn_enabled", "foreach", "cpu_affinity",
    "profile_id", "benchmark_metadata",
}
PARENT_STEP = 170700
TRAIN_STEPS = 100
LOSS_TOLERANCE = 0.01
DEAD_TOLERANCE_PP = 0.5
MIN_SPEEDUP_PERCENT = 3.0


def deep_merge(base: dict, delta: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def assert_execution_only(profile: dict) -> None:
    unexpected = set(profile) - ALLOWED_PROFILE_KEYS
    if unexpected:
        raise ValueError(f"non-runtime profile keys are forbidden: {sorted(unexpected)}")
    if any(key in profile for key in ("organs", "plasticity", "sleep_downscale",
                                      "sleep_interval_steps", "replay_compression",
                                      "delta_parallel_refinements", "delta_pre_normalize",
                                      "delta_reuse_write_update", "delta_always_apply_eligibility",
                                      "delta_trace_norm_diagnostics_on_demand",
                                      "delta_share_causal_masks", "homeostasis_fused_scatter")):
        raise ValueError("organ/architecture behavior cannot be changed in a speed-only tournament")


def run_training(root: Path, label: str, profile: dict, candidate: dict,
                 args, output: Path) -> dict:
    assert_execution_only(profile)
    profile_path = output / f"{label}.runtime_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    log_path = output / f"{label}.log"
    ckpt_dir = output / f"{label}-checkpoints"
    command = [
        sys.executable, "-m", "train.train",
        "--steps", str(TRAIN_STEPS),
        "--resume", str(args.checkpoint.resolve()),
        "--phase", "1", "--pilot-all-rows",
        "--data", str((args.prepared_root / "data/spanish-v2-plus-personality-250k.jsonl").resolve()),
        "--tokenizer", str((args.prepared_root / "artifacts/spanish-v2/tokenizer.json").resolve()),
        "--token-cache", str((args.prepared_root / "data/spanish-v2-plus-personality-token-cache.pt").resolve()),
        "--batch", "8", "--max-len", "128", "--seed", "42",
        "--runtime-profile", str(profile_path.resolve()),
        "--lr", "3e-5", "--warmup", "100", "--total-steps", "170800",
        "--scheduler-transition", "--restart-scheduler-transition",
        "--transition-start", str(PARENT_STEP), "--transition-end", str(PARENT_STEP + TRAIN_STEPS),
        "--warmup-transition", "10", "--transition-plateau-steps", "0",
        "--transition-min-lr-ratio", "0.5",
        "--log-every", str(TRAIN_STEPS), "--checkpoint-every", "1000",
        "--recovery-checkpoint-every", "0", "--parameter-check-every", "100",
        "--threads", "4", "--interop-threads", "2",
        "--ckpt-dir", str(ckpt_dir.resolve()),
    ]
    if candidate.get("compile"):
        command.extend(["--compile-mode", candidate["compile"]])

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(root.resolve()),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "OMP_WAIT_POLICY": "PASSIVE",
        "PYTHONUNBUFFERED": "1",
    })
    env.update(candidate.get("env", {}))
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=root, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    process_seconds = time.perf_counter() - started
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    profile_rows = re.findall(
        r"profile_100:.*?total=([0-9.]+)s", raw
    )
    final = re.findall(
        r"step=(\d+) loss=([0-9.eE+-]+).*?dead=([0-9.eE+-]+)%.*?tokens_seen=(\d+)",
        raw,
    )
    layer_line = re.findall(r"layer_diagnostics step=(\d+) (.+)", raw)
    dead_line = re.findall(r"effective_dead_by_layer step=(\d+) values=([0-9.,na-]+)", raw)
    nonfinite = bool(re.search(
        r"NONFINITE_DIAGNOSTICS|NONFINITE_PARAMETER|NONFINITE_BUFFER|NONFINITE_OPTIMIZER_STATE",
        raw,
    ))
    result = {
        "label": label,
        "exit_code": proc.returncode,
        "process_elapsed_seconds": round(process_seconds, 4),
        "profiled_windows": len(profile_rows),
        "profiled_seconds": round(sum(map(float, profile_rows)), 5) if profile_rows else None,
        "steps_per_second": round(TRAIN_STEPS / sum(map(float, profile_rows)), 6) if profile_rows else None,
        "nonfinite_diagnostics": nonfinite,
        "final_log_records": len(final),
        "layer_diagnostic_records": len(layer_line),
        "effective_dead_records": len(dead_line),
        "log_file": log_path.name,
        "profile_file": profile_path.name,
    }
    expected_final = PARENT_STEP + TRAIN_STEPS
    if final:
        step, loss, dead, tokens = final[-1]
        result.update({
            "final_step": int(step),
            "final_loss": float(loss),
            "final_dead_percent": float(dead),
            "final_tokens_seen": int(tokens),
        })
    if dead_line:
        result["dead_by_layer"] = dead_line[-1][1]
    if layer_line:
        # Keep the full four-layer activity/write diagnostic line as evidence.
        result["layer_diagnostics"] = layer_line[-1][1]
    if (proc.returncode == 0 and len(profile_rows) == 1 and len(final) >= 1
            and int(final[-1][0]) == expected_final and len(layer_line) >= 1
            and len(dead_line) >= 1 and not nonfinite
            and math.isfinite(float(final[-1][1]))
            and math.isfinite(float(final[-1][2]))):
        result["status"] = "ok"
    else:
        result["status"] = "error"
        result["reason"] = (
            f"exit={proc.returncode}; profile_windows={len(profile_rows)}/1; "
            f"final_step={final[-1][0] if final else 'missing'}; "
            f"layer_diagnostics={len(layer_line)}; dead_records={len(dead_line)}; "
            f"nonfinite={nonfinite}"
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=tuple(CANDIDATES))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    candidate = CANDIDATES[args.candidate]
    baseline_profile = json.loads((args.repo_root / "config/lumen_cpu_profile.json").read_text())
    candidate_profile = deep_merge(baseline_profile, candidate["profile"])
    # Both profiles retain the exact same Delta fastpaths and every LUMEN organ.
    assert_execution_only(candidate_profile)
    # Alternate order to reduce systematic page-cache and thermal bias.
    if int(args.candidate[:2]) % 2 == 0:
        candidate_result = run_training(args.repo_root, "candidate", candidate_profile,
                                        candidate, args, args.output)
        control_result = run_training(args.repo_root, "control", baseline_profile,
                                       {}, args, args.output)
    else:
        control_result = run_training(args.repo_root, "control", baseline_profile,
                                      {}, args, args.output)
        candidate_result = run_training(args.repo_root, "candidate", candidate_profile,
                                        candidate, args, args.output)

    reasons = []
    if control_result.get("status") != "ok":
        reasons.append("control run incomplete or unhealthy")
    if candidate_result.get("status") != "ok":
        reasons.append("candidate run incomplete or unhealthy")
    speedup = loss_delta = dead_delta = None
    if not reasons:
        speedup = 100.0 * (
            candidate_result["steps_per_second"] / control_result["steps_per_second"] - 1.0
        )
        loss_delta = candidate_result["final_loss"] - control_result["final_loss"]
        dead_delta = candidate_result["final_dead_percent"] - control_result["final_dead_percent"]
        if speedup < MIN_SPEEDUP_PERCENT:
            reasons.append(f"speedup {speedup:.3f}% below {MIN_SPEEDUP_PERCENT:.1f}% gate")
        if loss_delta > LOSS_TOLERANCE:
            reasons.append(f"final loss regression {loss_delta:.6f} exceeds {LOSS_TOLERANCE}")
        if dead_delta > DEAD_TOLERANCE_PP:
            reasons.append(f"dead activity regression {dead_delta:.3f} pp exceeds {DEAD_TOLERANCE_PP}")

    report = {
        "candidate": args.candidate,
        "status": "completed" if not (
            control_result.get("status") != "ok" or candidate_result.get("status") != "ok"
        ) else "incomplete",
        "decision": "APPROVED_FOR_LONGER_VALIDATION" if not reasons else "PENDING",
        "reasons": reasons,
        "control": control_result,
        "candidate_result": candidate_result,
        "speedup_percent": round(speedup, 5) if speedup is not None else None,
        "final_loss_delta": round(loss_delta, 8) if loss_delta is not None else None,
        "dead_delta_percentage_points": round(dead_delta, 5) if dead_delta is not None else None,
        "profile_delta": candidate["profile"],
        "compile_mode": candidate.get("compile"),
        "environment_delta": candidate.get("env", {}),
        "contract": {
            "parent_checkpoint_step": PARENT_STEP,
            "steps_per_run": TRAIN_STEPS,
            "same_runner_sequential_pair": True,
            "same_source_and_training_objective": True,
            "same_seed_rng_parent_corpus_tokenizer_and_token_cache": True,
            "same_batch_8_and_sequence_length_128": True,
            "all_organs_and_delta_fastpaths_preserved": True,
            "sleep_not_disabled": True,
            "measurement": "trainer profile_100 total wall time; includes sampling/collation, forward, loss, backward, optimizer and normal diagnostics; excludes model/data initialization and checkpoint write",
            "promotion_gate": f">={MIN_SPEEDUP_PERCENT}% faster; final loss regression <= {LOSS_TOLERANCE}; dead regression <= {DEAD_TOLERANCE_PP} percentage points; finite complete run and four-layer diagnostics",
            "holdout_used": False,
        },
    }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
