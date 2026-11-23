"""Run one 100-step, same-runner comparison: main control vs one LUMEN candidate."""
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

def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def load_main_profile(root: Path) -> dict:
    """Use main's committed runtime profile verbatim as the paired control."""
    path = root / "config/lumen_cpu_profile.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"main runtime profile must be an object: {path}")
    return profile


def candidate_delta(name: str) -> dict:
    candidates = {
        "01_tagging_capture": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True
                }
        },
        "02_intrinsic_threshold": {
                "plasticity": {
                        "mode": "custom",
                        "intrinsic_threshold_enabled": True
                }
        },
        "03_simulated_pruning": {
                "plasticity": {
                        "mode": "custom",
                        "simulated_pruning_enabled": True
                }
        },
        "04_tag_protected_ltd": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True,
                        "ltd_enabled": True
                }
        },
        "05_one_shot_memory": {
                "plasticity": {
                        "mode": "custom",
                        "one_shot_enabled": True
                }
        },
        "06_homeostatic_scaling": {
                "organs": {
                        "homeostasis": {
                                "synaptic_scaling": True
                        }
                }
        },
        "07_prioritized_replay": {
                "replay_compression": 0.5,
                "sleep_interval_steps": 50
        },
        "08_sleep_downscaling": {
                "sleep_downscale": 0.99,
                "sleep_interval_steps": 50
        },
        "09_tag_plus_intrinsic": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True,
                        "intrinsic_threshold_enabled": True
                }
        },
        "10_tag_plus_pruning": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True,
                        "simulated_pruning_enabled": True
                }
        },
        "11_tag_plus_one_shot": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True,
                        "one_shot_enabled": True
                }
        },
        "12_tag_plus_scaling": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True
                },
                "organs": {
                        "homeostasis": {
                                "synaptic_scaling": True
                        }
                }
        },
        "13_tag_plus_replay": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True
                },
                "replay_compression": 0.5,
                "sleep_interval_steps": 50
        },
        "14_tag_plus_downscale": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True
                },
                "sleep_downscale": 0.99,
                "sleep_interval_steps": 50
        },
        "15_intrinsic_plus_pruning": {
                "plasticity": {
                        "mode": "custom",
                        "intrinsic_threshold_enabled": True,
                        "simulated_pruning_enabled": True
                }
        },
        "16_one_shot_plus_scaling": {
                "plasticity": {
                        "mode": "custom",
                        "one_shot_enabled": True
                },
                "organs": {
                        "homeostasis": {
                                "synaptic_scaling": True
                        }
                }
        },
        "17_pruning_plus_ltd": {
                "plasticity": {
                        "mode": "custom",
                        "simulated_pruning_enabled": True,
                        "tagging_enabled": True,
                        "ltd_enabled": True
                }
        },
        "18_scaling_plus_downscale": {
                "organs": {
                        "homeostasis": {
                                "synaptic_scaling": True
                        }
                },
                "sleep_downscale": 0.99,
                "sleep_interval_steps": 50
        },
        "19_tag_replay_downscale": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True
                },
                "replay_compression": 0.5,
                "sleep_downscale": 0.99,
                "sleep_interval_steps": 50
        },
        "20_all_pending": {
                "plasticity": {
                        "mode": "custom",
                        "tagging_enabled": True,
                        "intrinsic_threshold_enabled": True,
                        "simulated_pruning_enabled": True,
                        "ltd_enabled": True,
                        "one_shot_enabled": True
                },
                "organs": {
                        "homeostasis": {
                                "synaptic_scaling": True
                        }
                },
                "replay_compression": 0.5,
                "sleep_downscale": 0.99,
                "sleep_interval_steps": 50
        }
};
    if name not in candidates:
        raise ValueError(f"unknown candidate: {name}")
    return candidates[name]

def deep_merge(base: dict, delta: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def run_training(root: Path, label: str, profile: dict, args, out: Path) -> dict:
    profile_path = out / f"{label}.runtime_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    log_path = out / f"{label}.log"
    ckpt_dir = out / f"{label}-checkpoints"
    command = [
        sys.executable, "-m", "train.train",
        "--steps", "100",
        "--resume", str(args.checkpoint.resolve()),
        "--phase", "1",
        "--pilot-all-rows",
        "--data", str((args.prepared_root / "data/spanish-v2-plus-personality-250k.jsonl").resolve()),
        "--tokenizer", str((args.prepared_root / "artifacts/spanish-v2/tokenizer.json").resolve()),
        "--token-cache", str((args.prepared_root / "data/spanish-v2-plus-personality-token-cache.pt").resolve()),
        "--batch", "8", "--max-len", "128", "--seed", "42",
        "--runtime-profile", str(profile_path.resolve()),
        "--lr", "3e-5", "--warmup", "100", "--total-steps", "235800",
        "--scheduler-transition", "--restart-scheduler-transition",
        "--transition-start", "235700", "--transition-end", "235800",
        "--warmup-transition", "10", "--transition-plateau-steps", "0",
        "--transition-min-lr-ratio", "0.5",
        "--log-every", "1", "--checkpoint-every", "10000",
        "--recovery-checkpoint-every", "0", "--parameter-check-every", "100",
        "--threads", "4", "--interop-threads", "2",
        "--ckpt-dir", str(ckpt_dir.resolve()),
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(root.resolve()),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
        "PYTHONUNBUFFERED": "1",
    })
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=root, env=env, stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - started
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"INFO step=(\d+) loss=([0-9.eE+-]+).*?dead=([0-9.eE+-]+)%",
        raw,
    )
    records = []
    for step, loss, dead in matches:
        values = (float(loss), float(dead))
        if all(math.isfinite(x) for x in values):
            records.append({"step": int(step), "loss": values[0], "dead_percent": values[1]})
    profile_totals = [
        float(value) for value in re.findall(
            r"profile_1: .*?total=([0-9.]+)s", raw
        )
    ]
    step_wall = sum(profile_totals)
    result = {
        "label": label,
        "exit_code": proc.returncode,
        "process_elapsed_seconds": round(elapsed, 4),
        "step_wall_seconds": round(step_wall, 4),
        "steps_per_second": round(len(records) / step_wall, 6) if step_wall > 0 else None,
        "logged_steps": len(records),
        "profiled_steps": len(profile_totals),
        "log_file": log_path.name,
        "profile_file": profile_path.name,
        "sleep_skipped": "Sleep skipped" in raw,
        "sleep_calls": raw.count("sleep_called:"),
        "nonfinite_marker": "NONFINITE" in raw or bool(re.search(r"\b(?:nan|inf)\b", raw, re.IGNORECASE)),
    }
    if (proc.returncode == 0 and len(records) == 100
            and len(profile_totals) == 100 and not result["nonfinite_marker"]):
        losses = [r["loss"] for r in records]
        windows = [sum(losses[i:i+20]) / 20 for i in range(0, 100, 20)]
        result.update({
            "status": "ok",
            "mean_loss_100_steps": round(sum(losses) / 100, 8),
            "loss_windows_20_steps": [round(x, 8) for x in windows],
            "max_dead_percent": round(max(r["dead_percent"] for r in records), 6),
            "first_step": records[0]["step"],
            "last_step": records[-1]["step"],
        })
    else:
        result.update({
            "status": "error",
            "reason": (
                f"exit_code={proc.returncode}; logged_steps={len(records)}/100; "
                f"profiled_steps={len(profile_totals)}/100; "
                f"nonfinite_marker={result['nonfinite_marker']}"
            ),
        })
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    delta = candidate_delta(args.candidate)
    base_profile = load_main_profile(args.baseline_root)
    candidate_profile = deep_merge(base_profile, delta)
    # Alternate order by candidate number to balance filesystem/page-cache warmup.
    candidate_first = int(args.candidate[:2]) % 2 == 0
    if candidate_first:
        candidate = run_training(args.candidate_root, "candidate", candidate_profile, args, args.output)
        control = run_training(args.baseline_root, "main-control", base_profile, args, args.output)
    else:
        control = run_training(args.baseline_root, "main-control", base_profile, args, args.output)
        candidate = None
    if control["status"] != "ok":
        report = {
            "candidate": args.candidate,
            "status": "blocked",
            "decision": "PENDING",
            "reason": "main control did not complete a valid 100-step run",
            "control": control,
            "candidate_result": None,
            "profile_delta": delta,
            "candidate_profile": candidate_profile,
            "baseline_profile": base_profile,
        }
    else:
        if candidate is None:
            candidate = run_training(args.candidate_root, "candidate", candidate_profile, args, args.output)
        reasons = []
        if "prioritized_replay" in args.candidate or "sleep_downscaling" in args.candidate or "tag_replay" in args.candidate or "plus_replay" in args.candidate or "plus_downscale" in args.candidate or "scaling_plus_downscale" in args.candidate or "all_pending" in args.candidate:
            if candidate.get("sleep_skipped"):
                candidate["status"] = "not_exercised"
                candidate["reason"] = "no había registros consolidados; el órgano de sueño no llegó a actuar"
            elif candidate.get("sleep_calls", 0) < 1:
                candidate["status"] = "not_exercised"
                candidate["reason"] = "el plan de 100 pasos no ejecutó ningún ciclo de sueño"
        if candidate["status"] != "ok":
            reasons.append(candidate.get("reason", "no completó 100 pasos válidos o produjo métricas no finitas"))
        else:
            if candidate["mean_loss_100_steps"] >= control["mean_loss_100_steps"]:
                reasons.append("no mejoró loss media")
            if candidate["steps_per_second"] <= control["steps_per_second"]:
                reasons.append("bajó pasos/segundo")
            if candidate["loss_windows_20_steps"][-1] > control["loss_windows_20_steps"][-1]:
                reasons.append("empeoró la última ventana de 20 pasos")
            if candidate["max_dead_percent"] > control["max_dead_percent"]:
                reasons.append("subió dead")
        approved = candidate["status"] == "ok" and not reasons
        report = {
            "candidate": args.candidate,
            "status": "completed",
            "decision": "APPROVED" if approved else "PENDING",
            "reasons": reasons,
            "control": control,
            "candidate_result": candidate,
            "loss_improvement_percent": (
                round(100 * (control["mean_loss_100_steps"] - candidate["mean_loss_100_steps"])
                      / control["mean_loss_100_steps"], 6)
                if candidate["status"] == "ok" and control["mean_loss_100_steps"] else None
            ),
            "throughput_improvement_percent": (
                round(100 * (candidate["steps_per_second"] / control["steps_per_second"] - 1), 6)
                if candidate["status"] == "ok" and control["steps_per_second"] else None
            ),
            "profile_delta": delta,
            "candidate_profile": candidate_profile,
            "comparison_contract": {
                "baseline": "main source and its committed config/lumen_cpu_profile.json",
                "baseline_sha": git_sha(args.baseline_root),
                "candidate_sha": git_sha(args.candidate_root),
                "candidate": "PR head; main frozen baseline plus pending-mechanism delta",
                "runtime_profile_source": "loaded verbatim from main/config/lumen_cpu_profile.json, then deep-merged with one candidate delta",
                "throughput_metric": "sum of trainer profile_1 per-step wall times; excludes process startup and data/model loading",
                "run_order": "alternates by candidate number to balance cache warmup",
                "parent_checkpoint_step": 235700,
                "training_steps_each": 100,
                "same_runner_sequential_pair": True,
                "same_seed": 42,
                "same_batch_sequence_from_checkpoint_rng": True,
                "same_corpus_tokenizer_cache": True,
                "same_cpu_profile": True,
                "loss_windows": 5,
                "holdout_used": False,
                "strict_gate": "mean loss lower AND steps/s higher AND dead not higher",
            },
        }
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    # Benchmark failures remain pending with their logs; don't cancel the other matrix jobs.

if __name__ == "__main__":
    main()
