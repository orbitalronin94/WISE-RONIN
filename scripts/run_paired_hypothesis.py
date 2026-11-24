"""Run one fresh-check-0, same-runner paired LUMEN neuroplasticity hypothesis."""
from __future__ import annotations
import argparse, copy, json, math, os, re, subprocess, sys, time
from pathlib import Path

HYPOTHESES = {
    "H01": {"label": "Pain + pleasure response combined", "delta": {"organs": {"pain": {"response": 0.20}, "pleasure": {"response": 0.50}}}},
    "H02": {"label": "Pain response", "delta": {"organs": {"pain": {"response": 0.20}}}},
    "H03": {"label": "Pleasure response", "delta": {"organs": {"pleasure": {"response": 0.50}}}},
    "H04": {"label": "Pain budget gain", "delta": {"organs": {"pain_budget_gain": 0.15}}},
    "H05": {"label": "Pleasure budget gain", "delta": {"organs": {"pleasure_budget_gain": 0.15}}},
    "H06": {"label": "LTP threshold", "delta": {"plasticity": {"mode": "custom", "threshold_scale": 0.85}}},
    "H07": {"label": "Disinhibition gain", "delta": {"plasticity": {"mode": "custom", "inhibition_scale": 0.80}}},
    "H08": {"label": "Modulator gain", "delta": {"plasticity": {"mode": "custom", "modulator_gain": 1.50}}},
    "H09": {"label": "Eligibility window", "delta": {"plasticity": {"mode": "custom", "eligibility_retention": 0.98}}},
    "H10": {"label": "Attention gate", "delta": {"plasticity": {"mode": "custom", "attention_gain": 0.40}}},
    "H11": {"label": "Tagging and capture", "delta": {"plasticity": {"mode": "custom", "tagging_enabled": True}}},
    "H12": {"label": "Intrinsic threshold", "delta": {"plasticity": {"mode": "custom", "intrinsic_threshold_enabled": True}}},
    "H13": {"label": "Non-destructive simulated pruning", "delta": {"plasticity": {"mode": "custom", "simulated_pruning_enabled": True}}},
    "H14": {"label": "Tag-protected LTD", "delta": {"plasticity": {"mode": "custom", "tagging_enabled": True, "ltd_enabled": True}}},
    "H15": {"label": "One-shot memory", "delta": {"plasticity": {"mode": "custom", "one_shot_enabled": True}}},
    "H16": {"label": "Homeostatic synaptic scaling", "delta": {"organs": {"homeostasis": {"synaptic_scaling": True}}}},
    "H17": {"label": "Prioritized replay", "delta": {"replay_compression": 0.50, "sleep_interval_steps": 50}},
    "H18": {"label": "Sleep downscaling", "delta": {"sleep_downscale": 0.99, "sleep_interval_steps": 50}},
    "H19": {"label": "Tagging + intrinsic threshold + protected LTD", "delta": {"plasticity": {"mode": "custom", "tagging_enabled": True, "intrinsic_threshold_enabled": True, "ltd_enabled": True}}},
    "H20": {"label": "Pain/pleasure modulation + extended eligibility", "delta": {"organs": {"pain": {"response": 0.20}, "pleasure": {"response": 0.50}, "pain_budget_gain": 0.15, "pleasure_budget_gain": 0.15}, "plasticity": {"mode": "custom", "eligibility_retention": 0.98, "modulator_gain": 1.50}}},
}

def deep_merge(base, delta):
    result = copy.deepcopy(base)
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

def run_arm(name, profile, args, out):
    profile_path = out / f"{name}.runtime_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    log_path = out / f"{name}.log"
    ckpt = out / f"{name}-checkpoints"
    cmd = [
        sys.executable, "-m", "train.train",
        "--steps", str(args.steps), "--phase", "1", "--pilot-all-rows",
        "--data", "artifacts/spanish-v1/train.jsonl",
        "--tokenizer", "data/spanish/tokenizer/tokenizer.json",
        "--token-cache", "data/spanish/train_tokens_128.pt",
        "--batch", str(args.batch), "--max-len", str(args.max_len),
        "--seed", str(args.seed), "--runtime-profile", str(profile_path),
        "--lr", str(args.lr), "--warmup", str(args.warmup),
        "--total-steps", str(args.steps), "--log-every", "10",
        "--checkpoint-every", str(args.steps), "--recovery-checkpoint-every", "250",
        "--parameter-check-every", "100", "--ckpt-dir", str(ckpt),
    ]
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(Path.cwd()), "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                "MKL_DYNAMIC": "FALSE", "PYTHONUNBUFFERED": "1"})
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{name}] {line}", end="", flush=True)
            log.write(line)
        code = proc.wait()
    elapsed = time.perf_counter() - started
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"INFO step=(\d+) loss=([0-9.eE+-]+).*?dead=([0-9.eE+-]+)%", raw)
    records = []
    for step, loss, dead in matches:
        values = (float(loss), float(dead))
        if all(math.isfinite(x) for x in values):
            records.append({"step": int(step), "loss": values[0], "dead_percent": values[1]})
    profile_totals = [float(x) for x in re.findall(r"profile_1: .*?total=([0-9.]+)s", raw)]
    measured_seconds = sum(profile_totals) if profile_totals else elapsed
    result = {
        "arm": name, "exit_code": code, "wall_seconds": round(elapsed, 3),
        "measured_seconds": round(measured_seconds, 3),
        "steps": args.steps, "logged_records": len(records),
        "steps_per_second": round(args.steps / measured_seconds, 5) if measured_seconds > 0 else None,
        "log": str(log_path),
        "sleep_calls": raw.count("sleep_called:"),
        "sleep_skipped": "Sleep skipped" in raw,
        "nonfinite": "NONFINITE" in raw or bool(re.search(r"\b(?:nan|inf)\b", raw, re.I)),
    }
    if code == 0 and not result["nonfinite"] and records:
        losses = [r["loss"] for r in records]
        result.update({
            "status": "ok",
            "mean_logged_loss": round(sum(losses) / len(losses), 8),
            "last_logged_loss": losses[-1],
            "last_20_logged_mean": round(sum(losses[-20:]) / len(losses[-20:]), 8),
            "max_dead_percent": round(max(r["dead_percent"] for r in records), 6),
            "first_logged_step": records[0]["step"],
            "last_logged_step": records[-1]["step"],
        })
    else:
        result.update({"status": "error", "reason": f"exit={code}; logged={len(records)}; nonfinite={result['nonfinite']}"})
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hypothesis", choices=HYPOTHESES, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path("paired-results") / args.hypothesis
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path("config/lumen_cpu_profile.json").read_text())
    spec = HYPOTHESES[args.hypothesis]
    candidate = deep_merge(base, spec["delta"])
    # Alternate which arm goes first to reduce systematic cache-warmup bias.
    if int(args.hypothesis[1:]) % 2:
        control = run_arm("main", base, args, root)
        candidate_result = run_arm("candidate", candidate, args, root)
    else:
        candidate_result = run_arm("candidate", candidate, args, root)
        control = run_arm("main", base, args, root)
    valid = control["status"] == "ok" and candidate_result["status"] == "ok"
    result = {
        "hypothesis": args.hypothesis, "label": spec["label"],
        "scope": "implemented neuroplasticity mechanism/profile; check-0 screening, not holdout validation",
        "start": "fresh initialization (step 0)", "seed": args.seed,
        "batch": args.batch, "max_len": args.max_len, "steps_per_arm": args.steps,
        "control": control, "candidate": candidate_result,
        "candidate_speedup_vs_main": round(control["measured_seconds"] / candidate_result["measured_seconds"], 5) if valid else None,
        "mean_loss_delta": round(candidate_result["mean_logged_loss"] - control["mean_logged_loss"], 8) if valid else None,
        "last_logged_loss_delta": round(candidate_result["last_logged_loss"] - control["last_logged_loss"], 8) if valid else None,
        "decision": "SCREENING_ONLY",
        "profile_delta": spec["delta"],
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
