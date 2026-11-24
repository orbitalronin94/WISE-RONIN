"""Run a fresh-start, same-runner paired LUMEN control/candidate experiment."""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

HYPOTHESES = {
    "H01": {"label": "Delta refinements = 1", "profile": {"delta_parallel_refinements": 1}},
    "H02": {"label": "Delta refinements = 3", "profile": {"delta_parallel_refinements": 3}},
    "H03": {"label": "Delta refinements = 4", "profile": {"delta_parallel_refinements": 4}},
    "H04": {"label": "Delta refinements = 6", "profile": {"delta_parallel_refinements": 6}},
    "H05": {"label": "Delta pre-normalization enabled", "profile": {"delta_pre_normalize": True}},
    "H06": {"label": "Delta write-update reuse disabled", "profile": {"delta_reuse_write_update": False}},
    "H07": {"label": "Persistent eligibility always-apply disabled", "profile": {"delta_always_apply_eligibility": False}},
    "H08": {"label": "On-demand trace norm diagnostics disabled", "profile": {"delta_trace_norm_diagnostics_on_demand": False}},
    "H09": {"label": "Shared causal masks disabled", "profile": {"delta_share_causal_masks": False}},
    "H10": {"label": "Fused homeostasis scatter enabled", "profile": {"homeostasis_fused_scatter": True}},
    "H11": {"label": "Shared CE/repetition normalization enabled", "profile": {"shared_loss_normalization": True}},
    "H12": {"label": "Lazy backward enabled", "cli": ["--lazy-backward"]},
    "H13": {"label": "Surprise gate at loss 1.8", "cli": ["--surprise-gate-loss", "1.8"]},
    "H14": {"label": "Lazy backward + surprise gate 1.8", "cli": ["--lazy-backward", "--surprise-gate-loss", "1.8"]},
    "H15": {"label": "MKLDNN enabled", "profile": {"mkldnn_enabled": True}},
    "H16": {"label": "AdamW foreach enabled", "profile": {"foreach": True}},
    "H17": {"label": "One intra-op thread", "profile": {"threads": 1, "interop_threads": 1}},
    "H18": {"label": "Two intra-op threads", "profile": {"threads": 2, "interop_threads": 1}},
    "H19": {"label": "One inter-op thread", "profile": {"interop_threads": 1}},
    "H20": {"label": "CPU affinity disabled", "profile": {"cpu_affinity": False}},
}

def run_arm(name, profile_path, cli_flags, args, root):
    log_path = root / f"{name}.log"
    ckpt = root / f"checkpoints-{name}"
    cmd = [
        sys.executable, "-m", "train.train",
        "--steps", str(args.steps), "--phase", "1", "--pilot-all-rows",
        "--data", "artifacts/spanish-v1/train.jsonl",
        "--tokenizer", "data/spanish/tokenizer/tokenizer.json",
        "--token-cache", "data/spanish/train_tokens_128.pt",
        "--batch", str(args.batch), "--max-len", str(args.max_len),
        "--seed", str(args.seed), "--runtime-profile", str(profile_path),
        "--lr", str(args.lr), "--warmup", str(args.warmup),
        "--total-steps", str(args.steps), "--log-every", "100",
        "--checkpoint-every", str(args.steps), "--recovery-checkpoint-every", "250",
        "--parameter-check-every", "100", "--ckpt-dir", str(ckpt),
    ] + cli_flags
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=os.environ.copy())
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{name}] {line}", end="", flush=True)
            log.write(line)
        code = proc.wait()
    elapsed = time.monotonic() - started
    if code:
        raise RuntimeError(f"{name} exited with code {code}; inspect {log_path}")
    content = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(?:loss|total_loss)\s*[=:]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", content)
    return {
        "arm": name, "wall_seconds": round(elapsed, 3),
        "steps": args.steps, "steps_per_second": round(args.steps / elapsed, 5),
        "last_logged_loss": float(matches[-1]) if matches else None,
        "log": str(log_path), "exit_code": code,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hypothesis", choices=HYPOTHESES, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--output", default="paired-result.json")
    args = p.parse_args()
    root = Path("paired-results") / args.hypothesis
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path("config/lumen_cpu_profile.json").read_text())
    candidate = json.loads(json.dumps(base))
    spec = HYPOTHESES[args.hypothesis]
    candidate.update(spec.get("profile", {}))
    base_path, candidate_path = root / "main-profile.json", root / "candidate-profile.json"
    base_path.write_text(json.dumps(base, indent=2) + "\n")
    candidate_path.write_text(json.dumps(candidate, indent=2) + "\n")
    # Same runner, same downloaded data/tokenizer/cache, same seed and hyperparameters.
    control = run_arm("main", base_path, [], args, root)
    candidate_result = run_arm("candidate", candidate_path, spec.get("cli", []), args, root)
    result = {
        "hypothesis": args.hypothesis, "label": spec["label"],
        "scope": "executable runtime/mechanism screening; not a claim that all 20 proposed biological hypotheses are implemented",
        "start": "fresh initialization (check 0)", "seed": args.seed,
        "batch": args.batch, "max_len": args.max_len, "steps_per_arm": args.steps,
        "control": control, "candidate": candidate_result,
        "candidate_speedup_vs_main": round(control["wall_seconds"] / candidate_result["wall_seconds"], 5),
        "candidate_loss_delta": (
            round(candidate_result["last_logged_loss"] - control["last_logged_loss"], 8)
            if candidate_result["last_logged_loss"] is not None and control["last_logged_loss"] is not None else None
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
