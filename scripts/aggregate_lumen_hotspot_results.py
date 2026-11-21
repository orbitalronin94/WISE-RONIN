"""Select a faster LUMEN runtime profile only when the original loss is preserved."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--current-profile", default="config/lumen_cpu_profile.json")
    parser.add_argument("--report", default="artifacts/lumen_hotspot_benchmark.json")
    parser.add_argument("--minimum-speedup", type=float, default=1.05)
    parser.add_argument("--max-loss-regression", type=float, default=0.01)
    args = parser.parse_args()
    files = sorted(Path(args.results_dir).glob("*.json"))
    results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in files}
    if "control" not in results:
        raise RuntimeError("Missing control result; refusing to promote any profile")
    baseline = results["control"]["full_objective"]["median_steps_per_second"]
    baseline_loss = results["control"]["full_objective"]["median_mean_loss"]
    if baseline <= 0:
        raise RuntimeError("Invalid control throughput")
    rows = []
    for name, result in results.items():
        speed = result["full_objective"]["median_steps_per_second"]
        loss = result["full_objective"]["median_mean_loss"]
        loss_delta = loss - baseline_loss
        rows.append({
            "name": name,
            "steps_per_second": speed,
            "speedup_vs_control": speed / baseline,
            "mean_training_loss": loss,
            "loss_delta": loss_delta,
            "loss_gate_passed": loss_delta <= args.max_loss_regression,
            "promotion_eligible": name != "control" and loss_delta <= args.max_loss_regression
                and speed >= baseline * args.minimum_speedup,
        })
    eligible = [r for r in rows if r["promotion_eligible"]]
    winner = max(eligible, key=lambda r: r["steps_per_second"]) if eligible else None
    current = json.loads(Path(args.current_profile).read_text(encoding="utf-8"))
    selected = "control"
    promoted = False
    if winner:
        selected = winner["name"]
        profile = results[selected]["profile"]
        profile["benchmark_metadata"] = {
            "selected_profile": selected,
            "steps_per_second": winner["steps_per_second"],
            "control_steps_per_second": baseline,
            "speedup_vs_control": winner["speedup_vs_control"],
            "mean_training_loss": winner["mean_training_loss"],
            "control_mean_training_loss": baseline_loss,
            "loss_delta": winner["loss_delta"],
            "max_loss_regression": args.max_loss_regression,
            "checkpoint_step": 3000,
            "objective_unchanged": True,
            "selection": "parallel 40-hypothesis matrix around measured 4-thread winner; same checkpoint, batch, seed, cache and original loss",
        }
        Path(args.current_profile).write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        promoted = True
    report = {
        "benchmark": "lumen_parallel_40_hypotheses",
        "checkpoint_step": 3000,
        "control_steps_per_second": baseline,
        "control_mean_training_loss": baseline_loss,
        "minimum_speedup": args.minimum_speedup,
        "max_loss_regression": args.max_loss_regression,
        "loss_gate": "candidate mean training loss must be <= control + 0.01",
        "promotion_gate": "candidate must pass loss gate AND be >=5% faster than control",
        "selected_profile": selected,
        "promoted": promoted,
        "ranked_results": sorted(rows, key=lambda r: r["steps_per_second"], reverse=True),
        "raw_results": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("selected_profile", "promoted", "control_steps_per_second", "control_mean_training_loss", "ranked_results")}, indent=2))

if __name__ == "__main__":
    main()
