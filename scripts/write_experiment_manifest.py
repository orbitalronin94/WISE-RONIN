#!/usr/bin/env python3
"""Write a reproducible LUMEN-vs-Transformer manifest for one matched checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumen", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lumen-parent")
    parser.add_argument("--transformer-parent")
    args = parser.parse_args()

    paths = {
        "lumen": Path(args.lumen),
        "transformer": Path(args.transformer),
        "data": Path(args.data),
        "tokenizer": Path(args.tokenizer),
        "evaluation": Path(args.evaluation),
    }
    states = {name: load_checkpoint(paths[name]) for name in ("lumen", "transformer")}
    data_hash = sha256(paths["data"])
    tokenizer_hash = sha256(paths["tokenizer"])
    for name, state in states.items():
        if state.get("data_sha256") != data_hash:
            raise ValueError(f"{name}: checkpoint corpus hash differs from current corpus")
        if state.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError(f"{name}: checkpoint tokenizer hash differs from current tokenizer")

    lumen, transformer = states["lumen"], states["transformer"]
    for field in ("step", "seed", "tokens_seen"):
        if lumen.get(field) != transformer.get(field):
            raise ValueError(
                f"Matched run mismatch for {field}: "
                f"LUMEN={lumen.get(field)} Transformer={transformer.get(field)}"
            )
    lumen_params = int(lumen.get("params") or sum(t.numel() for t in lumen["model"].values()))
    transformer_params = int(transformer.get("params") or sum(t.numel() for t in transformer["model"].values()))
    gap = abs(lumen_params - transformer_params) / max(1, lumen_params) * 100
    if gap >= 1.0:
        raise ValueError(f"Parameter gap {gap:.4f}% exceeds 1% matched-size tolerance")

    def lumen_health(state: dict) -> dict:
        tensors = state["model"]
        layer_ids = sorted({
            int(key.split(".")[1])
            for key in tensors
            if key.startswith("homeo.") and key.split(".")[1].isdigit()
        })
        layers = []
        for index in layer_ids:
            prefix = f"homeo.{index}."
            effective = tensors.get(prefix + "effective_m")
            pre_gate = tensors.get(prefix + "m")
            bias = tensors.get(prefix + "bias")
            layers.append({
                "layer": index,
                "effective_activity_ema_mean": float(effective.float().mean()) if effective is not None else None,
                "effective_dead_fraction": float((effective < 0.001).float().mean()) if effective is not None else None,
                "pre_gate_activity_ema_mean": float(pre_gate.float().mean()) if pre_gate is not None else None,
                "homeostasis_bias_norm": float(bias.float().norm()) if bias is not None else None,
                "homeostasis_bias_min": float(bias.float().min()) if bias is not None else None,
                "homeostasis_bias_max": float(bias.float().max()) if bias is not None else None,
            })
        return {
            "homeostasis_by_layer": layers,
            "layer_diagnostics_history": state.get("layer_diagnostics_history", []),
            "sleep_events": state.get("sleep_events", []),
            "sleep_executed_events": sum(bool(event.get("executed")) for event in state.get("sleep_events", [])),
        }

    def parent_record(raw_path: str | None):
        if not raw_path:
            return None
        parent = Path(raw_path)
        if not parent.is_file():
            raise FileNotFoundError(f"Missing parent checkpoint: {parent}")
        return {"path": str(parent), "sha256": sha256(parent)}

    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    step = int(lumen["step"])
    manifest = {
        "schema_version": 1,
        "experiment": "RONIN-LUMEN-vs-Transformer-longitudinal",
        "global_step": step,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "workflow": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "shared_training": {
            "seed": lumen.get("seed"),
            "tokens_seen": lumen.get("tokens_seen"),
            "corpus_sha256": data_hash,
            "tokenizer_sha256": tokenizer_hash,
            "parameter_gap_percent": round(gap, 6),
        },
        "models": {
            "lumen": {
                "architecture_id": lumen.get("architecture_id", "ronin_lumen_v3_pilot_legacy_checkpoint"),
                "architecture": lumen.get("architecture", {"config": lumen.get("config")}),
                "parameters": lumen_params,
                "checkpoint": {"path": str(paths["lumen"]), "sha256": sha256(paths["lumen"])},
                "parent_checkpoint": parent_record(args.lumen_parent),
                "health_and_memory_diagnostics": lumen_health(lumen),
                "training_config": {
                    "phase": lumen.get("phase"),
                    "pilot_all_rows": lumen.get("pilot_all_rows"),
                    "v32_ze": lumen.get("v32_ze"),
                    "delta_policy": lumen.get("delta_policy"),
                },
            },
            "transformer": {
                "architecture_id": transformer.get("architecture_id", "decoder_transformer_legacy_checkpoint"),
                "architecture": transformer.get("architecture"),
                "parameters": transformer_params,
                "checkpoint": {"path": str(paths["transformer"]), "sha256": sha256(paths["transformer"])},
                "parent_checkpoint": parent_record(args.transformer_parent),
                "training_config": transformer.get("training_config"),
            },
        },
        "evaluation": {
            "path": str(paths["evaluation"]),
            "sha256": sha256(paths["evaluation"]),
            "summary": evaluation,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "global_step": step,
        "lumen_architecture": manifest["models"]["lumen"]["architecture_id"],
        "transformer_architecture": manifest["models"]["transformer"]["architecture_id"],
        "parameter_gap_percent": round(gap, 6),
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
