#!/usr/bin/env python3
"""Rebase only the LR schedule of a promoted LUMEN checkpoint for a new horizon."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebase_state(state: dict, *, end_step: int, warmup_steps: int = 20) -> dict:
    rebased = copy.deepcopy(state)
    start_step = int(rebased["step"])
    if end_step <= start_step + warmup_steps:
        raise ValueError("end_step must leave room for the transition warmup")
    if rebased.get("scheduler_transition", {}).get("kind") != "cosine_transition_v1":
        raise ValueError("Expected a checkpoint with cosine_transition_v1 scheduler state")
    groups = rebased["optimizer"]["param_groups"]
    rates = {round(float(group["lr"]), 15) for group in groups}
    if len(rates) != 1:
        raise ValueError("Cannot rebase optimizer groups with different learning rates")
    current_lr = float(groups[0]["lr"])
    if not 0.0 < current_lr:
        raise ValueError(f"Expected a positive current learning rate, got {current_lr}")

    schedule = {
        "kind": "cosine_transition_v1",
        "start_step": start_step,
        "end_step": int(end_step),
        "old_lr": current_lr,
        "new_lr": current_lr,
        "warmup_steps": int(warmup_steps),
        "current_step": start_step,
    }
    for group in groups:
        group["lr"] = current_lr
    rebased["scheduler_transition"] = schedule
    rebased["scheduler"] = copy.deepcopy(schedule)
    config = rebased["training_config"]
    config.update({
        "total_steps": int(end_step),
        "transition_start": start_step,
        "transition_end": int(end_step),
        "warmup_transition": int(warmup_steps),
    })
    return rebased


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    source, output, manifest_path = Path(args.input), Path(args.output), Path(args.manifest)
    source_hash = sha256(source)
    state = torch.load(source, map_location="cpu", weights_only=False)
    original_step = int(state["step"])
    original_lr = float(state["optimizer"]["param_groups"][0]["lr"])
    rebased = rebase_state(state, end_step=args.end_step, warmup_steps=args.warmup_steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rebased, output)
    verified = torch.load(output, map_location="cpu", weights_only=False)
    if verified["step"] != original_step or verified["scheduler_transition"]["current_step"] != original_step:
        raise RuntimeError("Rebased checkpoint failed step validation")
    if verified["optimizer"]["param_groups"][0]["lr"] != original_lr:
        raise RuntimeError("Rebased checkpoint changed the optimizer's current LR")
    if verified["training_config"]["total_steps"] != args.end_step:
        raise RuntimeError("Rebased checkpoint has the wrong training horizon")

    manifest = {
        "operation": "scheduler_only_rebase",
        "source_checkpoint": str(source),
        "source_sha256": source_hash,
        "derived_checkpoint": str(output),
        "derived_sha256": sha256(output),
        "parent_step": original_step,
        "end_step": args.end_step,
        "warmup_steps": args.warmup_steps,
        "lr_at_rebase": original_lr,
        "schedule": "hold current LR for warmup_steps, then cosine decay to zero at end_step",
        "weights_changed": False,
        "optimizer_moments_changed": False,
        "rng_changed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
