#!/usr/bin/env python3
"""Validate and rebase the step-2000 checkpoints for a 2000 -> 4000 continuation.

Only scheduler metadata is changed. Model weights, Adam moments, RNG state,
token counters and the original checkpoint bytes remain auditable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

SOURCE_RUN = "44922965444"
SOURCE_ARTIFACT = "14715857574"
PARENT_STEP = 2000
END_STEP = 4000
WARMUP_STEPS = 20
EXPECTED = {
    "lumen": {
        "architecture_id": "ronin_lumen_v3_1_residual_homeostasis",
        "params": 2_281_841,
        "lr": 2e-5,
        "grad_clip": 0.5,
    },
    "transformer": {
        "architecture_id": "decoder_transformer_4l_d192_h4_ff576_tied_v1",
        "params": 2_295_168,
        "lr": 5e-5,
        "grad_clip": 1.0,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(value) -> str:
    """Stable digest for nested checkpoint state, including tensors."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda x: str(x)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def has_adam_moments(optimizer: dict) -> bool:
    slots = optimizer.get("state", {})
    return bool(slots) and any(
        isinstance(slot, dict) and "exp_avg" in slot and "exp_avg_sq" in slot
        for slot in slots.values()
    )


def expected_parent_config(kind: str) -> dict:
    item = EXPECTED[kind]
    return {
        "batch": 8,
        "max_len": 128,
        "lr": item["lr"],
        "warmup": 100,
        "total_steps": 2000,
        "weight_decay": 0.01,
        "scheduler_mode": "cosine_transition_v1",
        "transition_start": 1000,
        "transition_end": 2000,
        "warmup_transition": 20,
        "grad_clip": item["grad_clip"],
    }


def continuation_config(kind: str) -> dict:
    config = expected_parent_config(kind)
    config.update({
        "total_steps": END_STEP,
        "transition_start": PARENT_STEP,
        "transition_end": END_STEP,
    })
    return config


def rebase_state(state: dict, kind: str, data_hash: str, tokenizer_hash: str,
                 source_checkpoint_sha256: str) -> dict:
    if kind not in EXPECTED:
        raise ValueError(f"Unknown model kind: {kind}")
    spec = EXPECTED[kind]
    if int(state.get("step", -1)) != PARENT_STEP:
        raise ValueError(f"{kind}: expected parent step {PARENT_STEP}, got {state.get('step')}")
    if state.get("architecture_id") != spec["architecture_id"]:
        raise ValueError(f"{kind}: unexpected architecture_id {state.get('architecture_id')}")
    if state.get("params") != spec["params"]:
        raise ValueError(f"{kind}: parameter count mismatch: {state.get('params')}")
    if state.get("seed") != 42:
        raise ValueError(f"{kind}: expected seed 42")
    if state.get("data_sha256") != data_hash or state.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError(f"{kind}: corpus/tokenizer hash mismatch")
    if not state.get("optimizer") or not has_adam_moments(state["optimizer"]):
        raise ValueError(f"{kind}: Adam moments are absent; exact continuation is impossible")
    if not state.get("rng") or not state.get("scheduler"):
        raise ValueError(f"{kind}: scheduler or RNG state is missing")
    if state.get("training_config") != expected_parent_config(kind):
        raise ValueError(f"{kind}: step-2000 training configuration differs from audited parent")
    if kind == "lumen" and (
        state.get("phase") != 1 or state.get("pilot_all_rows") is not True
        or state.get("v32_ze", False) is not False
    ):
        raise ValueError("LUMEN: parent must be phase-1 all-rows V3.1, not V3.2")
    old = state.get("scheduler_transition") or {}
    if (
        old.get("kind") != "cosine_transition_v1"
        or old.get("start_step") != 1000
        or old.get("end_step") != PARENT_STEP
        or old.get("current_step") != PARENT_STEP
        or old.get("warmup_steps") != WARMUP_STEPS
        or not math.isclose(float(old.get("new_lr", -1)), spec["lr"], rel_tol=1e-10)
    ):
        raise ValueError(f"{kind}: parent scheduler is not the completed audited 1000 -> 2000 schedule")
    if state["scheduler"] != old:
        raise ValueError(f"{kind}: scheduler and scheduler_transition state differ")
    groups = state["optimizer"].get("param_groups", [])
    if not groups or any(not math.isclose(float(group.get("lr", -1)), 0.0, abs_tol=1e-12) for group in groups):
        raise ValueError(f"{kind}: expected completed cosine schedule with optimizer LR=0")

    weights_before = hash_tree(state["model"])
    optimizer_before = hash_tree(state["optimizer"])
    rng_before = hash_tree(state["rng"])
    tokens_before = int(state.get("tokens_seen", -1))
    rebased = copy.deepcopy(state)
    next_schedule = {
        "kind": "cosine_transition_v1",
        "start_step": PARENT_STEP,
        "end_step": END_STEP,
        "old_lr": 0.0,
        "new_lr": spec["lr"],
        "warmup_steps": WARMUP_STEPS,
        "current_step": PARENT_STEP,
    }
    rebased["training_config"] = continuation_config(kind)
    rebased["scheduler_transition"] = next_schedule
    rebased["scheduler"] = copy.deepcopy(next_schedule)
    rebased["continuation_parent"] = {
        "source_run": SOURCE_RUN,
        "source_artifact_id": SOURCE_ARTIFACT,
        "source_step": PARENT_STEP,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "weights_sha256": weights_before,
        "optimizer_state_sha256": optimizer_before,
        "rng_state_sha256": rng_before,
        "tokens_seen": tokens_before,
        "schedule": "20-step linear warmup from LR=0 to the model's established target LR, then cosine decay to zero at step 4000",
    }
    if hash_tree(rebased["model"]) != weights_before:
        raise AssertionError(f"{kind}: weights changed during scheduler rebase")
    if hash_tree(rebased["optimizer"]) != optimizer_before:
        raise AssertionError(f"{kind}: optimizer state changed during scheduler rebase")
    if hash_tree(rebased["rng"]) != rng_before or int(rebased["tokens_seen"]) != tokens_before:
        raise AssertionError(f"{kind}: RNG or token counter changed during scheduler rebase")
    return rebased


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumen", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_path, tokenizer_path = Path(args.data), Path(args.tokenizer)
    data_hash, tokenizer_hash = sha256_file(data_path), sha256_file(tokenizer_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = {
        "source_run": SOURCE_RUN,
        "source_artifact_id": SOURCE_ARTIFACT,
        "parent_step": PARENT_STEP,
        "target_step": END_STEP,
        "data_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "models": {},
    }
    token_counts = set()
    for kind, raw_path in (("lumen", args.lumen), ("transformer", args.transformer)):
        source = Path(raw_path)
        source_hash = sha256_file(source)
        state = torch.load(source, map_location="cpu", weights_only=False)
        rebased = rebase_state(state, kind, data_hash, tokenizer_hash, source_hash)
        token_counts.add(int(rebased["tokens_seen"]))
        destination = output_dir / f"{kind}_step_2000_continuation.pt"
        torch.save(rebased, destination)
        audit["models"][kind] = {
            "source_checkpoint_sha256": source_hash,
            "continuation_checkpoint": str(destination),
            "continuation_checkpoint_sha256": sha256_file(destination),
            "weights_sha256": rebased["continuation_parent"]["weights_sha256"],
            "optimizer_state_sha256": rebased["continuation_parent"]["optimizer_state_sha256"],
            "rng_state_sha256": rebased["continuation_parent"]["rng_state_sha256"],
            "tokens_seen": rebased["tokens_seen"],
            "parameters": rebased["params"],
            "optimizer_moments_preserved": True,
            "weights_preserved": True,
            "rng_preserved": True,
            "schedule": rebased["scheduler_transition"],
        }
    if len(token_counts) != 1:
        raise ValueError(f"Matched parent token counters differ: {sorted(token_counts)}")
    audit["tokens_seen_at_parent"] = token_counts.pop()
    (output_dir / "parent_validation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
