#!/usr/bin/env python3
"""Audit and prepare matched LUMEN/Transformer step-3000 parents for a long run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

PARENT_STEP = 3000
END_STEP = 8000
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
        "total_steps": 4000,
        "weight_decay": 0.01,
        "scheduler_mode": "cosine_transition_v1",
        "transition_start": 2000,
        "transition_end": 4000,
        "warmup_transition": 20,
        "grad_clip": item["grad_clip"],
    }


def lr_at_step(schedule: dict, step: int) -> float:
    start, end = int(schedule["start_step"]), int(schedule["end_step"])
    old_lr, new_lr = float(schedule["old_lr"]), float(schedule["new_lr"])
    warmup = int(schedule["warmup_steps"])
    if step <= start:
        return old_lr
    transition_end = start + warmup
    if step < transition_end:
        alpha = (step - start) / warmup
        return old_lr * (1.0 - alpha) + new_lr * alpha
    progress = min(1.0, max(0.0, (step - transition_end) / (end - transition_end)))
    return new_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def prepare(state: dict, kind: str, data_hash: str, tokenizer_hash: str,
            source_hash: str) -> dict:
    spec = EXPECTED[kind]
    if int(state.get("step", -1)) != PARENT_STEP:
        raise ValueError(f"{kind}: expected exact step-{PARENT_STEP} parent, got {state.get('step')}")
    if state.get("architecture_id") != spec["architecture_id"] or state.get("params") != spec["params"]:
        raise ValueError(f"{kind}: architecture or parameter count mismatch")
    if state.get("seed") != 42:
        raise ValueError(f"{kind}: expected seed 42")
    if state.get("data_sha256") != data_hash or state.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError(f"{kind}: corpus/tokenizer provenance mismatch")
    if state.get("training_config") != expected_parent_config(kind):
        raise ValueError(f"{kind}: parent training configuration differs from audited 2000->4000 run")
    if not state.get("optimizer") or not has_adam_moments(state["optimizer"]):
        raise ValueError(f"{kind}: Adam moments missing; exact continuation is impossible")
    if not state.get("rng") or not state.get("scheduler_transition"):
        raise ValueError(f"{kind}: scheduler or RNG state missing")
    if kind == "lumen":
        if state.get("phase") != 1 or state.get("pilot_all_rows") is not True or state.get("v32_ze", False):
            raise ValueError("LUMEN parent must be phase-1 all-rows V3.1")
    old = state["scheduler_transition"]
    if (
        old.get("kind") != "cosine_transition_v1"
        or old.get("start_step") != 2000
        or old.get("end_step") != 4000
        or old.get("current_step") != PARENT_STEP
        or old.get("warmup_steps") != 20
        or not math.isclose(float(old.get("new_lr", -1)), spec["lr"], rel_tol=1e-10)
        or state.get("scheduler") != old
    ):
        raise ValueError(f"{kind}: scheduler is not the expected step-2000-to-4000 schedule at step 3000")
    groups = state["optimizer"].get("param_groups", [])
    if not groups:
        raise ValueError(f"{kind}: optimizer parameter groups missing")
    current_lrs = [float(group["lr"]) for group in groups]
    if max(current_lrs) - min(current_lrs) > 1e-12:
        raise ValueError(f"{kind}: optimizer groups have inconsistent learning rates")
    current_lr = current_lrs[0]
    scheduled_lr = lr_at_step(old, PARENT_STEP)
    if not math.isclose(current_lr, scheduled_lr, rel_tol=1e-7, abs_tol=1e-12):
        raise ValueError(f"{kind}: optimizer LR {current_lr} differs from scheduler LR {scheduled_lr}")

    weights_hash = hash_tree(state["model"])
    optimizer_hash = hash_tree(state["optimizer"])
    rng_hash = hash_tree(state["rng"])
    tokens_seen = int(state.get("tokens_seen", -1))
    if tokens_seen <= 0:
        raise ValueError(f"{kind}: invalid tokens_seen")
    result = copy.deepcopy(state)
    # Continue smoothly from the actual step-3000 LR; do not restart warmup or
    # jump back to the peak LR. The one-step warmup is flat (same LR), then cosine decay.
    schedule = {
        "kind": "cosine_transition_v1",
        "start_step": PARENT_STEP,
        "end_step": END_STEP,
        "old_lr": current_lr,
        "new_lr": current_lr,
        "warmup_steps": 1,
        "current_step": PARENT_STEP,
    }
    config = dict(result["training_config"])
    config.update({
        "total_steps": END_STEP,
        "transition_start": PARENT_STEP,
        "transition_end": END_STEP,
        "warmup_transition": 1,
    })
    result["training_config"] = config
    result["scheduler_transition"] = schedule
    result["scheduler"] = copy.deepcopy(schedule)
    result["continuation_parent"] = {
        "source_step": PARENT_STEP,
        "source_checkpoint_sha256": source_hash,
        "weights_sha256": weights_hash,
        "optimizer_state_sha256": optimizer_hash,
        "rng_state_sha256": rng_hash,
        "tokens_seen": tokens_seen,
        "original_scheduler": old,
        "new_schedule": schedule,
        "policy": "resume from actual step-3000 LR and cosine-decay smoothly to zero at step 8000; preserve weights, Adam moments, RNG and token counter",
    }
    if hash_tree(result["model"]) != weights_hash:
        raise AssertionError(f"{kind}: weights changed during schedule preparation")
    if hash_tree(result["optimizer"]) != optimizer_hash:
        raise AssertionError(f"{kind}: optimizer state changed during schedule preparation")
    if hash_tree(result["rng"]) != rng_hash or int(result["tokens_seen"]) != tokens_seen:
        raise AssertionError(f"{kind}: RNG or token counter changed during preparation")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumen", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data_hash = sha256_file(Path(args.data))
    tokenizer_hash = sha256_file(Path(args.tokenizer))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema_version": 1,
        "experiment": "RONIN-LUMEN-vs-Transformer-longrun",
        "parent_step": PARENT_STEP,
        "target_step": END_STEP,
        "data_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "models": {},
    }
    token_counts = set()
    for kind, raw in (("lumen", args.lumen), ("transformer", args.transformer)):
        source = Path(raw)
        source_hash = sha256_file(source)
        state = torch.load(source, map_location="cpu", weights_only=False)
        prepared = prepare(state, kind, data_hash, tokenizer_hash, source_hash)
        token_counts.add(int(prepared["tokens_seen"]))
        destination = out / f"{kind}_step_3000_longrun.pt"
        torch.save(prepared, destination)
        audit["models"][kind] = {
            "source_checkpoint_sha256": source_hash,
            "prepared_checkpoint_sha256": sha256_file(destination),
            "weights_sha256": prepared["continuation_parent"]["weights_sha256"],
            "optimizer_state_sha256": prepared["continuation_parent"]["optimizer_state_sha256"],
            "rng_state_sha256": prepared["continuation_parent"]["rng_state_sha256"],
            "tokens_seen": prepared["tokens_seen"],
            "parameters": prepared["params"],
            "weights_preserved": True,
            "optimizer_moments_preserved": True,
            "rng_preserved": True,
            "learning_rate_at_parent": prepared["scheduler_transition"]["old_lr"],
            "schedule": prepared["scheduler_transition"],
        }
    if len(token_counts) != 1:
        raise ValueError(f"Matched parent token counters differ: {sorted(token_counts)}")
    audit["tokens_seen_at_parent"] = token_counts.pop()
    (out / "parent_validation.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
