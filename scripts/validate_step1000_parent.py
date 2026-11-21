#!/usr/bin/env python3
"""Fail-closed validation of the immutable step-1000 parent artifact."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

EXPECTED_RUN = "44915003982"
EXPECTED_STEP = 1000
EXPECTED_ARCH = {
    "lumen": "ronin_lumen_v3_1_residual_homeostasis",
    "transformer": "decoder_transformer_4l_d192_h4_ff576_tied_v1",
}


def tensor_hash(mapping):
    digest = hashlib.sha256()
    for name in sorted(mapping):
        value = mapping[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_has_moments(optimizer):
    state = optimizer.get("state", {})
    return bool(state) and any(
        isinstance(slot, dict) and "exp_avg" in slot and "exp_avg_sq" in slot
        for slot in state.values()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--lumen", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    if args.source_run != EXPECTED_RUN:
        raise SystemExit(f"Refusing checkpoint source {args.source_run}; expected {EXPECTED_RUN}")

    data_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    states = {}
    for kind, raw_path in (("lumen", args.lumen), ("transformer", args.transformer)):
        path = Path(raw_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"{kind}: missing/empty checkpoint {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        required = ("model", "optimizer", "scheduler", "rng", "step", "tokens_seen",
                    "data_sha256", "tokenizer_sha256", "training_config")
        missing = [key for key in required if key not in state]
        if missing:
            raise SystemExit(f"{kind}: checkpoint lacks resumable fields {missing}")
        if int(state["step"]) != EXPECTED_STEP:
            raise SystemExit(f"{kind}: expected step 1000, got {state['step']}")
        if state.get("architecture_id") != EXPECTED_ARCH[kind]:
            raise SystemExit(f"{kind}: unexpected architecture {state.get('architecture_id')}")
        if state["data_sha256"] != data_hash or state["tokenizer_sha256"] != tokenizer_hash:
            raise SystemExit(f"{kind}: corpus/tokenizer hash mismatch")
        if not optimizer_has_moments(state["optimizer"]):
            raise SystemExit(f"{kind}: Adam moments are absent; cannot preserve optimizer state")
        if not state["scheduler"] or not state["rng"]:
            raise SystemExit(f"{kind}: scheduler/RNG state missing")
        weights_hash = tensor_hash(state["model"])
        print(
            f"checkpoint_recovered: run={EXPECTED_RUN}, model={kind}, step=1000, "
            f"weights_hash={weights_hash}, optimizer_moments_preserved=True"
        )
        states[kind] = state

    for key in ("seed", "tokens_seen", "data_sha256", "tokenizer_sha256"):
        if states["lumen"].get(key) != states["transformer"].get(key):
            raise SystemExit(f"Matched parent mismatch for {key}")
    print(
        "parent_pair_validated: step=1000, tokens_seen="
        f"{states['lumen']['tokens_seen']}, corpus_sha256={data_hash}, "
        f"tokenizer_sha256={tokenizer_hash}"
    )
    print(
        "historical_step1000_train_loss_reference: "
        "lumen=6.35334, transformer=7.83796 "
        "(run-log observations; not recomputed as checkpoint identity hashes)"
    )


if __name__ == "__main__":
    main()
