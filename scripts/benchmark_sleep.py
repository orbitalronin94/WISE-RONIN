"""Isolated sleep/replay benchmark using real LUMEN activations.

The promoted step-3000 checkpoint has an empty slow-memory bank. This benchmark
first measures that production no-op, then creates a diagnostic bank from
frozen training-corpus activations (never holdout data) and compares:
  - no sleep,
  - the current three-update random-noise replay,
  - a 50-update denoising replay candidate.

The diagnostic bank is explicitly not written back to a training checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model.lumen import RoninLumen
from train.dataset import load_jsonl, load_token_cache, pack_token_cache, collate_cached_indices
from train.sleep import sueno


def load_model(checkpoint_path: Path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RoninLumen(v32_ze=bool(state.get("v32_ze", False))).cpu()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return state, model


def eval_metrics(model, tokens, targets):
    model.eval()
    with torch.no_grad():
        output = model(tokens, reset_state=True, collect_diagnostics=False)
        logits = output["logits"][:, :-1].contiguous()
        labels = targets[:, 1:].contiguous()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0
        )
        return float(loss), logits.detach(), output["activations"].detach()


def pooled_activations(activations, tokens):
    mask = tokens.ne(0).unsqueeze(-1).to(activations.dtype)
    return (activations * mask).sum(1) / mask.sum(1).clamp_min(1)


def denoising_sleep(model, records, steps=50, lr=1e-4, noise_std=0.1):
    """Diagnostic candidate: learn noisy-real-activation -> clean activation."""
    memory_mlp = model.memoria.mlp
    optimizer = torch.optim.AdamW(memory_mlp.parameters(), lr=lr)
    last_loss = 0.0
    for _ in range(steps):
        noisy = records + torch.randn_like(records) * noise_std
        prediction = memory_mlp(noisy)
        loss = F.mse_loss(prediction, records)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return last_loss


def reconstruction_mse(model, records, seed=991, noise_std=0.1):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noisy = records + torch.randn(records.shape, generator=generator) * noise_std
    with torch.no_grad():
        return float(F.mse_loss(model.memoria.mlp(noisy), records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank-size", type=int, default=96)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--sleep-steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.bank_size < 8 or args.eval_size < 4 or args.sleep_steps < 1 or args.repeats < 1:
        parser.error("bank-size >= 8, eval-size >= 4, sleep-steps >= 1 and repeats >= 1 are required")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(42)
    random.seed(42)

    checkpoint, reference = load_model(Path(args.checkpoint))
    rows, data_hash = load_jsonl(args.data)
    tokenizer_hash = checkpoint["tokenizer_sha256"]
    cache = load_token_cache(
        args.token_cache, data_hash, tokenizer_hash, 128, len(rows)
    )
    packed, lengths = pack_token_cache(cache, 128)
    needed = args.bank_size + args.eval_size
    if len(rows) < needed:
        raise ValueError(f"Need {needed} frozen training rows; found {len(rows)}")
    indices = list(range(needed))
    tokens = collate_cached_indices(packed, lengths, indices)
    with torch.no_grad():
        reference_output = reference(tokens, reset_state=True, collect_diagnostics=False)
        vectors = pooled_activations(reference_output["activations"], tokens)
    bank_records = vectors[:args.bank_size].clone()
    eval_records = vectors[args.bank_size:].clone()
    if not torch.isfinite(bank_records).all() or (bank_records.norm(dim=-1) == 0).any():
        raise ValueError("Frozen corpus produced invalid memory vectors")

    # Record the real checkpoint state before constructing a diagnostic bank.
    original_count = int((reference.memoria.contador_lenta > 0).sum())
    no_op_started = time.perf_counter()
    original_noop_loss = sueno(reference, device="cpu")
    original_noop_seconds = time.perf_counter() - no_op_started

    def make_variant():
        _, model = load_model(Path(args.checkpoint))
        for vector in bank_records:
            if not model.memoria.registrar_nueva(vector):
                raise RuntimeError("Could not seed diagnostic slow-memory bank")
        if int((model.memoria.contador_lenta > 0).sum()) != args.bank_size:
            raise RuntimeError("Diagnostic memory bank size mismatch")
        return model

    baseline = make_variant()
    baseline_loss, baseline_logits, _ = eval_metrics(baseline, tokens[args.bank_size:], tokens[args.bank_size:])
    baseline_recon = reconstruction_mse(baseline, eval_records)
    baseline_mlp = {k: v.detach().clone() for k, v in baseline.memoria.mlp.state_dict().items()}

    variants = {}
    variant_specs = {
        "current_random_noise_3": ("current", 3),
        "denoising_replay_3": ("denoising", 3),
        "denoising_replay_50": ("denoising", args.sleep_steps),
    }
    for name, (method, updates) in variant_specs.items():
        samples = []
        for repeat in range(args.repeats):
            model = make_variant()
            before = {k: v.detach().clone() for k, v in model.memoria.mlp.state_dict().items()}
            started = time.perf_counter()
            torch.manual_seed(42 + repeat)
            if method == "current":
                replay_loss = float(sueno(model, device="cpu"))
            else:
                replay_loss = denoising_sleep(model, bank_records, steps=updates)
            elapsed = time.perf_counter() - started
            eval_loss, logits, _ = eval_metrics(
                model, tokens[args.bank_size:], tokens[args.bank_size:]
            )
            recon = reconstruction_mse(model, eval_records)
            changed = any(
                not torch.equal(before[k], value)
                for k, value in model.memoria.mlp.state_dict().items()
            )
            changed_non_memory = []
            base_state = baseline.state_dict()
            for key, value in model.state_dict().items():
                if not key.startswith("memoria.mlp.") and not torch.equal(value, base_state[key]):
                    changed_non_memory.append(key)
            samples.append({
                "elapsed_seconds": round(elapsed, 6),
                "replay_loss_last": replay_loss,
                "heldout_denoising_mse": recon,
                "heldout_denoising_mse_delta_vs_no_sleep": round(recon - baseline_recon, 8),
                "next_token_eval_loss": eval_loss,
                "next_token_eval_loss_delta_vs_no_sleep": round(eval_loss - baseline_loss, 8),
                "logit_rms_delta_vs_no_sleep": float((logits - baseline_logits).pow(2).mean().sqrt()),
                "memory_mlp_changed": changed,
                "non_memory_state_keys_changed": changed_non_memory,
                "bank_records_after_sleep": int((model.memoria.contador_lenta > 0).sum()),
            })
        variants[name] = {
            "updates_per_run": updates,
            "repeats": args.repeats,
            "median_elapsed_seconds": statistics.median(x["elapsed_seconds"] for x in samples),
            "median_heldout_denoising_mse": statistics.median(x["heldout_denoising_mse"] for x in samples),
            "median_next_token_eval_loss": statistics.median(x["next_token_eval_loss"] for x in samples),
            "runs": samples,
        }

    report = {
        "schema_version": 1,
        "experiment": "lumen_sleep_replay_ablation",
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "corpus_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "seed": 42,
        "repeats": args.repeats,
        "diagnostic_only": True,
        "diagnostic_bank_source": "pooled hidden activations from frozen training-corpus rows; no holdout rows",
        "original_checkpoint_slow_memory_records": original_count,
        "original_checkpoint_occupancy": original_count / 128,
        "original_checkpoint_sleep_noop": {
            "loss": float(original_noop_loss),
            "duration_seconds": original_noop_seconds,
            "executed_replay": original_count > 0,
        },
        "diagnostic_bank_records": args.bank_size,
        "diagnostic_eval_records": args.eval_size,
        "baseline_no_sleep": {
            "heldout_denoising_mse": baseline_recon,
            "next_token_eval_loss": baseline_loss,
        },
        "variants": variants,
        "interpretation_guard": (
            "This is a mechanism benchmark with a diagnostic bank, not evidence that "
            "the production trainer naturally writes slow-memory records. The diagnostic "
            "bank is not persisted to any training checkpoint."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
