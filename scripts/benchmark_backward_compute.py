"""Eight-way 100-step LUMEN backward-cost tournament.

Each candidate is paired with an unmodified control on the same runner, using
the same frozen checkpoint, token cache, row order, seed and optimizer state.
Experimental masks are deliberately labelled by what they do: post-backward
sparsification is not counted as backward-compute savings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import deque
from pathlib import Path

import torch

from model.config import CONFIG
from model.lumen import RoninLumen
from train.dataset import load_jsonl, load_token_cache, pack_token_cache, collate_cached_indices
from train.loss import perdida_total
from train.sleep import sueno
from train.scheduler_transition import CosineTransitionScheduler
from train.plasticity import PlasticityProfile
from train.train import apply_organ_profile, enable_fake_int8

CANDIDATES = {
    "01_surprise_gate": "Skip backward and optimizer update when scalar loss <= 1.8.",
    "02_topk10": "Keep only the largest 10% of each parameter gradient after dense backward.",
    "03_wta8": "Straight-through top-1 activation mask in groups of eight after homeostasis.",
    "04_stopgrad2": "Detach the output of the first two homeostasis layers.",
    "05_lowrank8": "Project each matrix gradient to its rank-8 SVD approximation.",
    "06_lazy_backward": "Compute backward on alternate steps; reuse previous gradients at 0.9 scale.",
    "07_homeostatic_norm": "Normalize each homeostasis output by per-feature RMS.",
    "08_sleep_replay20": "At step 100, replay top 20% consolidated records at 2x parent LR.",
}

def resolve_file(root: Path, candidates: list[str]) -> Path:
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"none of the expected files exist under {root}: {candidates}")

def prepare_inputs(root: Path, max_len: int):
    data_path = resolve_file(root, [
        "data/spanish-v2-plus-personality-250k.jsonl",
        "inputs/data/spanish-v2-plus-personality-250k.jsonl",
    ])
    cache_path = resolve_file(root, [
        "data/spanish-v2-plus-personality-token-cache.pt",
        "inputs/data/spanish-v2-plus-personality-token-cache.pt",
    ])
    tokenizer_path = resolve_file(root, [
        "artifacts/tokenizer.json", "artifacts/spanish-v2/tokenizer.json",
        "inputs/artifacts/tokenizer.json",
    ])
    rows, data_hash = load_jsonl(str(data_path))
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    tokenized = load_token_cache(str(cache_path), data_hash, tokenizer_hash, max_len, len(rows))
    packed, lengths = pack_token_cache(tokenized, max_len)
    return packed, lengths, len(rows), data_hash, tokenizer_hash

def build_model_and_optimizer(checkpoint_path: Path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(state.get("step", -1)) != 389700:
        raise ValueError(f"expected step-389700 checkpoint, got {state.get('step')}")
    if state.get("architecture_id") != "ronin_lumen_v3_3_neuroplasticity":
        raise ValueError(f"unexpected architecture_id: {state.get('architecture_id')}")
    model = RoninLumen(v32_ze=bool(state.get("v32_ze", False))).cpu()
    model.load_state_dict(state["model"], strict=True)
    optimizer_state = state["optimizer"]
    lr = float(optimizer_state["param_groups"][0]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG.weight_decay, foreach=False)
    optimizer.load_state_dict(optimizer_state)
    transition = state["scheduler_transition"]
    scheduler = CosineTransitionScheduler(
        optimizer,
        start_step=transition["start_step"],
        end_step=transition["end_step"],
        old_lr=transition["old_lr"],
        new_lr=transition["new_lr"],
        warmup_steps=transition["warmup_steps"],
        plateau_steps=transition.get("plateau_steps", 0),
        min_lr_ratio=transition.get("min_lr_ratio", 0.0),
        current_step=transition["current_step"],
    )
    scheduler.load_state_dict(transition)
    model.train()
    return state, model, optimizer, scheduler, lr

def attach_candidate(model, name: str):
    handles = []
    if name == "02_topk10":
        def sparse_gradient(grad):
            flat = grad.abs().reshape(-1)
            k = max(1, int(flat.numel() * 0.10))
            threshold = torch.topk(flat, k, sorted=False).values.min()
            return grad * (grad.abs() >= threshold)
        for parameter in model.parameters():
            if parameter.requires_grad:
                handles.append(parameter.register_hook(sparse_gradient))
    elif name == "03_wta8":
        def make_wta(_module, _inputs, output):
            width = output.shape[-1]
            usable = (width // 8) * 8
            if usable == 0:
                return output
            core = output[..., :usable].reshape(*output.shape[:-1], -1, 8)
            winner = core.abs().argmax(dim=-1, keepdim=True)
            mask = torch.zeros_like(core).scatter_(-1, winner, 1.0)
            sparse = core * mask
            return torch.cat((sparse.reshape(*output.shape[:-1], usable), output[..., usable:]), dim=-1)
        for module in model.homeo:
            handles.append(module.register_forward_hook(make_wta))
    elif name == "04_stopgrad2":
        for module in list(model.homeo)[:2]:
            handles.append(module.register_forward_hook(lambda _m, _i, out: out.detach()))
    elif name == "07_homeostatic_norm":
        def rms_normalize(_module, _inputs, output):
            rms = output.float().pow(2).mean(dim=(0, 1), keepdim=True).sqrt().clamp_min(1e-4)
            return output / rms.to(output.dtype)
        for module in model.homeo:
            handles.append(module.register_forward_hook(rms_normalize))
    return handles

def project_low_rank_gradients(model, rank=8):
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None or grad.ndim != 2 or min(grad.shape) <= rank:
            continue
        u, singular, vh = torch.linalg.svd(grad, full_matrices=False)
        parameter.grad.copy_((u[:, :rank] * singular[:rank]) @ vh[:rank, :])

def apply_checkpoint_runtime_profile(model, state):
    profile = state.get("runtime_profile", {})
    plasticity = PlasticityProfile.from_config(profile.get("plasticity", {}))
    for layer in model.delta_layers:
        layer.plasticity_profile = plasticity
        layer.tagging.decay = plasticity.eligibility_retention
        layer.pre_normalize_sequence = bool(profile.get("delta_pre_normalize", False))
        refinements = int(profile.get("delta_parallel_refinements", layer.parallel_refinements))
        if not 1 <= refinements <= 6:
            raise ValueError("delta_parallel_refinements must be between 1 and 6")
        layer.parallel_refinements = refinements
        layer.reuse_write_update = bool(profile.get("delta_reuse_write_update", False))
        layer.always_apply_eligibility = bool(profile.get("delta_always_apply_eligibility", False))
        layer.trace_norm_diagnostics_on_demand = bool(profile.get("delta_trace_norm_diagnostics_on_demand", False))
        layer.share_causal_masks = bool(profile.get("delta_share_causal_masks", False))
    for organ in model.homeo:
        organ.fused_scatter_add = bool(profile.get("homeostasis_fused_scatter", False))
    model.memoria.one_shot_enabled = plasticity.one_shot_enabled
    apply_organ_profile(model, profile.get("organs", {}))
    organs = profile.get("organs", {})
    pain_budget_gain = float(organs.get("pain_budget_gain", 0.10))
    pleasure_budget_gain = float(organs.get("pleasure_budget_gain", 0.05))
    pain_value = float(model.pain.pain.item())
    pleasure_value = float(model.pleasure.pleasure.item())
    model.presupuesto.affect_scale = max(
        0.85, min(1.05, 1.0 - pain_budget_gain * pain_value + pleasure_budget_gain * pleasure_value)
    )
    if int(state["step"]) >= 5000:
        enable_fake_int8(model)


def one_run(label: str, candidate: str | None, checkpoint_path: Path,
            packed, lengths, row_count: int, steps: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    state, model, optimizer, scheduler, parent_lr = build_model_and_optimizer(checkpoint_path)
    apply_checkpoint_runtime_profile(model, state)
    # Match the trainer's exact checkpoint RNG restoration and sampler stream.
    random.setstate(state["rng"]["python"])
    torch.set_rng_state(state["rng"]["torch_cpu"])
    sampler = random.Random()
    sampler.setstate(state["rng"]["local_sampler"])
    hooks = attach_candidate(model, candidate) if candidate else []
    cached_gradients = None
    losses = []
    glucose_loss_window = deque(maxlen=100)
    timings = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0, "replay": 0.0}
    skipped_backward = 0
    replay_status = "not_requested"
    started = time.perf_counter()

    try:
        for index in range(steps):
            indices = sampler.choices(range(row_count), k=8)
            tokens = collate_cached_indices(packed, lengths, indices)
            model.reset_states()
            global_step = int(state["step"]) + index + 1
            collect_diagnostics = (global_step % 100 == 0 or index == 0)
            forward_started = time.perf_counter()
            output = model(tokens, reset_state=True, collect_diagnostics=collect_diagnostics)
            loss, _ = perdida_total(
                output, tokens, model, use_jepa=True, use_rep=True,
                use_free_energy=bool(state.get("v32_ze", False)),
            )
            timings["forward"] += time.perf_counter() - forward_started
            value = float(loss.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at candidate step {index + 1}")
            losses.append(value)
            glucose_loss_window.append(value)

            if candidate == "01_surprise_gate" and value <= 1.8:
                optimizer.zero_grad(set_to_none=True)
                skipped_backward += 1
                continue

            if candidate == "06_lazy_backward" and index % 2 == 1 and cached_gradients is not None:
                optimizer.zero_grad(set_to_none=True)
                for parameter, cached in zip(model.parameters(), cached_gradients):
                    if cached is not None:
                        parameter.grad = cached.mul(0.9)
                skipped_backward += 1
            else:
                optimizer.zero_grad(set_to_none=True)
                backward_started = time.perf_counter()
                loss.backward()
                timings["backward"] += time.perf_counter() - backward_started
                if candidate == "05_lowrank8":
                    project_low_rank_gradients(model, rank=8)
                if candidate == "06_lazy_backward":
                    cached_gradients = [
                        None if p.grad is None else p.grad.detach().clone()
                        for p in model.parameters()
                    ]

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            optimizer_started = time.perf_counter()
            optimizer.step()
            scheduler.step()
            timings["optimizer"] += time.perf_counter() - optimizer_started

            if global_step % 100 == 0 and glucose_loss_window:
                shared_signal = model.presupuesto.check_loss_signal(statistics.fmean(glucose_loss_window))
                pain_state = model.pain.update_signal(shared_signal["regression"])
                pleasure_state = model.pleasure.update_signal(shared_signal["improvement"])
                model.set_learning_modulator(pleasure_state["pleasure"] - pain_state["pain"])
                organs = state.get("runtime_profile", {}).get("organs", {})
                pain_gain = float(organs.get("pain_budget_gain", 0.10))
                pleasure_gain = float(organs.get("pleasure_budget_gain", 0.05))
                model.presupuesto.affect_scale = max(
                    0.85, min(1.05, 1.0 - pain_gain * pain_state["pain"] + pleasure_gain * pleasure_state["pleasure"])
                )

            if candidate == "08_sleep_replay20" and index == steps - 1:
                replay_started = time.perf_counter()
                occupancy = model.memoria.occupancy()
                if occupancy > 0:
                    replay_loss = sueno(
                        model, device="cpu", compression_ratio=0.20,
                        lr=2.0 * parent_lr, downscale=1.0,
                    )
                    replay_status = {"executed": True, "occupancy": occupancy, "loss": float(replay_loss)}
                else:
                    replay_status = {"executed": False, "occupancy": occupancy, "reason": "no consolidated records"}
                timings["replay"] += time.perf_counter() - replay_started
    finally:
        for handle in hooks:
            handle.remove()

    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "candidate": candidate or "main_control",
        "status": "ok",
        "steps": len(losses),
        "mean_loss": statistics.fmean(losses),
        "last_20_loss": statistics.fmean(losses[-20:]),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "elapsed_seconds": elapsed,
        "steps_per_second": len(losses) / elapsed,
        "timings_seconds": timings,
        "skipped_backward_steps": skipped_backward,
        "replay": replay_status,
        "losses": losses,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps != 100:
        raise ValueError("the tournament contract is exactly 100 steps per arm")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = True
    packed, lengths, row_count, data_hash, tokenizer_hash = prepare_inputs(args.inputs, 128)

    # Alternate arm order to reduce systematic cache/warmup bias.
    if int(args.candidate[:2]) % 2 == 0:
        candidate_result = one_run("candidate", args.candidate, args.checkpoint, packed, lengths, row_count, 100, args.seed)
        control_result = one_run("control", None, args.checkpoint, packed, lengths, row_count, 100, args.seed)
    else:
        control_result = one_run("control", None, args.checkpoint, packed, lengths, row_count, 100, args.seed)
        candidate_result = one_run("candidate", args.candidate, args.checkpoint, packed, lengths, row_count, 100, args.seed)

    loss_better = candidate_result["mean_loss"] < control_result["mean_loss"] and candidate_result["last_20_loss"] <= control_result["last_20_loss"]
    time_better = candidate_result["elapsed_seconds"] < control_result["elapsed_seconds"]
    if args.candidate == "08_sleep_replay20":
        exercised = bool(candidate_result["replay"].get("executed", False))
    elif args.candidate in {"01_surprise_gate", "06_lazy_backward"}:
        exercised = candidate_result["skipped_backward_steps"] > 0
    else:
        exercised = True
    approved = bool(loss_better and time_better and exercised)
    report = {
        "candidate": args.candidate,
        "hypothesis": CANDIDATES[args.candidate],
        "decision": "APPROVED" if approved else "PENDING",
        "criteria": {
            "mean_loss_lower": loss_better,
            "last_20_loss_not_higher": candidate_result["last_20_loss"] <= control_result["last_20_loss"],
            "elapsed_time_lower": time_better,
            "mechanism_exercised": exercised,
        },
        "loss_improvement_percent": 100 * (control_result["mean_loss"] - candidate_result["mean_loss"]) / max(abs(control_result["mean_loss"]), 1e-12),
        "time_improvement_percent": 100 * (control_result["elapsed_seconds"] - candidate_result["elapsed_seconds"]) / max(control_result["elapsed_seconds"], 1e-12),
        "parent_checkpoint_step": 389700,
        "parent_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "corpus_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "seed": args.seed,
        "control": control_result,
        "candidate_result": candidate_result,
        "interpretation": {
            "02_topk10": "Dense backward still runs; this tests sparse gradient updates, not reduced backward FLOPs.",
            "03_wta8": "Mask is applied after dense layer computation; no dense-matmul FLOP saving is claimed.",
            "05_lowrank8": "SVD projects dense gradients after backward; this tests update quality/cost, not LoRA compute savings.",
            "04_stopgrad2": "This changes the learning graph and is not an equivalent optimizer shortcut.",
            "01_surprise_gate": "The scalar threshold is a hypothesis; skipped steps still contribute measured loss.",
            "06_lazy_backward": "Stale gradients are an approximation; inspect loss windows and skipped-step count.",
            "07_homeostatic_norm": "This changes activation dynamics and is not a drop-in normalization equivalence.",
            "08_sleep_replay20": "Only counts as tested if consolidated records exist and replay executes.",
        }.get(args.candidate, ""),
        "contract": {
            "steps_per_arm": 100,
            "same_runner_sequential_pair": True,
            "same_parent_checkpoint": True,
            "same_seed_and_batch_order": True,
            "same_token_cache_and_tokenizer": True,
            "optimizer_state_restored_from_parent": True,
            "runtime": "CPU, 4 intra-op threads, 2 inter-op threads, MKLDNN on",
            "timing": "end-to-end measured training loop; forward/backward/optimizer/replay separately",
            "holdout": "not used; this is a short training microbenchmark",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
