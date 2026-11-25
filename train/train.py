"""CPU-only RONIN-LUMEN trainer with reproducible batch-boundary resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from collections import deque
from pathlib import Path

import torch

from model.config import CONFIG, LumenConfig
from model.lumen import RoninLumen
from model.plasticity import PlasticityProfile
from train.curriculum import datos_para_fase, sample_phase2
from train.dataset import (BPETokenizer, IndexedJsonlCorpus, collate_rows, collate_cached_indices,
                            load_jsonl, load_token_cache, pack_token_cache)
from train.loss import perdida_total
from train.sleep import sueno
from train.quant import fake_int8_tensor
from train.scheduler_transition import CosineTransitionScheduler
from train.data_transition import resolve_data_transition

LOG = logging.getLogger("lumen.train")


def should_surprise_gate(loss_value: float, threshold: float | None) -> bool:
    """Skip a parameter update when the observed loss is already below threshold."""
    return threshold is not None and float(loss_value) <= float(threshold)


def should_reuse_cached_gradient(
    lazy_backward: bool, local_step_index: int, cached_gradients
) -> bool:
    """Reuse the previous full gradient on alternating eligible local steps."""
    return bool(lazy_backward and local_step_index % 2 == 1 and cached_gradients is not None)



def _tensor_health(value):
    if not isinstance(value, torch.Tensor):
        return str(value)
    tensor = value.detach()
    if tensor.numel() == 0:
        return f"shape={tuple(tensor.shape)} empty"
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    finite_values = tensor[finite]
    if finite_count:
        max_abs = float(finite_values.abs().max().item())
        rms = float(finite_values.float().pow(2).mean().sqrt().item())
    else:
        max_abs = float("nan")
        rms = float("nan")
    return (
        f"shape={tuple(tensor.shape)} finite={finite_count}/{tensor.numel()} "
        f"max_abs={max_abs:.6g} rms={rms:.6g}"
    )


def log_nonfinite_diagnostics(model, output, metrics, loss, optimizer, step, reason):
    """Emit the first useful numerical boundary when a training step becomes invalid."""
    LOG.error("NONFINITE_DIAGNOSTICS step=%d reason=%s loss=%s", step, reason, _tensor_health(loss))
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            LOG.error("NONFINITE_OUTPUT step=%d %s %s", step, key, _tensor_health(value))
    for key, value in metrics.items():
        LOG.error("NONFINITE_LOSS_COMPONENT step=%d %s %s", step, key, _tensor_health(value))
    for index, layer in enumerate(model.delta_layers):
        LOG.error("NONFINITE_DELTA_STATE step=%d layer=%d %s", step, index, _tensor_health(layer.state))
    LOG.error(
        "NONFINITE_MEMORY step=%d occupancy=%.4f lenta=%s counters=%s",
        step, model.memoria.occupancy(),
        _tensor_health(model.memoria.lenta),
        _tensor_health(model.memoria.contador_lenta),
    )
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            LOG.error("NONFINITE_PARAMETER step=%d name=%s %s", step, name, _tensor_health(parameter))
    for index, (name, buffer) in enumerate(model.named_buffers()):
        if buffer.is_floating_point() and buffer.numel() and not torch.isfinite(buffer).all():
            LOG.error("NONFINITE_BUFFER step=%d name=%s %s", step, name, _tensor_health(buffer))
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel():
                if not torch.isfinite(value).all():
                    LOG.error("NONFINITE_OPTIMIZER_STATE step=%d key=%s %s", step, key, _tensor_health(value))


def capture_rng(local_rng):
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "local_sampler": local_rng.getstate(),
    }


def restore_rng(state, local_rng):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    local_rng.setstate(state["local_sampler"])


def save_checkpoint(path, model, optimizer, scheduler, step, args, data_hash, local_rng, tokens_seen, tokenizer_hash, layer_diagnostics_history, sleep_events):
    model.reset_states()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "architecture_id": "ronin_lumen_v3_3_neuroplasticity",
        "architecture": {
            "model": "RoninLumen", "config": model.config.to_dict(),
            "v32_ze": args.v32_ze, "pilot_all_rows": args.pilot_all_rows,
            "no_sleep": args.no_sleep, "no_jepa": args.no_jepa,
            "no_rep": args.no_rep,
            "neuroplasticity": {
                "parallel_delta": True,
                "persistent_eligibility_traces": True,
                "delayed_modulator": "pleasure_minus_pain",
                "adaptive_chunking": True,
                "chunk_size_bounds": [4, 16],
                "replay_enabled": not args.no_sleep,
                "homeostasis_enabled": True,
                "plasticity_profile": getattr(args, "runtime_profile_settings", {}).get("plasticity", {"mode": "neutral"}),
            },
        },
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": optimizer.state_dict(),
        "compute_policy": {
            "lazy_backward": bool(args.lazy_backward),
            "lazy_gradient_decay": 0.9 if args.lazy_backward else None,
            "surprise_gate_loss": args.surprise_gate_loss,
            "scheduler_advances_on_observation": True,
            "cached_gradients_checkpointed": False,
        },
        "scheduler": scheduler.state_dict(),
        "step": step,
        "phase": args.phase,
        "seed": args.seed,
        "config": model.config.to_dict(),
        "data_sha256": data_hash,
        "data_lineage": list(getattr(args, "data_lineage", [])),
        "rng": capture_rng(local_rng),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "runtime_profile": getattr(args, "runtime_profile_settings", {}),
        "delta_policy": "transient_state_reset_persistent_eligibility_trace",
        "v32_ze": args.v32_ze,
        "pilot_all_rows": args.pilot_all_rows,
        "tokens_seen": tokens_seen,
        "tokenizer_sha256": tokenizer_hash,
        "training_config": {
            "batch": args.batch, "max_len": args.max_len, "lr": args.lr,
            "warmup": args.warmup, "total_steps": args.total_steps,
            "weight_decay": CONFIG.weight_decay,
            **({
                "scheduler_mode": "cosine_transition_v1",
                "transition_start": args.transition_start,
                "transition_end": args.transition_end,
                "warmup_transition": args.warmup_transition,
                "transition_plateau_steps": args.transition_plateau_steps,
                "transition_min_lr_ratio": args.transition_min_lr_ratio,
                "grad_clip": args.grad_clip,
            } if args.scheduler_transition else {}),
        },
        "scheduler_transition": scheduler.state_dict() if args.scheduler_transition else None,
        "layer_diagnostics_history": list(layer_diagnostics_history),
        "sleep_events": list(sleep_events),
    }, path)
    LOG.info("checkpoint=%s step=%d", path, step)


def make_scheduler(optimizer, warmup, total):
    def scale(step):
        if step < warmup:
            return max(step, 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def enable_fake_int8(model):
    model.heads.kan.register_forward_hook(
        lambda _module, _inputs, output: fake_int8_tensor(output)
    )
    model.memoria.mlp.register_forward_hook(
        lambda _module, _inputs, output: fake_int8_tensor(output)
    )
    LOG.info("fake-int8 activation simulation enabled")


def migrate_legacy_homeostasis(model, optimizer):
    """Reset legacy sparse-gate thresholds when entering residual V3.1.

    V3.0 learned large subtractive biases while Top-K output replaced the
    representation. V3.1 adds the gated update to the residual stream, so
    carrying those thresholds forward can suppress every update. Reset only
    the homeostasis-specific state; preserve all other model and optimizer
    state from the step-500 parent.
    """
    reset_units = 0
    with torch.no_grad():
        for organ in model.homeo:
            organ.bias.zero_()
            organ.m.fill_(organ.m_target)
            organ.effective_m.zero_()
            reset_units += organ.dim
            state = optimizer.state.get(organ.bias, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    value.zero_()
    LOG.warning(
        "V3.1 migration: reset homeostasis bias/EMA and AdamW moments "
        "for %d units; all other parent optimizer state preserved",
        reset_units,
    )



def apply_organ_profile(model, profile):
    """Apply bounded organ-parameter hypotheses without disabling organs."""
    allowed = {"glucose", "pain", "pleasure", "homeostasis", "memory", "pain_budget_gain", "pleasure_budget_gain"}
    unknown = set(profile) - allowed
    if unknown: raise ValueError(f"Unknown organ profile keys: {sorted(unknown)}")
    glucose_map = {"step": "GLUCOSE_STEP", "decay": "GLUCOSE_DECAY", "improvement_threshold": "IMPROVEMENT_THRESHOLD"}
    for key, value in profile.get("glucose", {}).items():
        if key not in glucose_map: raise ValueError(f"Unknown glucose parameter: {key}")
        value = float(value)
        if key == "step" and value < 0: raise ValueError("glucose step must be non-negative")
        if key == "decay" and not 0 <= value < 1: raise ValueError("glucose decay must be in [0,1)")
        if key == "improvement_threshold" and value < 0: raise ValueError("glucose threshold must be non-negative")
        setattr(model.presupuesto, glucose_map[key], value)
    for name in ("pain", "pleasure"):
        organ = getattr(model, name)
        for key, value in profile.get(name, {}).items():
            if key not in {"ema", "deadband", "full_scale", "response", "decay"}: raise ValueError(f"Unknown {name} parameter: {key}")
            setattr(organ, key, float(value))
        if not (0 <= organ.ema < 1 and organ.deadband >= 0 and organ.full_scale > organ.deadband and 0 < organ.response <= 1 and 0 <= organ.decay < 1):
            raise ValueError(f"Invalid {name} organ profile")
    for key, value in profile.get("homeostasis", {}).items():
        if key == "eta":
            value = float(value)
            if value < 0: raise ValueError("homeostasis eta must be non-negative")
            for organ in model.homeo: organ.eta = value
        elif key == "target":
            value = float(value)
            if not 0 <= value <= 1: raise ValueError("homeostasis target must be in [0,1]")
            for organ in model.homeo: organ.m_target = value
        elif key == "k_top":
            value = int(value)
            if not 1 <= value <= model.config.model_dim: raise ValueError("homeostasis k_top out of range")
            for organ in model.homeo: organ.k_top = value
        elif key == "synaptic_scaling":
            if not isinstance(value, bool): raise ValueError("homeostasis synaptic_scaling must be boolean")
            for organ in model.homeo: organ.synaptic_scaling_enabled = value
        else: raise ValueError(f"Unknown homeostasis parameter: {key}")
    for key, value in profile.get("memory", {}).items():
        if key not in {"small_pe", "large_pe"}: raise ValueError(f"Unknown memory parameter: {key}")
        setattr(model.memoria, key, float(value))
    if not 0 <= model.memoria.small_pe < model.memoria.large_pe: raise ValueError("memory thresholds must satisfy 0 <= small_pe < large_pe")
    for key in ("pain_budget_gain", "pleasure_budget_gain"):
        if key in profile and not 0 <= float(profile[key]) <= 1: raise ValueError(f"{key} must be in [0,1]")
    LOG.info("organ_profile applied: %s", json.dumps(profile, sort_keys=True))

def main():
    global CONFIG
    parser = argparse.ArgumentParser(description="RONIN-LUMEN CPU training")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100, help="Training metric log interval")
    parser.add_argument("--checkpoint-every", type=int, default=250, help="Named checkpoint interval")
    parser.add_argument("--recovery-checkpoint-every", type=int, default=0, help="Overwrite latest_good.pt every N successful updates; 0 disables")
    parser.add_argument("--parameter-check-every", type=int, default=25, help="Full parameter and optimizer-state finite scan interval; loss and gradients are still checked every step")
    parser.add_argument("--threads", type=int, default=2,
                        help="PyTorch intra-op CPU threads; benchmark 1/2/4 on the target runner")
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--disable-mkldnn", action="store_true")
    parser.add_argument("--disable-foreach", action="store_true")
    parser.add_argument("--cpu-affinity", action="store_true")
    parser.add_argument("--runtime-profile", default="config/lumen_cpu_profile.json", help="JSON runtime profile generated by the hotspot benchmark")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead"),
                        help="Compile only non-diagnostic forward passes; checkpoints remain eager-model state")
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--model-config", help="JSON LumenConfig override for isolated architecture experiments")
    parser.add_argument("--resume")
    parser.add_argument("--data", default="data/train_25k.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--token-cache", help="Validated pre-tokenized corpus cache")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--batch", type=int, default=CONFIG.batch_size)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=CONFIG.warmup_steps)
    parser.add_argument("--total-steps", type=int, default=CONFIG.total_steps)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--allow-data-change-on-resume", action="store_true",
                        help="Allow an explicitly declared corpus expansion while preserving optimizer/model state")
    parser.add_argument("--parent-data-sha256", default=None)
    parser.add_argument("--data-change-label", default="explicit corpus transition")
    parser.add_argument("--lazy-backward", action="store_true",
                        help="Reuse the previous full gradient, decayed by 0.9, on alternating eligible steps")
    parser.add_argument("--surprise-gate-loss", type=float, default=None,
                        help="Skip backward and optimizer update when scalar loss is <= this threshold")
    parser.add_argument("--scheduler-transition", action="store_true")
    parser.add_argument("--restart-scheduler-transition", action="store_true",
                        help="Start a new cosine segment from an exhausted transition checkpoint")
    parser.add_argument("--transition-start", type=int, default=1000)
    parser.add_argument("--transition-end", type=int, default=2000)
    parser.add_argument("--warmup-transition", type=int, default=20)
    parser.add_argument("--transition-plateau-steps", type=int, default=0)
    parser.add_argument("--transition-min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--wall-clock-file")
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--no-jepa", action="store_true")
    parser.add_argument("--no-rep", action="store_true")
    parser.add_argument("--orpo", action="store_true")
    parser.add_argument("--v32-ze", action="store_true",
                        help="Enable experimental V3.2 dual-state/LC-NE/energy mechanisms")
    parser.add_argument("--pilot-all-rows", action="store_true",
                        help="Use every row in the explicitly authorized provisional pilot corpus; phase 1 only")
    parser.add_argument("--dummy-m1", action="store_true",
                        help="Explicit synthetic smoke test only; never a training dataset")
    args = parser.parse_args()

    if args.model_config:
        model_config_data = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        CONFIG = LumenConfig(**model_config_data)
        LOG.info("model_config loaded: %s", args.model_config)
        args.lr = CONFIG.learning_rate
        args.batch = CONFIG.batch_size
        args.warmup = CONFIG.warmup_steps
        args.total_steps = CONFIG.total_steps
        args.seed = CONFIG.seed

    if args.steps < 1 or (args.steps > 10000 and not args.dummy_m1):
        parser.error("--steps must be between 1 and 10000")
    if args.log_every < 1 or args.checkpoint_every < 1 or args.recovery_checkpoint_every < 0 or args.parameter_check_every < 1:
        parser.error("log/checkpoint/parameter-check intervals must be positive; recovery interval may be zero")
    if args.surprise_gate_loss is not None and args.surprise_gate_loss < 0:
        parser.error("--surprise-gate-loss must be non-negative")
    if args.threads < 1 or args.interop_threads < 1:
        parser.error("--threads and --interop-threads must be at least 1")
    if args.orpo:
        parser.error("ORPO is not implemented: preference pairs and objective need a separate approved specification")
    if args.dummy_m1 and args.steps > 10:
        parser.error("--dummy-m1 is limited to blocks of at most 10 steps")
    if args.pilot_all_rows and (args.dummy_m1 or args.phase != 1):
        parser.error("--pilot-all-rows is only valid for real-data phase 1")
    if args.scheduler_transition and not args.resume:
        parser.error("--scheduler-transition requires an exact resumable parent checkpoint")
    if args.restart_scheduler_transition and not args.scheduler_transition:
        parser.error("--restart-scheduler-transition requires --scheduler-transition")

    runtime_profile = {}
    if args.runtime_profile and Path(args.runtime_profile).is_file():
        runtime_profile = json.loads(Path(args.runtime_profile).read_text(encoding="utf-8"))
        if not isinstance(runtime_profile, dict):
            raise ValueError("CPU runtime profile must be a JSON object")
        args.threads = int(runtime_profile.get("threads", args.threads))
        args.interop_threads = int(runtime_profile.get("interop_threads", args.interop_threads))
        args.disable_mkldnn = not bool(runtime_profile.get("mkldnn_enabled", not args.disable_mkldnn))
        args.disable_foreach = not bool(runtime_profile.get("foreach", not args.disable_foreach))
        args.cpu_affinity = bool(runtime_profile.get("cpu_affinity", args.cpu_affinity))
        LOG.info("runtime_profile loaded and applied: %s", args.runtime_profile)
    elif args.runtime_profile:
        LOG.warning("runtime profile not found; using explicit/default CLI CPU settings: %s", args.runtime_profile)
    args.runtime_profile_settings = dict(runtime_profile)
    organ_profile = runtime_profile.get("organs", {})
    sleep_downscale = float(runtime_profile.get("sleep_downscale", 1.0))
    replay_compression = float(runtime_profile.get("replay_compression", 1.0))
    sleep_interval_steps = int(runtime_profile.get("sleep_interval_steps", CONFIG.sleep_interval_steps))
    if not 0.90 <= sleep_downscale <= 1.0:
        raise ValueError("sleep_downscale must be in [0.90,1]")
    if not 0.1 <= replay_compression <= 1.0:
        raise ValueError("replay_compression must be in [0.1,1]")
    if sleep_interval_steps < 1:
        raise ValueError("sleep_interval_steps must be positive")
    pain_budget_gain = float(organ_profile.get("pain_budget_gain", 0.10))
    pleasure_budget_gain = float(organ_profile.get("pleasure_budget_gain", 0.05))
    if not 0.0 <= pain_budget_gain <= 1.0 or not 0.0 <= pleasure_budget_gain <= 1.0:
        raise ValueError("organ budget gains must be in [0,1]")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.backends.mkldnn.enabled = not args.disable_mkldnn
    affinity_cpus = None
    if args.cpu_affinity and hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        affinity_cpus = available[:args.threads]
        os.sched_setaffinity(0, set(affinity_cpus))
    LOG.info("cpu_runtime: intraop_threads=%d interop_threads=%d mkldnn=%s foreach=%s affinity=%s",
             args.threads, args.interop_threads, torch.backends.mkldnn.enabled,
             not args.disable_foreach, affinity_cpus)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")
    model = RoninLumen(config=CONFIG, v32_ze=args.v32_ze).to(device)
    plasticity_profile = PlasticityProfile.from_config(runtime_profile.get("plasticity", {}))
    for layer in model.delta_layers:
        layer.plasticity_profile = plasticity_profile
        layer.tagging.decay = plasticity_profile.eligibility_retention
        layer.pre_normalize_sequence = bool(runtime_profile.get("delta_pre_normalize", False))
        refinements = int(runtime_profile.get("delta_parallel_refinements", layer.parallel_refinements))
        if not 1 <= refinements <= 6:
            raise ValueError("delta_parallel_refinements must be between 1 and 6")
        layer.parallel_refinements = refinements
        # Exact-equivalence/runtime Delta fastpaths are controlled by the
        # frozen runtime profile; defaults preserve the promoted baseline.
        layer.reuse_write_update = bool(runtime_profile.get("delta_reuse_write_update", False))
        layer.always_apply_eligibility = bool(runtime_profile.get("delta_always_apply_eligibility", False))
        layer.trace_norm_diagnostics_on_demand = bool(runtime_profile.get("delta_trace_norm_diagnostics_on_demand", False))
        layer.share_causal_masks = bool(runtime_profile.get("delta_share_causal_masks", False))
    for organ in model.homeo:
        organ.fused_scatter_add = bool(runtime_profile.get("homeostasis_fused_scatter", False))
    model.memoria.one_shot_enabled = plasticity_profile.one_shot_enabled
    LOG.info("plasticity_profile: %s", plasticity_profile)
    LOG.info("lumen_fastpaths: %s", {
        "delta_pre_normalize": bool(runtime_profile.get("delta_pre_normalize", False)),
        "delta_parallel_refinements": refinements,
        "delta_reuse_write_update": bool(runtime_profile.get("delta_reuse_write_update", False)),
        "delta_always_apply_eligibility": bool(runtime_profile.get("delta_always_apply_eligibility", False)),
        "delta_trace_norm_diagnostics_on_demand": bool(runtime_profile.get("delta_trace_norm_diagnostics_on_demand", False)),
        "delta_share_causal_masks": bool(runtime_profile.get("delta_share_causal_masks", False)),
        "homeostasis_fused_scatter": bool(runtime_profile.get("homeostasis_fused_scatter", False)),
    })

    if args.dummy_m1:
        data_hash, tokenizer, phase_data = "dummy-m1-no-corpus", None, None
    else:
        rows, data_hash = load_jsonl(args.data)
        if args.token_cache and isinstance(rows, IndexedJsonlCorpus):
            raise ValueError("Large sharded corpora use lazy indexed sampling; omit --token-cache")
        tokenizer = BPETokenizer(args.tokenizer)
        data_path = Path(args.data)
        manifest_path = data_path / "manifest.json" if data_path.is_dir() else (
            data_path if data_path.name == "manifest.json" else None
        )
        if manifest_path is not None:
            corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual_tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
            if corpus_manifest.get("tokenizer_sha256") != actual_tokenizer_hash:
                raise ValueError("Corpus manifest tokenizer SHA-256 does not match --tokenizer")
        if tokenizer.tokenizer.get_vocab_size() != CONFIG.vocab_size:
            raise ValueError("Tokenizer vocabulary size must match model config")
        if tokenizer.tokenizer.token_to_id("[PAD]") != 0:
            raise ValueError("[PAD] must be token ID 0")
        phase_data = (
            rows if args.pilot_all_rows or isinstance(rows, IndexedJsonlCorpus)
            else datos_para_fase(rows, args.phase)
        )
        if isinstance(rows, IndexedJsonlCorpus):
            LOG.info("using lazy indexed corpus rows=%d; no full-corpus row materialization", len(rows))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=CONFIG.weight_decay,
        foreach=not args.disable_foreach,
    )
    scheduler = None if args.scheduler_transition else make_scheduler(optimizer, args.warmup, args.total_steps)
    start, state = 0, None
    tokens_seen, tokenizer_hash = 0, "dummy-no-tokenizer"
    layer_diagnostics_history, sleep_events = [], []
    if not args.dummy_m1:
        tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    tokenized_rows, row_indices = None, {}
    packed_token_rows, token_lengths = None, None
    if args.token_cache:
        if args.dummy_m1:
            raise ValueError("--token-cache cannot be used with --dummy-m1")
        tokenized_rows = load_token_cache(
            args.token_cache, data_hash, tokenizer_hash, args.max_len, len(rows)
        )
        row_indices = {id(row): index for index, row in enumerate(rows)}
        packed_token_rows, token_lengths = pack_token_cache(tokenized_rows, args.max_len)
        tokenized_rows = None
        LOG.info("loaded packed token cache rows=%d max_len=%d", len(rows), args.max_len)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        parent_arch = state.get("architecture_id", "legacy_preversioned")
        allowed_parent_arches = {
            "ronin_lumen_v3_1_residual_homeostasis",
            "ronin_lumen_v3_2_parallel_delta",
            "ronin_lumen_v3_3_neuroplasticity",
            "ronin_lumen_v3_pilot",
            "legacy_preversioned",
        }
        if parent_arch not in allowed_parent_arches:
            raise ValueError(f"Unsupported LUMEN parent architecture: {parent_arch}")
        needs_homeostasis_migration = parent_arch in {"ronin_lumen_v3_pilot", "legacy_preversioned"}
        if needs_homeostasis_migration and int(state.get("step", 0)) != 500:
            raise ValueError("Only the frozen 500-step legacy pilot checkpoint may migrate to residual homeostasis")
        if needs_homeostasis_migration:
            LOG.warning("Legacy architecture transition at step 500: sparse homeostatic updates now preserve the residual stream")
        if state.get("v32_ze", False) != args.v32_ze:
            raise ValueError("Checkpoint V3.2 feature flag mismatch")
        expected_training_config = {
            "batch": args.batch, "max_len": args.max_len, "lr": args.lr,
            "warmup": args.warmup, "total_steps": args.total_steps,
            "weight_decay": CONFIG.weight_decay,
            **({
                "scheduler_mode": "cosine_transition_v1",
                "transition_start": args.transition_start,
                "transition_end": args.transition_end,
                "warmup_transition": args.warmup_transition,
                "transition_plateau_steps": args.transition_plateau_steps,
                "transition_min_lr_ratio": args.transition_min_lr_ratio,
                "grad_clip": args.grad_clip,
            } if args.scheduler_transition else {}),
        }
        if args.scheduler_transition and not state.get("scheduler_transition") and int(state.get("step", -1)) == args.transition_start:
            parent_config = state.get("training_config") or {}
            required_parent = {
                "batch": args.batch, "max_len": args.max_len, "lr": args.lr,
                "warmup": args.warmup, "weight_decay": CONFIG.weight_decay,
            }
            if any(parent_config.get(key) != value for key, value in required_parent.items()):
                raise ValueError("Parent checkpoint batch/length/LR/warmup/weight-decay mismatch")
            if int(parent_config.get("total_steps", -1)) < args.transition_start:
                raise ValueError("Parent scheduler horizon ends before the transition step")
        elif args.restart_scheduler_transition:
            parent_config = state.get("training_config") or {}
            stable_keys = ("batch", "max_len", "lr", "warmup", "weight_decay", "grad_clip")
            mismatches = [key for key in stable_keys
                          if parent_config.get(key) != expected_training_config.get(key)]
            saved_transition = state.get("scheduler_transition")
            saved_step = int(state.get("step", -1))
            scheduler_step = int(saved_transition.get("current_step", -1)) if saved_transition else -1
            if mismatches or not saved_transition or scheduler_step not in (saved_step, saved_step + 1):
                raise ValueError(
                    f"Cannot restart scheduler from this checkpoint; mismatches={mismatches}, "
                    f"checkpoint_step={saved_step}, scheduler_step={scheduler_step}"
                )
        elif state.get("training_config") not in (None, expected_training_config):
            raise ValueError("Checkpoint training hyperparameters mismatch")
        if state.get("seed", args.seed) != args.seed:
            raise ValueError("Checkpoint seed mismatch")
        saved_model_config = dict(state.get("config") or {})
        # Backward-compatible defaults for architecture fields added after
        # older checkpoints were written. Missing keys resolve only to the
        # original base defaults, never to an experimental override.
        for key, value in LumenConfig().to_dict().items():
            saved_model_config.setdefault(key, value)
        if saved_model_config != CONFIG.to_dict():
            raise ValueError("Checkpoint model config mismatch")
        args.data_lineage = list(state.get("data_lineage", []))
        transition = resolve_data_transition(
            state.get("data_sha256"), data_hash,
            allow_change=args.allow_data_change_on_resume,
            expected_parent_hash=args.parent_data_sha256,
            step=int(state.get("step", -1)),
            label=args.data_change_label,
        )
        if transition:
            args.data_lineage.append(transition)
            LOG.warning("explicit_data_transition %s", json.dumps(transition, sort_keys=True))
        if state["phase"] != args.phase:
            raise ValueError("Cannot change curriculum phase when resuming a checkpoint")
        if args.pilot_all_rows and (
            state.get("pilot_all_rows") is not True
            or state.get("tokenizer_sha256") != tokenizer_hash
        ):
            raise ValueError("Pilot resume requires the same all-row mode and tokenizer hash")
        incompatible = model.load_state_dict(state["model"], strict=False)
        allowed_missing = {f"homeo.{i}.effective_m" for i in range(len(model.homeo))}
        allowed_missing |= {"pain.loss_ema", "pain.pain", "pain.last_regression",
                            "pleasure.loss_ema", "pleasure.pleasure", "pleasure.last_improvement"}
        # Step-155700 v3.2 parents predate persistent Delta eligibility buffers.
        # The model registers these buffers as zeros, so retaining their neutral
        # initialization is an exact, parameter-free state migration.
        allowed_missing |= {
            f"delta_layers.{i}.eligibility_trace" for i in range(len(model.delta_layers))
        }
        allowed_missing |= {
            f"delta_layers.{i}.eligibility_modulator" for i in range(len(model.delta_layers))
        }
        allowed_missing |= {f"homeo.{i}.synaptic_scale" for i in range(len(model.homeo))}
        # New experimental plasticity buffers are neutral on migration from
        # pre-organ checkpoints; no trained parameters or optimizer moments are reset.
        for i in range(len(model.delta_layers)):
            allowed_missing |= {
                f"delta_layers.{i}.tagging.tag",
                f"delta_layers.{i}.tagging.captured",
                f"delta_layers.{i}.intrinsic_threshold.threshold",
                f"delta_layers.{i}.simulated_pruner.usage",
                f"delta_layers.{i}.simulated_pruner.mask",
            }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing - allowed_missing or unexpected:
            raise ValueError(
                f"Checkpoint architecture mismatch: missing={sorted(missing - allowed_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if missing:
            LOG.warning(
                "Checkpoint state migration: initialized %d new persistent buffers to their neutral defaults: %s",
                len(missing), sorted(missing),
            )
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["foreach"] = not args.disable_foreach
        if needs_homeostasis_migration:
            migrate_legacy_homeostasis(model, optimizer)
        start = int(state["step"])
        if args.scheduler_transition:
            saved_transition = state.get("scheduler_transition")
            if args.restart_scheduler_transition:
                old_lr = float(optimizer.param_groups[0]["lr"])
                scheduler = CosineTransitionScheduler(
                    optimizer, start_step=start, end_step=args.transition_end,
                    old_lr=old_lr, new_lr=args.lr, warmup_steps=args.warmup_transition,
                    plateau_steps=args.transition_plateau_steps,
                    min_lr_ratio=args.transition_min_lr_ratio,
                )
                scheduler.step()
                LOG.info("lr_transition_restart: step=%d from=%.8g to=%.8g optimizer_moments_preserved=True",
                         start, old_lr, args.lr)
            elif saved_transition:
                scheduler = CosineTransitionScheduler(
                    optimizer, start_step=saved_transition["start_step"],
                    end_step=saved_transition["end_step"], old_lr=saved_transition["old_lr"],
                    new_lr=saved_transition["new_lr"], warmup_steps=saved_transition["warmup_steps"],
                    plateau_steps=saved_transition.get("plateau_steps", 0),
                    min_lr_ratio=saved_transition.get("min_lr_ratio", 0.0),
                    current_step=saved_transition["current_step"],
                )
                scheduler.load_state_dict(saved_transition)
            elif start == args.transition_start:
                old_lr = float(optimizer.param_groups[0]["lr"])
                scheduler = CosineTransitionScheduler(
                    optimizer, start_step=args.transition_start, end_step=args.transition_end,
                    old_lr=old_lr, new_lr=args.lr, warmup_steps=args.warmup_transition,
                    plateau_steps=args.transition_plateau_steps,
                    min_lr_ratio=args.transition_min_lr_ratio,
                )
                LOG.info(
                    "lr_transition: step=%d, from=%.8g to=%.8g (LUMEN), "
                    "weights_preserved=True, optimizer_moments_preserved=True",
                    start, old_lr, args.lr,
                )
                scheduler.step()  # Set the LR for the first update after the parent checkpoint.
            else:
                raise ValueError("scheduler-transition requires the exact step-1000 parent or a transition checkpoint")
        else:
            scheduler = make_scheduler(optimizer, args.warmup, args.total_steps)
            scheduler.load_state_dict(state["scheduler"])
        tokens_seen = int(state.get("tokens_seen", 0))
        layer_diagnostics_history = list(state.get("layer_diagnostics_history", []))
        sleep_events = list(state.get("sleep_events", []))
        pain_value = float(model.pain.pain.item())
        pleasure_value = float(model.pleasure.pleasure.item())
        model.presupuesto.affect_scale = max(0.85, min(1.05, 1.0 - pain_budget_gain * pain_value + pleasure_budget_gain * pleasure_value))
        LOG.info("resumed step=%d tokens_seen=%d pain=%.3f pleasure=%.3f affect_scale=%.3f",
                 start, tokens_seen, pain_value, pleasure_value, model.presupuesto.affect_scale)

    apply_organ_profile(model, organ_profile)

    phase_ranges = {1: (0, 2000), 2: (2000, 8000), 3: (8000, 10000)}
    low, high = ((0, args.total_steps) if args.pilot_all_rows else phase_ranges[args.phase])
    if not args.dummy_m1 and not (low <= start < high and start + args.steps <= high):
        raise ValueError(
            f"Phase {args.phase} accepts start in [{low},{high}) and must end by {high}; "
            f"got start={start}, steps={args.steps}"
        )

    local_rng = random.Random(args.seed + start)
    if state is not None:
        restore_rng(state["rng"], local_rng)

    quant_enabled = start >= 5000 and not args.dummy_m1
    if quant_enabled:
        enable_fake_int8(model)

    sleep_done = bool(model.memoria.occupancy() >= 0.8)
    model.train()
    compiled_model = None
    if args.compile_mode:
        compiled_model = torch.compile(model, mode=args.compile_mode)
        LOG.info("torch_compile enabled: mode=%s; diagnostic forwards remain eager", args.compile_mode)
    glucose_loss_window = deque(maxlen=args.log_every)
    profile = {"forward": 0.0, "backward": 0.0, "sleep": 0.0, "wall": 0.0}
    cached_gradients = None
    optimizer_updates = 0
    surprise_gated_steps = 0
    lazy_reused_steps = 0
    backward_calls = 0
    if args.lazy_backward or args.surprise_gate_loss is not None:
        LOG.info(
            "compute_saving_policy lazy_backward=%s gradient_decay=%.2f surprise_gate_loss=%s "
            "scheduler_advances_on_observation=True cached_gradients_checkpointed=False",
            args.lazy_backward, 0.9, args.surprise_gate_loss,
        )
    block_started = time.perf_counter()
    elapsed_offset = 0.0
    if args.wall_clock_file and Path(args.wall_clock_file).is_file():
        elapsed_offset = float(json.loads(Path(args.wall_clock_file).read_text()).get("elapsed_seconds", 0.0))
    wall_state = json.loads(Path(args.wall_clock_file).read_text()) if args.wall_clock_file and Path(args.wall_clock_file).is_file() else {}
    reached_targets = set(wall_state.get("reached_targets", []))
    for step in range(start + 1, start + args.steps + 1):
        step_started = time.perf_counter()
        if args.dummy_m1:
            tokens = torch.randint(4, CONFIG.vocab_size,
                                   (args.batch, args.max_len), device=device)
        else:
            chosen = (sample_phase2(phase_data, args.batch, local_rng)
                      if args.phase == 2 else local_rng.choices(phase_data, k=args.batch))
            if packed_token_rows is None:
                tokens = collate_rows(chosen, tokenizer, args.max_len)
            else:
                tokens = collate_cached_indices(
                    packed_token_rows, token_lengths,
                    [row_indices[id(row)] for row in chosen],
                )

        tokens_seen += int(tokens.ne(0).sum())
        forward_started = time.perf_counter()
        collect_diagnostics = (step % 100 == 0 or step == start + 1)
        forward_model = (model if collect_diagnostics or compiled_model is None else compiled_model)
        output = forward_model(tokens, reset_state=True, collect_diagnostics=collect_diagnostics)
        profile["forward"] += time.perf_counter() - forward_started
        loss, metrics = perdida_total(
            output, tokens, model,
            use_jepa=not args.no_jepa,
            use_rep=not args.no_rep,
            use_free_energy=args.v32_ze,
            beta=0.01,
            pad_id=0,
            shared_loss_normalization=bool(
                args.runtime_profile_settings.get("shared_loss_normalization", False)
            ),
        )
        glucose_loss_window.append(float(loss.detach().item()))
        if not torch.isfinite(loss.detach()).all():
            log_nonfinite_diagnostics(model, output, metrics, loss, optimizer, step, "non-finite loss; optimizer update skipped")
            raise FloatingPointError(
                f"non-finite loss at global step {step}; optimizer update skipped"
            )
        loss_value = float(loss.detach().item())
        surprise_gate = should_surprise_gate(loss_value, args.surprise_gate_loss)
        optimizer.zero_grad(set_to_none=True)
        if surprise_gate:
            # Keep the observation for loss/affect/homeostasis, but do not
            # compute gradients or mutate model/optimizer parameters.
            surprise_gated_steps += 1
        else:
            local_step_index = step - start - 1
            lazy_reuse = should_reuse_cached_gradient(
                args.lazy_backward, local_step_index, cached_gradients
            )
            if lazy_reuse:
                for parameter, cached in zip(model.parameters(), cached_gradients):
                    if cached is not None:
                        parameter.grad = cached.mul(0.9)
                lazy_reused_steps += 1
            else:
                backward_started = time.perf_counter()
                loss.backward()
                profile["backward"] += time.perf_counter() - backward_started
                backward_calls += 1
                if args.lazy_backward:
                    # Cache the un-clipped gradient; clipping is reapplied to
                    # each reused copy immediately before the optimizer step.
                    cached_gradients = [
                        None if parameter.grad is None else parameter.grad.detach().clone()
                        for parameter in model.parameters()
                    ]
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip,
                    foreach=not args.disable_foreach, error_if_nonfinite=True,
                )
            except RuntimeError:
                log_nonfinite_diagnostics(model, output, metrics, loss, optimizer, step, "non-finite gradient norm")
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        LOG.error("NONFINITE_GRADIENT step=%d name=%s %s", step, name, _tensor_health(parameter.grad))
                raise
            optimizer.step()
            optimizer_updates += 1
        # Keep cheap numerical guards on every update (finite loss and
        # error_if_nonfinite gradient clipping above); amortize the full model
        # and Adam-state scans, which otherwise read every parameter each step.
        full_health_check = (
            step % args.parameter_check_every == 0
            or step == start + args.steps
        )
        invalid_parameters = False
        invalid_optimizer_state = False
        if full_health_check:
            invalid_parameters = any(
                not torch.isfinite(parameter.detach()).all()
                for parameter in model.parameters()
            )
            invalid_optimizer_state = any(
                isinstance(value, torch.Tensor) and value.is_floating_point()
                and value.numel() and not torch.isfinite(value).all()
                for state_values in optimizer.state.values()
                for value in state_values.values()
            )
        if invalid_parameters or invalid_optimizer_state:
            log_nonfinite_diagnostics(
                model, output, metrics, loss, optimizer, step,
                "optimizer produced non-finite parameters or moments",
            )
            raise FloatingPointError(
                f"optimizer produced non-finite state at global step {step}"
            )
        # The LR schedule follows observed training steps, including gated
        # observations, so checkpoint scheduler_step remains aligned with step.
        scheduler.step()
        for target in (6.0, 5.8):
            if target not in reached_targets and loss.item() <= target:
                elapsed = elapsed_offset + time.perf_counter() - block_started
                reached_targets.add(target)
                LOG.info("time_to_target: model=LUMEN, target_loss=%.1f, step=%d, reached=True, wall_seconds=%.3f",
                         target, step, elapsed)

        if step >= 5000 and not quant_enabled and not args.dummy_m1:
            enable_fake_int8(model)
            quant_enabled = True

        if step % args.log_every == 0 or step == start + 1:
            glucose_signal = "not_checked"
            if step % args.log_every == 0 and glucose_loss_window:
                check_loss = sum(glucose_loss_window) / len(glucose_loss_window)
                shared_loss_signal = model.presupuesto.check_loss_signal(check_loss)
                improved = shared_loss_signal["improved"]
                glucose_level = shared_loss_signal["reserve"]
                glucose_signal = (
                    "baseline" if improved is None else
                    "fed" if improved else "metabolized"
                )
                LOG.info(
                    "glucose_check step=%d mean_loss=%.6f improved=%s signal=%s reserve=%.3f",
                    step, check_loss, improved, glucose_signal, glucose_level,
                )
                pain_state = model.pain.update_signal(shared_loss_signal["regression"])
                pleasure_state = model.pleasure.update_signal(shared_loss_signal["improvement"])
                eligibility_signal = (
                    pleasure_state["pleasure"] - pain_state["pain"]
                )
                model.set_learning_modulator(eligibility_signal)
                LOG.info(
                    "eligibility_feedback step=%d signal=%.4f trace_norms=%s",
                    step, eligibility_signal,
                    [round(layer.last_eligibility_norm, 6) for layer in model.delta_layers],
                )
                model.presupuesto.affect_scale = max(
                    0.85, min(1.05,
                        1.0 - pain_budget_gain * pain_state["pain"] + pleasure_budget_gain * pleasure_state["pleasure"])
                )
                LOG.info(
                    "affect_check step=%d pain=%.3f regression=%.5f pleasure=%.3f improvement=%.5f affect_scale=%.3f",
                    step, pain_state["pain"], pain_state["regression"],
                    pleasure_state["pleasure"], pleasure_state["improvement"],
                    model.presupuesto.affect_scale,
                )
            entropy = float(model.presupuesto.entropy(output["energy"]).detach())
            ze_conflict = 0.0
            ze_erase = 0
            if output["ze"] is not None:
                ze_conflict = float(output["ze"]["conflict"].mean().detach())
                ze_erase = int(model.ze.erase_count.item())
            LOG.info(
                "compute_saving step=%d optimizer_updates=%d surprise_gated=%d lazy_reused=%d backward_calls=%d",
                step, optimizer_updates, surprise_gated_steps, lazy_reused_steps, backward_calls,
            )
            LOG.info(
                "step=%d loss=%.5f token=%.5f latent=%.5f div=%.5f "
                "router=%.5f rep=%.5f "
                "energy_entropy=%.4f PE_mean=%.4f PE_max=%.4f slow_mem=%.3f ZE_conflict=%.4f ZE_erase=%d tokens_seen=%d lr=%.3e",
                step, loss.item(), metrics["L_token"].item(),
                metrics["L_latente"].item(), metrics["L_div"].item(),
                metrics["L_router"].item(), metrics["L_rep"].item(),
                entropy,
                float(output["surprise"].mean().detach()), float(output["surprise"].max().detach()),
                model.memoria.occupancy(), ze_conflict, ze_erase,
                tokens_seen, scheduler.get_last_lr()[0],
            )
            LOG.info(
                "neuroplasticity step=%d chunk_sizes=%s eligibility_norms=%s modulator=%s",
                step, output["chunk_sizes"],
                [round(layer.last_eligibility_norm, 6) for layer in model.delta_layers],
                [round(float(layer.eligibility_modulator.item()), 5) for layer in model.delta_layers],
            )
            layer_records = []
            for i, (delta, homeo) in enumerate(zip(model.delta_layers, model.homeo)):
                layer_records.append({
                    "layer": i,
                    "write_rate": delta.last_write_rate,
                    "delta_state_norm": delta.last_state_norm,
                    "delta_output_norm": delta.last_output_norm,
                    "input_positive_rate": homeo.last_input_positive,
                    "output_positive_rate": homeo.last_output_positive,
                    "gate_positive_rate": homeo.last_gate_positive,
                    "homeostasis_output_norm": homeo.last_output_norm,
                    "effective_activity_mean": float(homeo.effective_m.mean().detach()),
                    "homeostasis_bias_norm": float(homeo.bias.detach().norm()),
                    "homeostasis_bias_min": float(homeo.bias.detach().min()),
                    "homeostasis_bias_max": float(homeo.bias.detach().max()),
                })
            layer_diagnostics_history.append({"step": step, "layers": layer_records})
            layer_diag = " | ".join(
                f"L{record['layer']}:write={record['write_rate']:.3f},"
                f"state={record['delta_state_norm']:.3e},"
                f"delta_out={record['delta_output_norm']:.3e},"
                f"in+={record['input_positive_rate']:.3f},"
                f"out+={record['output_positive_rate']:.3f},"
                f"gate+={record['gate_positive_rate']:.3f},"
                f"homeo_out={record['homeostasis_output_norm']:.3e},"
                f"bias_norm={record['homeostasis_bias_norm']:.3e}"
                for record in layer_records
            )
            LOG.info("layer_diagnostics step=%d %s", step, layer_diag)
        if not args.no_sleep and not args.dummy_m1:
            if args.v32_ze:
                if not sleep_done and model.memoria.occupancy() >= 0.8:
                    records_before_sleep = int((model.memoria.contador_lenta > 0).sum().item())
                    sleep_started = time.perf_counter()
                    sleep_loss = sueno(model, device="cpu", compression_ratio=replay_compression, downscale=sleep_downscale)
                    sleep_duration = time.perf_counter() - sleep_started
                    profile["sleep"] += sleep_duration
                    LOG.info("sleep_called: step=%d, records=%d, duration_s=%.3f", step, records_before_sleep, sleep_duration)
                    sleep_events.append({"step": step, "records": records_before_sleep,
                                         "executed": records_before_sleep > 0, "loss": float(sleep_loss),
                                         "duration_s": sleep_duration})
                    sleep_done = True
                    LOG.info("adaptive IRI sleep step=%d occupancy=%.1f%% loss=%.6f",
                             step, model.memoria.occupancy() * 100, sleep_loss)
            elif step % sleep_interval_steps == 0:
                records_before_sleep = int((model.memoria.contador_lenta > 0).sum().item())
                sleep_started = time.perf_counter()
                sleep_loss = sueno(model, device="cpu", compression_ratio=replay_compression, downscale=sleep_downscale)
                sleep_duration = time.perf_counter() - sleep_started
                profile["sleep"] += sleep_duration
                LOG.info("sleep_called: step=%d, duration_s=%.3f", step, sleep_duration)
                sleep_events.append({"step": step, "records": records_before_sleep,
                                     "executed": records_before_sleep > 0, "loss": float(sleep_loss),
                                     "duration_s": sleep_duration})
                LOG.info("sleep step=%d loss=%.6f", step, sleep_loss)

        if step % 1000 == 0 and not args.dummy_m1:
            candidates = model.emb.podar()
            LOG.info("vocabulary prune candidates=%d (tokenizer remap required before applying)",
                     len(candidates))

        if step % args.checkpoint_every == 0:
            save_checkpoint(Path(args.ckpt_dir) / f"step_{step}.pt",
                            model, optimizer, scheduler, step, args, data_hash, local_rng,
                            tokens_seen, tokenizer_hash, layer_diagnostics_history, sleep_events)

        if args.recovery_checkpoint_every and step % args.recovery_checkpoint_every == 0:
            save_checkpoint(Path(args.ckpt_dir) / "latest_good.pt",
                            model, optimizer, scheduler, step, args, data_hash, local_rng,
                            tokens_seen, tokenizer_hash, layer_diagnostics_history, sleep_events)

        profile["wall"] += time.perf_counter() - step_started
        if step % args.log_every == 0:
            total = profile["wall"]
            known = profile["forward"] + profile["backward"] + profile["sleep"]
            LOG.info(
                "profile_%d: forward=%.3fs, backward=%.3fs, sleep=%.3fs, "
                "python_overhead=%.3fs, total=%.3fs",
                args.log_every, profile["forward"], profile["backward"], profile["sleep"],
                max(0.0, total - known), total,
            )
            for key in profile:
                profile[key] = 0.0

    final_step = start + args.steps
    if final_step % args.checkpoint_every:
        save_checkpoint(Path(args.ckpt_dir) / f"step_{final_step}.pt",
                        model, optimizer, scheduler, final_step, args, data_hash, local_rng,
                        tokens_seen, tokenizer_hash, layer_diagnostics_history, sleep_events)
    block_elapsed = time.perf_counter() - block_started
    total_elapsed = elapsed_offset + block_elapsed
    for target in (6.0, 5.8):
        if target not in reached_targets:
            LOG.info("time_to_target: model=LUMEN, target_loss=%.1f, reached=False, wall_seconds=%.3f",
                     target, total_elapsed)
    if args.wall_clock_file:
        wall_path = Path(args.wall_clock_file)
        wall_path.parent.mkdir(parents=True, exist_ok=True)
        wall_path.write_text(json.dumps({"model": "LUMEN", "elapsed_seconds": total_elapsed,
                                         "reached_targets": sorted(reached_targets)}, indent=2) + "\n")
    LOG.info("wall_clock_accumulated: model=LUMEN, seconds=%.3f", total_elapsed)
    LOG.info("completed step=%d device=cpu", final_step)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
