"""Matched-size decoder-only Transformer baseline with exact block-boundary resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model.config import CONFIG
from train.dataset import (BPETokenizer, collate_rows, collate_cached_indices,
                            load_jsonl, load_token_cache, pack_token_cache)
from train.scheduler_transition import CosineTransitionScheduler
from train.data_transition import resolve_data_transition

LOG = logging.getLogger("lumen.baseline")
ARCHITECTURE_ID = "decoder_transformer_4l_d192_h4_ff576_tied_v1"


class CausalBlock(nn.Module):
    def __init__(self, width=192, heads=4, ff_width=576, residual_groups=0):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        if residual_groups < 0 or residual_groups > width:
            raise ValueError("residual_groups must be between 0 and model width")
        self.residual_gain = nn.Parameter(torch.ones(residual_groups)) if residual_groups else None
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, ff_width)
        self.ff2 = nn.Linear(ff_width, width)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(b, t, c)
        if self.residual_gain is None:
            gain = 1.0
        else:
            channel_groups = torch.arange(c, device=x.device) % self.residual_gain.numel()
            gain = self.residual_gain[channel_groups].view(1, 1, c)
        x = x + self.proj(attended) * gain
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x)))) * gain


class SmallTransformerLM(nn.Module):
    """Four-layer decoder-only Transformer with optional grouped residual gains."""
    def __init__(self, vocab_size=4096, max_seq=128, width=192, heads=4,
                 ff_width=None, residual_groups=0):
        super().__init__()
        ff_width = int(ff_width or width * 3)
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
        if residual_groups < 0 or residual_groups > width:
            raise ValueError("residual_groups must be between 0 and model width")
        self.token_emb = nn.Embedding(vocab_size, width)
        self.position_emb = nn.Embedding(max_seq, width)
        self.blocks = nn.ModuleList([
            CausalBlock(width, heads, ff_width, residual_groups) for _ in range(4)
        ])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        # GPT-style small embedding initialization is essential with tied output
        # weights; the nn.Embedding default N(0,1) makes initial logits explode.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_emb.weight, mean=0.0, std=0.02)
        self.lm_head.weight = self.token_emb.weight
        self.max_seq = max_seq

    def forward(self, token_ids):
        _, t = token_ids.shape
        if t > self.max_seq:
            raise ValueError(f"Sequence length {t} exceeds {self.max_seq}")
        pos = torch.arange(t, device=token_ids.device)
        x = self.token_emb(token_ids) + self.position_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def make_scheduler(optimizer, warmup=100, total=10000):
    def scale(step):
        if step < warmup:
            return max(step, 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


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


def save_checkpoint(path, model, optimizer, scheduler, step, args,
                    data_hash, tokenizer_hash, local_rng, tokens_seen, params):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "seed": args.seed,
        "data_sha256": data_hash,
        "data_lineage": list(getattr(args, "data_lineage", [])),
        "tokenizer_sha256": tokenizer_hash,
        "tokens_seen": tokens_seen,
        "params": params,
        "architecture_id": ARCHITECTURE_ID,
        "model_spec": {
            "model_dim": int(getattr(args, "model_dim", 192)),
            "heads": int(getattr(args, "heads", 4)),
            "ff_width": int(getattr(args, "ff_width", None) or getattr(args, "model_dim", 192) * 3),
            "residual_groups": int(getattr(args, "residual_groups", 0)),
            "layers": 4,
            "tied_embeddings": True,
        },
        "architecture": f"4-layer decoder-only Transformer, d_model={getattr(args, 'model_dim', 192)}, heads={getattr(args, 'heads', 4)}, ffn={getattr(args, 'ff_width', None) or getattr(args, 'model_dim', 192) * 3}, grouped_residual_gains={getattr(args, 'residual_groups', 0)}, tied embeddings",
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
            } if getattr(args, 'scheduler_transition', False) else {}),
        },
        "scheduler_transition": scheduler.state_dict() if getattr(args, 'scheduler_transition', False) else None,
        "rng": capture_rng(local_rng),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }, path)
    LOG.info("checkpoint=%s step=%d", path, step)


def main():
    p = argparse.ArgumentParser(description="Matched-size Transformer pilot baseline")
    p.add_argument("--data", default="data/pilot_train_19k.jsonl")
    p.add_argument("--tokenizer", default="data/pilot_tokenizer/tokenizer.json")
    p.add_argument("--token-cache", help="Validated pre-tokenized corpus cache")
    p.add_argument("--steps", type=int, default=500,
                   help="Number of additional steps in this block (maximum 1000)")
    p.add_argument("--threads", type=int, default=2,
                   help="PyTorch intra-op CPU threads; benchmark 1/2/4 on the target runner")
    p.add_argument("--interop-threads", type=int, default=1,
                   help="PyTorch inter-op CPU threads")
    p.add_argument("--disable-mkldnn", action="store_true")
    p.add_argument("--disable-foreach", action="store_true")
    p.add_argument("--cpu-affinity", action="store_true")
    p.add_argument("--resume", help="Previous global-step checkpoint")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--model-dim", type=int, default=192)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ff-width", type=int, default=None)
    p.add_argument("--residual-groups", type=int, default=0,
                   help="Trainable grouped channel gains in each residual block")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--parameter-check-every", type=int, default=25,
                   help="Full parameter and AdamW-state finite scan interval")
    p.add_argument("--checkpoint-every", type=int, default=250,
                   help="Named checkpoint interval; final block checkpoint is always saved")
    p.add_argument("--allow-data-change-on-resume", action="store_true",
                   help="Allow an explicitly declared corpus expansion while preserving optimizer/model state")
    p.add_argument("--parent-data-sha256", default=None)
    p.add_argument("--data-change-label", default="explicit corpus transition")
    p.add_argument("--scheduler-transition", action="store_true")
    p.add_argument("--restart-scheduler-transition", action="store_true")
    p.add_argument("--transition-start", type=int, default=1000)
    p.add_argument("--transition-end", type=int, default=2000)
    p.add_argument("--warmup-transition", type=int, default=20)
    p.add_argument("--transition-plateau-steps", type=int, default=0)
    p.add_argument("--transition-min-lr-ratio", type=float, default=0.0)
    p.add_argument("--wall-clock-file")
    p.add_argument("--ckpt-dir", default="checkpoints/transformer")
    args = p.parse_args()
    if not 1 <= args.steps <= 10000:
        p.error("--steps must be between 1 and 10000")
    if args.threads < 1 or args.interop_threads < 1 or args.parameter_check_every < 1 or args.checkpoint_every < 1:
        p.error("--threads, --interop-threads, --parameter-check-every and --checkpoint-every must be at least 1")
    if getattr(args, 'scheduler_transition', False) and not args.resume:
        p.error("--scheduler-transition requires an exact resumable parent checkpoint")
    if args.restart_scheduler_transition and not args.scheduler_transition:
        p.error("--restart-scheduler-transition requires --scheduler-transition")

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
    rows, data_hash = load_jsonl(args.data)
    tokenizer = BPETokenizer(args.tokenizer)
    if tokenizer.tokenizer.get_vocab_size() != CONFIG.vocab_size or tokenizer.tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("Baseline requires the shared 4k pilot tokenizer with PAD=0")
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    data_path = Path(args.data)
    manifest_path = data_path / "manifest.json" if data_path.is_dir() else (
        data_path if data_path.name == "manifest.json" else None
    )
    if manifest_path is not None:
        corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if corpus_manifest.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError("Corpus manifest tokenizer SHA-256 does not match --tokenizer")
    tokenized_rows, row_indices = None, {}
    packed_token_rows, token_lengths = None, None
    if args.token_cache:
        tokenized_rows = load_token_cache(
            args.token_cache, data_hash, tokenizer_hash, args.max_len, len(rows)
        )
        row_indices = {id(row): index for index, row in enumerate(rows)}
        packed_token_rows, token_lengths = pack_token_cache(tokenized_rows, args.max_len)
        tokenized_rows = None
        LOG.info("loaded packed token cache rows=%d max_len=%d", len(rows), args.max_len)

    if args.model_dim < 1 or args.heads < 1 or args.model_dim % args.heads:
        p.error("--model-dim must be positive and divisible by --heads")
    ff_width = args.ff_width or args.model_dim * 3
    model = SmallTransformerLM(
        CONFIG.vocab_size, args.max_len, args.model_dim, args.heads,
        ff_width, args.residual_groups,
    ).cpu()
    global ARCHITECTURE_ID
    if args.residual_groups:
        ARCHITECTURE_ID = (
            f"decoder_transformer_4l_d{args.model_dim}_h{args.heads}_"
            f"ff{ff_width}_rg{args.residual_groups}_tied_v2"
        )
    else:
        ARCHITECTURE_ID = f"decoder_transformer_4l_d{args.model_dim}_h{args.heads}_ff{ff_width}_tied_v1"
    params = sum(x.numel() for x in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=CONFIG.weight_decay,
        foreach=not args.disable_foreach,
    )
    scheduler = None if getattr(args, 'scheduler_transition', False) else make_scheduler(optimizer, args.warmup, args.total_steps)
    local_rng = random.Random(args.seed)
    tokens_seen, start, state = 0, 0, None

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected_config = {
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
            } if getattr(args, 'scheduler_transition', False) else {}),
        }
        expected_resume_config = expected_config
        if getattr(args, 'scheduler_transition', False) and not state.get("scheduler_transition") and int(state.get("step", -1)) == args.transition_start:
            parent_config = state.get("training_config") or {}
            required_parent = {
                "batch": args.batch, "max_len": args.max_len, "lr": args.lr,
                "warmup": args.warmup, "weight_decay": CONFIG.weight_decay,
            }
            if any(parent_config.get(key) != value for key, value in required_parent.items()):
                raise ValueError("Parent checkpoint batch/length/LR/warmup/weight-decay mismatch")
            if int(parent_config.get("total_steps", -1)) < args.transition_start:
                raise ValueError("Parent scheduler horizon ends before the transition step")
            expected_resume_config = parent_config
        if args.restart_scheduler_transition:
            parent_config = state.get("training_config") or {}
            stable_keys = ("batch", "max_len", "lr", "warmup", "weight_decay", "grad_clip")
            mismatches = [key for key in stable_keys
                          if parent_config.get(key) != expected_config.get(key)]
            saved_transition = state.get("scheduler_transition")
            saved_step = int(state.get("step", -1))
            scheduler_step = int(saved_transition.get("current_step", -1)) if saved_transition else -1
            if mismatches or not saved_transition or scheduler_step not in (saved_step, saved_step + 1):
                raise ValueError(
                    f"Cannot restart scheduler from this checkpoint; mismatches={mismatches}, "
                    f"checkpoint_step={saved_step}, scheduler_step={scheduler_step}"
                )
            expected_resume_config = parent_config
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
        checks = {
            "architecture_id": (state.get("architecture_id"), ARCHITECTURE_ID),
            "seed": (state.get("seed"), args.seed),
            "tokenizer_sha256": (state.get("tokenizer_sha256"), tokenizer_hash),
            "training_config": (state.get("training_config"), expected_resume_config),
            "params": (state.get("params"), params),
        }
        mismatches = [key for key, (actual, expected) in checks.items() if actual != expected]
        if mismatches:
            raise ValueError(f"Transformer resume metadata mismatch: {mismatches}")
        if "optimizer" not in state or "scheduler" not in state or "rng" not in state:
            raise ValueError("Checkpoint lacks optimizer/scheduler/RNG state; exact resume is impossible")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["foreach"] = not args.disable_foreach
        start = int(state["step"])
        if getattr(args, 'scheduler_transition', False):
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
                    "lr_transition: step=%d, from=%.8g to=%.8g (Transformer), "
                    "weights_preserved=True, optimizer_moments_preserved=True",
                    start, old_lr, args.lr,
                )
                scheduler.step()  # Set the LR for the first update after the parent checkpoint.
            else:
                raise ValueError("scheduler-transition requires the exact step-1000 parent or a transition checkpoint")
        else:
            scheduler = make_scheduler(optimizer, args.warmup, args.total_steps)
            scheduler.load_state_dict(state["scheduler"])
        tokens_seen = int(state["tokens_seen"])
        restore_rng(state["rng"], local_rng)
        LOG.info("resumed step=%d tokens_seen=%d", start, tokens_seen)

    if start + args.steps > args.total_steps:
        raise ValueError(f"Requested global step {start + args.steps} exceeds total_steps={args.total_steps}")

    LOG.info("baseline=decoder_transformer params=%d data_rows=%d data_sha256=%s tokenizer_sha256=%s",
             params, len(rows), data_hash, tokenizer_hash)
    model.train()
    profile = {"forward": 0.0, "backward": 0.0, "sleep": 0.0,
               "homeostasis_in_forward": 0.0, "wall": 0.0}
    block_started = time.perf_counter()
    elapsed_offset = 0.0
    if args.wall_clock_file and Path(args.wall_clock_file).is_file():
        elapsed_offset = float(json.loads(Path(args.wall_clock_file).read_text()).get("elapsed_seconds", 0.0))
    wall_state = json.loads(Path(args.wall_clock_file).read_text()) if args.wall_clock_file and Path(args.wall_clock_file).is_file() else {}
    reached_targets = set(wall_state.get("reached_targets", []))
    for step in range(start + 1, start + args.steps + 1):
        step_started = time.perf_counter()
        chosen = local_rng.choices(rows, k=args.batch)
        if packed_token_rows is None:
            tokens = collate_rows(chosen, tokenizer, args.max_len)
        else:
            tokens = collate_cached_indices(
                packed_token_rows, token_lengths,
                [row_indices[id(row)] for row in chosen],
            )
        tokens_seen += int(tokens.ne(0).sum())
        forward_started = time.perf_counter()
        logits = model(tokens)
        profile["forward"] += time.perf_counter() - forward_started
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=0,
        )
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(f"non-finite Transformer loss at global step {step}")
        optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        loss.backward()
        profile["backward"] += time.perf_counter() - backward_started
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, foreach=True, error_if_nonfinite=True
        )
        optimizer.step()
        full_health_check = (
            step % args.parameter_check_every == 0
            or step == start + args.steps
        )
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
                raise FloatingPointError(
                    f"non-finite Transformer parameters/AdamW state at global step {step}"
                )
        scheduler.step()
        for target in (7.0, 6.0, 5.8):
            if target not in reached_targets and loss.item() <= target:
                elapsed = elapsed_offset + time.perf_counter() - block_started
                reached_targets.add(target)
                LOG.info("time_to_target: model=Transformer, target_loss=%.1f, step=%d, reached=True, wall_seconds=%.3f",
                         target, step, elapsed)
        if step == start + 1 or step % 100 == 0 or step == start + args.steps:
            LOG.info("baseline=transformer step=%d loss=%.5f tokens_seen=%d lr=%.3e",
                     step, loss.item(), tokens_seen, scheduler.get_last_lr()[0])
        if step % args.checkpoint_every == 0:
            save_checkpoint(Path(args.ckpt_dir) / f"step_{step}.pt",
                            model, optimizer, scheduler, step, args, data_hash,
                            tokenizer_hash, local_rng, tokens_seen, params)

        profile["wall"] += time.perf_counter() - step_started
        if step % 100 == 0:
            total = profile["wall"]
            known = profile["forward"] + profile["backward"] + profile["sleep"]
            LOG.info(
                "profile_100: forward=%.3fs, backward=%.3fs, sleep=%.3fs, "
                "homeostasis_in_forward=%.3fs, python_overhead=%.3fs, total=%.3fs",
                profile["forward"], profile["backward"], profile["sleep"],
                profile["homeostasis_in_forward"], max(0.0, total - known), total,
            )
            for key in profile:
                profile[key] = 0.0

    final_step = start + args.steps
    if final_step % args.checkpoint_every:
        save_checkpoint(Path(args.ckpt_dir) / f"step_{final_step}.pt",
                        model, optimizer, scheduler, final_step, args, data_hash,
                        tokenizer_hash, local_rng, tokens_seen, params)
    block_elapsed = time.perf_counter() - block_started
    total_elapsed = elapsed_offset + block_elapsed
    for target in (7.0, 6.0, 5.8):
        if target not in reached_targets:
            LOG.info("time_to_target: model=Transformer, target_loss=%.1f, reached=False, wall_seconds=%.3f",
                     target, total_elapsed)
    if args.wall_clock_file:
        wall_path = Path(args.wall_clock_file)
        wall_path.parent.mkdir(parents=True, exist_ok=True)
        wall_path.write_text(json.dumps({"model": "Transformer", "elapsed_seconds": total_elapsed,
                                         "reached_targets": sorted(reached_targets)}, indent=2) + "\n")
    LOG.info("wall_clock_accumulated: model=Transformer, seconds=%.3f", total_elapsed)
    LOG.info("completed step=%d device=cpu", final_step)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
