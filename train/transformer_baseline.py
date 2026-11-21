"""Matched-size decoder-only Transformer baseline with exact block-boundary resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model.config import CONFIG
from train.dataset import BPETokenizer, collate_rows, load_jsonl
from train.scheduler_transition import CosineTransitionScheduler

LOG = logging.getLogger("lumen.baseline")
ARCHITECTURE_ID = "decoder_transformer_4l_d192_h4_ff576_tied_v1"


class CausalBlock(nn.Module):
    def __init__(self, width=192, heads=4, ff_width=576):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
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
        x = x + self.proj(attended)
        return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))


class SmallTransformerLM(nn.Module):
    """2,295,168 parameters at the frozen 4k vocabulary and 128 context."""
    def __init__(self, vocab_size=4096, max_seq=128):
        super().__init__()
        width = 192
        self.token_emb = nn.Embedding(vocab_size, width)
        self.position_emb = nn.Embedding(max_seq, width)
        self.blocks = nn.ModuleList([CausalBlock(width, 4, 576) for _ in range(4)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
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
        "tokenizer_sha256": tokenizer_hash,
        "tokens_seen": tokens_seen,
        "params": params,
        "architecture_id": ARCHITECTURE_ID,
        "architecture": "4-layer decoder-only Transformer, d_model=192, heads=4, ffn=576, tied embeddings",
        "training_config": {
            "batch": args.batch, "max_len": args.max_len, "lr": args.lr,
            "warmup": args.warmup, "total_steps": args.total_steps,
            "weight_decay": CONFIG.weight_decay,
            **({
                "scheduler_mode": "cosine_transition_v1",
                "transition_start": args.transition_start,
                "transition_end": args.transition_end,
                "warmup_transition": args.warmup_transition,
                "grad_clip": args.grad_clip,
            } if args.scheduler_transition else {}),
        },
        "scheduler_transition": scheduler.state_dict() if args.scheduler_transition else None,
        "rng": capture_rng(local_rng),
        "torch_version": torch.__version__,
    }, path)
    LOG.info("checkpoint=%s step=%d", path, step)


def main():
    p = argparse.ArgumentParser(description="Matched-size Transformer pilot baseline")
    p.add_argument("--data", default="data/pilot_train_19k.jsonl")
    p.add_argument("--tokenizer", default="data/pilot_tokenizer/tokenizer.json")
    p.add_argument("--steps", type=int, default=500,
                   help="Number of additional steps in this block (maximum 500)")
    p.add_argument("--resume", help="Previous global-step checkpoint")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--scheduler-transition", action="store_true")
    p.add_argument("--transition-start", type=int, default=1000)
    p.add_argument("--transition-end", type=int, default=2000)
    p.add_argument("--warmup-transition", type=int, default=20)
    p.add_argument("--wall-clock-file")
    p.add_argument("--ckpt-dir", default="checkpoints/transformer")
    args = p.parse_args()
    if not 1 <= args.steps <= 500:
        p.error("--steps must be between 1 and 500")
    if args.scheduler_transition and not args.resume:
        p.error("--scheduler-transition requires an exact resumable parent checkpoint")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows, data_hash = load_jsonl(args.data)
    tokenizer = BPETokenizer(args.tokenizer)
    if tokenizer.tokenizer.get_vocab_size() != CONFIG.vocab_size or tokenizer.tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("Baseline requires the shared 4k pilot tokenizer with PAD=0")
    tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()

    model = SmallTransformerLM(CONFIG.vocab_size, args.max_len).cpu()
    params = sum(x.numel() for x in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CONFIG.weight_decay)
    scheduler = None if args.scheduler_transition else make_scheduler(optimizer, args.warmup, args.total_steps)
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
                "grad_clip": args.grad_clip,
            } if args.scheduler_transition else {}),
        }
        expected_resume_config = expected_config
        if args.scheduler_transition and not state.get("scheduler_transition") and int(state.get("step", -1)) == args.transition_start:
            parent_config = {
                "batch": args.batch, "max_len": args.max_len, "lr": 1.5e-4,
                "warmup": 100, "total_steps": 10000,
                "weight_decay": CONFIG.weight_decay,
            }
            if state.get("training_config") != parent_config:
                raise ValueError("Step-1000 parent training config differs from the frozen baseline")
            expected_resume_config = parent_config
        checks = {
            "architecture_id": (state.get("architecture_id"), ARCHITECTURE_ID),
            "seed": (state.get("seed"), args.seed),
            "data_sha256": (state.get("data_sha256"), data_hash),
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
        start = int(state["step"])
        if args.scheduler_transition:
            saved_transition = state.get("scheduler_transition")
            if saved_transition:
                scheduler = CosineTransitionScheduler(
                    optimizer, start_step=saved_transition["start_step"],
                    end_step=saved_transition["end_step"], old_lr=saved_transition["old_lr"],
                    new_lr=saved_transition["new_lr"], warmup_steps=saved_transition["warmup_steps"],
                    current_step=saved_transition["current_step"],
                )
                scheduler.load_state_dict(saved_transition)
            elif start == args.transition_start:
                old_lr = float(optimizer.param_groups[0]["lr"])
                scheduler = CosineTransitionScheduler(
                    optimizer, start_step=args.transition_start, end_step=args.transition_end,
                    old_lr=old_lr, new_lr=args.lr, warmup_steps=args.warmup_transition,
                )
                LOG.info(
                    "lr_transition: step=%d, from=%.8g to=%.8g (Transformer), "
                    "weights_preserved=True, optimizer_moments_preserved=True",
                    start, old_lr, args.lr,
                )
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
    reached_targets = set()
    for step in range(start + 1, start + args.steps + 1):
        step_started = time.perf_counter()
        chosen = local_rng.choices(rows, k=args.batch)
        tokens = collate_rows(chosen, tokenizer, args.max_len)
        tokens_seen += int(tokens.ne(0).sum())
        forward_started = time.perf_counter()
        logits = model(tokens)
        profile["forward"] += time.perf_counter() - forward_started
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=0,
        )
        optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        loss.backward()
        profile["backward"] += time.perf_counter() - backward_started
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
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
        if step % 250 == 0:
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
    if final_step % 250:
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
        wall_path.write_text(json.dumps({"model": "Transformer", "elapsed_seconds": total_elapsed}, indent=2) + "\n")
    LOG.info("wall_clock_accumulated: model=Transformer, seconds=%.3f", total_elapsed)
    LOG.info("completed step=%d device=cpu", final_step)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
