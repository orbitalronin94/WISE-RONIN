"""Matched CPU forward/backward/optimizer throughput benchmark (synthetic fixed batch)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model.config import CONFIG
from model.lumen import RoninLumen
from train.loss import perdida_total
from train.transformer_baseline import SmallTransformerLM


def run_one(name, model, tokens, steps, warmup, grad_clip):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=CONFIG.weight_decay,
        foreach=True,
    )

    def update():
        optimizer.zero_grad(set_to_none=True)
        if name == "lumen":
            output = model(tokens, reset_state=True, collect_diagnostics=False)
            loss, _ = perdida_total(output, tokens, model)
        else:
            logits = model(tokens)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                ignore_index=0,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
        optimizer.step()

    for _ in range(warmup):
        update()
    started = time.perf_counter()
    for _ in range(steps):
        update()
    elapsed = time.perf_counter() - started
    return {
        "model": name,
        "steps": steps,
        "elapsed_seconds": round(elapsed, 4),
        "steps_per_second": round(steps / elapsed, 4),
        "tokens_per_second": round(steps * tokens.numel() / elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model compute throughput on CPU; excludes data loading."
    )
    parser.add_argument("--model", choices=("lumen", "transformer", "both"), default="both")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    if min(args.threads, args.batch, args.seq_len, args.steps) < 1 or args.warmup < 0:
        parser.error("threads, batch, seq-len and steps must be positive; warmup cannot be negative")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokens = torch.randint(4, CONFIG.vocab_size, (args.batch, args.seq_len), dtype=torch.long)
    results = []
    if args.model in ("lumen", "both"):
        results.append(run_one("lumen", RoninLumen().cpu(), tokens, args.steps, args.warmup, 1.0))
    if args.model in ("transformer", "both"):
        results.append(run_one(
            "transformer", SmallTransformerLM(CONFIG.vocab_size, args.seq_len).cpu(),
            tokens, args.steps, args.warmup, 1.0,
        ))
    report = {
        "device": "cpu",
        "threads": args.threads,
        "batch": args.batch,
        "sequence_length": args.seq_len,
        "includes_data_loading": False,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
