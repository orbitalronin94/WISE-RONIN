"""M1 CPU smoke trainer with deterministic checkpoint/resume."""
import argparse
import logging
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from model.config import CONFIG
from model.lumen import RoninLumen

LOG = logging.getLogger("lumen.train")

def save_checkpoint(path: Path, model, optimizer, step: int, seed: int) -> None:
    """Persist all M1 state required to resume the next update."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "seed": seed,
        "config": CONFIG.to_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
    }, path)

def run(steps: int, resume: str | None, output_dir: Path, seed: int) -> Path:
    """Run dummy CPU updates and save a resumable checkpoint."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = RoninLumen().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate,
                                  weight_decay=CONFIG.weight_decay)
    start_step = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["config"] != CONFIG.to_dict():
            raise ValueError("Checkpoint config does not match current config")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        seed = int(state["seed"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        LOG.info("Resumed checkpoint at step %d", start_step)
    model.train()
    for step in range(start_step + 1, start_step + steps + 1):
        token_ids = torch.randint(0, CONFIG.vocab_size,
                                  (CONFIG.batch_size, CONFIG.sequence_length),
                                  device=device)
        logits = model(token_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, CONFIG.vocab_size),
            token_ids[:, 1:].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == start_step + 1 or step % 10 == 0 or step == start_step + steps:
            LOG.info("step=%d loss=%.5f device=cpu", step, loss.item())
    checkpoint = output_dir / f"step_{start_step + steps}.pt"
    save_checkpoint(checkpoint, model, optimizer, start_step + steps, seed)
    LOG.info("Saved %s", checkpoint)
    return checkpoint

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.steps, args.resume, Path(args.output_dir), args.seed)

if __name__ == "__main__":
    main()
