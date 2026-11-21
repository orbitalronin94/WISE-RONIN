"""CPU-only RONIN-LUMEN training loop with exact batch-boundary resume."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from model.config import CONFIG
from model.lumen import RoninLumen
from train.dataset import BPETokenizer, load_jsonl, collate
from train.curriculum import datos_para_fase, sample_phase2
from train.loss import perdida_total
from train.sleep import sueno
from train.quant import fake_int8_tensor

LOG = logging.getLogger("lumen.train")

def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state()}

def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])

def save_checkpoint(path, model, optimizer, scheduler, step, args, data_hash, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "step": step, "phase": args.phase,
        "seed": args.seed, "config": CONFIG.to_dict(), "data_sha256": data_hash,
        "rng": rng_state(rng), "torch_version": torch.__version__,
        "delta_policy": "reset_at_each_batch_boundary",
    }, path)
    LOG.info("checkpoint=%s step=%d", path, step)

def make_scheduler(optimizer, warmup, total):
    def scale(step):
        if step < warmup:
            return max(step, 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)

def main():
    parser = argparse.ArgumentParser(description="RONIN-LUMEN CPU training")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--resume")
    parser.add_argument("--data", default="data/train_25k.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--batch", type=int, default=CONFIG.batch_size)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=CONFIG.warmup_steps)
    parser.add_argument("--total-steps", type=int, default=CONFIG.total_steps)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--no-jepa", action="store_true")
    parser.add_argument("--no-rep", action="store_true")
    parser.add_argument("--orpo", action="store_true")
    parser.add_argument("--dummy-m1", action="store_true", help="Explicit 10-step smoke mode only; not real training")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.orpo:
        parser.error("ORPO is not implemented: preference pairs and objective are not yet approved")
    if args.dummy_m1 and (args.steps > 10 or args.resume):
        parser.error("--dummy-m1 is limited to a fresh run of at most 10 steps")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")
    model = RoninLumen().to(device)

    if args.dummy_m1:
        data_hash = "dummy-m1-no-corpus"
        tokenizer = None
        rows = None
        phase_data = None
    else:
        rows, data_hash = load_jsonl(args.data)
        tokenizer = BPETokenizer(args.tokenizer)
        if tokenizer.tokenizer.get_vocab_size() != CONFIG.vocab_size:
            raise ValueError("Tokenizer vocabulary size must match model config")
        phase_data = datos_para_fase(rows, args.phase)

    start = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CONFIG.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup, args.total_steps)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["config"] != CONFIG.to_dict() or state["data_sha256"] != data_hash:
            raise ValueError("Checkpoint config or training corpus hash mismatch")
        if state["phase"] != args.phase:
            raise ValueError("Cannot change curriculum phase when resuming a checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = int(state["step"])
        restore_rng(state["rng"])
        LOG.info("resumed step=%d", start)

    if start >= 5000 and not args.dummy_m1:
        model.heads.kan = FakeInt8(model.heads.kan)
        model.memoria.mlp = FakeInt8(model.memoria.mlp)
        LOG.info("fake-int8 activation simulation enabled at step >=5000")

    model.train()
    rng = random.Random(args.seed + start)\n    if args.resume and state.get("rng", {}).get("local") is not None:\n        rng.setstate(state["rng"]["local"])
    for step in range(start + 1, start + args.steps + 1):
        if args.dummy_m1:
            tokens = torch.randint(4, CONFIG.vocab_size, (args.batch, args.max_len))
        else:
            if args.phase == 2:
                chosen = sample_phase2(phase_data, args.batch, rng)
            else:
                chosen = rng.choices(phase_data, k=args.batch)
            tokens = collate(chosen, tokenizer, args.batch, args.max_len, rng)
        output = model(tokens, reset_state=True)
        loss, metrics = perdida_total(output, tokens, model,
                                      use_jepa=not args.no_jepa,
                                      use_rep=not args.no_rep,
                                      pad_id=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if model.memoria.sorpresa_acum.item() >= CONFIG.elastic_memory_threshold:
                model.memoria.consolidar(output["hidden"])
                model.memoria.sorpresa_acum.zero_()

        if step % 100 == 0 or step == start + 1:
            dead = max(h.neuronas_muertas() for h in model.homeo)
            entropy = float(model.presupuesto.entropy(output["energy"]).detach())
            LOG.info("step=%d loss=%.5f token=%.5f latent=%.5f div=%.5f router=%.5f rep=%.5f dead=%.2f%% energy_entropy=%.4f lr=%.3e",
                     step, loss.item(), metrics["L_token"].item(), metrics["L_latente"].item(),
                     metrics["L_div"].item(), metrics["L_router"].item(), metrics["L_rep"].item(),
                     dead * 100, entropy, scheduler.get_last_lr()[0])
            if dead > 0.20:
                save_checkpoint(Path(args.ckpt_dir) / f"step_{step}_STOP.pt", model, optimizer,
                                scheduler, step, args, data_hash, rng)
                raise RuntimeError(f"STOP: dead neurons {dead:.1%} exceeds 20%")

        if step % CONFIG.sleep_interval_steps == 0 and not args.no_sleep and not args.dummy_m1:
            sleep_loss = sueno(model, CONFIG.sleep_steps, device="cpu")
            LOG.info("sleep step=%d loss=%.6f", step, sleep_loss)
        if step % 500 == 0:
            save_checkpoint(Path(args.ckpt_dir) / f"step_{step}.pt", model, optimizer,
                            scheduler, step, args, data_hash, rng)

    final_step = start + args.steps
    if final_step % 500:
        save_checkpoint(Path(args.ckpt_dir) / f"step_{final_step}.pt", model, optimizer,
                        scheduler, final_step, args, data_hash, rng)
    LOG.info("completed step=%d device=cpu", final_step)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
