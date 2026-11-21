"""CPU-only RONIN-LUMEN trainer with reproducible batch-boundary resume."""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
from pathlib import Path

import torch

from model.config import CONFIG
from model.lumen import RoninLumen
from train.curriculum import datos_para_fase, sample_phase2
from train.dataset import BPETokenizer, collate_rows, load_jsonl
from train.loss import perdida_total
from train.sleep import sueno
from train.quant import fake_int8_tensor

LOG = logging.getLogger("lumen.train")


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


def save_checkpoint(path, model, optimizer, scheduler, step, args, data_hash, local_rng, tokens_seen, tokenizer_hash):
    model.reset_states()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "architecture_id": "ronin_lumen_v3_pilot",
        "architecture": {
            "model": "RoninLumen", "config": CONFIG.to_dict(),
            "v32_ze": args.v32_ze, "pilot_all_rows": args.pilot_all_rows,
            "no_sleep": args.no_sleep, "no_jepa": args.no_jepa,
            "no_rep": args.no_rep,
        },
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "phase": args.phase,
        "seed": args.seed,
        "config": CONFIG.to_dict(),
        "data_sha256": data_hash,
        "rng": capture_rng(local_rng),
        "torch_version": torch.__version__,
        "delta_policy": "reset_at_each_batch_boundary",
        "v32_ze": args.v32_ze,
        "pilot_all_rows": args.pilot_all_rows,
        "tokens_seen": tokens_seen,
        "tokenizer_sha256": tokenizer_hash,
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
    parser.add_argument("--v32-ze", action="store_true",
                        help="Enable experimental V3.2 dual-state/LC-NE/energy mechanisms")
    parser.add_argument("--pilot-all-rows", action="store_true",
                        help="Use every row in the explicitly authorized provisional pilot corpus; phase 1 only")
    parser.add_argument("--dummy-m1", action="store_true",
                        help="Explicit synthetic smoke test only; never a training dataset")
    args = parser.parse_args()

    if args.steps < 1 or (args.steps > 500 and not args.dummy_m1):
        parser.error("--steps must be between 1 and 500")
    if args.orpo:
        parser.error("ORPO is not implemented: preference pairs and objective need a separate approved specification")
    if args.dummy_m1 and args.steps > 10:
        parser.error("--dummy-m1 is limited to blocks of at most 10 steps")
    if args.pilot_all_rows and (args.dummy_m1 or args.phase != 1):
        parser.error("--pilot-all-rows is only valid for real-data phase 1")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cpu")
    model = RoninLumen(v32_ze=args.v32_ze).to(device)

    if args.dummy_m1:
        data_hash, tokenizer, phase_data = "dummy-m1-no-corpus", None, None
    else:
        rows, data_hash = load_jsonl(args.data)
        tokenizer = BPETokenizer(args.tokenizer)
        if tokenizer.tokenizer.get_vocab_size() != CONFIG.vocab_size:
            raise ValueError("Tokenizer vocabulary size must match model config")
        if tokenizer.tokenizer.token_to_id("[PAD]") != 0:
            raise ValueError("[PAD] must be token ID 0")
        phase_data = rows if args.pilot_all_rows else datos_para_fase(rows, args.phase)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=CONFIG.weight_decay
    )
    scheduler = make_scheduler(optimizer, args.warmup, args.total_steps)
    start, state = 0, None
    tokens_seen, tokenizer_hash = 0, "dummy-no-tokenizer"
    if not args.dummy_m1:
        tokenizer_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("v32_ze", False) != args.v32_ze:
            raise ValueError("Checkpoint V3.2 feature flag mismatch")
        if state["config"] != CONFIG.to_dict() or state["data_sha256"] != data_hash:
            raise ValueError("Checkpoint config or training corpus hash mismatch")
        if state["phase"] != args.phase:
            raise ValueError("Cannot change curriculum phase when resuming a checkpoint")
        if args.pilot_all_rows and (
            state.get("pilot_all_rows") is not True
            or state.get("tokenizer_sha256") != tokenizer_hash
        ):
            raise ValueError("Pilot resume requires the same all-row mode and tokenizer hash")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = int(state["step"])
        tokens_seen = int(state.get("tokens_seen", 0))
        LOG.info("resumed step=%d tokens_seen=%d", start, tokens_seen)

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
    for step in range(start + 1, start + args.steps + 1):
        if args.dummy_m1:
            tokens = torch.randint(4, CONFIG.vocab_size,
                                   (args.batch, args.max_len), device=device)
        else:
            chosen = (sample_phase2(phase_data, args.batch, local_rng)
                      if args.phase == 2 else local_rng.choices(phase_data, k=args.batch))
            tokens = collate_rows(chosen, tokenizer, args.max_len)

        tokens_seen += int(tokens.ne(0).sum())
        output = model(tokens, reset_state=True)
        loss, metrics = perdida_total(
            output, tokens, model,
            use_jepa=not args.no_jepa,
            use_rep=not args.no_rep,
            use_free_energy=args.v32_ze,
            beta=0.01,
            pad_id=0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step >= 5000 and not quant_enabled and not args.dummy_m1:
            enable_fake_int8(model)
            quant_enabled = True

        if step % 100 == 0 or step == start + 1:
            dead_by_layer = [organ.neuronas_muertas() for organ in model.homeo]
            dead = max(dead_by_layer, default=1.0)
            entropy = float(model.presupuesto.entropy(output["energy"]).detach())
            ze_conflict = 0.0
            ze_erase = 0
            if output["ze"] is not None:
                ze_conflict = float(output["ze"]["conflict"].mean().detach())
                ze_erase = int(model.ze.erase_count.item())
            LOG.info(
                "step=%d loss=%.5f token=%.5f latent=%.5f div=%.5f "
                "router=%.5f rep=%.5f F_complexity=%.5f dead=%.2f%% "
                "energy_entropy=%.4f PE=%.4f ZE_conflict=%.4f ZE_erase=%d tokens_seen=%d lr=%.3e",
                step, loss.item(), metrics["L_token"].item(),
                metrics["L_latente"].item(), metrics["L_div"].item(),
                metrics["L_router"].item(), metrics["L_rep"].item(),
                metrics["F_complexity"].item(), dead * 100, entropy,
                float(output["surprise"].mean().detach()), ze_conflict, ze_erase,
                tokens_seen, scheduler.get_last_lr()[0],
            )
            layer_diag = " | ".join(
                f"L{i}:write={delta.last_write_rate:.3f},"
                f"state={delta.last_state_norm:.3e},"
                f"delta_out={delta.last_output_norm:.3e},"
                f"in+={homeo.last_input_positive:.3f},"
                f"out+={homeo.last_output_positive:.3f},"
                f"homeo_out={homeo.last_output_norm:.3e},"
                f"dead={homeo.neuronas_muertas():.1%}"
                for i, (delta, homeo) in enumerate(zip(model.delta_layers, model.homeo))
            )
            LOG.info("layer_diagnostics step=%d %s", step, layer_diag)
            LOG.info("effective_dead_by_layer step=%d values=%s", step,
                     ",".join(f"{value:.3f}" for value in dead_by_layer))
            if step >= 100 and dead > 0.20:
                save_checkpoint(
                    Path(args.ckpt_dir) / f"step_{step}_STOP.pt",
                    model, optimizer, scheduler, step, args, data_hash, local_rng,
                    tokens_seen, tokenizer_hash,
                )
                raise RuntimeError(f"STOP: dead neurons {dead:.1%} exceeds 20%")

        if not args.no_sleep and not args.dummy_m1:
            if args.v32_ze:
                if not sleep_done and model.memoria.occupancy() >= 0.8:
                    sleep_loss = sueno(model, device="cpu")
                    sleep_done = True
                    LOG.info("adaptive IRI sleep step=%d occupancy=%.1f%% loss=%.6f",
                             step, model.memoria.occupancy() * 100, sleep_loss)
            elif step % CONFIG.sleep_interval_steps == 0:
                sleep_loss = sueno(model, device="cpu")
                LOG.info("sleep step=%d loss=%.6f", step, sleep_loss)

        if step % 1000 == 0 and not args.dummy_m1:
            candidates = model.emb.podar()
            LOG.info("vocabulary prune candidates=%d (tokenizer remap required before applying)",
                     len(candidates))

        if step % 500 == 0:
            save_checkpoint(Path(args.ckpt_dir) / f"step_{step}.pt",
                            model, optimizer, scheduler, step, args, data_hash, local_rng,
                            tokens_seen, tokenizer_hash)

    final_step = start + args.steps
    if final_step % 500:
        save_checkpoint(Path(args.ckpt_dir) / f"step_{final_step}.pt",
                        model, optimizer, scheduler, final_step, args, data_hash, local_rng,
                        tokens_seen, tokenizer_hash)
    LOG.info("completed step=%d device=cpu", final_step)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
