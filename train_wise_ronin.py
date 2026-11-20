#!/usr/bin/env python3
"""WISE-RONIN: small decoder-only Transformer and reproducible SFT/ORPO trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer

CONFIG = {"vocab": 6500, "dim": 256, "layers": 6, "q_heads": 8,
          "kv_heads": 2, "head_dim": 32, "hidden": 640, "ctx": 1024}
SYSTEM_FIXED = ("Eres WISE-RONIN. Respondón y antisicofántico. Da hechos ciertos "
                "con rodeo y trolleo. No adules ni des moralejas. No afirmes ser "
                "Claude, GPT ni otro modelo.")
SPECIAL = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|pad|>"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: JSON inválido: {exc}") from exc
                if not isinstance(row, dict) or not row.get("user") or not (row.get("assistant") or (row.get("chosen") and row.get("rejected"))):
                    raise ValueError(f"{path}:{line_no}: se requiere user/assistant o user/chosen/rejected")
                rows.append(row)
                count += 1
        print(f"[data] {path}: {count:,} ejemplos")
    return rows


def train_tokenizer(data_paths: list[str], output_path: Path, vocab_size: int) -> Tokenizer:
    print(f"[tokenizer] Entrenando BPE exclusivamente con train: {data_paths}")
    tmp = output_path.parent / "tokenizer_corpus.tmp"
    with tmp.open("w", encoding="utf-8") as out:
        for path in data_paths:
            with open(path, encoding="utf-8") as src:
                for line in src:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out.write(row.get("system", SYSTEM_FIXED) + "\n")
                    out.write(row.get("user", "") + "\n")
                    out.write(row.get("assistant", "") + "\n")
                    if row.get("chosen") and row.get("rejected"):
                        out.write(row["chosen"] + "\n" + row["rejected"] + "\n")
    bpe = ByteLevelBPETokenizer()
    bpe.train(files=[str(tmp)], vocab_size=vocab_size, min_frequency=2,
              special_tokens=SPECIAL)
    bpe.save(str(output_path))
    tmp.unlink(missing_ok=True)
    tokenizer = Tokenizer.from_file(str(output_path))
    for token in SPECIAL:
        if tokenizer.token_to_id(token) is None:
            tokenizer.add_special_tokens([token])
    tokenizer.save(str(output_path))
    print(f"[tokenizer] vocab real={tokenizer.get_vocab_size()} guardado={output_path}")
    return tokenizer


def get_tokenizer(data_paths: list[str], output_path: Path, vocab_size: int,
                  reuse: bool) -> Tokenizer:
    if output_path.exists() and reuse:
        tok = Tokenizer.from_file(str(output_path))
        missing = [s for s in SPECIAL if tok.token_to_id(s) is None]
        if missing:
            raise ValueError(f"Tokenizer existente sin tokens especiales: {missing}")
        print(f"[tokenizer] Reutilizando {output_path}, vocab={tok.get_vocab_size()}")
        return tok
    return train_tokenizer(data_paths, output_path, vocab_size)


def encode_record(tok: Tokenizer, row: dict, max_len: int,
                  answer_key: str = "assistant") -> tuple[list[int], list[int]]:
    system = row.get("system", SYSTEM_FIXED)
    user = row["user"]
    answer = row[answer_key]
    sys_ids = tok.encode("<|system|>" + system).ids
    user_ids = tok.encode("<|user|>" + user).ids
    answer_ids = tok.encode("<|assistant|>" + answer + "<|end|>").ids
    ids = sys_ids + user_ids + answer_ids
    labels = [-100] * (len(sys_ids) + len(user_ids)) + answer_ids
    if len(ids) > max_len:
        # Preserve the assistant target when a long prompt+answer exceeds the cap.
        ids = ids[-max_len:]
        labels = labels[-max_len:]
    return ids, labels


def collate(items: list[tuple[list[int], list[int]]], pad_id: int,
            device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(x[0]) for x in items)
    ids, labels = [], []
    for x, y in items:
        n = width - len(x)
        ids.append(x + [pad_id] * n)
        labels.append(y + [-100] * n)
    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_freqs(head_dim: int, max_len: int, device):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_len, device=device).float()
    angles = torch.outer(pos, inv)
    return angles.cos(), angles.sin()


def apply_rope(q, k, cos, sin):
    def rotate(x):
        c = cos[:x.size(1)].unsqueeze(0).unsqueeze(2)
        s = sin[:x.size(1)].unsqueeze(0).unsqueeze(2)
        a, b = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = a * c - b * s
        out[..., 1::2] = a * s + b * c
        return out
    return rotate(q), rotate(k)


class GQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, hd = cfg["dim"], cfg["head_dim"]
        self.qh, self.kvh, self.hd = cfg["q_heads"], cfg["kv_heads"], hd
        self.q = nn.Linear(d, self.qh * hd, bias=False)
        self.k = nn.Linear(d, self.kvh * hd, bias=False)
        self.v = nn.Linear(d, self.kvh * hd, bias=False)
        self.o = nn.Linear(self.qh * hd, d, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.qh, self.hd)
        k = self.k(x).view(b, t, self.kvh, self.hd)
        v = self.v(x).view(b, t, self.kvh, self.hd)
        q, k = apply_rope(q, k, cos, sin)
        repeat = self.qh // self.kvh
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
        scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.hd)
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.einsum("bhts,bshd->bthd", probs, v).reshape(b, t, -1)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["dim"], cfg["hidden"]
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg["dim"]), GQAAttention(cfg)
        self.norm2, self.ffn = RMSNorm(cfg["dim"]), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class WISERonin(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab"], cfg["dim"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["layers"])])
        self.norm = RMSNorm(cfg["dim"])
        cos, sin = rope_freqs(cfg["head_dim"], cfg["ctx"], torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, ids):
        if ids.size(1) > self.cfg["ctx"]:
            raise ValueError(f"Secuencia {ids.size(1)} > contexto {self.cfg['ctx']}")
        x = self.tok_emb(ids)
        cos, sin = self.rope_cos[:ids.size(1)], self.rope_sin[:ids.size(1)]
        for block in self.blocks:
            x = block(x, cos, sin)
        return F.linear(self.norm(x), self.tok_emb.weight)


def sft_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                           labels[:, 1:].contiguous().view(-1), ignore_index=-100)


def sequence_logprob(logits, labels):
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    target = labels[:, 1:]
    mask = target.ne(-100)
    picked = logp.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def orpo_loss(chosen_logits, chosen_labels, rejected_logits, rejected_labels, beta: float):
    """Length-normalized ORPO-style chosen SFT + odds preference objective."""
    chosen_lp = sequence_logprob(chosen_logits, chosen_labels)
    rejected_lp = sequence_logprob(rejected_logits, rejected_labels)
    chosen_odds = chosen_lp - torch.log1p(-torch.exp(chosen_lp).clamp(max=1 - 1e-6))
    rejected_odds = rejected_lp - torch.log1p(-torch.exp(rejected_lp).clamp(max=1 - 1e-6))
    preference = -F.logsigmoid(chosen_odds - rejected_odds).mean()
    supervised = sft_loss(chosen_logits, chosen_labels)
    return supervised + beta * preference, supervised.detach(), preference.detach()


def save_checkpoint(path: Path, model, optimizer, step: int, cfg, tokenizer_path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": cfg, "tokenizer": str(tokenizer_path)}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="Uno o varios JSONL de entrenamiento")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=None, help="Epochs completos sobre el dataset; calcula las actualizaciones")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer-wise-ronin.json")
    ap.add_argument("--vocab-size", type=int, default=6500)
    ap.add_argument("--reuse-tokenizer", action="store_true")
    ap.add_argument("--orpo", action="store_true",
                    help="Exige filas con user, chosen y rejected; no sirve con el JSONL SFT actual")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if args.steps < 1 or args.batch_size < 1 or not 1 <= args.max_seq_len <= CONFIG["ctx"]:
        ap.error("steps/batch-size deben ser positivos y max-seq-len debe estar entre 1 y 1024")
    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tok_path = Path(args.tokenizer)
    tok_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.data)
    if args.orpo:
        missing = [i for i, r in enumerate(rows) if not r.get("chosen") or not r.get("rejected")]
        if missing:
            raise ValueError("--orpo requiere chosen/rejected en cada fila; el dataset actual es SFT user/assistant. Usa SFT sin --orpo.")
    # Remove exact duplicates only; do not silently discard distinct assistant targets.
    seen, unique = set(), []
    for row in rows:
        key = (row.get("system", ""), row["user"], row.get("assistant", ""), row.get("chosen", ""), row.get("rejected", ""))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    rows = unique
    if not rows:
        raise ValueError("Dataset vacío")
    print(f"[data] total={len(rows):,} tras deduplicación exacta; device={device}")

    tokenizer = get_tokenizer(args.data, tok_path, args.vocab_size, args.reuse_tokenizer)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("No existe <|pad|> en el tokenizer")
    cfg = dict(CONFIG)
    cfg["vocab"] = tokenizer.get_vocab_size()
    cfg["ctx"] = CONFIG["ctx"]
    if args.max_seq_len > cfg["ctx"]:
        raise ValueError("max-seq-len supera el contexto del modelo")

    encoded = []
    for row in rows:
        chosen = dict(row)
        chosen["assistant"] = row["chosen"] if args.orpo else row["assistant"]
        encoded.append(encode_record(tokenizer, chosen, args.max_seq_len))
    rejected = ([encode_record(tokenizer, {**r, "assistant": r["rejected"]}, args.max_seq_len)
                 for r in rows] if args.orpo else None)

    model = WISERonin(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={params:,}; fp32 weights≈{params*4/1024/1024:.2f} MiB; config={cfg}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95), eps=1e-8)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt["config"] != cfg:
            raise ValueError("La configuración del checkpoint no coincide con la actual")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] reanudando desde step={start_step}")
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs debe ser positivo")
        updates_per_epoch = math.ceil(len(encoded) / args.batch_size)
        args.steps = start_step + args.epochs * updates_per_epoch
        print(f"[schedule] epochs={args.epochs}; updates_per_epoch={updates_per_epoch}; "
              f"additional_updates={args.epochs * updates_per_epoch}; target_global_step={args.steps}")
    else:
        updates_per_epoch = math.ceil(len(encoded) / args.batch_size)

    if start_step >= args.steps:
        raise ValueError("El checkpoint ya alcanzó o superó --steps")

    model.train()
    started = time.time()
    running = 0.0
    for step in range(start_step, args.steps):
        if args.epochs is not None:
            local_step = step - start_step
            epoch_index = local_step // updates_per_epoch
            batch_index = local_step % updates_per_epoch
            order = list(range(len(encoded)))
            random.Random(args.seed + epoch_index).shuffle(order)
            begin = batch_index * args.batch_size
            indices = order[begin:begin + args.batch_size]
        else:
            indices = random.choices(range(len(encoded)), k=args.batch_size)
        batch, labels = collate([encoded[i] for i in indices], pad_id, device)
        progress = step / max(1, args.steps - 1)
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / max(1, args.warmup_steps)
        else:
            # On resume, schedule only the remaining updates instead of restarting the cosine at the old global step.
            decay = (step - start_step) / max(1, args.steps - start_step)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, decay))))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        chosen_logits = model(batch)
        if args.orpo:
            rbatch, rlabels = collate([rejected[i] for i in indices], pad_id, device)
            rejected_logits = model(rbatch)
            loss, nll, pref = orpo_loss(chosen_logits, labels, rejected_logits, rlabels, args.beta)
        else:
            loss = sft_loss(chosen_logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss no finita en step={step+1}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += float(loss.detach())
        completed = step + 1
        if completed % 50 == 0 or completed == 1:
            elapsed = time.time() - started
            avg = running / (50 if completed % 50 == 0 else completed)
            print(f"[train] step={completed}/{args.steps} loss={avg:.5f} lr={lr:.3e} "
                  f"elapsed={elapsed/60:.1f}m", flush=True)
            running = 0.0
        if completed % args.save_every == 0 or completed == args.steps:
            ckpt_path = output / f"wise_ronin_step_{completed:05d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, completed, cfg, tok_path)
            print(f"[checkpoint] {ckpt_path}", flush=True)

    summary = {"steps": args.steps, "batch_size": args.batch_size,
               "max_seq_len": args.max_seq_len, "lr": args.lr, "seed": args.seed,
               "train_examples": len(rows), "train_files": args.data,
               "tokenizer_vocab": tokenizer.get_vocab_size(), "parameters": params,
               "orpo": args.orpo, "epochs": args.epochs, "elapsed_seconds": round(time.time() - started, 2)}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
