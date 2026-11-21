"""Matched-size decoder-only Transformer baseline for RONIN-LUMEN pilot."""
from __future__ import annotations
import argparse, hashlib, logging, random
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from model.config import CONFIG
from train.dataset import BPETokenizer, collate_rows, load_jsonl

LOG = logging.getLogger("lumen.baseline")

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

def main():
    p = argparse.ArgumentParser(description="Matched-size Transformer pilot baseline")
    p.add_argument("--data", default="data/pilot_train_19k.jsonl")
    p.add_argument("--tokenizer", default="data/pilot_tokenizer/tokenizer.json")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--ckpt-dir", default="checkpoints/transformer")
    args = p.parse_args()
    if not 1 <= args.steps <= 500:
        p.error("Pilot baseline is limited to 1..500 steps")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows, data_hash = load_jsonl(args.data)
    tokenizer = BPETokenizer(args.tokenizer)
    if tokenizer.tokenizer.get_vocab_size() != 4096 or tokenizer.tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("Baseline requires the shared 4k pilot tokenizer with PAD=0")
    model = SmallTransformerLM(CONFIG.vocab_size, args.max_len).cpu()
    params = sum(x.numel() for x in model.parameters())
    LOG.info("baseline=decoder_transformer params=%d data_rows=%d data_sha256=%s",
             params, len(rows), data_hash)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CONFIG.weight_decay)
    scheduler = make_scheduler(optimizer, args.warmup, args.total_steps)
    rng = random.Random(args.seed)
    tokens_seen = 0
    model.train()
    for step in range(1, args.steps + 1):
        chosen = rng.choices(rows, k=args.batch)
        tokens = collate_rows(chosen, tokenizer, args.max_len)
        tokens_seen += int(tokens.ne(0).sum())
        logits = model(tokens)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            LOG.info("baseline=transformer step=%d/%d loss=%.5f tokens_seen=%d lr=%.3e",
                     step, args.steps, loss.item(), tokens_seen, scheduler.get_last_lr()[0])
    out = Path(args.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"step_{args.steps}.pt"
    torch.save({
        "model": model.state_dict(), "step": args.steps, "seed": args.seed,
        "data_sha256": data_hash, "tokenizer_sha256": hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
        "params": params, "tokens_seen": tokens_seen,
        "architecture": "4-layer decoder-only Transformer, d_model=192, heads=4, ffn=576, tied embeddings",
    }, ckpt)
    LOG.info("checkpoint=%s", ckpt)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
