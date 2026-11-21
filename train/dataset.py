"""Strict JSONL training data loading and frozen BPE tokenization."""
import hashlib
import json
from pathlib import Path
import torch

class BPETokenizer:
    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install tokenizers to use the trained BPE tokenizer") from exc
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text, max_len):
        return self.tokenizer.encode(text).ids[:max_len]

def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {path}")
    rows = []
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            digest.update(line.encode("utf-8"))
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "holdout" or row.get("source") == "holdout_500.jsonl":
                raise ValueError(f"Holdout contamination marker at line {line_no}")
            if "prompt" not in row or "completion" not in row:
                raise ValueError(f"Missing prompt/completion at line {line_no}")
            rows.append(row)
    if not rows:
        raise ValueError("Training JSONL is empty")
    return rows, digest.hexdigest()

def collate_rows(rows, tokenizer, max_len):
    """Encode an already selected batch; shared by both pilot architectures."""
    encoded = []
    for row in rows:
        text = (str(row["prompt"]) + "\n" + str(row["completion"])).strip()
        ids = tokenizer.encode(text, max_len)
        if len(ids) >= 2:
            encoded.append(ids)
    if not encoded:
        raise ValueError("Batch contains no sequences with at least two BPE tokens")
    pad_id = tokenizer.tokenizer.token_to_id("[PAD]")
    if pad_id != 0:
        raise ValueError("Tokenizer must define [PAD] at token ID 0")
    width = max(map(len, encoded))
    batch = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    for index, ids in enumerate(encoded):
        batch[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


def collate(rows, tokenizer, batch_size, max_len, rng):
    """Backward-compatible random sampler plus deterministic batch encoder."""
    return collate_rows(rng.choices(rows, k=batch_size), tokenizer, max_len)
