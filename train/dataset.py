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

def encode_rows(rows, tokenizer, max_len):
    """Tokenize the immutable corpus once; preserve row order for RNG sampling."""
    return [
        tokenizer.encode(
            (str(row["prompt"]) + "\\n" + str(row["completion"])).strip(),
            max_len,
        )
        for row in rows
    ]


def collate_token_ids(encoded_rows, pad_id=0):
    """Pad pre-tokenized rows without invoking the tokenizer in the train loop."""
    encoded = [ids for ids in encoded_rows if len(ids) >= 2]
    if not encoded:
        raise ValueError("Batch contains no sequences with at least two BPE tokens")
    width = max(map(len, encoded))
    batch = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    for index, ids in enumerate(encoded):
        batch[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


def load_token_cache(path, data_hash, tokenizer_hash, max_len, row_count):
    """Load a cache only when it is tied to the exact corpus/tokenizer/length."""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "data_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "max_len": int(max_len),
        "row_count": int(row_count),
    }
    actual = {key: cache.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Token cache metadata mismatch: expected={expected}, actual={actual}")
    rows = cache.get("encoded_rows")
    if not isinstance(rows, list) or len(rows) != row_count:
        raise ValueError("Token cache row count/content is invalid")
    return rows


def collate_rows(rows, tokenizer, max_len):
    """Backward-compatible encoder for callers without a frozen token cache."""
    return collate_token_ids(encode_rows(rows, tokenizer, max_len))


def collate(rows, tokenizer, batch_size, max_len, rng):
    """Backward-compatible random sampler plus deterministic batch encoder."""
    return collate_rows(rng.choices(rows, k=batch_size), tokenizer, max_len)
