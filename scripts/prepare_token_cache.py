#!/usr/bin/env python3
"""Build a content-addressed token cache for deterministic CPU training."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# When invoked as python scripts/prepare_token_cache.py, Python places
# scripts/ (not the repository root) on sys.path. Add the project root so the
# shared train package resolves consistently in GitHub Actions and locally.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from train.dataset import BPETokenizer, encode_rows, load_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_path, tokenizer_path = Path(args.data), Path(args.tokenizer)
    rows, data_hash = load_jsonl(data_path)
    tokenizer = BPETokenizer(tokenizer_path)
    if tokenizer.tokenizer.token_to_id("[PAD]") != 0:
        raise ValueError("Tokenizer must define [PAD] at token ID 0")
    encoded = encode_rows(rows, tokenizer, args.max_len)
    payload = {
        "format": "ronin_token_cache_v1",
        "data_sha256": data_hash,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "max_len": int(args.max_len),
        "row_count": len(rows),
        "encoded_rows": encoded,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"token_cache={output} rows={len(rows)} max_len={args.max_len} "
        f"data_sha256={data_hash} tokenizer_sha256={payload['tokenizer_sha256']} "
        f"cache_bytes={output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
