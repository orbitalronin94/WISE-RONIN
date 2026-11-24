#!/usr/bin/env python3
"""Build a non-authorized v3.1 candidate by adding audited Spanish QA sources.

The script preserves the existing 250k JSONL rows and their cached token IDs,
then appends only normalized, non-duplicate records from two Apache-2.0 datasets.
It does not alter validation/holdout and does not authorize training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer


SOURCES = [
    {
        "repo_id": "somosnlp/LingComp_QA",
        "name": "LingComp_QA",
        "license": "apache-2.0",
        "question_field": "pregunta",
        "answer_field": "respuesta",
        "task": "computational_linguistics_qa",
        "language_field": None,
    },
    {
        "repo_id": "somosnlp/lenguaje-claro-dataset",
        "name": "lenguaje-claro-dataset",
        "license": "apache-2.0",
        "question_field": "question",
        "answer_field": "answer",
        "task": "plain_language_rewrite",
        "language_field": "idioma",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_jsonl(path: Path):
    rows = []
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            digest.update(line.encode("utf-8"))
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "prompt" not in row or "completion" not in row:
                raise ValueError(f"Invalid training row at {path}:{line_no}")
            rows.append(row)
    return rows, digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jsonl", required=True)
    ap.add_argument("--base-token-cache", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output-dir", default="artifacts/spanish-v3-1-candidate")
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    base_path = Path(args.base_jsonl)
    cache_path = Path(args.base_token_cache)
    tokenizer_path = Path(args.tokenizer)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_rows, base_hash = read_jsonl(base_path)
    tokenizer_hash = sha256_file(tokenizer_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    required_cache = {
        "format": "ronin_token_cache_v1",
        "data_sha256": base_hash,
        "tokenizer_sha256": tokenizer_hash,
        "max_len": args.max_len,
        "row_count": len(base_rows),
    }
    for key, expected in required_cache.items():
        if cache.get(key) != expected:
            raise ValueError(f"Base token-cache mismatch for {key}: {cache.get(key)!r} != {expected!r}")
    encoded_rows = cache.get("encoded_rows")
    if not isinstance(encoded_rows, list) or len(encoded_rows) != len(base_rows):
        raise ValueError("Base token cache has no valid encoded_rows list")

    existing = set()
    for row in base_rows:
        key = norm(str(row["prompt"])) + "\u241f" + norm(str(row["completion"]))
        existing.add(hashlib.sha256(key.encode()).hexdigest())

    api = HfApi()
    tok = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tok.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("Frozen tokenizer does not define [EOS]")

    additions = []
    source_stats = {}
    review = []
    for source in SOURCES:
        info = api.dataset_info(source["repo_id"])
        card = info.card_data or {}
        actual_license = str(card.get("license", "")).lower()
        if actual_license != source["license"]:
            raise ValueError(f"License changed for {source['repo_id']}: {actual_license!r}")
        revision = info.sha
        dataset = load_dataset(source["repo_id"], split="train", revision=revision)
        seen_source = set()
        rejected = {"empty": 0, "language": 0, "duplicate": 0}
        accepted = 0
        for index, row in enumerate(dataset):
            if source["language_field"]:
                lang = str(row.get(source["language_field"], "")).lower()
                if lang not in {"es", "es-es", "spanish", "castellano"}:
                    rejected["language"] += 1
                    continue
            question = str(row.get(source["question_field"], "")).strip()
            answer = str(row.get(source["answer_field"], "")).strip()
            if not question or not answer:
                rejected["empty"] += 1
                continue
            key = norm(question) + "\u241f" + norm(answer)
            digest = hashlib.sha256(key.encode()).hexdigest()
            if digest in existing or digest in seen_source:
                rejected["duplicate"] += 1
                continue
            seen_source.add(digest)
            record = {
                "id": f"v31-{source['name']}-{index:06d}",
                "prompt": question,
                "completion": answer,
                "task": source["task"],
                "language": "es",
                "source": source["repo_id"],
                "source_id": str(index),
                "license": "Apache-2.0",
                "source_revision": revision,
            }
            additions.append(record)
            existing.add(digest)
            accepted += 1
            if len(review) < 400:
                review.append(record)
        source_stats[source["repo_id"]] = {
            "revision": revision,
            "license": actual_license,
            "rows_seen": len(dataset),
            "rows_added": accepted,
            "rejected": rejected,
        }

    if not additions:
        raise ValueError("No new rows were admitted; refusing to publish an empty extension")

    combined_path = out / "spanish-v3-1-candidate.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for row in base_rows + additions:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    combined_hash = sha256_file(combined_path)

    new_encoded = []
    for row in additions:
        text = (str(row["prompt"]) + "\n" + str(row["completion"])).strip()
        ids = list(tok.encode(text).ids)
        if len(ids) >= args.max_len:
            ids = ids[:args.max_len - 1]
        if not ids or ids[-1] != eos_id:
            ids = ids[:args.max_len - 1]
            ids.append(eos_id)
        new_encoded.append(ids)

    new_cache = {
        "format": "ronin_token_cache_v1",
        "data_sha256": combined_hash,
        "tokenizer_sha256": tokenizer_hash,
        "max_len": int(args.max_len),
        "row_count": len(base_rows) + len(additions),
        "encoded_rows": encoded_rows + new_encoded,
    }
    torch.save(new_cache, out / "spanish-v3-1-token-cache.pt")

    with (out / "new_rows_manual_review.jsonl").open("w", encoding="utf-8") as f:
        for row in review:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "name": "WISE-RONIN Spanish corpus v3.1 candidate",
        "version": "3.1.0-candidate",
        "status": "CANDIDATE_NOT_AUTHORIZED",
        "training_authorized": False,
        "language": "es",
        "shared_by": ["Lumen", "Transformer"],
        "base_rows": len(base_rows),
        "added_rows": len(additions),
        "total_rows": len(base_rows) + len(additions),
        "max_len": args.max_len,
        "tokenizer_sha256": tokenizer_hash,
        "base_data_sha256": base_hash,
        "combined_data_sha256": combined_hash,
        "sources": source_stats,
        "validation_and_holdout": "not modified; copied from the frozen base artifact by the workflow",
        "review_required": [
            "Manual semantic review of the original 50k personality/factual subset.",
            "Human spot-check of all admitted source families and a stratified sample of additions.",
            "Near-duplicate and answer-consistency review before changing training_authorized.",
            "Confirm that no evaluation examples have entered training.",
        ],
        "note": "The original base rows and cached token IDs are preserved. Only new admitted rows are tokenized with the frozen tokenizer.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
