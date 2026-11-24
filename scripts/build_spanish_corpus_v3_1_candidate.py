#!/usr/bin/env python3
"""Build a conservative Spanish v3.1 corpus candidate.

Rows flagged as personality/identity metadata or likely personal data are
quarantined from training. Original token IDs are retained for accepted base
rows; only newly admitted source rows are tokenized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
import unicodedata
from collections import Counter
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

PERSONALITY_RE = re.compile(
    r"personalidad|personality|identidad|identity|conducta|behavior|"
    r"comportamiento|assistant|asistente|honestidad|honesty|"
    r"incertidumbre|uncertainty|preferencia|preference|rol|role",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"https?://\S+", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
DNI_RE = re.compile(r"\b\d{8}[A-Z]\b", re.I)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def metadata_text(row: dict) -> str:
    keys = ("source", "dataset", "category", "domain", "task", "type",
            "subset", "tags", "metadata", "license")
    return " ".join(str(row.get(k, "")) for k in keys)


def privacy_flags(text: str) -> list[str]:
    flags = []
    if EMAIL_RE.search(text):
        flags.append("email_like")
    if DNI_RE.search(text):
        flags.append("dni_like")
    if PHONE_RE.search(text):
        flags.append("phone_like")
    return flags


def read_jsonl(path: Path):
    rows, digest = [], hashlib.sha256()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            digest.update(line.encode("utf-8"))
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
                raise ValueError(f"Invalid prompt/completion row at {path}:{line_no}")
            rows.append(row)
    return rows, digest.hexdigest()


def encode_new_row(tok, question: str, answer: str, max_len: int, eos_id: int):
    ids = list(tok.encode((question + "\n" + answer).strip()).ids)
    # Do not silently truncate questions or answers: reject instead.
    if len(ids) >= max_len:
        return None
    if not ids or ids[-1] != eos_id:
        ids.append(eos_id)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jsonl", required=True)
    ap.add_argument("--base-token-cache", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output-dir", default="artifacts/spanish-v3-1-candidate")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--review-per-source", type=int, default=200)
    args = ap.parse_args()

    base_path, cache_path, tokenizer_path = map(Path, (args.base_jsonl, args.base_token_cache, args.tokenizer))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_rows, base_hash = read_jsonl(base_path)
    tokenizer_hash = sha256_file(tokenizer_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {
        "format": "ronin_token_cache_v1",
        "data_sha256": base_hash,
        "tokenizer_sha256": tokenizer_hash,
        "max_len": int(args.max_len),
        "row_count": len(base_rows),
    }
    for key, expected in required.items():
        if cache.get(key) != expected:
            raise ValueError(f"Base token-cache mismatch for {key}: {cache.get(key)!r} != {expected!r}")
    encoded_rows = cache.get("encoded_rows")
    if not isinstance(encoded_rows, list) or len(encoded_rows) != len(base_rows):
        raise ValueError("Base token cache has no valid encoded_rows list")

    accepted_base, accepted_tokens = [], []
    quarantine = Counter()
    seen_pairs, prompt_answers = set(), {}
    for row, ids in zip(base_rows, encoded_rows):
        text = row["prompt"] + "\n" + row["completion"]
        flags = privacy_flags(text)
        personality = bool(PERSONALITY_RE.search(metadata_text(row)))
        if personality:
            quarantine["personality_identity_metadata"] += 1
            continue
        if flags:
            for flag in flags:
                quarantine[flag] += 1
            continue
        if not isinstance(ids, list) or len(ids) < 2 or len(ids) > args.max_len or any(not isinstance(x, int) for x in ids):
            quarantine["invalid_cached_tokens"] += 1
            continue
        pair = norm(row["prompt"]) + "\u241f" + norm(row["completion"])
        prompt = norm(row["prompt"])
        if pair in seen_pairs:
            quarantine["base_duplicate_pair"] += 1
            continue
        # Repeated prompts are legitimate in the pretraining/chunk corpus.
        # Only exact prompt+completion duplicates are excluded at this stage.
        seen_pairs.add(pair)
        accepted_base.append(row)
        accepted_tokens.append(ids)

    if not accepted_base:
        raise ValueError("Conservative filters removed the entire base corpus")

    # Conflict checks apply only among the newly admitted QA rows, not to the
    # base corpus where repeated prompts are expected (e.g. text chunks).
    new_prompt_answers = {}
    api, tok = HfApi(), Tokenizer.from_file(str(tokenizer_path))
    eos_id = tok.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("Frozen tokenizer does not define [EOS]")

    additions, new_encoded = [], []
    source_stats, review_by_source = {}, {}
    for source in SOURCES:
        info = api.dataset_info(source["repo_id"])
        card = info.card_data or {}
        actual_license = str(card.get("license", "")).lower()
        if actual_license != source["license"]:
            raise ValueError(f"License changed for {source['repo_id']}: {actual_license!r}")
        revision = info.sha
        dataset = load_dataset(source["repo_id"], split="train", revision=revision)
        rejected = Counter()
        added, review_candidates = 0, []
        for index, row in enumerate(dataset):
            lang_field = source["language_field"]
            if lang_field:
                lang = str(row.get(lang_field, "")).strip().lower()
                if lang not in {"es", "es-es", "spanish", "castellano"}:
                    rejected["language"] += 1
                    continue
            question_value = row.get(source["question_field"])
            answer_value = row.get(source["answer_field"])
            if not isinstance(question_value, str) or not isinstance(answer_value, str):
                rejected["invalid_type"] += 1
                continue
            question, answer = question_value.strip(), answer_value.strip()
            if len(question) < 8 or len(answer) < 8:
                rejected["empty_or_too_short"] += 1
                continue
            if privacy_flags(question + "\n" + answer):
                rejected["privacy_marker"] += 1
                continue
            if norm(question) == norm(answer):
                rejected["identical_question_answer"] += 1
                continue
            pair = norm(question) + "\u241f" + norm(answer)
            prompt = norm(question)
            if pair in seen_pairs:
                rejected["duplicate_pair"] += 1
                continue
            if prompt in new_prompt_answers and new_prompt_answers[prompt] != norm(answer):
                rejected["conflicting_prompt"] += 1
                continue
            ids = encode_new_row(tok, question, answer, args.max_len, eos_id)
            if ids is None:
                rejected["over_context_no_truncation"] += 1
                continue
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
            new_encoded.append(ids)
            seen_pairs.add(pair)
            new_prompt_answers[prompt] = norm(answer)
            added += 1
            review_candidates.append(record)
        source_stats[source["repo_id"]] = {
            "revision": revision,
            "license": actual_license,
            "rows_seen": len(dataset),
            "rows_added": added,
            "rejected": dict(rejected),
        }
        if len(review_candidates) > args.review_per_source:
            review_candidates = random.Random(42).sample(review_candidates, args.review_per_source)
        review_by_source[source["repo_id"]] = review_candidates

    if not additions:
        raise ValueError("No new rows passed conservative admission filters")

    combined_path = out / "spanish-v3-1-candidate.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for row in accepted_base + additions:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    combined_hash = sha256_file(combined_path)
    all_tokens = accepted_tokens + new_encoded
    new_cache = {
        "format": "ronin_token_cache_v1",
        "data_sha256": combined_hash,
        "tokenizer_sha256": tokenizer_hash,
        "max_len": int(args.max_len),
        "row_count": len(accepted_base) + len(additions),
        "encoded_rows": all_tokens,
    }
    torch.save(new_cache, out / "spanish-v3-1-token-cache.pt")

    with (out / "new_rows_manual_review.jsonl").open("w", encoding="utf-8") as f:
        for source_name in sorted(review_by_source):
            for row in review_by_source[source_name]:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "name": "WISE-RONIN Spanish corpus v3.1 conservative candidate",
        "version": "3.1.0-candidate",
        "status": "CANDIDATE_NOT_AUTHORIZED",
        "training_authorized": False,
        "language": "es",
        "shared_by": ["Lumen", "Transformer"],
        "original_base_rows": len(base_rows),
        "verified_base_rows_retained": len(accepted_base),
        "quarantined_base_rows": len(base_rows) - len(accepted_base),
        "quarantine_reasons": dict(quarantine),
        "added_rows": len(additions),
        "total_rows": len(accepted_base) + len(additions),
        "max_len": int(args.max_len),
        "tokenizer_sha256": tokenizer_hash,
        "original_base_data_sha256": base_hash,
        "combined_data_sha256": combined_hash,
        "sources": source_stats,
        "quarantine_policy": "Personality/identity metadata and likely personal-data markers are excluded pending row-level human review; no automatic deletion from the frozen source artifact.",
        "validation_and_holdout": "must be copied byte-for-byte from the frozen base artifact by the workflow",
        "review_required": [
            "Human spot-check of stratified additions from every admitted source.",
            "Review quarantined personality/identity rows before any possible later re-admission.",
            "Inspect false positives among phone-like markers; quarantined rows remain excluded until reviewed.",
            "Verify byte-for-byte protected validation and holdout files.",
            "Run cache/data/tokenizer hash checks before authorizing training.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
