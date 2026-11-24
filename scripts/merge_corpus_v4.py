#!/usr/bin/env python3
"""Merge the frozen v3.2.005 base with v4 candidate shards and enforce final gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

try:
    from .build_corpus_v4_openculture import (
        ShardWriter, encode_chunks, find_attractors, load_denylist,
        normalize, sha256_file,
    )
except ImportError:
    from build_corpus_v4_openculture import (
        ShardWriter, encode_chunks, find_attractors, load_denylist,
        normalize, sha256_file,
    )

CHUNKABLE_TASKS = {
    "raw_text_pretraining", "lm_text", "persona_profile",
    "persona_dialogue", "synthetic_persona_voice",
}
PERSONALITY_TASKS = {
    "persona_profile", "persona_dialogue", "synthetic_persona_voice",
    "persona_dialogue_lm", "dialogue_and_assistant_style_lm",
}


def iter_jsonl_file(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def iter_candidate_rows(root: Path):
    for path in sorted(root.glob("*.jsonl")):
        yield from iter_jsonl_file(path)


def row_token_count(tokenizer: Tokenizer, prompt: str, completion: str) -> int:
    text = (prompt + "\n" + completion).strip()
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def normalize_base(row: dict, index: int):
    row = dict(row)
    if not isinstance(row.get("completion"), str):
        row["completion"] = str(row.get("text") or "")
    if not isinstance(row.get("prompt"), str):
        row["prompt"] = ""
    if not row.get("task"):
        row["task"] = row.get("task_type") or "unknown"
    row.setdefault("language", "es")
    row.setdefault("id", hashlib.sha256(
        f"{row.get('source','base')}|{row.get('source_id',row.get('source_row',index))}".encode()
    ).hexdigest()[:24])
    row.pop("text", None)
    row.pop("task_type", None)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-jsonl", required=True, type=Path)
    parser.add_argument("--openculture-dir", required=True, type=Path)
    parser.add_argument("--personality-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--denylist", type=Path, default=Path("data/corpus_v4_attractor_denylist.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corpus-v4-frozen"))
    parser.add_argument("--chunk-tokens", type=int, default=126)
    parser.add_argument("--max-shard-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--minimum-raw-text-rows", type=int, default=1_800_000)
    parser.add_argument("--minimum-personality-dialogue-rows", type=int, default=160_000)
    args = parser.parse_args()
    if not 32 <= args.chunk_tokens <= 126:
        parser.error("chunk-tokens must be 32..126")
    if args.max_shard_bytes > 100 * 1024 * 1024:
        parser.error("max-shard-bytes cannot exceed 100 MiB")

    for path in (args.output_dir,):
        path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    denylist, patterns = load_denylist(args.denylist)
    writer = ShardWriter(args.output_dir, args.max_shard_bytes)
    rejected_path = args.output_dir / "rejected_rows.jsonl"
    rejected_handle = rejected_path.open("w", encoding="utf-8")
    db = sqlite3.connect(args.output_dir / "dedup.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS rows (sha256 TEXT PRIMARY KEY)")
    db.commit()

    counts = Counter()
    task_counts = Counter()
    source_counts = Counter()
    license_counts = Counter()
    reject_counts = Counter()
    attractor_counts = Counter()
    base_rows_seen = 0
    candidate_rows_seen = Counter()
    raw_text_rows = 0
    personality_rows = 0

    def reject(row, reason, hits=None):
        reject_counts[reason] += 1
        if hits:
            attractor_counts.update(hits)
        payload = dict(row)
        payload["_admission_rejection"] = {
            "reason": reason,
            "matched_terms": hits or []
        }
        rejected_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def admit(row, origin):
        nonlocal raw_text_rows, personality_rows
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        if not completion.strip():
            reject(row, "empty_completion")
            return

        hits = find_attractors(prompt + "\n" + completion, patterns)
        if hits:
            reject(row, "lumen_attractor_denylist", hits)
            return

        task = str(row.get("task") or "unknown")
        source = str(row.get("source") or origin)
        license_name = str(row.get("license") or "unknown")

        pieces = []
        if not prompt.strip() and task in CHUNKABLE_TASKS:
            encoded = tokenizer.encode(completion, add_special_tokens=False)
            if len(encoded.ids) > args.chunk_tokens:
                pieces = [
                    chunk for _, chunk, _ in encode_chunks(
                        completion, tokenizer, args.chunk_tokens
                    )
                ]
            else:
                pieces = [completion.strip()]
        else:
            if row_token_count(tokenizer, prompt, completion) > args.chunk_tokens:
                reject(row, "overlength_prompt_completion")
                return
            pieces = [completion.strip()]

        if not pieces:
            reject(row, "no_valid_token_chunks")
            return

        for chunk_index, piece in enumerate(pieces):
            if not piece:
                continue
            final_row = dict(row)
            final_row["prompt"] = prompt if len(pieces) == 1 else ""
            final_row["completion"] = piece
            final_row["task"] = task
            final_row["language"] = str(row.get("language") or "es")
            if len(pieces) > 1:
                final_row["id"] = hashlib.sha256(
                    f"{row.get('id','')}|v4-chunk|{chunk_index}".encode()
                ).hexdigest()[:24]
                final_row["chunk_index"] = chunk_index

            full_text = (final_row["prompt"] + "\n" + piece).strip()
            if row_token_count(tokenizer, final_row["prompt"], piece) > args.chunk_tokens:
                reject(final_row, "token_limit_violation")
                continue
            digest = hashlib.sha256(normalize(full_text).encode("utf-8")).hexdigest()
            try:
                db.execute("INSERT INTO rows(sha256) VALUES (?)", (digest,))
            except sqlite3.IntegrityError:
                reject_counts["duplicate_exact"] += 1
                continue

            writer.write(final_row)
            counts["accepted_rows"] += 1
            task_counts[task] += 1
            source_counts[source] += 1
            license_counts[license_name] += 1
            if task == "raw_text_pretraining":
                raw_text_rows += 1
            if task in PERSONALITY_TASKS:
                personality_rows += 1

    candidate_manifest_sha = {}
    try:
        for index, row in enumerate(iter_jsonl_file(args.base_jsonl)):
            base_rows_seen += 1
            admit(normalize_base(row, index), "v3.2.005_base")
            if base_rows_seen % 25_000 == 0:
                db.commit()
                print(f"BASE {base_rows_seen}", flush=True)

        for label, root in (
            ("openculture", args.openculture_dir),
            ("personality", args.personality_dir),
        ):
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if source_manifest.get("training_authorized") is not False:
                raise ValueError(f"{label} candidate must remain unapproved for training")
            if source_manifest.get("row_target_met") is not True:
                raise ValueError(f"{label} candidate did not meet its row target")
            candidate_manifest_sha[label] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            for row in iter_candidate_rows(root):
                candidate_rows_seen[label] += 1
                admit(row, label)
                if counts["accepted_rows"] % 25_000 == 0:
                    db.commit()
                    print(f"{label} candidates={candidate_rows_seen[label]} accepted={counts['accepted_rows']}", flush=True)
            counts[f"{label}_manifest_rows"] = int(source_manifest.get("actual_rows", 0))

        db.commit()
    finally:
        writer.close()
        db.close()
        rejected_handle.close()

    shard_rows = sum(item["rows"] for item in writer.shards)
    manifest = {
        "schema": "wise-ronin-corpus-v4-frozen-manifest-v1",
        "version": "v4.0.0-candidate",
        "status": "FROZEN_CANDIDATE_PENDING_REVIEW",
        "training_authorized": False,
        "base_jsonl_sha256": sha256_file(args.base_jsonl),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "denylist_sha256": sha256_file(args.denylist),
        "denylist_terms": denylist["terms"],
        "base_rows_seen": base_rows_seen,
        "candidate_rows_seen": dict(candidate_rows_seen),
        "candidate_manifest_sha256": candidate_manifest_sha,
        "total_rows": shard_rows,
        "raw_text_pretraining_rows": raw_text_rows,
        "personality_dialogue_rows": personality_rows,
        "task_counts": dict(task_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "license_counts": dict(license_counts.most_common()),
        "rejection_counts": dict(reject_counts.most_common()),
        "attractor_rejections_by_term": dict(attractor_counts.most_common()),
        "minimum_raw_text_rows": args.minimum_raw_text_rows,
        "minimum_personality_dialogue_rows": args.minimum_personality_dialogue_rows,
        "raw_text_target_met": raw_text_rows >= args.minimum_raw_text_rows,
        "personality_target_met": personality_rows >= args.minimum_personality_dialogue_rows,
        "max_sequence_length": 128,
        "chunk_tokens": args.chunk_tokens,
        "max_shard_bytes": args.max_shard_bytes,
        "shards": writer.shards,
        "sha256": {
            item["file"]: item["sha256"] for item in writer.shards
        },
        "notes": [
            "The original v3.2.005 training JSONL is filtered with the hard LUMEN attractor denylist.",
            "Raw text and persona text are re-chunked to at most 126 tokens using the exact shared tokenizer.",
            "Prompt/completion SFT rows longer than the context budget are rejected rather than silently truncated.",
            "Exact duplicates are removed across the base and both candidate collections.",
            "Final corpus is split into immutable shards of at most 100 MiB.",
            "This artifact is frozen as a candidate but is not authorized for training until explicit review.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    if not manifest["raw_text_target_met"]:
        raise SystemExit(
            f"Raw text target not met: {raw_text_rows} < {args.minimum_raw_text_rows}"
        )
    if not manifest["personality_target_met"]:
        raise SystemExit(
            f"Personality/dialogue target not met: {personality_rows} < {args.minimum_personality_dialogue_rows}"
        )


if __name__ == "__main__":
    main()
