#!/usr/bin/env python3
"""Build a provenance-preserving Spanish dialogue/persona expansion for corpus v4."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

try:
    from .build_corpus_v4_openculture import (
        ShardWriter, encode_chunks, find_attractors, load_denylist, metadata_text,
        normalize, sha256_file,
    )
except ImportError:
    from build_corpus_v4_openculture import (
        ShardWriter, encode_chunks, find_attractors, load_denylist, metadata_text,
        normalize, sha256_file, text_quality,
    )


SOURCES = [
    {
        "key": "m_personas_es",
        "repo": "BSC-LT/m-personas",
        "revision": "6fab205a86a9244c90a2782e28055027f45ddd03",
        "split": "es",
        "quota_share": 0.6586944444,
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "attribution": "BSC-LT mPersonas dataset, Apache-2.0",
        "task": "persona_dialogue_lm",
    },
    {
        "key": "sft_general_spanish",
        "repo": "OysterCoreAI/SFT-General-Spanish-50K",
        "revision": "25d2187",
        "split": "train",
        "quota_share": 0.3413055556,
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "SFT-General-Spanish-50K v1.0.0, authors/compilers, CC BY 4.0",
        "task": "dialogue_and_assistant_style_lm",
    },
]


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
DNI_RE = re.compile(r"\b(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+34[\s.-]?)?(?:[6789]\d{2})[\s.-]?\d{3}[\s.-]?\d{3}(?!\w)")


def conversation_quality(text: str) -> tuple[bool, str]:
    text = text.strip()
    words = re.findall(r"\b\w+\b", text, re.UNICODE)
    if len(text) < 180:
        return False, "conversation_too_short"
    if len(words) < 25:
        return False, "conversation_too_few_words"
    alpha_ratio = sum(ch.isalpha() for ch in text) / max(1, len(text))
    if alpha_ratio < 0.52:
        return False, "conversation_low_alphabetic_density"
    if EMAIL_RE.search(text) or DNI_RE.search(text) or PHONE_RE.search(text):
        return False, "conversation_personal_identifier_pattern"
    import langid
    if langid.classify(text[:4000])[0] != "es":
        return False, "conversation_not_spanish"
    return True, "ok"


def serialize_persona(item):
    persona = str(item.get("persona") or "").strip()
    messages = item.get("messages") or []
    parts = []
    if persona:
        parts.append("Persona ficticia: " + persona)
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        label = {
            "user": "Usuario", "assistant": "Asistente",
            "system": "Sistema", "tool": "Herramienta"
        }.get(role, "Interlocutor")
        parts.append(f"{label}: {content}")
    return "\n".join(parts), persona


def serialize_sft(item):
    messages = item.get("messages") or []
    metadata = item.get("metadata") or {}
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        label = {
            "user": "Usuario", "assistant": "Asistente",
            "system": "Sistema", "tool": "Herramienta"
        }.get(role, "Interlocutor")
        parts.append(f"{label}: {content}")
    return "\n".join(parts), metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--denylist", type=Path, default=Path("data/corpus_v4_attractor_denylist.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corpus-v4-personality"))
    parser.add_argument("--target-rows", type=int, default=180_000)
    parser.add_argument("--chunk-tokens", type=int, default=120)
    parser.add_argument("--max-shard-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target_rows < 1 or not 32 <= args.chunk_tokens <= 126:
        parser.error("target-rows must be positive and chunk-tokens must be 32..126")
    if args.max_shard_bytes > 100 * 1024 * 1024:
        parser.error("max-shard-bytes cannot exceed 100 MiB")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer_sha = sha256_file(args.tokenizer)
    denylist, patterns = load_denylist(args.denylist)
    writer = ShardWriter(args.output_dir, args.max_shard_bytes)
    db = sqlite3.connect(args.output_dir / "dedup.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (sha256 TEXT PRIMARY KEY)")
    db.commit()

    total_rows = 0
    docs_seen = Counter()
    docs_accepted = Counter()
    rows_by_source = Counter()
    rejects = Counter()
    source_stats = {}

    try:
        for source in SOURCES:
            quota = round(args.target_rows * source["quota_share"])
            dataset = load_dataset(
                source["repo"], split=source["split"], streaming=True,
                revision=source["revision"]
            )
            if source["key"] == "m_personas_es":
                dataset = dataset.shuffle(seed=args.seed, buffer_size=5000)
            else:
                dataset = dataset.shuffle(seed=args.seed, buffer_size=5000)

            for index, item in enumerate(dataset):
                if rows_by_source[source["key"]] >= quota or total_rows >= args.target_rows:
                    break
                docs_seen[source["key"]] += 1

                if source["key"] == "m_personas_es":
                    if str(item.get("lang", "es")).lower() != "es":
                        rejects["m_personas_not_es"] += 1
                        continue
                    text, persona = serialize_persona(item)
                    source_id = str(item.get("id") or index)
                    task = source["task"]
                    extra = {
                        "persona_type": str(item.get("type") or ""),
                        "persona_theme": str(item.get("theme") or ""),
                        "parent_persona_id": str(item.get("parent_persona_id") or ""),
                    }
                else:
                    text, metadata = serialize_sft(item)
                    source_id = str(item.get("id") or index)
                    task = source["task"]
                    extra = {
                        "task_family": str(metadata.get("task_family") or ""),
                        "domain": str(metadata.get("domain") or ""),
                        "content_origin": str(metadata.get("content_origin") or "original_synthetic_editorial"),
                        "source_attribution": str(metadata.get("source_attribution") or source["attribution"]),
                    }

                if not text.strip():
                    rejects["empty_conversation"] += 1
                    continue
                metadata_probe = metadata_text(item)
                hits = find_attractors(text + "\n" + metadata_probe, patterns)
                if hits:
                    rejects["attractor:" + "|".join(hits)] += 1
                    continue
                ok, reason = conversation_quality(text)
                if not ok:
                    rejects[reason] += 1
                    continue

                docs_accepted[source["key"]] += 1
                for chunk_index, chunk, _token_ids in encode_chunks(text, tokenizer, args.chunk_tokens):
                    if rows_by_source[source["key"]] >= quota or total_rows >= args.target_rows:
                        break
                    digest = hashlib.sha256(normalize(chunk).encode("utf-8")).hexdigest()
                    try:
                        db.execute("INSERT INTO chunks(sha256) VALUES (?)", (digest,))
                    except sqlite3.IntegrityError:
                        rejects["duplicate_chunk"] += 1
                        continue
                    row = {
                        "id": hashlib.sha256(
                            f"{source['key']}|{source['revision']}|{source_id}|{chunk_index}".encode("utf-8")
                        ).hexdigest()[:24],
                        "prompt": "",
                        "completion": chunk,
                        "task": task,
                        "language": "es",
                        "source": source["repo"],
                        "source_revision": source["revision"],
                        "source_document_id": source_id,
                        "chunk_index": chunk_index,
                        "license": source["license"],
                        "license_url": source["license_url"],
                        "attribution": source["attribution"],
                        **{k: v for k, v in extra.items() if v},
                    }
                    writer.write(row)
                    rows_by_source[source["key"]] += 1
                    total_rows += 1
                    if total_rows % 10_000 == 0:
                        db.commit()
                        print(f"PROGRESS personality_dialogue_rows={total_rows}/{args.target_rows}", flush=True)

            db.commit()
            source_stats[source["key"]] = {
                "repo": source["repo"],
                "revision": source["revision"],
                "split": source["split"],
                "license": source["license"],
                "license_url": source["license_url"],
                "rows": rows_by_source[source["key"]],
                "quota": quota,
                "documents_seen": docs_seen[source["key"]],
                "documents_accepted": docs_accepted[source["key"]],
            }
            print(f"END {source['key']} rows={rows_by_source[source['key']]}/{quota}", flush=True)
    finally:
        writer.close()
        db.close()
    dedup_path = args.output_dir / "dedup.sqlite"
    if dedup_path.exists():
        dedup_path.unlink()

    manifest = {
        "schema": "wise-ronin-corpus-v4-personality-manifest-v1",
        "status": "candidate_build_not_training_authorized",
        "target_rows": args.target_rows,
        "actual_rows": total_rows,
        "row_target_met": total_rows >= args.target_rows,
        "max_sequence_length": 128,
        "chunk_tokens": args.chunk_tokens,
        "max_shard_bytes": args.max_shard_bytes,
        "tokenizer_sha256": tokenizer_sha,
        "denylist_sha256": sha256_file(args.denylist),
        "denylist_terms": denylist["terms"],
        "source_stats": source_stats,
        "documents_seen": dict(docs_seen),
        "documents_accepted": dict(docs_accepted),
        "rejections": dict(rejects.most_common()),
        "shards": writer.shards,
        "training_authorized": False,
        "notes": [
            "m-personas source_text seed field is deliberately excluded; only synthetic persona and conversation text is used.",
            "SFT-General-Spanish train split only; validation and test splits remain excluded.",
            "Each serialized conversation is split into at most 126 BPE-token chunks.",
            "The LUMEN attractor denylist is applied to the complete conversation before chunking.",
            "Chunks are exact-deduplicated in SQLite across both source datasets.",
            "Source license, revision and attribution are preserved per row.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if total_rows < args.target_rows:
        raise SystemExit(f"Target not met: {total_rows} < {args.target_rows}")


if __name__ == "__main__":
    main()

# Shared utilities are imported from the scripts package.
