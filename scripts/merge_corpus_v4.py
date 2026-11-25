#!/usr/bin/env python3
"""Merge the frozen v3.2.005 base with v4 candidate shards and enforce final gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import shutil
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
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
DNI_RE = re.compile(r"\b(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+34[\s.-]?)?(?:[6789]\d{2})[\s.-]?\d{3}[\s.-]?\d{3}(?!\w)")


def has_personal_identifier(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or DNI_RE.search(text) or PHONE_RE.search(text))


PERSONALITY_TASKS = {
    "persona_profile", "persona_dialogue", "synthetic_persona_voice",
    "persona_dialogue_lm", "dialogue_and_assistant_style_lm",
}

# Strict admission: only explicit public-domain, Apache-2.0 or CC-BY-4.0 records.
# SA/GFDL, NC, internal/unclear and missing-license rows are held back for review.
ALLOWED_LICENSES = {"public_domain", "Apache-2.0", "CC-BY-4.0"}


def is_license_admissible(value: object) -> bool:
    return str(value or "").strip() in ALLOWED_LICENSES



def iter_jsonl_file(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def iter_candidate_rows(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        yield from iter_jsonl_file(path)


def row_token_count(tokenizer: Tokenizer, prompt: str, completion: str) -> int:
    text = (prompt + "\n" + completion).strip()
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def metadata_text(row: dict) -> str:
    """Flatten every metadata value so denylist checks cannot miss nested fields."""
    excluded = {"prompt", "completion", "text", "id", "content_sha256", "sha256"}
    values = []

    def visit(value, key=""):
        if key.lower() in excluded:
            return
        if isinstance(value, str):
            if value.strip():
                values.append(value)
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child_value in value:
                visit(child_value, key)

    visit(row)
    return "\n".join(values)


def find_blocked_base_documents(path: Path, patterns):
    """Reject every base row from a source document if any sibling row hits the denylist."""
    blocked = {}
    for index, raw in enumerate(iter_jsonl_file(path)):
        row = normalize_base(raw, index)
        source_name = str(row.get("source") or "unknown")
        source_row = str(row.get("source_row") or "").strip()
        if source_name == "BSC-LT/m-personas" and source_row:
            source_row = source_row.split(".", 1)[0]
        document_id = (
            source_row if source_name == "BSC-LT/m-personas" and source_row else
            row.get("source_document_id")
            or row.get("document_id")
            or row.get("source_url")
            or row.get("document_url")
            or row.get("article_url")
            or row.get("source_id")
            or source_row
            or row.get("id")
        )
        if document_id is None or not str(document_id).strip():
            document_id = f"row:{index}"
        document_key = f"{source_name}::{document_id}"
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        hits = find_attractors(
            prompt + "\n" + completion + "\n" + metadata_text(row), patterns
        )
        if hits:
            blocked[document_key] = sorted(set(blocked.get(document_key, [])) | set(hits))
    return blocked


def missing_candidate_provenance(row: dict):
    required = (
        "source_revision", "source_document_id", "license",
        "license_url", "attribution",
    )
    return [field for field in required if not str(row.get(field) or "").strip()]


def normalize_base(row: dict, index: int):
    row = dict(row)
    if not isinstance(row.get("completion"), str):
        row["completion"] = str(row.get("text") or "")
    if not isinstance(row.get("prompt"), str):
        row["prompt"] = ""
    if not row.get("task"):
        row["task"] = row.get("task_type") or "unknown"
    row.setdefault("language", "es")
    if not row.get("source_document_id"):
        if str(row.get("source") or "") == "BSC-LT/m-personas":
            source_row = str(row.get("source_row") or "").strip()
            parent_persona_id = source_row.split(".", 1)[0] if source_row else ""
            document_id = row.get("document_id") or row.get("source_id") or parent_persona_id
        else:
            document_id = row.get("document_id") or row.get("source_id") or row.get("source_row")
        if document_id is not None and str(document_id).strip():
            row["source_document_id"] = str(document_id).strip()
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
    parser.add_argument("--source-plan", type=Path, default=Path("data/corpus_v4_source_plan.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corpus-v4-frozen"))
    parser.add_argument("--chunk-tokens", type=int, default=126)
    parser.add_argument("--max-shard-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--minimum-raw-text-rows", type=int, default=1_800_000)
    parser.add_argument("--minimum-personality-dialogue-rows", type=int, default=175_000)
    parser.add_argument("--minimum-personality-share", type=float, default=0.10)
    parser.add_argument("--maximum-personality-share", type=float, default=0.15)
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
    total_tokens = 0

    def reject(row, reason, hits=None):
        reject_counts[reason] += 1
        if hits:
            attractor_counts.update(hits)
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or row.get("text") or "")
        payload = {
            "id": row.get("id"),
            "source": row.get("source"),
            "source_document_id": (row.get("source_document_id") or row.get("document_id") or row.get("source_url") or row.get("document_url") or row.get("article_url") or row.get("source_id") or row.get("source_row")),
            "task": row.get("task") or row.get("task_type"),
            "content_sha256": hashlib.sha256(
                normalize(prompt + "\n" + completion).encode("utf-8")
            ).hexdigest(),
            "content_characters": len(prompt) + len(completion),
            "admission_rejection": {
                "reason": reason,
                "matched_terms": hits or []
            }
        }
        rejected_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def admit(row, origin):
        nonlocal raw_text_rows, personality_rows, total_tokens
        prompt = str(row.get("prompt") or "")
        completion = str(row.get("completion") or "")
        if not completion.strip():
            reject(row, "empty_completion")
            return
        if row.get("split") == "holdout" or str(row.get("source") or "") == "holdout_500.jsonl":
            reject(row, "holdout_contamination")
            return
        if str(row.get("language") or "") != "es":
            reject(row, "non_spanish_or_unverified_language")
            return
        if has_personal_identifier(prompt + "\n" + completion):
            reject(row, "personal_identifier_pattern")
            return

        hits = find_attractors(
            prompt + "\n" + completion + "\n" + metadata_text(row), patterns
        )
        if hits:
            reject(row, "lumen_attractor_denylist", hits)
            return

        task = str(row.get("task") or "unknown")
        source = str(row.get("source") or origin)
        license_name = str(row.get("license") or "unknown")
        if not is_license_admissible(license_name):
            reject(row, "license_unverified_restricted_or_complex")
            return

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
            final_row["admission_origin"] = origin
            if len(pieces) > 1:
                final_row["id"] = hashlib.sha256(
                    f"{row.get('id','')}|v4-chunk|{chunk_index}".encode()
                ).hexdigest()[:24]
                final_row["chunk_index"] = chunk_index

            full_text = (final_row["prompt"] + "\n" + piece).strip()
            token_count = row_token_count(tokenizer, final_row["prompt"], piece)
            if token_count > args.chunk_tokens:
                reject(final_row, "token_limit_violation")
                continue
            final_row["token_count"] = token_count
            digest = hashlib.sha256(normalize(full_text).encode("utf-8")).hexdigest()
            try:
                db.execute("INSERT INTO rows(sha256) VALUES (?)", (digest,))
            except sqlite3.IntegrityError:
                reject_counts["duplicate_exact"] += 1
                continue

            writer.write(final_row)
            total_tokens += token_count
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
        blocked_documents = find_blocked_base_documents(args.base_jsonl, patterns)
        print(f"BASE_DOCUMENTS_BLOCKED {len(blocked_documents)}", flush=True)
        for index, row in enumerate(iter_jsonl_file(args.base_jsonl)):
            base_rows_seen += 1
            normalized_row = normalize_base(row, index)
            source_name = str(normalized_row.get("source") or "unknown")
            source_row = str(normalized_row.get("source_row") or "").strip()
            if source_name == "BSC-LT/m-personas" and source_row:
                source_row = source_row.split(".", 1)[0]
            document_id = (
                source_row if source_name == "BSC-LT/m-personas" and source_row else
                normalized_row.get("source_document_id")
                or normalized_row.get("document_id")
                or normalized_row.get("source_url")
                or normalized_row.get("document_url")
                or normalized_row.get("article_url")
                or normalized_row.get("source_id")
                or source_row
                or normalized_row.get("id")
            )
            if document_id is None or not str(document_id).strip():
                document_id = f"row:{index}"
            document_key = f"{source_name}::{document_id}"
            if document_key in blocked_documents:
                reject(normalized_row, "source_document_contains_lumen_attractor",
                       blocked_documents[document_key])
            else:
                admit(normalized_row, "v3.2.005_base")
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
            if source_manifest.get("tokenizer_sha256") != sha256_file(args.tokenizer):
                raise ValueError(f"{label} candidate tokenizer hash does not match the paired-model tokenizer")
            candidate_terms = set(source_manifest.get("denylist_terms", []))
            final_terms = set(denylist["terms"])
            if candidate_terms != final_terms:
                raise ValueError(f"{label} candidate denylist terms do not exactly match the frozen v4 denylist")
            if source_manifest.get("denylist_sha256") != sha256_file(args.denylist):
                raise ValueError(f"{label} candidate denylist SHA-256 does not match the frozen v4 denylist")
            if int(source_manifest.get("chunk_tokens", 999)) > args.chunk_tokens:
                raise ValueError(f"{label} candidate chunks exceed the final merge token budget")
            if source_manifest.get("row_target_met") is not True:
                raise ValueError(f"{label} candidate did not meet its row target")
            if label == "personality":
                expected_sources = {
                    "m_personas_es": (
                        "BSC-LT/m-personas",
                        "6fab205a86a9244c90a2782e28055027f45ddd03",
                        "Apache-2.0",
                        "https://www.apache.org/licenses/LICENSE-2.0",
                        20_000,
                    ),
                    "sft_general_spanish": (
                        "OysterCoreAI/SFT-General-Spanish-50K",
                        "25d2187",
                        "CC-BY-4.0",
                        "https://creativecommons.org/licenses/by/4.0/",
                        20_000,
                    ),
                }
                stats = source_manifest.get("source_stats", {})
                for source_key, (repo, revision, license_name, license_url, minimum_rows) in expected_sources.items():
                    source_info = stats.get(source_key, {})
                    if (
                        source_info.get("repo") != repo
                        or source_info.get("revision") != revision
                        or source_info.get("license") != license_name
                        or source_info.get("license_url") != license_url
                    ):
                        raise ValueError(f"Personality source lineage/license mismatch: {source_key}")
                    if int(source_info.get("rows", 0)) < minimum_rows:
                        raise ValueError(f"Personality source contribution too small: {source_key}")
            if label == "openculture":
                expected_sources = {
                    "spanish_pd_books": (
                        "PleIAs/Spanish-PD-Books",
                        "001eaf13681f483069361dd82195ce279e12ed63",
                        "public_domain",
                        "https://huggingface.co/datasets/PleIAs/Spanish-PD-Books",
                        1_800_000,
                    ),
                }
                stats = source_manifest.get("source_stats", {})
                for source_key, (repo, revision, license_name, license_url, minimum_rows) in expected_sources.items():
                    source_info = stats.get(source_key, {})
                    if (
                        source_info.get("repo") != repo
                        or source_info.get("revision") != revision
                        or source_info.get("license") != license_name
                        or source_info.get("license_url") != license_url
                    ):
                        raise ValueError(f"OpenCulture source lineage/license mismatch: {source_key}")
                    if int(source_info.get("rows", 0)) < minimum_rows:
                        raise ValueError(f"OpenCulture source contribution too small: {source_key}")
            candidate_manifest_sha[label] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            candidate_files = {path.name: path for path in root.rglob("*.jsonl")}
            for shard in source_manifest.get("shards", []):
                shard_path = candidate_files.get(shard["file"])
                if shard_path is None:
                    raise FileNotFoundError(f"Missing {label} shard: {shard['file']}")
                if shard_path.stat().st_size != int(shard["bytes"]):
                    raise ValueError(f"{label} shard byte count mismatch: {shard['file']}")
                if sha256_file(shard_path) != shard["sha256"]:
                    raise ValueError(f"{label} shard SHA-256 mismatch: {shard['file']}")
            for row in iter_candidate_rows(root):
                candidate_rows_seen[label] += 1
                missing = missing_candidate_provenance(row)
                if missing:
                    raise ValueError(
                        f"{label} candidate row lacks provenance fields {missing}: {row.get('id')}"
                    )
                admit(row, label)
                if counts["accepted_rows"] and counts["accepted_rows"] % 25_000 == 0:
                    db.commit()
                    print(f"{label} candidates={candidate_rows_seen[label]} accepted={counts['accepted_rows']}", flush=True)
            manifest_rows = int(source_manifest.get("actual_rows", 0))
            if candidate_rows_seen[label] != manifest_rows:
                raise ValueError(
                    f"{label} shard row count mismatch: {candidate_rows_seen[label]} != {manifest_rows}"
                )
            counts[f"{label}_manifest_rows"] = manifest_rows

        db.commit()
    finally:
        writer.close()
        db.close()
        rejected_handle.close()

    dedup_path = args.output_dir / "dedup.sqlite"
    if dedup_path.exists():
        dedup_path.unlink()
    shutil.copy2(args.denylist, args.output_dir / "attractor_denylist.json")
    shutil.copy2(args.tokenizer, args.output_dir / "tokenizer.json")
    if args.source_plan.is_file():
        shutil.copy2(args.source_plan, args.output_dir / "source_plan.json")

    shard_rows = sum(item["rows"] for item in writer.shards)
    personality_share = personality_rows / max(1, shard_rows)
    manifest = {
        "schema": "wise-ronin-corpus-v4-frozen-manifest-v1",
        "version": "v4.0.0-candidate",
        "status": "FROZEN_CANDIDATE_PENDING_REVIEW",
        "training_authorized": False,
        "base_jsonl_sha256": sha256_file(args.base_jsonl),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "tokenizer_file": "tokenizer.json",
        "denylist_sha256": sha256_file(args.denylist),
        "source_plan_sha256": sha256_file(args.source_plan) if args.source_plan.is_file() else None,
        "denylist_terms": denylist["terms"],
        "base_rows_seen": base_rows_seen,
        "candidate_rows_seen": dict(candidate_rows_seen),
        "candidate_manifest_sha256": candidate_manifest_sha,
        "total_rows": shard_rows,
        "token_count_without_special_tokens": total_tokens,
        "estimated_tokens_with_one_eos_per_row": total_tokens + shard_rows,
        "raw_text_pretraining_rows": raw_text_rows,
        "personality_dialogue_rows": personality_rows,
        "personality_dialogue_share": personality_share,
        "task_counts": dict(task_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "license_counts": dict(license_counts.most_common()),
        "rejection_counts": dict(reject_counts.most_common()),
        "attractor_rejections_by_term": dict(attractor_counts.most_common()),
        "minimum_raw_text_rows": args.minimum_raw_text_rows,
        "minimum_personality_dialogue_rows": args.minimum_personality_dialogue_rows,
        "minimum_personality_share": args.minimum_personality_share,
        "maximum_personality_share": args.maximum_personality_share,
        "raw_text_target_met": raw_text_rows >= args.minimum_raw_text_rows,
        "personality_target_met": (
            personality_rows >= args.minimum_personality_dialogue_rows
            and personality_share >= args.minimum_personality_share
            and personality_share <= args.maximum_personality_share
        ),
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
            f"Personality/dialogue target not met: rows={personality_rows}, share={personality_share:.4%}, required rows>={args.minimum_personality_dialogue_rows} and share in [{args.minimum_personality_share:.0%},{args.maximum_personality_share:.0%}]"
        )


if __name__ == "__main__":
    main()
