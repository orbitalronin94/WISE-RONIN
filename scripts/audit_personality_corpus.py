#!/usr/bin/env python3
"""Audit the frozen 250k corpus and extract a reviewable personality subset.

This script does not modify training data. It emits deterministic audit reports and
redacted review samples so a human can inspect the original examples before v3.1.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

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


def clean_for_review(value: str) -> str:
    value = EMAIL_RE.sub("[EMAIL_REDACTED]", value)
    value = URL_RE.sub("[URL_REDACTED]", value)
    value = PHONE_RE.sub("[PHONE_REDACTED]", value)
    return DNI_RE.sub("[ID_REDACTED]", value)


def stable_text(row: dict) -> str:
    parts = []
    for key in ("system", "prompt", "instruction", "input", "context", "completion", "output", "response", "answer", "messages"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def metadata_text(row: dict) -> str:
    return " ".join(
        str(row.get(k, "")) for k in
        ("source", "dataset", "category", "domain", "task", "type", "subset", "tags", "metadata", "license")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="artifacts/personality-audit")
    ap.add_argument("--sample-per-source", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_counts = collections.Counter()
    task_counts = collections.Counter()
    schema_counts = collections.Counter()
    candidates = []
    rows = 0
    empty = 0
    exact_hashes = collections.Counter()
    length_stats = []
    pii_counts = collections.Counter()
    personality_key_hits = collections.Counter()
    samples_by_source = collections.defaultdict(list)

    with src.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            if not isinstance(row, dict):
                raise TypeError(f"line {line_no}: expected JSON object")
            schema_counts[",".join(sorted(row.keys()))] += 1
            source = str(row.get("source", row.get("dataset", "unknown")))
            task = str(row.get("task", row.get("category", row.get("domain", "unknown"))))
            source_counts[source] += 1
            task_counts[task] += 1
            body = stable_text(row)
            if not body:
                empty += 1
            length_stats.append(len(body))
            if body:
                exact_hashes[hashlib.sha256(" ".join(body.casefold().split()).encode()).hexdigest()] += 1
            metadata = metadata_text(row)
            if PERSONALITY_RE.search(metadata):
                candidates.append((line_no, row, body))
                for key in row:
                    if PERSONALITY_RE.search(key):
                        personality_key_hits[key] += 1
            if EMAIL_RE.search(body): pii_counts["email"] += 1
            if PHONE_RE.search(body): pii_counts["phone_like"] += 1
            if DNI_RE.search(body): pii_counts["dni_like"] += 1
            samples_by_source[source].append((line_no, row, body))

    rng = random.Random(args.seed)
    review_rows = []
    for source in sorted(samples_by_source):
        items = samples_by_source[source]
        rng.shuffle(items)
        for line_no, row, body in items[:args.sample_per_source]:
            review_rows.append({
                "line": line_no,
                "source": source,
                "id": row.get("id"),
                "task": row.get("task", row.get("category")),
                "prompt": clean_for_review(str(row.get("prompt", row.get("instruction", "")))),
                "completion": clean_for_review(str(row.get("completion", row.get("response", row.get("answer", ""))))),
                "text_preview": clean_for_review(body)[:1600],
                "metadata": {k: row.get(k) for k in row if k not in {"prompt", "completion", "input", "context", "answer", "response", "output", "messages"}},
            })

    with (out / "personality_candidates_redacted.jsonl").open("w", encoding="utf-8") as f:
        for line_no, row, body in candidates:
            item = {
                "line": line_no,
                "id": row.get("id"),
                "source": row.get("source", row.get("dataset")),
                "task": row.get("task", row.get("category")),
                "prompt": clean_for_review(str(row.get("prompt", row.get("instruction", "")))),
                "completion": clean_for_review(str(row.get("completion", row.get("response", row.get("answer", ""))))),
                "text_preview": clean_for_review(body)[:2000],
                "metadata": {k: row.get(k) for k in row if k not in {"prompt", "completion", "input", "context", "answer", "response", "output", "messages"}},
            }
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    with (out / "manual_review_sample_redacted.jsonl").open("w", encoding="utf-8") as f:
        for item in review_rows:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    duplicates = sum(n - 1 for n in exact_hashes.values() if n > 1)
    report = {
        "input": str(src),
        "rows": rows,
        "empty_text_rows": empty,
        "unique_normalized_texts": len(exact_hashes),
        "exact_duplicate_excess": duplicates,
        "mean_chars": round(sum(length_stats) / max(1, len(length_stats)), 2),
        "median_chars": sorted(length_stats)[len(length_stats)//2] if length_stats else 0,
        "source_counts": dict(source_counts.most_common()),
        "task_counts": dict(task_counts.most_common()),
        "schema_counts": dict(schema_counts.most_common(20)),
        "personality_candidate_rows_by_metadata_keyword": len(candidates),
        "personality_candidate_key_hits": dict(personality_key_hits),
        "potential_pii_row_counts": dict(pii_counts),
        "manual_review_sample_rows": len(review_rows),
        "sample_seed": args.seed,
        "warning": "Keyword candidates are a retrieval aid, not a semantic personality classification. Human review is required. Review exports redact likely email, URL, phone and DNI patterns.",
    }
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
