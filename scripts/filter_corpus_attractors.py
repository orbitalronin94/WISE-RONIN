#!/usr/bin/env python3
"""Reject JSONL documents containing current LUMEN attractor terms/phrases."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value).strip()


def compile_terms(terms: list[str]):
    compiled = []
    for raw in terms:
        term = normalize(raw)
        if term:
            compiled.append((raw, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")))
    return compiled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--denylist", type=Path,
                        default=Path("data/corpus_v4_attractor_denylist.json"))
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    policy = json.loads(args.denylist.read_text(encoding="utf-8"))
    compiled = compile_terms(policy["terms"])
    counts: Counter[str] = Counter()
    total = accepted = rejected = malformed = 0

    for path in (args.accepted, args.rejected, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8") as src, \
         args.accepted.open("w", encoding="utf-8") as good, \
         args.rejected.open("w", encoding="utf-8") as bad:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                bad.write(json.dumps({
                    "_filter_error": "invalid_json",
                    "_source_line": line_number,
                    "_raw": line.rstrip("\n")
                }, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            text = row.get(args.text_field)
            if not isinstance(text, str):
                text = row.get("completion", row.get("text", ""))
            if not isinstance(text, str):
                text = ""
            normalized_text = normalize(text)
            hits = [raw for raw, pattern in compiled if pattern.search(normalized_text)]
            if hits:
                text_hash = hashlib.sha256(
                    normalize(text).encode("utf-8")
                ).hexdigest()
                bad.write(json.dumps({
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "source_document_id": row.get("source_document_id"),
                    "task": row.get("task") or row.get("task_type"),
                    "content_sha256": text_hash,
                    "content_characters": len(text),
                    "admission_rejection": {
                        "reason": "lumen_attractor_denylist",
                        "matched_terms": hits
                    }
                }, ensure_ascii=False) + "\n")
                counts.update(hits)
                rejected += 1
            else:
                good.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1

    report = {
        "schema": "wise-ronin-attractor-filter-report-v1",
        "input": str(args.input),
        "denylist": str(args.denylist),
        "total_rows": total,
        "accepted_rows": accepted,
        "rejected_rows": rejected,
        "malformed_rows": malformed,
        "rejected_by_term": dict(counts.most_common()),
        "policy_terms": policy["terms"],
        "note": "A document is rejected if any denylist term or phrase appears. Counts by term can overlap."
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
