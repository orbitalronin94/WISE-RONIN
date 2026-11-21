#!/usr/bin/env python3
"""Build a provisional 19k-row pilot corpus from the immutable candidate artifact.

This is deliberately separate from the production 25k builder. It preserves
source attribution, checks source hashes/row counts, deduplicates, and audits
the frozen factual holdout. It does not mark third-party rights as cleared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from data.audit_holdout import audit_rows, read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def language(row: dict) -> str:
    declared = str(row.get("language", "")).lower().split("-")[0]
    if declared in {"es", "en", "ca", "pt", "fr", "de", "it"}:
        return declared
    words = set(norm(row.get("prompt", "") + " " + row.get("completion", "")).split())
    es = len(words & {"que", "de", "la", "el", "en", "los", "una", "para", "con", "por", "como", "qué", "del"})
    en = len(words & {"the", "of", "and", "to", "in", "is", "for", "with", "that", "this", "from", "what"})
    if es == en == 0:
        return "und"
    return "es" if es > en else "en"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--holdout", default="data/holdout_500.jsonl")
    parser.add_argument("--output", default="data/pilot_train_19k.jsonl")
    parser.add_argument("--manifest", default="data/pilot_train_19k.manifest.json")
    parser.add_argument("--pilot-authorization", action="store_true",
                        help="Record the owner's explicit authorization for this provisional private pilot.")
    args = parser.parse_args()
    if not args.pilot_authorization:
        parser.error("--pilot-authorization is required; this is not the production corpus builder")

    root = Path(args.sources)
    source_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "source_candidates_only_not_final_training_dataset":
        raise ValueError("Unexpected candidate artifact status")
    if source_manifest.get("downloaded_external_rows") != 19000:
        raise ValueError("Pilot expects exactly 19,000 externally staged rows")

    rows: list[dict] = []
    provenance = []
    for source in source_manifest["sources"]:
        path = root / source["file"]
        digest = sha256(path)
        if digest != source["sha256"]:
            raise ValueError(f"{source['key']}: source SHA-256 mismatch")
        part = read_jsonl(path)
        if len(part) != source["rows"] or len(part) != source["quota"]:
            raise ValueError(f"{source['key']}: row-count mismatch")
        for index, row in enumerate(part):
            if row.get("source_key") != source["key"] or row.get("source") != source["repo"]:
                raise ValueError(f"{source['key']} row {index}: provenance mismatch")
            if row.get("license") != source["license"]:
                raise ValueError(f"{source['key']} row {index}: license mismatch")
            if not str(row.get("completion", "")).strip():
                raise ValueError(f"{source['key']} row {index}: empty completion")
            row["language"] = language(row)
        rows.extend(part)
        provenance.append({
            "key": source["key"], "repository": source["repo"],
            "revision": source["dataset_revision"], "rows": len(part),
            "license": source["license"], "license_url": source["license_url"],
            "license_reviewed": source.get("license_reviewed", False),
            "rights_confirmed": source.get("rights_confirmed", False),
            "file": source["file"], "sha256": digest,
        })

    if len(rows) != 19000:
        raise ValueError(f"Expected 19,000 rows, got {len(rows)}")

    seen: dict[str, int] = {}
    prefixes = Counter()
    for index, row in enumerate(rows):
        text = norm(str(row.get("prompt", "")) + " " + str(row.get("completion", "")))
        if not text:
            raise ValueError(f"Empty normalized record at row {index}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate row {index}; first occurrence {seen[digest]}")
        seen[digest] = index
        prefixes[" ".join(text.split()[:5])] += 1
    repeated = {key: count for key, count in prefixes.items() if count > 10}
    if repeated:
        raise ValueError(f"Repeated five-token prefixes exceed 10: {list(repeated.items())[:10]}")

    holdout = read_jsonl(args.holdout)
    audit = audit_rows(rows, holdout)
    if not audit["lexical_gate_pass"] or not audit["candidate_integrity_pass"]:
        raise ValueError("Pilot corpus failed duplicate/holdout lexical audit")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    type_counts = Counter(str(row.get("type", "unknown")) for row in rows)
    language_counts = Counter(str(row.get("language", "und")) for row in rows)
    pilot_manifest = {
        "project": "RONIN-LUMEN V1 provisional pilot",
        "status": "provisional_private_pilot_only",
        "owner_pilot_authorization": True,
        "training_rows": len(rows),
        "training_sha256": sha256(out),
        "source_artifact_seed": source_manifest.get("seed"),
        "source_artifact_created_at_utc": source_manifest.get("created_at_utc"),
        "source_artifact_status": source_manifest.get("status"),
        "sources": provenance,
        "type_counts": dict(type_counts),
        "language_counts_heuristic": dict(language_counts),
        "holdout_audit": {
            "rows": audit["holdout_rows"],
            "unique_prompts": audit["holdout_unique_user_prompts"],
            "exact_matches": audit["exact_match_count"],
            "long_substring_matches": audit["long_user_substring_match_count"],
            "duplicate_candidates": audit["duplicate_candidate_count"],
            "lexical_gate_pass": audit["lexical_gate_pass"],
            "method": "NFKC/casefold exact + long user-prompt substring; not semantic proof",
        },
        "rights_review_status": "pending_human_review; owner authorization is not a third-party license clearance",
        "production_corpus": False,
        "production_training_authorized": False,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(pilot_manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(pilot_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
