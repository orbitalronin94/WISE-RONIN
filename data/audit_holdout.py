#!/usr/bin/env python3
"""Audit exact and high-confidence textual overlap with the frozen LUMEN holdout.

This is a conservative lexical gate, not a semantic plagiarism detector. Any
reported match must be reviewed before a final corpus can be frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def candidate_text(row: dict[str, Any]) -> str:
    return normalize(str(row.get("prompt", "")) + " " + str(row.get("completion", "")))


def audit_rows(candidates: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> dict[str, Any]:
    if len(holdout) != 500:
        raise ValueError(f"Frozen holdout must contain exactly 500 rows, got {len(holdout)}")

    holdout_ids = [str(row.get("id", "")) for row in holdout]
    if len(set(holdout_ids)) != 500 or any(not item for item in holdout_ids):
        raise ValueError("Holdout IDs must be present and unique")

    # The historical holdout schema uses user/hecho_esperado, not
    # prompt/completion. Compare those fields explicitly; never hash empty keys.
    targets: dict[str, list[tuple[str, str]]] = {}
    prompt_counts = Counter()
    for row in holdout:
        hid = str(row["id"])
        user = normalize(row.get("user", ""))
        expected = normalize(row.get("hecho_esperado", ""))
        if user:
            targets.setdefault(user, []).append((hid, "user_exact"))
            prompt_counts[user] += 1
        if expected:
            targets.setdefault(expected, []).append((hid, "expected_exact"))

    # Normalize the 500 frozen prompts once. Re-normalizing each prompt for
    # every candidate made the 19k x 500 lexical audit needlessly expensive.
    holdout_users = [
        (hid, normalize(row.get("user", "")))
        for hid, row in zip(holdout_ids, holdout)
    ]
    exact_matches = []
    substring_matches = []
    seen_candidate_hashes: dict[str, int] = {}
    duplicate_candidates = []
    for index, row in enumerate(candidates):
        text = candidate_text(row)
        prompt = normalize(row.get("prompt", ""))
        completion = normalize(row.get("completion", ""))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen_candidate_hashes:
            duplicate_candidates.append({
                "row": index, "duplicate_of": seen_candidate_hashes[digest],
                "source_key": row.get("source_key", ""),
            })
        else:
            seen_candidate_hashes[digest] = index

        matched = set()
        for field_name, field_value in (("prompt", prompt), ("completion", completion), ("combined", text)):
            if not field_value:
                continue
            for target, refs in targets.items():
                if field_value == target:
                    for hid, kind in refs:
                        matched.add((hid, kind, field_name))
        for hid, user in holdout_users:
            if user and len(user) >= 28 and user in text:
                substring_matches.append({
                    "candidate_row": index, "holdout_id": hid,
                    "matched_field": "user_substring",
                    "source_key": row.get("source_key", ""),
                    "excerpt": user[:180],
                })
        for hid, kind, field_name in sorted(matched):
            exact_matches.append({
                "candidate_row": index, "holdout_id": hid,
                "match": kind, "candidate_field": field_name,
                "source_key": row.get("source_key", ""),
            })

    unique_prompts = len(prompt_counts)
    return {
        "candidate_rows": len(candidates),
        "holdout_rows": len(holdout),
        "holdout_unique_user_prompts": unique_prompts,
        "exact_matches": exact_matches,
        "long_user_substring_matches": substring_matches,
        "duplicate_candidate_rows": duplicate_candidates,
        "exact_match_count": len(exact_matches),
        "long_user_substring_match_count": len(substring_matches),
        "duplicate_candidate_count": len(duplicate_candidates),
        "lexical_gate_pass": not exact_matches and not substring_matches,
        "candidate_integrity_pass": not duplicate_candidates,
        "limitations": [
            "Lexical audit only; it does not prove semantic independence.",
            "Long user-prompt substring matches are conservative flags and require manual review.",
            "The historical holdout contains repeated prompt variants; unique prompt count is reported separately.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Directory containing candidate JSONL files")
    parser.add_argument("--holdout", default="data/holdout_500.jsonl")
    parser.add_argument("--report", default="artifacts/holdout-overlap-report.json")
    args = parser.parse_args()

    root = Path(args.candidates)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "source_candidates_only_not_final_training_dataset":
        raise ValueError("Unexpected source artifact status")
    if manifest.get("final_dataset_ready") is not False:
        raise ValueError("Candidate artifact must not claim final dataset readiness")

    candidates = []
    source_results = []
    for source in manifest.get("sources", []):
        path = root / source["file"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source.get("sha256"):
            raise ValueError(f"{source['key']}: file SHA-256 mismatch")
        rows = read_jsonl(path)
        if len(rows) != source.get("quota") or len(rows) != source.get("rows"):
            raise ValueError(f"{source['key']}: row count mismatch")
        for row in rows:
            if row.get("source_key") != source["key"] or row.get("source") != source["repo"]:
                raise ValueError(f"{source['key']}: row provenance mismatch")
            if row.get("license") != source.get("license"):
                raise ValueError(f"{source['key']}: row license differs from manifest")
        candidates.extend(rows)
        source_results.append({
            "key": source["key"], "rows": len(rows), "sha256": digest,
            "license": source.get("license"), "revision": source.get("dataset_revision"),
        })

    if len(candidates) != manifest.get("downloaded_external_rows"):
        raise ValueError("Aggregate external row count differs from manifest")

    holdout = read_jsonl(args.holdout)
    report = audit_rows(candidates, holdout)
    report.update({
        "artifact_status": manifest.get("status"),
        "artifact_seed": manifest.get("seed"),
        "external_sources": source_results,
        "external_rows": len(candidates),
        "target_external_rows": manifest.get("external_target_rows"),
        "missing_external_rows": manifest.get("external_target_rows", 0) - len(candidates),
        "owner_curated_rows_required": manifest.get("owner_curated_rows_required"),
        "final_dataset_ready": False,
    })
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "candidate_rows", "holdout_rows", "holdout_unique_user_prompts",
        "exact_match_count", "long_user_substring_match_count",
        "duplicate_candidate_count", "lexical_gate_pass",
        "missing_external_rows", "owner_curated_rows_required",
    )}, ensure_ascii=False, indent=2))
    if not report["lexical_gate_pass"]:
        raise SystemExit("Holdout lexical-overlap gate failed; inspect the report.")
    if not report["candidate_integrity_pass"]:
        raise SystemExit("Candidate duplicate gate failed; rebuild source staging.")


if __name__ == "__main__":
    main()
