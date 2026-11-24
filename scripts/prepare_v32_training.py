#!/usr/bin/env python3
"""Assemble the explicit v3.2.005 training corpus from frozen v3.1 + immutable deltas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_BASE_SHA = "93e8c997535ca2bdc5215937e3ada6bc21e39a89d3131daa71c93b605582cd13"
EXPECTED_BASE_ROWS = 197788
EXPECTED_DELTA_ROWS = 81825
EXPECTED_VERSIONS = [f"v3.2.{i:03d}" for i in range(1, 6)]
ALLOWED_LICENSES = {"Apache-2.0", "MIT", "CC-BY-4.0", "public-domain"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            yield line_no, row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-jsonl", required=True)
    p.add_argument("--base-manifest", required=True)
    p.add_argument("--chain-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--user-authorized", action="store_true", required=True)
    a = p.parse_args()

    base = Path(a.base_jsonl)
    base_manifest = json.loads(Path(a.base_manifest).read_text(encoding="utf-8"))
    chain_dir = Path(a.chain_dir)
    chain = json.loads((chain_dir / "lumen-corpus-v3-2-version-chain.json").read_text(encoding="utf-8"))
    if sha256(base) != EXPECTED_BASE_SHA:
        raise SystemExit("Frozen v3.1 base SHA-256 mismatch")
    if base_manifest.get("total_rows") != EXPECTED_BASE_ROWS:
        raise SystemExit(f"Unexpected v3.1 base row count: {base_manifest.get('total_rows')}")
    if chain.get("cumulative_chain") is not True:
        raise SystemExit("v3.2 delta chain is not marked cumulative")
    versions = chain.get("versions", [])
    if [v.get("version") for v in versions] != EXPECTED_VERSIONS:
        raise SystemExit("Expected the complete v3.2.001-v3.2.005 chain")
    if sum(int(v.get("delta_rows", 0)) for v in versions) != EXPECTED_DELTA_ROWS:
        raise SystemExit("v3.2 delta row count does not match the reviewed ledger")

    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_rows = delta_rows = 0
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for line_no, row in read_rows(base):
            if "prompt" not in row or "completion" not in row:
                raise SystemExit(f"Base row {line_no} lacks prompt/completion")
            if row.get("split") == "holdout":
                raise SystemExit(f"Holdout contamination in base row {line_no}")
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            base_rows += 1

        for version in versions:
            delta = chain_dir / version["delta_file"]
            if sha256(delta) != version["delta_sha256"]:
                raise SystemExit(f"Delta SHA mismatch: {delta.name}")
            count = 0
            for line_no, row in read_rows(delta):
                if row.get("language") != "es":
                    raise SystemExit(f"Non-Spanish delta row {delta.name}:{line_no}")
                if row.get("review_status") not in {"accept", "auto_accept"}:
                    raise SystemExit(f"Unreviewed delta row {delta.name}:{line_no}")
                if row.get("review_status") == "auto_accept" and row.get("review_method") != "batch_policy_v1":
                    raise SystemExit(f"Unrecognized auto-approval policy {delta.name}:{line_no}")
                if row.get("rights_status") != "verified" or row.get("privacy_status") != "clear" or row.get("quality_status") != "approved":
                    raise SystemExit(f"Delta admission gate failed {delta.name}:{line_no}")
                if row.get("license") not in ALLOWED_LICENSES:
                    raise SystemExit(f"Unallowlisted license {delta.name}:{line_no}")
                text = str(row.get("text", "")).strip()
                if len(text) < 20:
                    raise SystemExit(f"Delta text too short {delta.name}:{line_no}")
                converted = dict(row)
                converted["prompt"] = ""
                converted["completion"] = text
                converted["split"] = "train"
                out.write(json.dumps(converted, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            if count != int(version["delta_rows"]):
                raise SystemExit(f"Delta row count mismatch {delta.name}: {count}")
            delta_rows += count

    if base_rows != EXPECTED_BASE_ROWS or delta_rows != EXPECTED_DELTA_ROWS:
        raise SystemExit(f"Unexpected assembled counts: base={base_rows}, delta={delta_rows}")
    digest = sha256(output)
    result = {
        "corpus": "WISE-RONIN Spanish v3.2.005",
        "base_version": "v3.1.0-candidate",
        "delta_versions": EXPECTED_VERSIONS,
        "base_rows": base_rows,
        "delta_rows": delta_rows,
        "total_train_rows": base_rows + delta_rows,
        "data_sha256": digest,
        "user_training_authorized": True,
        "authorization_note": "Explicit user authorization for this 7,500-step paired experiment; source manifests retain their original audit status.",
        "tokenizer_sha256_must_match_parent": True,
        "holdout_included": False,
        "validation_included": False,
        "source_chain_manifest": "lumen-corpus-v3-2-version-chain.json",
    }
    Path(a.manifest).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
