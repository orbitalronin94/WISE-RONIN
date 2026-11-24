#!/usr/bin/env python3
"""Build sequential, provenance-locked corpus blocks capped at 100 MiB.

Input is normalized JSONL. A row is admitted only after explicit human/source
review fields are present. This script never downloads sources and never
authorizes model training.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

LIMIT = 100 * 1024 * 1024
ALLOWED = {"Apache-2.0", "MIT", "CC-BY-4.0", "public-domain"}
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")


def canonical_text(row):
    value = row.get("text")
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Reviewed normalized JSONL")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--block-prefix", default="lumen-v3-2")
    p.add_argument("--max-block-bytes", type=int, default=LIMIT)
    a = p.parse_args()
    if a.max_block_bytes < 1 or a.max_block_bytes > LIMIT:
        raise SystemExit("max-block-bytes must be between 1 and 104857600")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen = set()
    blocks, rejects = [], {}
    current = None
    block_no = 1

    def open_block():
        nonlocal block_no
        path = out / f"{a.block_prefix}-block-{block_no:04d}.jsonl"
        return {"path": path, "handle": path.open("wb"), "bytes": 0,
                "rows": 0, "sources": {}, "hash": hashlib.sha256()}

    def close_block(b):
        nonlocal block_no
        if not b or b["rows"] == 0:
            if b:
                b["handle"].close()
                b["path"].unlink(missing_ok=True)
            return
        b["handle"].close()
        blocks.append({
            "block": block_no, "file": b["path"].name,
            "bytes": b["bytes"], "rows": b["rows"],
            "sha256": hashlib.sha256(b["path"].read_bytes()).hexdigest(),
            "sources": b["sources"]
        })
        block_no += 1

    with open(a.input, encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                rejects["invalid_json"] = rejects.get("invalid_json", 0) + 1
                continue
            text = canonical_text(row)
            reason = None
            if not text:
                reason = "empty_text"
            elif row.get("review_status") != "accept":
                reason = "not_human_accepted"
            elif row.get("rights_status") != "verified":
                reason = "rights_not_verified"
            elif row.get("license") not in ALLOWED:
                reason = "license_not_allowlisted"
            elif not row.get("source") or not row.get("source_revision"):
                reason = "missing_source_provenance"
            elif EMAIL.search(text) or NIF.search(text) or PHONE.search(text):
                reason = "possible_personal_data_review"
            digest = sha(text) if text else None
            if not reason and digest in seen:
                reason = "exact_duplicate"
            payload = dict(row)
            payload["text"] = text
            payload["text_sha256"] = digest
            encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            if not reason and len(encoded) > a.max_block_bytes:
                reason = "single_record_exceeds_block_limit"
            if reason:
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            if current is None:
                current = open_block()
            if current["bytes"] + len(encoded) > a.max_block_bytes:
                close_block(current)
                current = open_block()
            current["handle"].write(encoded)
            current["hash"].update(encoded)
            current["bytes"] += len(encoded)
            current["rows"] += 1
            source_key = f'{row["source"]}@{row["source_revision"]}'
            current["sources"][source_key] = current["sources"].get(source_key, 0) + 1
            seen.add(digest)
    close_block(current)

    manifest = {
        "schema": "lumen-corpus-block-manifest-v1",
        "block_limit_bytes": a.max_block_bytes,
        "blocks": blocks,
        "accepted_rows": sum(x["rows"] for x in blocks),
        "accepted_bytes": sum(x["bytes"] for x in blocks),
        "rejected_rows_by_reason": rejects,
        "training_authorized": False,
        "note": "Block admission is not training authorization; verify source-level rights and human review before promotion."
    }
    manifest_path = out / f"{a.block_prefix}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"blocks": len(blocks), "accepted_rows": manifest["accepted_rows"],
                      "accepted_bytes": manifest["accepted_bytes"],
                      "manifest": str(manifest_path), "training_authorized": False}))


if __name__ == "__main__":
    main()
