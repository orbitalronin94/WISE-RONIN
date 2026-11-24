#!/usr/bin/env python3
"""Create cumulative LUMEN corpus versions from reviewed 100 MiB delta blocks.

Each v3.2.NNN version points to its predecessor and adds one immutable JSONL
delta. The chain is cumulative without rewriting prior versions.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

LIMIT = 100 * 1024 * 1024
ALLOWED = {"Apache-2.0", "MIT", "CC-BY-4.0", "public-domain"}


def canonical_text(row):
    values = []
    def walk(value, depth=0):
        if depth > 5 or len(values) > 200:
            return
        if isinstance(value, str) and len(value.strip()) >= 20:
            values.append(" ".join(value.split()))
        elif isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() not in {"id", "uuid", "url", "link", "license", "language", "lang", "split", "label", "labels"}:
                    walk(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)
    if isinstance(row.get("text"), str):
        return " ".join(row["text"].split())
    walk(row)
    return "\\n".join(dict.fromkeys(values))


def digest_bytes(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Human-reviewed normalized JSONL")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--parent-version", default="v3.1")
    p.add_argument("--base-jsonl", required=True, help="Frozen v3.1 JSONL used to seed exact deduplication")
    p.add_argument("--start-version", type=int, default=1)
    p.add_argument("--max-block-bytes", type=int, default=LIMIT)
    p.add_argument("--dedup-db", default=None, help="Persistent SQLite DB shared across all versions")
    p.add_argument("--allow-auto-approved", action="store_true",
                   help="Allow auto_accept rows only when review_method=batch_policy_v1")
    a = p.parse_args()
    if a.max_block_bytes < 1 or a.max_block_bytes > LIMIT:
        raise SystemExit("max-block-bytes must be between 1 and 104857600")
    if a.start_version < 1:
        raise SystemExit("start-version must be >= 1")

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = Path(a.dedup_db) if a.dedup_db else out / "v3-2-chain-dedup.sqlite"
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE IF NOT EXISTS accepted (text_sha256 TEXT PRIMARY KEY, version TEXT, source TEXT)")
    if db.execute("SELECT COUNT(*) FROM accepted").fetchone()[0] == 0:
        with Path(a.base_jsonl).open(encoding="utf-8") as base:
            for line in base:
                if not line.strip():
                    continue
                try:
                    base_row = json.loads(line)
                except Exception:
                    continue
                base_text = canonical_text(base_row)
                if base_text:
                    base_hash = hashlib.sha256(base_text.encode("utf-8")).hexdigest()
                    db.execute("INSERT OR IGNORE INTO accepted VALUES (?,?,?)", (base_hash, a.parent_version, "v3.1"))
        db.commit()

    versions = []
    rejected = {}
    current = None
    version_no = a.start_version
    parent = a.parent_version

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    def open_version():
        version = f"v3.2.{version_no:03d}"
        path = out / f"lumen-corpus-{version}-delta.jsonl"
        return {"version": version, "path": path, "handle": path.open("wb"),
                "bytes": 0, "rows": 0, "sources": {}, "tokens": 0}

    def close_version(v):
        nonlocal version_no, parent
        if not v:
            return
        if v["rows"] == 0:
            v["handle"].close()
            v["path"].unlink(missing_ok=True)
            return
        v["handle"].close()
        entry = {
            "schema": "lumen-corpus-version-v1",
            "version": v["version"],
            "parent_version": parent,
            "delta_file": v["path"].name,
            "delta_bytes": v["bytes"],
            "delta_rows": v["rows"],
            "delta_whitespace_tokens": v["tokens"],
            "delta_sha256": digest_bytes(v["path"]),
            "sources": v["sources"],
            "cumulative": True,
            "training_authorized": False
        }
        manifest = out / f"lumen-corpus-{v['version']}-manifest.json"
        manifest.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        versions.append({**entry, "manifest_file": manifest.name,
                         "manifest_sha256": digest_bytes(manifest)})
        parent = v["version"]
        version_no += 1

    with Path(a.input).open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                reject("invalid_json")
                continue
            text = canonical_text(row)
            if not text:
                reject("empty_text")
                continue
            status = row.get("review_status")
            auto_ok = (a.allow_auto_approved and status == "auto_accept"
                       and row.get("review_method") == "batch_policy_v1")
            if status != "accept" and not auto_ok:
                reject("not_approved_or_unrecognized_review_method")
                continue
            if row.get("rights_status") != "verified":
                reject("rights_not_verified")
                continue
            if row.get("privacy_status") != "clear":
                reject("privacy_not_cleared")
                continue
            if row.get("quality_status") != "approved":
                reject("quality_not_approved")
                continue
            if row.get("license") not in ALLOWED:
                reject("license_not_allowlisted")
                continue
            if not row.get("source") or not row.get("source_revision"):
                reject("missing_source_provenance")
                continue

            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if db.execute("SELECT 1 FROM accepted WHERE text_sha256=?", (text_hash,)).fetchone():
                reject("duplicate_against_previous_or_current_versions")
                continue

            payload = dict(row)
            payload["text"] = text
            payload["text_sha256"] = text_hash
            encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            if len(encoded) > a.max_block_bytes:
                reject("single_record_exceeds_block_limit")
                continue

            if current is None:
                current = open_version()
            if current["bytes"] + len(encoded) > a.max_block_bytes:
                close_version(current)
                current = open_version()

            current["handle"].write(encoded)
            current["bytes"] += len(encoded)
            current["rows"] += 1
            current["tokens"] += len(text.split())
            source_key = f'{row["source"]}@{row["source_revision"]}'
            current["sources"][source_key] = current["sources"].get(source_key, 0) + 1
            db.execute("INSERT INTO accepted VALUES (?,?,?)",
                       (text_hash, current["version"], row["source"]))
            if current["rows"] % 5000 == 0:
                db.commit()

    close_version(current)
    db.commit()
    db.close()

    summary = {
        "schema": "lumen-corpus-version-chain-v1",
        "starting_parent": a.parent_version,
        "versions_created": len(versions),
        "versions": versions,
        "rejected_rows_by_reason": rejected,
        "block_limit_bytes": a.max_block_bytes,
        "cumulative_chain": True,
        "training_authorized": False,
        "admission_mode": "mixed" if any(x.get("review_method") == "batch_policy_v1" for x in []) else ("automated_batch_policy_v1" if a.allow_auto_approved else "human_reviewed"),
        "note": "Each version is its parent plus one immutable delta. Automated admission is limited to rows passing the source-specific policy; it does not authorize training."
    }
    summary_path = out / "lumen-corpus-v3-2-version-chain.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"versions_created": len(versions), "versions": [x["version"] for x in versions],
                      "summary": str(summary_path), "training_authorized": False}))


if __name__ == "__main__":
    main()
