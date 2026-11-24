#!/usr/bin/env python3
"""Batch-admit audited Spanish SFT chat rows while excluding all near-duplicate candidates."""
import argparse, hashlib, json, re, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = "OysterCoreAI/SFT-General-Spanish-50K"
REVISION = "c2b887e207e2bb721c05aedd57cd50b636ffed83"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "SFT-General-Spanish-50K v1.0.0, authors/compilers of this release, CC BY 4.0"
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")

def near_duplicate_rows(audit_db):
    """Reproduce the pinned audit's 64-bit SimHash <=3-bit candidate quarantine."""
    db = sqlite3.connect(audit_db)
    rows = {}
    buckets = defaultdict(list)
    query = """SELECT band,value,fingerprint,source,source_row FROM sim
               WHERE source IN ('v3.1', ?)"""
    for band, value, fingerprint, source, source_row in db.execute(query, (SOURCE,)):
        key = (source, int(source_row))
        fp = int(fingerprint, 16)
        buckets[(band, value)].append((key, fp))
        if source == SOURCE:
            rows[key] = fp
    db.close()
    flagged = set()
    for key, fp in rows.items():
        row_index = key[1]
        for band in range(4):
            value = (fp >> (band * 16)) & 65535
            for other, other_fp in buckets[(band, value)]:
                if other == key:
                    continue
                if other[0] != "v3.1" and not (other[0] == SOURCE and other[1] < row_index):
                    continue
                if (fp ^ other_fp).bit_count() <= 3:
                    flagged.add(row_index)
                    break
            if row_index in flagged:
                break
    if len(rows) != 48397 or len(flagged) != 13489:
        raise SystemExit(f"audit DB mismatch: rows={len(rows)}, near_duplicates={len(flagged)}")
    return flagged

def normalize(value):
    return " ".join(str(value or "").split())

def normalize_row(row, index):
    metadata = row.get("metadata")
    messages = row.get("messages")
    if not isinstance(metadata, dict):
        return None, "metadata_missing"
    if metadata.get("content_origin") != "original_synthetic_editorial":
        return None, "content_origin_not_original_synthetic_editorial"
    if str(metadata.get("license", "")).lower() != "cc-by-4.0":
        return None, "row_license_mismatch"
    if not isinstance(messages, list) or len(messages) != 3:
        return None, "invalid_message_count"
    expected_roles = ["system", "user", "assistant"]
    normalized_messages = []
    for message, expected in zip(messages, expected_roles):
        if not isinstance(message, dict) or str(message.get("role", "")).lower() != expected:
            return None, "invalid_role_sequence"
        content = normalize(message.get("content"))
        if not content:
            return None, "empty_message"
        normalized_messages.append({"role": expected, "content": content})
    scanned = "\n".join(m["content"] for m in normalized_messages)
    if EMAIL.search(scanned):
        return None, "privacy_flag:possible_email"
    if NIF.search(scanned):
        return None, "privacy_flag:possible_nif"
    if PHONE.search(scanned):
        return None, "privacy_flag:possible_phone"
    if len(normalized_messages[1]["content"].split()) < 8:
        return None, "user_turn_too_short"
    if len(normalized_messages[2]["content"].split()) < 20:
        return None, "assistant_turn_too_short"
    parts = []
    for message in normalized_messages:
        label = {"system": "Sistema", "user": "Usuario", "assistant": "Asistente"}[message["role"]]
        parts.append(f"{label}:\n{message['content']}")
    text = "\n\n".join(parts)
    digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
    safe_metadata = dict(metadata)
    return {
        "text": text, "text_sha256": digest, "task_type": "general_sft_chat",
        "language": "es", "source": SOURCE, "source_revision": REVISION,
        "source_row": str(row.get("id", index)), "license": LICENSE,
        "attribution": ATTRIBUTION, "attribution_url": "https://creativecommons.org/licenses/by/4.0/",
        "review_status": "auto_accept", "review_method": "batch_policy_v1",
        "rights_status": "verified", "privacy_status": "clear", "quality_status": "approved",
        "metadata": safe_metadata,
        "quality_checks": ["pinned_train_split", "original_synthetic_editorial",
            "system_user_assistant_roles_preserved", "release_rubric_metadata_preserved",
            "email_nif_phone_scan_clear", "near_duplicate_candidates_quarantined"]
    }, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(SOURCE, revision=REVISION)
    card = info.card_data.to_dict() if info.card_data else {}
    if info.sha != REVISION:
        raise SystemExit(f"revision mismatch: expected {REVISION}, got {info.sha}")
    if str(card.get("license", "")).lower() != "cc-by-4.0":
        raise SystemExit(f"pinned dataset license is not CC-BY-4.0: {card.get('license')!r}")
    flagged = near_duplicate_rows(args.audit_db)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows_seen = rows_accepted = tokens = 0
    rejected = Counter()
    with output_path.open("w", encoding="utf-8") as out:
        stream = load_dataset(SOURCE, name=None, split="train", streaming=True, revision=REVISION)
        for index, row in enumerate(stream):
            if args.max_rows and index >= args.max_rows:
                break
            rows_seen += 1
            if index in flagged:
                rejected["near_duplicate_candidate_quarantined"] += 1
                continue
            normalized, reason = normalize_row(row, index)
            if reason:
                rejected[reason] += 1
                continue
            digest = normalized["text_sha256"]
            if digest in seen:
                rejected["exact_duplicate_within_source"] += 1
                continue
            seen.add(digest)
            out.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
            rows_accepted += 1
            tokens += len(normalized["text"].split())
    report = {
        "schema": "lumen-batch-admission-report-v1", "source": SOURCE,
        "source_revision": REVISION, "split": "train",
        "license_verified_at_pinned_revision": LICENSE, "rows_seen": rows_seen,
        "near_duplicate_candidate_rows_quarantined": len(flagged),
        "rows_auto_accepted": rows_accepted, "rows_rejected": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "accepted_whitespace_tokens": tokens, "review_status": "auto_accept",
        "review_method": "batch_policy_v1", "training_authorized": False,
        "attribution": ATTRIBUTION, "attribution_url": "https://creativecommons.org/licenses/by/4.0/",
        "note": "The pinned audit's 13,489 near-duplicate candidates are quarantined. Original synthetic editorial chat only; system/user/assistant roles and source quality metadata are retained. CC BY 4.0 attribution is required."
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_seen": rows_seen, "near_duplicate_rows": len(flagged),
        "rows_auto_accepted": rows_accepted, "rows_rejected": sum(rejected.values()),
        "whitespace_tokens": tokens, "training_authorized": False}))

if __name__ == "__main__":
    main()
