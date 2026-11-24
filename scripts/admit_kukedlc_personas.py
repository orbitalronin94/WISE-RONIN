#!/usr/bin/env python3
"""Conservatively batch-admit Spanish persona descriptions from a pinned source revision."""
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

SOURCE = "Kukedlc/Personas_Spanish-40k"
REVISION = "2f78060ecab15d65ba8890f4f642fd0b18260405"
LICENSE = "Apache-2.0"
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
ES_WORDS = {"que","de","la","el","en","los","una","para","con","por","como","del","este","esta","son"}
EN_WORDS = {"the","of","and","to","in","is","for","with","that","this","from","are"}
WORD = re.compile(r"[^\W_]+", re.UNICODE)

def normalize(value):
    return " ".join(str(value).split())

def language_is_spanish(text):
    words = {x.casefold() for x in WORD.findall(text)}
    es = len(words & ES_WORDS)
    en = len(words & EN_WORDS)
    return es >= 3 and es > en

def normalized_hash_text(text):
    return " ".join(WORD.findall(text.casefold()))

def normalize_row(row):
    description = normalize(row.get("Persona_Descripcion") or row.get("descripcion") or row.get("text") or "")
    if len(description) < 40:
        return None, "description_missing_or_too_short"
    if not language_is_spanish(description):
        return None, "language_unresolved"
    if EMAIL.search(description):
        return None, "privacy_flag:possible_email"
    if NIF.search(description):
        return None, "privacy_flag:possible_nif"
    if PHONE.search(description):
        return None, "privacy_flag:possible_phone"
    if sum(c.isalpha() for c in description) / max(1, len(description)) < 0.60:
        return None, "low_text_density"
    text = "Descripción de persona:\n" + description
    digest = hashlib.sha256(normalized_hash_text(text).encode("utf-8")).hexdigest()
    return {
        "text": text,
        "text_sha256": digest,
        "task_type": "persona_profile",
        "language": "es",
        "source": SOURCE,
        "source_revision": REVISION,
        "source_row": str(row.get("id", "")),
        "license": LICENSE,
        "review_status": "auto_accept",
        "review_method": "batch_policy_v1",
        "rights_status": "verified",
        "privacy_status": "clear",
        "quality_status": "approved",
        "quality_checks": ["pinned_train_split", "spanish_stopword_heuristic",
            "description_min_40_chars", "alphabetic_density_min_60_percent",
            "email_nif_phone_scan_clear", "exact_duplicate_filter"]
    }, None

def main():
    parser = argparse.ArgumentParser()
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
    if str(card.get("license", "")).lower() != "apache-2.0":
        raise SystemExit(f"pinned dataset license is not Apache-2.0: {card.get('license')!r}")
    seen = set()
    rows_seen = rows_accepted = tokens = 0
    rejected = Counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        stream = load_dataset(SOURCE, name=None, split="train", streaming=True, revision=REVISION)
        for index, row in enumerate(stream):
            if args.max_rows and index >= args.max_rows:
                break
            rows_seen += 1
            normalized, reason = normalize_row(row)
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
        "rows_auto_accepted": rows_accepted, "rows_rejected": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "accepted_whitespace_tokens": tokens, "review_status": "auto_accept",
        "review_method": "batch_policy_v1", "training_authorized": False,
        "note": "Unresolved-language and PII-flagged rows are quarantined. Exact duplicates are removed. This source contains persona descriptions focused partly on arts/culture/theatre; it is a narrow persona slice, not a population sample."
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_seen": rows_seen, "rows_auto_accepted": rows_accepted,
        "rows_rejected": sum(rejected.values()), "whitespace_tokens": tokens,
        "training_authorized": False}))

if __name__ == "__main__":
    main()
