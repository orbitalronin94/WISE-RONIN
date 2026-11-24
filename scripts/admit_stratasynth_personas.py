#!/usr/bin/env python3
"""Batch-admit only non-sensitive narrative fields from synthetic Spanish StrataSynth personas."""
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

SOURCE = "StrataSynth/stratasynth-personas-spain"
REVISION = "aa06f584e2932f28ead13ead309711573605ca87"
LICENSE = "CC-BY-4.0"
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")

def normalize(value):
    return " ".join(str(value or "").split())

def normalize_row(row, index):
    if row.get("country") != "ES" or row.get("language") != "es":
        return None, "not_spanish_spain"
    persona = normalize(row.get("persona"))
    routine = normalize(row.get("daily_routine"))
    style = normalize(row.get("speaking_style"))
    if len(persona) < 100:
        return None, "persona_missing_or_too_short"
    if len(routine) < 100:
        return None, "routine_missing_or_too_short"
    if len(style) < 80:
        return None, "speaking_style_missing_or_too_short"
    scanned = "\n".join((persona, routine, style))
    if EMAIL.search(scanned):
        return None, "privacy_flag:possible_email"
    if NIF.search(scanned):
        return None, "privacy_flag:possible_nif"
    if PHONE.search(scanned):
        return None, "privacy_flag:possible_phone"
    text = (
        "Persona sintética:\n" + persona +
        "\n\nRutina diaria:\n" + routine +
        "\n\nEstilo de habla:\n" + style
    )
    digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
    return {
        "text": text, "text_sha256": digest, "task_type": "synthetic_persona_voice",
        "language": "es", "source": SOURCE, "source_revision": REVISION,
        "source_row": str(row.get("id", index)), "license": LICENSE,
        "attribution": "StrataSynth Personas Spain v2 — CC BY 4.0",
        "review_status": "auto_accept", "review_method": "batch_policy_v1",
        "rights_status": "verified", "privacy_status": "clear", "quality_status": "approved",
        "metadata": {"country": "ES", "language": "es"},
        "excluded_source_fields": ["name", "age", "gender", "region", "education",
            "occupation", "income_level", "household_type", "marital_status",
            "children", "housing", "employment_status", "consumer", "personality",
            "hobbies_and_interests", "skills", "goals"],
        "quality_checks": ["pinned_train_split", "country_ES_language_es",
            "persona_min_100_chars", "routine_min_100_chars", "speaking_style_min_80_chars",
            "email_nif_phone_scan_clear", "demographic_and_psychometric_fields_excluded"]
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
    if str(card.get("license", "")).lower() != "cc-by-4.0":
        raise SystemExit(f"pinned dataset license is not CC-BY-4.0: {card.get('license')!r}")
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
        "rows_auto_accepted": rows_accepted, "rows_rejected": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "accepted_whitespace_tokens": tokens, "review_status": "auto_accept",
        "review_method": "batch_policy_v1", "training_authorized": False,
        "excluded_fields": ["demographics", "consumer", "personality", "hobbies_and_interests", "skills", "goals"],
        "note": "All people are synthetic. Only persona, daily_routine and speaking_style are admitted; demographic, consumer and psychometric fields are excluded. Attribute StrataSynth under CC BY 4.0."
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_seen": rows_seen, "rows_auto_accepted": rows_accepted,
        "rows_rejected": sum(rejected.values()), "whitespace_tokens": tokens,
        "training_authorized": False}))

if __name__ == "__main__":
    main()
