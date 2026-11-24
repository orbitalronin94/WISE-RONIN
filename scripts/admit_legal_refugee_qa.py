#!/usr/bin/env python3
"""Conservative batch admission for pinned context-grounded Spanish refugee-law QA."""
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

SOURCE = "somosnlp/instruct-legal-refugiados-es"
REVISION = "c5e0373049d50c382a75ee0677e3f6b1b4b63dc2"
LICENSE = "Apache-2.0"
NEAR_DUPLICATE_QUARANTINE = frozenset({124,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,320,397,400,407,416,488,564,570,639,645,653,667,724,802,848,917,950,993,1072,1158,1183,1207,1217,1283,1298,1328,1473,1534,1656,2011,2029,2080,2082,2111,2146,2173,2199,2200,2201,2202,2219,2311,2410,2569,2570,2571,2572,2578,2579,2580,2582,2613,2627,2698,2787,2828,2958,3029,3047,3203,3204,3205,3206,3366,3367,3368,3369,3398,3457,3525,3526,3527,3528,3640,3641,3642,3643,3934,4010,4207,4210,4246,4247,4248,4249,4289,4392,4393,4394,4395,4452,4453,4473,4474,4475,4476,4487,4488,4489,4490,4554,4599,4620,4621,4921,4947,5001,5009,5101,5210,5240,5245,5271,5554,5748,5775,5836,5902,5945,5991,6020,6028,6042,6058,6084,6231,6232,6236,6237,6238,6496,6667,6668,6670,6723,6724,6725,6726,6753,6754,6755,6756,6757,6758,6759,6760,6846,6847,6852,6879,6943,7046,7549,7621,7630,7874,8158,8198,8296,8413,8652,8706,8722,8998,9098,9249,9288,9298,9301,9302,9303,9326,9347,9414,9428})
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
# Require phone-like grouping or a +34 prefix; bare legal references and BOE IDs are not phones.
PHONE_FORMATTED = re.compile(r"(?<!\w)(?:\+34[\s.-]?)?[6789]\d{2}[\s.-]\d{2,3}[\s.-]\d{2,3}(?!\w)")
PHONE_CONTEXT = re.compile(r"(?:tel[eé]fono|m[oó]vil|contacto|llamar|fax)\D{0,30}[6789]\d{8}", re.I)

def normalize(value):
    return " ".join(str(value or "").split())

def normalize_row(row, index):
    if index in NEAR_DUPLICATE_QUARANTINE:
        return None, "near_duplicate_candidate_quarantined"
    if str(row.get("lang", "")).lower() != "es":
        return None, "language_not_es"
    question = normalize(row.get("question"))
    context = normalize(row.get("context"))
    answer = normalize(row.get("answer"))
    if len(question) < 20:
        return None, "question_missing_or_too_short"
    if len(context) < 40:
        return None, "context_missing_or_too_short"
    if len(answer) < 60 or len(answer.split()) < 25:
        return None, "answer_missing_or_too_short"
    scanned = "\n".join((question, context, answer))
    if EMAIL.search(scanned):
        return None, "privacy_flag:possible_email"
    if NIF.search(scanned):
        return None, "privacy_flag:possible_nif"
    if PHONE_FORMATTED.search(scanned) or PHONE_CONTEXT.search(scanned):
        return None, "privacy_flag:possible_phone"
    if sum(c.isalpha() for c in scanned) / max(1, len(scanned)) < 0.45:
        return None, "low_text_density"
    text = (
        "Pregunta:\n" + question +
        "\n\nContexto documental:\n" + context +
        "\n\nRespuesta generada:\n" + answer
    )
    normalized = " ".join(text.split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    metadata = {
        "jurisdiction_origin": normalize(row.get("pais_origen")),
        "record_type": normalize(row.get("registro")),
        "period_label": normalize(row.get("periodo")),
        "domain": normalize(row.get("dominio")),
        "task": normalize(row.get("tarea")),
        "answer_origin": "machine_generated_from_context",
        "legal_validity": "not_independently_verified"
    }
    return {
        "text": text, "text_sha256": digest, "task_type": "contextual_legal_qa",
        "language": "es", "source": SOURCE, "source_revision": REVISION,
        "source_row": str(index), "license": LICENSE,
        "review_status": "auto_accept", "review_method": "batch_policy_v1",
        "rights_status": "verified", "privacy_status": "clear", "quality_status": "approved",
        "metadata": metadata,
        "quality_checks": ["pinned_train_split", "question_context_answer_present",
            "answer_min_25_words", "near_duplicate_candidates_quarantined",
            "email_nif_formatted_phone_scan_clear", "legal_validity_not_asserted"]
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
        "note": "Near-duplicate candidate rows from the pinned audit are quarantined. Phone detection requires formatting or explicit phone context to avoid treating statutory references as telephone numbers. Answers are machine-generated and legal validity is not independently verified."
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_seen": rows_seen, "rows_auto_accepted": rows_accepted,
        "rows_rejected": sum(rejected.values()), "whitespace_tokens": tokens,
        "training_authorized": False}))

if __name__ == "__main__":
    main()
