#!/usr/bin/env python3
"""Conservatively batch-admit the pinned Spanish mPersonas dialogue split."""
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

SOURCE = "BSC-LT/m-personas"
REVISION = "6fab205a86a9244c90a2782e28055027f45ddd03"
LICENSE = "Apache-2.0"
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")

def normalize(value):
    return " ".join(str(value).split())

def content_flags(text):
    flags = []
    if EMAIL.search(text): flags.append("possible_email")
    if NIF.search(text): flags.append("possible_nif")
    if PHONE.search(text): flags.append("possible_phone")
    return flags

def normalize_row(row):
    if str(row.get("lang", "")).lower() != "es":
        return None, "not_spanish_split"
    persona = normalize(row.get("persona", ""))
    prompt = normalize(row.get("prompt", ""))
    messages = row.get("messages")
    if len(persona) < 80:
        return None, "persona_missing_or_too_short"
    if not isinstance(messages, list) or not 2 <= len(messages) <= 10:
        return None, "invalid_message_count"
    normalized_messages = []
    expected = "user"
    for message in messages:
        if not isinstance(message, dict):
            return None, "invalid_message_object"
        role = str(message.get("role", "")).lower()
        content = normalize(message.get("content", ""))
        if role != expected or not content:
            return None, "invalid_role_sequence_or_empty_turn"
        normalized_messages.append({"role": role, "content": content})
        expected = "assistant" if expected == "user" else "user"
    if normalized_messages[0]["role"] != "user" or normalized_messages[-1]["role"] != "assistant":
        return None, "invalid_conversation_boundaries"
    # source_text is deliberately never read: it contains original FineWeb2 seed text.
    scanned = "\n".join([persona, prompt] + [m["content"] for m in normalized_messages])
    flags = content_flags(scanned)
    if flags:
        return None, "privacy_flag:" + ",".join(flags)
    conversation_words = sum(len(m["content"].split()) for m in normalized_messages)
    if conversation_words < 30:
        return None, "conversation_too_short"
    parts = [f"Persona:\n{persona}"]
    if prompt:
        parts.append(f"Prompt:\n{prompt}")
    parts.append("Conversation:")
    for message in normalized_messages:
        label = "Usuario" if message["role"] == "user" else "Asistente"
        parts.append(f"{label}:\n{message['content']}")
    text = "\n\n".join(parts)
    digest = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
    output = {
        "text": text, "text_sha256": digest, "task_type": "persona_dialogue",
        "language": "es", "source": SOURCE, "source_revision": REVISION,
        "source_row": str(row.get("id", "")), "license": LICENSE,
        "review_status": "auto_accept", "review_method": "batch_policy_v1",
        "rights_status": "verified", "privacy_status": "clear", "quality_status": "approved",
        "quality_checks": ["pinned_spanish_split", "persona_min_80_chars",
            "alternating_user_assistant_roles", "2_to_10_nonempty_messages",
            "conversation_min_30_words", "email_nif_phone_scan_clear", "source_text_excluded"]
    }
    return output, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-jsonl", required=True, help="Frozen v3.1 JSONL; verified by workflow")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(SOURCE, revision=REVISION)
    card = info.card_data.to_dict() if info.card_data else {}
    if info.sha != REVISION:
        raise SystemExit(f"revision mismatch: expected {REVISION}, got {info.sha}")
    if str(card.get("license", "")).lower() != "apache-2.0":
        raise SystemExit(f"pinned dataset license is not Apache-2.0: {card.get('license')!r}")
    if not Path(args.base_jsonl).is_file():
        raise SystemExit("frozen v3.1 base JSONL is missing")
    seen = set()
    rows_seen = rows_accepted = tokens = 0
    rejected = Counter()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        stream = load_dataset(SOURCE, name=None, split="es", streaming=True, revision=REVISION)
        for index, row in enumerate(stream):
            if args.max_rows and index >= args.max_rows: break
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
        "source_revision": REVISION, "split": "es",
        "license_verified_at_pinned_revision": LICENSE, "rows_seen": rows_seen,
        "rows_auto_accepted": rows_accepted, "rows_rejected": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "accepted_whitespace_tokens": tokens, "source_text_included": False,
        "raw_seed_text_exported": False, "review_status": "auto_accept",
        "review_method": "batch_policy_v1", "training_authorized": False,
        "note": "Policy admission is source- and revision-specific. Regex flags are conservatively quarantined; no raw source_text is included. This report does not authorize training."
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows_seen": rows_seen, "rows_auto_accepted": rows_accepted,
        "rows_rejected": sum(rejected.values()), "whitespace_tokens": tokens,
        "training_authorized": False}))

if __name__ == "__main__":
    main()
