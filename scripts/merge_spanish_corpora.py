#!/usr/bin/env python3
"""Merge the current Spanish instruction corpus with canonical Spanish facts/math
extracted from the archived WISE-RONIN expanded corpus.

The legacy source has many paraphrase/style variants per fact_id. We deliberately
keep one deterministic representative per independent fact_id and use the
canonical `hecho` field as the completion, not the repetitive stylistic answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import langid


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def write_jsonl(path: Path, rows):
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def copy_jsonl(source: Path, destination: Path):
    """Preserve evaluation files byte-for-byte, not merely record-for-record."""
    payload = source.read_bytes()
    destination.write_bytes(payload)
    count = sum(1 for line in payload.splitlines() if line.strip())
    return count, hashlib.sha256(payload).hexdigest()


def normalize(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def is_spanish(text: str) -> bool:
    text = str(text or "").strip()
    if len(text) < 12:
        return False
    try:
        return langid.classify(text[:5000])[0] == "es"
    except Exception:
        return False


def extract_legacy(path: Path):
    """Return one Spanish canonical example per fact_id, plus audit counters."""
    groups = {}
    counts = Counter()
    for row in read_jsonl(path):
        counts["legacy_rows_seen"] += 1
        question = str(row.get("user", "")).strip()
        answer = str(row.get("hecho", "")).strip()
        fact_id = str(row.get("fact_id", "")).strip()
        if not question or not answer or not fact_id:
            counts["legacy_rows_missing_fields"] += 1
            continue
        if not is_spanish(question):
            counts["legacy_rows_non_spanish"] += 1
            continue
        counts["legacy_rows_spanish"] += 1
        candidate = {
            "fact_id": fact_id,
            "variant_id": int(row.get("variant_id", 0) or 0),
            "system": str(row.get("system", "")).strip(),
            "prompt": question,
            "completion": answer.rstrip(".") + ".",
            "legacy_type": str(row.get("tipo", "legacy_unknown")),
        }
        previous = groups.get(fact_id)
        # Prefer the lowest variant id, making extraction deterministic and
        # preventing dozens of stylistic paraphrases from dominating training.
        if previous is None or candidate["variant_id"] < previous["variant_id"]:
            groups[fact_id] = candidate
    counts["legacy_fact_groups"] = len(groups)
    return groups, counts


def merge(base_dir: Path, legacy_path: Path, output_dir: Path):
    base_train = base_dir / "train.jsonl"
    base_validation = base_dir / "validation.jsonl"
    base_holdout = base_dir / "holdout_multiple_choice.jsonl"
    manifest_path = base_dir / "manifest.json"
    for required in (base_train, base_validation, base_holdout, manifest_path, legacy_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = list(read_jsonl(base_train))
    validation = list(read_jsonl(base_validation))
    holdout = list(read_jsonl(base_holdout))

    # Never move validation or holdout examples into training.
    protected_prompts = {
        normalize(row.get("prompt", ""))
        for row in validation + holdout
        if row.get("prompt")
    }
    seen = {
        (normalize(row.get("prompt", "")), normalize(row.get("completion", "")))
        for row in train
    }
    seen_prompts = {prompt for prompt, _ in seen if prompt}

    groups, counts = extract_legacy(legacy_path)
    added = []
    skipped_protected = 0
    skipped_duplicate = 0
    for fact_id in sorted(groups):
        row = groups[fact_id]
        system = row["system"]
        prompt = (
            f"Sistema: {system}\nUsuario: {row['prompt']}\nAsistente:"
            if system else f"Usuario: {row['prompt']}\nAsistente:"
        )
        key = (normalize(prompt), normalize(row["completion"]))
        if key[0] in protected_prompts:
            skipped_protected += 1
            continue
        if key in seen or key[0] in seen_prompts:
            skipped_duplicate += 1
            continue
        added.append({
            "id": hashlib.sha256(f"legacy|{fact_id}".encode()).hexdigest()[:20],
            "prompt": prompt,
            "completion": row["completion"],
            "task": "legacy_" + row["legacy_type"],
            "language": "es",
            "source": "WISE-RONIN archived expanded corpus",
            "source_id": fact_id,
            "license": "user-generated",
            "legacy_fact_id": fact_id,
            "legacy_variant_id": row["variant_id"],
        })
        seen.add(key)
        seen_prompts.add(key[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    total_train = train + added
    train_count, train_hash = write_jsonl(output_dir / "train.jsonl", total_train)
    val_count, val_hash = copy_jsonl(base_validation, output_dir / "validation.jsonl")
    holdout_count, holdout_hash = copy_jsonl(base_holdout, output_dir / "holdout_multiple_choice.jsonl")

    source_counts = Counter(row.get("source", "unknown") for row in total_train)
    task_counts = Counter(row.get("task", "unknown") for row in total_train)
    manifest = dict(base_manifest)
    manifest.update({
        "name": "WISE-RONIN merged Spanish corpus v1",
        "legacy_extraction": {
            **dict(counts),
            "legacy_examples_added": len(added),
            "legacy_skipped_protected_validation_or_holdout": skipped_protected,
            "legacy_skipped_duplicate_prompt": skipped_duplicate,
            "policy": "Spanish prompt only; one lowest-variant representative per fact_id; canonical hecho used as completion; legacy data added to train only.",
            "source_file": str(legacy_path),
        },
        "rows_raw": int(base_manifest.get("rows_raw", 0)) + counts["legacy_rows_spanish"],
        "rows_deduplicated": train_count + val_count,
        "train_rows": train_count,
        "validation_rows": val_count,
        "holdout_rows": holdout_count,
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "sha256": {
            "train": train_hash,
            "validation": val_hash,
            "holdout_multiple_choice": holdout_hash,
        },
        "notes": list(base_manifest.get("notes", [])) + [
            "Legacy expanded corpus is synthetic: row counts are not independent facts.",
            "Only Spanish legacy prompts were considered; one example per fact_id was retained.",
            "Legacy style-generated assistant text was replaced with the canonical hecho field.",
            "Legacy examples were added to train only; new validation and holdout were preserved unchanged.",
        ],
    })
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "train_rows": train_count,
        "validation_rows": val_count,
        "holdout_rows": holdout_count,
        "legacy_examples_added": len(added),
        "legacy_audit": dict(counts),
        "skipped_protected": skipped_protected,
        "skipped_duplicate": skipped_duplicate,
        "source_counts": dict(source_counts),
        "task_counts": dict(task_counts),
        "sha256": manifest["sha256"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = merge(Path(args.base_dir), Path(args.legacy), Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
