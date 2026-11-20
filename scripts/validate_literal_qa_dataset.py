#!/usr/bin/env python3
"""Validate manually authored Ronin answers and emit a train-only SFT export."""
from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path

REQUIRED = {
    "source_dataset", "source_revision", "source_split", "source_id",
    "title_verbatim", "question_verbatim", "source_question_sha256", "context_verbatim", "reference_answers_verbatim",
    "answer_starts", "ronin_answer_manual", "author_status",
}


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def validate(rows, source_rows=None):
    source_by_id = {}
    if source_rows is not None:
        for source in source_rows:
            source_id = source.get("source_id")
            if not source_id or source_id in source_by_id:
                raise ValueError(f"authoring queue has missing or duplicate source_id: {source_id}")
            source_by_id[source_id] = source
    seen = set()
    accepted = []
    for index, row in enumerate(rows, 1):
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"row {index}: missing fields {sorted(missing)}")
        if row["source_split"] != "train":
            raise ValueError(f"row {index}: non-train split is forbidden in manual training data")
        if row["author_status"] != "manually_reviewed":
            raise ValueError(f"row {index}: answer has not been manually reviewed")
        if not all(isinstance(row[k], str) and row[k].strip() for k in
                   ("source_dataset", "source_id", "question_verbatim", "context_verbatim", "ronin_answer_manual")):
            raise ValueError(f"row {index}: required text field is empty or not a string")
        if not isinstance(row["reference_answers_verbatim"], list) or not row["reference_answers_verbatim"]:
            raise ValueError(f"row {index}: at least one source reference answer is required")
        if not isinstance(row["answer_starts"], list) or len(row["answer_starts"]) != len(row["reference_answers_verbatim"]):
            raise ValueError(f"row {index}: reference answer offsets are not aligned")
        for answer, start in zip(row["reference_answers_verbatim"], row["answer_starts"]):
            if not isinstance(answer, str) or not isinstance(start, int):
                raise ValueError(f"row {index}: invalid reference answer or offset")
            if row["context_verbatim"][start:start + len(answer)] != answer:
                raise ValueError(f"row {index}: source answer does not match context offset")
        actual_question_hash = hashlib.sha256(row["question_verbatim"].encode("utf-8")).hexdigest()
        if row["source_question_sha256"] != actual_question_hash:
            raise ValueError(f"row {index}: source question hash mismatch")
        if source_rows is not None:
            source = source_by_id.get(row["source_id"])
            if source is None:
                raise ValueError(f"row {index}: source ID is absent from the pinned train queue")
            for field in ("source_dataset", "source_revision", "source_split", "question_verbatim",
                          "title_verbatim", "source_question_sha256", "context_verbatim", "reference_answers_verbatim", "answer_starts"):
                if row[field] != source[field]:
                    raise ValueError(f"row {index}: {field} differs from the original source row")
        key = (row["source_dataset"], row["source_revision"], row["source_id"])
        if key in seen:
            raise ValueError(f"row {index}: duplicate source identity {key}")
        seen.add(key)
        # Do not silently pass off copied reference text as a manual rewrite.
        if norm(row["ronin_answer_manual"]) in {norm(x) for x in row["reference_answers_verbatim"]}:
            raise ValueError(f"row {index}: Ronin answer is identical to a reference answer")
        accepted.append(row)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path)
    parser.add_argument("--source-queue", type=Path,
                         help="Optional generated queue; verifies question/context/reference are unchanged")
    args = parser.parse_args()
    source_rows = read_jsonl(args.source_queue) if args.source_queue else None
    rows = validate(read_jsonl(args.input), source_rows=source_rows)
    if args.sft_output:
        args.sft_output.parent.mkdir(parents=True, exist_ok=True)
        with args.sft_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                item = {
                    "system": "Eres WISE-RONIN. Responde con precisión y concisión. El dato manda; el estilo nunca debe alterar el significado.",
                    "user": row["question_verbatim"],
                    "assistant": row["ronin_answer_manual"],
                    "tipo": "external_qa_manual",
                    "source_dataset": row["source_dataset"],
                    "source_id": row["source_id"],
                    "source_revision": row["source_revision"],
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"validated_manual_rows": len(rows), "training_split_only": True,
                      "sft_output": str(args.sft_output) if args.sft_output else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
