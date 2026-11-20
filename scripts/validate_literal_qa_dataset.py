#!/usr/bin/env python3
"""Validate manual answers against a pinned source queue and emit train-only SFT."""
from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path

MANUAL_REQUIRED = {
    "source_dataset", "source_revision", "source_split", "source_id",
    "title_verbatim", "question_verbatim", "source_question_sha256",
    "ronin_answer_manual", "author_status",
}
SOURCE_REQUIRED = {
    "source_dataset", "source_revision", "source_split", "source_id",
    "title_verbatim", "question_verbatim", "source_question_sha256",
    "context_verbatim", "reference_answers_verbatim", "answer_starts",
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


def validate(rows, source_rows):
    source_by_id = {}
    for source in source_rows:
        missing = SOURCE_REQUIRED - source.keys()
        if missing:
            raise ValueError(f"source row missing fields: {sorted(missing)}")
        source_id = source.get("source_id")
        if not source_id or source_id in source_by_id:
            raise ValueError(f"authoring queue has missing or duplicate source_id: {source_id}")
        source_by_id[source_id] = source

    seen = set()
    accepted = []
    for index, row in enumerate(rows, 1):
        missing = MANUAL_REQUIRED - row.keys()
        if missing:
            raise ValueError(f"manual row {index}: missing fields {sorted(missing)}")
        if row["source_split"] != "train":
            raise ValueError(f"manual row {index}: non-train split is forbidden")
        if row["author_status"] != "manually_reviewed":
            raise ValueError(f"manual row {index}: answer has not been manually reviewed")
        if not all(isinstance(row[k], str) and row[k].strip() for k in
                   ("source_dataset", "source_revision", "source_id", "question_verbatim", "ronin_answer_manual")):
            raise ValueError(f"manual row {index}: required text field is empty or not a string")

        source = source_by_id.get(row["source_id"])
        if source is None:
            raise ValueError(f"manual row {index}: source ID is absent from pinned train queue")
        for field in ("source_dataset", "source_revision", "source_split", "title_verbatim",
                      "question_verbatim", "source_question_sha256"):
            if row[field] != source[field]:
                raise ValueError(f"manual row {index}: {field} differs from original source")
        actual_hash = hashlib.sha256(row["question_verbatim"].encode("utf-8")).hexdigest()
        if row["source_question_sha256"] != actual_hash:
            raise ValueError(f"manual row {index}: source question hash mismatch")

        answers = source["reference_answers_verbatim"]
        starts = source["answer_starts"]
        context = source["context_verbatim"]
        if not isinstance(answers, list) or not answers or not isinstance(starts, list) or len(answers) != len(starts):
            raise ValueError(f"source row {index}: reference answers and offsets are not aligned")
        for answer, start in zip(answers, starts):
            if not isinstance(answer, str) or not isinstance(start, int):
                raise ValueError(f"source row {index}: invalid reference answer or offset")
            if context[start:start + len(answer)] != answer:
                raise ValueError(f"source row {index}: answer does not match source context offset")

        key = (row["source_dataset"], row["source_revision"], row["source_id"])
        if key in seen:
            raise ValueError(f"manual row {index}: duplicate source identity {key}")
        seen.add(key)
        if norm(row["ronin_answer_manual"]) in {norm(x) for x in answers}:
            raise ValueError(f"manual row {index}: Ronin answer is identical to a reference answer")
        accepted.append({**source, **row})
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-queue", type=Path, required=True,
                        help="Pinned authoring queue; required to verify questions and recover evidence")
    parser.add_argument("--sft-output", type=Path)
    args = parser.parse_args()
    rows = validate(read_jsonl(args.input), source_rows=read_jsonl(args.source_queue))
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
