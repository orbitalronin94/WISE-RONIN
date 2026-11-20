#!/usr/bin/env python3
"""Build verbatim-question manual-authoring queues from licensed Spanish QA sources.

No answer generation occurs here. The only output is a queue containing the
source question, context and reference answer for a human author to rewrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

SQAC_REVISION = "9a6e701a8487c35c1805f9f29eac247875a6be05"
SQAC_TRAIN_URL = (
    "https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/"
    f"{SQAC_REVISION}/train.json"
)
SQAC_TRAIN_SHA256 = "1d5c76176646e2ae7bdcd8b5ec6f18349102a9363aa25ad7d0e48262d7480d43"
SQAC_EXPECTED_TRAIN_ROWS = 15036


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "WISE-RONIN-dataset-builder/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer_arrays(answers: Any) -> tuple[list[str], list[int]]:
    """Normalize SQAC's SQuAD-like dict shape without altering answer text."""
    if not isinstance(answers, dict):
        raise ValueError("SQAC answers must be an object with text and answer_start arrays")
    texts = answers.get("text")
    starts = answers.get("answer_start")
    if not isinstance(texts, list) or not isinstance(starts, list) or len(texts) != len(starts):
        raise ValueError("SQAC answers.text and answers.answer_start must be aligned arrays")
    return texts, starts


def iter_sqac_rows(payload: dict[str, Any], split: str):
    if split != "train":
        raise ValueError("Only SQAC train is eligible for the authoring/training pipeline")
    articles = payload.get("data")
    if not isinstance(articles, list):
        raise ValueError("SQAC file must contain a top-level data array")
    for article in articles:
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                texts, starts = _answer_arrays(qa.get("answers"))
                if not texts:
                    continue
                for answer, start in zip(texts, starts):
                    if not isinstance(answer, str) or not isinstance(start, int):
                        raise ValueError(f"Invalid answer annotation in SQAC row {qa.get('id')}")
                    if context[start : start + len(answer)] != answer:
                        raise ValueError(f"Answer offset mismatch in SQAC row {qa.get('id')}")
                question = qa.get("question", "")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"Empty question in SQAC row {qa.get('id')}")
                yield {
                    "source_dataset": "PlanTL-GOB-ES/SQAC",
                    "source_revision": SQAC_REVISION,
                    "source_split": "train",
                    "source_id": str(qa.get("id", "")),
                    "title_verbatim": title,
                    "question_verbatim": question,
                    "source_question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                    "context_verbatim": context,
                    "reference_answers_verbatim": texts,
                    "answer_starts": starts,
                    "ronin_answer_manual": "",
                    "author_status": "pending_manual_authorship",
                }


def write_jsonl(rows, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the pinned SQAC train.json")
    parser.add_argument("--input", type=Path, help="Use an already downloaded SQAC train.json")
    parser.add_argument("--output", type=Path, default=Path("build/sqac_train_authoring_queue.jsonl"))
    parser.add_argument("--expected-rows", type=int, default=SQAC_EXPECTED_TRAIN_ROWS)
    args = parser.parse_args()

    source = args.input
    if args.download:
        source = Path("build/source/sqac_train.json")
        download(SQAC_TRAIN_URL, source)
        actual_sha = sha256(source)
        if actual_sha != SQAC_TRAIN_SHA256:
            raise SystemExit(f"SQAC SHA256 mismatch: expected {SQAC_TRAIN_SHA256}, got {actual_sha}")
    if source is None:
        parser.error("pass --download or --input")

    payload = json.loads(source.read_text(encoding="utf-8"))
    count = write_jsonl(iter_sqac_rows(payload, "train"), args.output)
    if args.expected_rows and count != args.expected_rows:
        raise SystemExit(f"SQAC row count mismatch: expected {args.expected_rows}, got {count}")
    print(json.dumps({"source": "PlanTL-GOB-ES/SQAC", "split": "train", "rows": count,
                      "output": str(args.output), "sha256": sha256(source)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
