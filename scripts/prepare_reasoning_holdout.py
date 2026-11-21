#!/usr/bin/env python3
"""Pin and prepare a clean 500-question GSM8K test holdout for pilot evaluation only."""
from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from data.audit_holdout import audit_rows, read_jsonl

def norm(value):
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text).split())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="data/pilot_train_19k.jsonl")
    p.add_argument("--output", default="artifacts/reasoning_gsm8k_500.jsonl")
    p.add_argument("--manifest", default="artifacts/reasoning_gsm8k_500.manifest.json")
    p.add_argument("--count", type=int, default=500)
    args = p.parse_args()
    train_rows = read_jsonl(args.train_data)
    train_texts = [norm(str(r.get("prompt", "")) + " " + str(r.get("completion", ""))) for r in train_rows]
    # Pin the official Hugging Face dataset commit; never follow a moving main ref.
    revision = "740312add88f781978c0658806c59bc2815b9866"
    ds = load_dataset("openai/gsm8k", "main", split="test", revision=revision)
    selected, skipped, seen_questions = [], 0, set()
    for row in ds:
        question = str(row["question"]).strip()
        answer = str(row["answer"])
        if "####" not in answer:
            continue
        expected = answer.rsplit("####", 1)[1].strip().replace(",", "")
        qn = norm(question)
        if (not qn or qn in seen_questions
                or any(qn == text or (len(qn) >= 28 and qn in text) for text in train_texts)):
            skipped += 1
            continue
        seen_questions.add(qn)
        selected.append({
            "id": f"GSM8K-TEST-{len(selected)+1:03d}",
            "categoria": "reasoning_gsm8k",
            "user": question,
            "hecho_esperado": expected,
            "reasoning_reference": answer.rsplit("####", 1)[0].strip(),
            "dataset": "openai/gsm8k",
            "dataset_revision": revision,
            "split": "test",
        })
        if len(selected) >= args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"Could only prepare {len(selected)} clean test questions; requested {args.count}")
    audit = audit_rows(train_rows, selected)
    if not audit["lexical_gate_pass"]:
        raise RuntimeError("GSM8K test slice overlaps pilot training data")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "dataset": "openai/gsm8k", "revision": revision, "split": "test",
        "rows": len(selected), "unique_questions": len(seen_questions),
        "sha256": digest, "skipped_training_overlap": skipped,
        "training_corpus_sha256": hashlib.sha256(Path(args.train_data).read_bytes()).hexdigest(),
        "audit": {
            "exact_matches": audit["exact_match_count"],
            "long_substring_matches": audit["long_user_substring_match_count"],
            "lexical_gate_pass": audit["lexical_gate_pass"],
        },
        "use": "evaluation_only_never_training",
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
