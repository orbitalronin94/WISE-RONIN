#!/usr/bin/env python3
"""Build a provenance-preserving WISE-RONIN master SFT corpus."""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path

QA_REVIEW = {
    "¿Qué metal es segundo mejor conductor?": "La respuesta aportada dice plata; la plata es el mejor conductor eléctrico entre los metales, mientras que el cobre suele citarse como segundo. Revisión humana necesaria.",
    "¿Cuál es la capital de Israel?": "Pregunta geopolíticamente disputada; la respuesta depende del reconocimiento y del marco institucional. Reescribir con contexto o excluir.",
}

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()

def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"{path}:{n}: JSON inválido: {e}")
            yield n, row

def standardize(row, source, line_no):
    if "user" in row and "assistant" in row:
        user, assistant = row["user"], row["assistant"]
    elif "prompt" in row and "completion" in row:
        user, assistant = row["prompt"], row["completion"]
    else:
        raise ValueError(f"{source}:{line_no}: faltan user/assistant o prompt/completion")
    if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
        raise ValueError(f"{source}:{line_no}: prompt/respuesta vacíos o no textuales")
    out = dict(row)
    out["user"], out["assistant"] = user.strip(), assistant.strip()
    out["source_dataset"], out["source_line"] = source, line_no
    if not out.get("tipo"):
        out["tipo"] = "qa_ronin_manual" if source == "qa_ronin" else "legacy_untyped"
    return out

def write_rows(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", nargs=3, required=True)
    ap.add_argument("--balanced", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--output", default="data/merged/wise_ronin_master.jsonl")
    ap.add_argument("--report", default="data/merged/wise_ronin_master_report.json")
    ap.add_argument("--variants", default="data/merged/wise_ronin_prompt_variants.jsonl")
    ap.add_argument("--review", default="data/merged/wise_ronin_qa_manual_review.jsonl")
    args = ap.parse_args()

    sources = [
        ("legacy_part1", Path(args.legacy[0])),
        ("legacy_part2", Path(args.legacy[1])),
        ("legacy_part3", Path(args.legacy[2])),
        ("balanced_80k", Path(args.balanced)),
        ("qa_ronin", Path(args.qa)),
    ]
    accepted, review = [], []
    source_counts, type_counts = collections.Counter(), collections.Counter()
    seen_pairs = set()
    prompt_groups = collections.defaultdict(list)
    exact_duplicates = 0

    for source, path in sources:
        for line_no, row in load_jsonl(path):
            item = standardize(row, source, line_no)
            source_counts[source] += 1
            type_counts[item["tipo"]] += 1

            if source == "qa_ronin" and item["user"] in QA_REVIEW:
                review.append({**item, "review_reason": QA_REVIEW[item["user"]]})
                continue

            pair_key = (normalize(item["user"]), normalize(item["assistant"]))
            if pair_key in seen_pairs:
                exact_duplicates += 1
                continue
            seen_pairs.add(pair_key)
            accepted.append(item)
            prompt_groups[pair_key[0]].append(item)

    variants = []
    rows_in_multi_completion_prompts = 0
    for prompt_key, rows in prompt_groups.items():
        completions = {normalize(row["assistant"]) for row in rows}
        if len(completions) > 1:
            rows_in_multi_completion_prompts += len(rows)
            by_source = collections.Counter(row["source_dataset"] for row in rows)
            variants.append({
                "normalized_prompt": prompt_key,
                "rows_preserved": len(rows),
                "distinct_completions": len(completions),
                "source_counts": dict(by_source),
                "completion_examples": list(dict.fromkeys(row["assistant"] for row in rows))[:5],
                "policy": "All variants retained; reviewed as intentional style/answer variation, not auto-discarded."
            })

    write_rows(args.output, accepted)
    write_rows(args.variants, variants)
    write_rows(args.review, review)
    report = {
        "master_rows": len(accepted),
        "input_rows_by_source": dict(source_counts),
        "input_rows_total": sum(source_counts.values()),
        "rows_by_tipo_in_inputs": dict(type_counts),
        "exact_prompt_answer_duplicates_removed": exact_duplicates,
        "prompts_with_multiple_completions_preserved": len(variants),
        "rows_in_multi_completion_prompts_preserved": rows_in_multi_completion_prompts,
        "qa_rows_quarantined_for_manual_review": len(review),
        "qa_manual_review_prompts": QA_REVIEW,
        "policy": "Deduplicate only identical normalized prompt+completion pairs. Preserve multiple completions for the same prompt and report them for inspection. Quarantine only the two explicitly flagged QA records. Original source files remain untouched. No training or holdout modification."
    }
    p = Path(args.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
