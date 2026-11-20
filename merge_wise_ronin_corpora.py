#!/usr/bin/env python3
"""Build a provenance-preserving WISE-RONIN master SFT corpus."""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path

QA_REVIEW = {
    "¿Qué metal es segundo mejor conductor?": "La respuesta aportada dice plata; la plata es el mejor conductor eléctrico entre los metales, mientras que el cobre suele citarse como segundo. Revisión humana necesaria.",
    "¿Cuál es la capital de Israel?": "Pregunta geopolíticamente disputada; la respuesta depende del reconocimiento y del marco institucional. Reescribir con contexto o excluir.",
}

def norm_prompt(s: str) -> str:
    return re.sub(r"\\s+", " ", s).strip().casefold()

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", nargs=3, required=True)
    ap.add_argument("--balanced", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--output", default="data/merged/wise_ronin_master.jsonl")
    ap.add_argument("--report", default="data/merged/wise_ronin_master_report.json")
    ap.add_argument("--conflicts", default="data/merged/wise_ronin_prompt_conflicts.jsonl")
    ap.add_argument("--review", default="data/merged/wise_ronin_qa_manual_review.jsonl")
    args = ap.parse_args()

    sources = [
        ("legacy_part1", Path(args.legacy[0])),
        ("legacy_part2", Path(args.legacy[1])),
        ("legacy_part3", Path(args.legacy[2])),
        ("balanced_80k", Path(args.balanced)),
        ("qa_ronin", Path(args.qa)),
    ]
    grouped = collections.defaultdict(list)
    source_counts, type_counts = collections.Counter(), collections.Counter()
    for source, path in sources:
        for line_no, row in load_jsonl(path):
            item = standardize(row, source, line_no)
            source_counts[source] += 1
            type_counts[item["tipo"]] += 1
            grouped[norm_prompt(item["user"])].append(item)

    accepted, conflicts, review = [], [], []
    duplicate_same = duplicate_conflict = 0
    for items in grouped.values():
        blocked = [x for x in items if x["source_dataset"] == "qa_ronin" and x["user"] in QA_REVIEW]
        for x in blocked:
            review.append({**x, "review_reason": QA_REVIEW[x["user"]]})
        eligible = [x for x in items if x not in blocked]
        if not eligible:
            continue
        answers = {re.sub(r"\\s+", " ", x["assistant"]).strip().casefold() for x in eligible}
        if len(answers) > 1:
            duplicate_conflict += 1
            conflicts.extend({**x, "conflict_reason": "same normalized prompt has different answers"} for x in eligible)
            continue
        accepted.append(eligible[0])
        duplicate_same += len(eligible) - 1

    def write_rows(path, rows):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\\n")

    write_rows(args.output, accepted)
    write_rows(args.conflicts, conflicts)
    write_rows(args.review, review)
    report = {
        "master_rows": len(accepted),
        "input_rows_by_source": dict(source_counts),
        "input_rows_total": sum(source_counts.values()),
        "rows_by_tipo_in_inputs": dict(type_counts),
        "exact_prompt_duplicates_removed": duplicate_same,
        "conflicting_prompt_groups_quarantined": duplicate_conflict,
        "conflict_rows_quarantined": len(conflicts),
        "qa_rows_quarantined_for_manual_review": len(review),
        "qa_manual_review_prompts": QA_REVIEW,
        "policy": "One row per normalized prompt. Exact same-answer duplicates collapse. Conflicting-answer groups and flagged QA rows are quarantined. Original source files remain untouched. No training or holdout modification."
    }
    p = Path(args.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
