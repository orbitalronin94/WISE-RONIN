#!/usr/bin/env python3
"""Audit WISE-RONIN SFT data and build a conservative, non-destructive candidate.

The source JSONL files are never modified. The candidate removes exact duplicate
records and caps repeated normalized assistant targets; all removals are logged.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def category(row: dict) -> str:
    for key in ("categoria", "category", "tipo", "type", "skill"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "sin_categoria"


def digest(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--output-dir", default="audit_artifacts")
    ap.add_argument("--answer-cap-fraction", type=float, default=0.01)
    ap.add_argument("--min-answer-cap", type=int, default=5)
    args = ap.parse_args()
    if not 0 < args.answer_cap_fraction <= 1:
        ap.error("answer-cap-fraction must be in (0, 1]")
    rows = []
    per_file = {}
    for name in args.data:
        count = 0
        with open(name, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{name}:{line_no}: invalid JSON: {exc}")
                if not isinstance(row, dict) or not row.get("user") or not row.get("assistant"):
                    raise SystemExit(f"{name}:{line_no}: expected user/assistant")
                rows.append({**row, "_source_file": name, "_source_line": line_no})
                count += 1
        per_file[name] = count

    exact_seen = set()
    exact_unique = []
    exact_dupes = []
    for row in rows:
        key = digest({k: v for k, v in row.items() if not k.startswith("_source_")})
        if key in exact_seen:
            exact_dupes.append(row)
        else:
            exact_seen.add(key)
            exact_unique.append(row)

    answer_groups = collections.defaultdict(list)
    prefix_groups = collections.Counter()
    category_counts = collections.Counter()
    for row in exact_unique:
        ans = normalize(row["assistant"])
        answer_groups[ans].append(row)
        words = ans.split()
        if words:
            prefix_groups[" ".join(words[:4])] += 1
        category_counts[category(row)] += 1

    cap = max(args.min_answer_cap, math.ceil(len(exact_unique) * args.answer_cap_fraction))
    kept = []
    capped = []
    for normalized, group in answer_groups.items():
        # Deterministic ordering favors distinct prompts, then stable source order.
        group = sorted(group, key=lambda r: (normalize(r["user"]), r["_source_file"], r["_source_line"]))
        seen_prompts = set()
        diverse = []
        for row in group:
            prompt = normalize(row["user"])
            if prompt not in seen_prompts:
                diverse.append(row)
                seen_prompts.add(prompt)
        selected = diverse[:cap]
        kept.extend(selected)
        capped.extend(diverse[cap:])
        # Repeated prompts with same answer are also removed from candidate.
        capped.extend([r for r in group if normalize(r["user"]) in seen_prompts and r not in diverse])

    # Preserve original corpus order in the candidate.
    source_order = {(r["_source_file"], r["_source_line"]): i for i, r in enumerate(rows)}
    kept.sort(key=lambda r: source_order[(r["_source_file"], r["_source_line"])])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_path = out / "dataset_wise_ronin_capped_candidate.jsonl"
    with candidate_path.open("w", encoding="utf-8") as f:
        for row in kept:
            clean = {k: v for k, v in row.items() if not k.startswith("_source_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    top_answers = sorted(
        ((len(group), answer, group[0]["assistant"][:240]) for answer, group in answer_groups.items()),
        reverse=True,
    )[:100]
    near_prefixes = prefix_groups.most_common(100)
    total = len(rows)
    report = {
        "source_files": per_file,
        "raw_rows": total,
        "exact_duplicate_rows": len(exact_dupes),
        "exact_unique_rows": len(exact_unique),
        "normalized_answer_groups": len(answer_groups),
        "unique_prompt_count": len({normalize(r["user"]) for r in exact_unique}),
        "category_counts": dict(category_counts),
        "answer_cap_fraction": args.answer_cap_fraction,
        "answer_cap_per_normalized_target": cap,
        "candidate_rows": len(kept),
        "candidate_removed_after_exact_dedupe": total - len(kept),
        "candidate_removed_by_target_cap": len(capped),
        "top_normalized_answers": [
            {"count": n, "normalized": ans, "example": example}
            for n, ans, example in top_answers
        ],
        "top_four_word_prefixes": [
            {"prefix": prefix, "count": n} for prefix, n in near_prefixes
        ],
        "notes": [
            "Candidate is diagnostic, not approved for training.",
            "Only exact duplicate records and repeated normalized assistant targets are capped.",
            "Near-duplicate semantic templates require manual review; four-word prefixes are indicators, not automatic deletion rules.",
            "Category counts use explicit metadata only; missing labels are not guessed.",
            "Original source files are unchanged.",
        ],
    }
    (out / "dataset_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "dataset_top_answers.csv").open("w", encoding="utf-8") as f:
        f.write("count,normalized,example\n")
        for item in report["top_normalized_answers"]:
            vals = [str(item["count"]), item["normalized"], item["example"]]
            f.write(",".join('"' + v.replace('"', '""') + '"' for v in vals) + "\n")
    print(json.dumps({k: report[k] for k in (
        "raw_rows", "exact_duplicate_rows", "exact_unique_rows",
        "normalized_answer_groups", "unique_prompt_count", "category_counts",
        "answer_cap_per_normalized_target", "candidate_rows",
        "candidate_removed_after_exact_dedupe"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
