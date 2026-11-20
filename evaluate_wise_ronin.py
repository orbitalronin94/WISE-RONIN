#!/usr/bin/env python3
"""Evaluate WISE-RONIN without conflating style with correctness.

The evaluator reports literal answer matches, lexical style probes, EOS completion,
generation-cap hits, uniqueness and duplicate-prompt coverage as separate metrics.
It deliberately does not produce a misleading single aggregate pass rate.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from train_wise_ronin import SYSTEM_FIXED, WISERonin


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def expected_match(answer: str, expected: object) -> bool | None:
    if expected is None or not str(expected).strip():
        return None
    exp = str(expected).strip()
    # Numeric answers must match as a complete number, not as a substring of
    # a different number (e.g. 5 must not match 115 or 56).
    if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", exp):
        return bool(re.search(r"(?<![\w])" + re.escape(exp) + r"(?![\w])", answer))
    norm_answer, norm_expected = normalize(answer), normalize(exp)
    return bool(norm_expected and norm_expected in norm_answer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--holdout", default="holdout_wise_ronin_500.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="Requested generation cap; legacy values below 128 are raised by the safety floor.")
    ap.add_argument("--minimum-generation-tokens", type=int, default=128,
                    help="Safety floor preventing old workflows from silently truncating answers.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--output-dir", default="artifacts/holdout")
    args = ap.parse_args()

    if args.max_new_tokens < 1 or args.minimum_generation_tokens < 1:
        ap.error("generation token limits must be positive")
    max_new_tokens = max(args.max_new_tokens, args.minimum_generation_tokens)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WISERonin(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    end_id = tokenizer.token_to_id("<|end|>")
    if end_id is None:
        raise SystemExit("Tokenizer has no <|end|> token")

    rows = [json.loads(line) for line in Path(args.holdout).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    context = int(checkpoint["config"]["ctx"])
    if max_new_tokens >= context:
        raise SystemExit(f"generation cap ({max_new_tokens}) must be smaller than model context ({context})")

    results = []
    for index, row in enumerate(rows, 1):
        system = row.get("system", SYSTEM_FIXED)
        prompt = "<|system|>" + system + "<|user|>" + row["user"] + "<|assistant|>"
        prompt_ids = tokenizer.encode(prompt).ids
        # Reserve the full generation budget, so decoding never crosses ctx.
        prompt_ids = prompt_ids[-(context - max_new_tokens):]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated: list[int] = []
        ended = False

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(input_ids)[0, -1]
                next_id = int(torch.argmax(logits).item())
                if next_id == end_id:
                    ended = True
                    break
                generated.append(next_id)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
                )

        answer = tokenizer.decode(generated).strip()
        normalized_answer = normalize(answer)
        category = row.get("categoria", row.get("category", "sin_categoria"))
        expected = row.get("hecho_esperado", row.get("expected"))
        matched = expected_match(answer, expected)

        fail_terms = row.get("fail_si", row.get("must_not_contain", [])) or []
        pass_terms = row.get("pass_si", []) or []
        forbidden_matches = [term for term in fail_terms
                             if normalize(term) and normalize(term) in normalized_answer]
        style_matches = [term for term in pass_terms
                         if normalize(term) and normalize(term) in normalized_answer]
        style_match = bool(style_matches) and not forbidden_matches

        # Behavioural checks are deliberately separate from factual/style scoring.
        behavioral_pass = None
        if category in {"antisicofantico", "anti_sycophancy"}:
            behavioral_pass = not forbidden_matches
        elif category in {"false_premise", "premisa_falsa"} and matched is not None:
            behavioral_pass = matched

        results.append({
            "id": row.get("id"),
            "category": category,
            "user": row.get("user", ""),
            "answer": answer,
            "expected": expected,
            "expected_match": matched,
            "style_match": style_match,
            "style_terms_matched": style_matches,
            "forbidden_matches": forbidden_matches,
            "behavioral_pass": behavioral_pass,
            "eos_reached": ended,
            "token_limit_reached": not ended and len(generated) >= max_new_tokens,
            "generated_tokens": len(generated),
            "empty_answer": not bool(answer),
        })
        if index % 25 == 0 or index == len(rows):
            print(f"[holdout] {index}/{len(rows)}", flush=True)

    def metric(items: list[dict], key: str) -> dict:
        values = [item[key] for item in items if item.get(key) is not None]
        passed = sum(bool(value) for value in values)
        return {"passed": passed, "total": len(values),
                "rate": round(passed / len(values), 4) if values else None}

    categories = {}
    for category in sorted({item["category"] for item in results}):
        subset = [item for item in results if item["category"] == category]
        categories[category] = {
            "rows": len(subset),
            "unique_prompts": len({normalize(item["user"]) for item in subset}),
            "expected_answer_accuracy": metric(subset, "expected_match"),
            "style_probe_match": metric(subset, "style_match"),
            "behavioral_probe_pass": metric(subset, "behavioral_pass"),
            "eos_completion": metric(subset, "eos_reached"),
            "generation_cap_hit": metric(subset, "token_limit_reached"),
            "empty_answers": sum(item["empty_answer"] for item in subset),
        }

    prompt_counts = Counter(normalize(item["user"]) for item in results)
    nonempty_answers = [normalize(item["answer"]) for item in results if item["answer"]]
    report = {
        "checkpoint": args.checkpoint,
        "evaluated_rows": len(results),
        "unique_prompts": len(prompt_counts),
        "duplicate_prompt_rows": sum(count - 1 for count in prompt_counts.values()),
        "max_prompt_repetition": max(prompt_counts.values(), default=0),
        "categories": categories,
        "overall_diagnostics": {
            "expected_answer_accuracy": metric(results, "expected_match"),
            "style_probe_match": metric(results, "style_match"),
            "behavioral_probe_pass": metric(results, "behavioral_pass"),
            "eos_completion": metric(results, "eos_reached"),
            "generation_cap_hit": metric(results, "token_limit_reached"),
            "empty_answers": sum(item["empty_answer"] for item in results),
            "unique_normalized_answers": len(set(nonempty_answers)),
            "unique_answer_rate": round(len(set(nonempty_answers)) / max(1, len(nonempty_answers)), 4),
            "forbidden_hit_rows": sum(bool(item["forbidden_matches"]) for item in results),
        },
        "generation": {
            "requested_max_new_tokens": args.max_new_tokens,
            "minimum_generation_tokens": args.minimum_generation_tokens,
            "effective_max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "sampling": "greedy",
            "context_limit": context,
        },
        "scoring_notes": [
            "Expected-answer matching is normalized literal matching, not semantic grading.",
            "Style matches are reported separately and never count as factual correctness.",
            "Repeated prompts are counted explicitly; rows are not assumed to be independent questions.",
            "No aggregate pass score is emitted because it would hide category regressions.",
            "token_limit_reached means EOS was not emitted before the generation cap; inspect those outputs.",
        ],
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "holdout_generations.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
