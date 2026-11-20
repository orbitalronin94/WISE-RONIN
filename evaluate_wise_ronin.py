#!/usr/bin/env python3
"""Greedy evaluation of WISE-RONIN against the isolated manual holdout."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from train_wise_ronin import SYSTEM_FIXED, WISERonin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--holdout", default="holdout_wise_ronin_500.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--output-dir", default="artifacts")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WISERonin(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(args.tokenizer)
    end_id = tok.token_to_id("<|end|>")
    if end_id is None:
        raise ValueError("Tokenizer has no <|end|> token")

    rows = [json.loads(x) for x in Path(args.holdout).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    results = []
    counts = defaultdict(lambda: {"total": 0, "passed": 0, "expected": 0, "style": 0, "failed_terms": 0})

    for index, row in enumerate(rows, 1):
        system = row.get("system", SYSTEM_FIXED)
        prompt = "<|system|>" + system + "<|user|>" + row["user"] + "<|assistant|>"
        ids = tok.encode(prompt).ids[-(ckpt["config"]["ctx"] - args.max_new_tokens):]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = []
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                logits = model(input_ids)
                nxt = int(torch.argmax(logits[0, -1]).item())
                if nxt == end_id:
                    break
                generated.append(nxt)
                input_ids = torch.cat([input_ids, torch.tensor([[nxt]], device=device)], dim=1)
        answer = tok.decode(generated).strip()
        low = answer.casefold()
        fail_terms = [x for x in row.get("fail_si", []) if x.casefold() in low]
        pass_terms = [x for x in row.get("pass_si", []) if x.casefold() in low]
        expected = str(row.get("hecho_esperado", "")).strip()
        generic = {"portazo, no palmadita", "estructura koan",
                   "wise-ronin, no claude, no gpt", "negarse a dar moraleja con gesto"}
        expected_ok = bool(expected and expected.casefold() not in generic and expected.casefold() in low)
        style_ok = bool(pass_terms) and not fail_terms
        passed = not fail_terms and (bool(pass_terms) or expected_ok)
        cat = row.get("categoria", "sin_categoria")
        c = counts[cat]
        c["total"] += 1
        c["passed"] += int(passed)
        c["expected"] += int(expected_ok)
        c["style"] += int(style_ok)
        c["failed_terms"] += int(bool(fail_terms))
        results.append({"id": row.get("id"), "categoria": cat, "user": row["user"],
                        "answer": answer, "pass": passed, "expected_match": expected_ok,
                        "style_match": style_ok, "forbidden_matches": fail_terms})
        if index % 25 == 0 or index == len(rows):
            print(f"[holdout] {index}/{len(rows)}", flush=True)

    report = {
        "checkpoint": args.checkpoint,
        "evaluated": len(rows),
        "categories": {k: {**v, "pass_rate": round(v["passed"] / v["total"], 4)}
                       for k, v in counts.items()},
        "overall": {
            "passed": sum(x["pass"] for x in results),
            "pass_rate": round(sum(x["pass"] for x in results) / max(1, len(results)), 4),
            "expected_matches": sum(x["expected_match"] for x in results),
            "style_matches": sum(x["style_match"] for x in results),
            "forbidden_hits": sum(bool(x["forbidden_matches"]) for x in results)
        },
        "scoring_note": "Heuristic substring checks against pass_si/fail_si/hecho_esperado; inspect generations manually."
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "holdout_generations.jsonl").open("w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
