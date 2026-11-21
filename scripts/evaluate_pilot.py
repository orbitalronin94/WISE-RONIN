#!/usr/bin/env python3
"""Greedy, matched-budget evaluation of LUMEN and the parameter-matched Transformer."""
from __future__ import annotations
import argparse, json, re, hashlib, unicodedata, random
from collections import OrderedDict
from pathlib import Path
import torch
from model.config import CONFIG
from model.lumen import RoninLumen
from train.dataset import BPETokenizer
from train.transformer_baseline import SmallTransformerLM
from data.audit_holdout import read_jsonl

def norm(s):
    return " ".join(re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", str(s or "")).casefold()).split())

def numeric(s):
    text = str(s or "")
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    matches = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return ""
    value = matches[-1].replace("$", "").replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    return value

def generate_batch(model, prompts, tokenizer, kind, max_new=64, max_prompt=63):
    bos = tokenizer.tokenizer.token_to_id("[BOS]")
    eos = tokenizer.tokenizer.token_to_id("[EOS]")
    pad = tokenizer.tokenizer.token_to_id("[PAD]")
    sequences = []
    for prompt in prompts:
        ids = tokenizer.tokenizer.encode(str(prompt), add_special_tokens=False).ids[:max_prompt]
        sequences.append([bos] + ids)
    initial_lengths = [len(s) for s in sequences]
    finished = [False] * len(sequences)
    for _ in range(max_new):
        width = max(len(s) for s in sequences)
        batch = torch.full((len(sequences), width), pad, dtype=torch.long)
        lengths = []
        for i, seq in enumerate(sequences):
            lengths.append(len(seq))
            batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        with torch.no_grad():
            out = model(batch, reset_state=True) if kind == "lumen" else model(batch)
            logits = out["logits"] if kind == "lumen" else out
            positions = torch.tensor(lengths, dtype=torch.long) - 1
            next_ids = logits[torch.arange(len(sequences)), positions].argmax(dim=-1).tolist()
        for i, token in enumerate(next_ids):
            if finished[i]:
                continue
            sequences[i].append(int(token))
            if token == eos:
                finished[i] = True
        if all(finished):
            break
    outputs = []
    for seq, prompt_len in zip(sequences, initial_lengths):
        generated = seq[prompt_len:]
        outputs.append(tokenizer.tokenizer.decode(generated, skip_special_tokens=True))
    return outputs

def load_model(kind, checkpoint, tokenizer_path):
    if kind == "lumen":
        model = RoninLumen(v32_ze=False)
    else:
        model = SmallTransformerLM(CONFIG.vocab_size, 128)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot_train_19k.jsonl")
    p.add_argument("--tokenizer", default="data/pilot_tokenizer/tokenizer.json")
    p.add_argument("--lumen-ckpt", default="checkpoints/lumen/step_500.pt")
    p.add_argument("--transformer-ckpt", default="checkpoints/transformer/step_500.pt")
    p.add_argument("--gsm8k", default="artifacts/reasoning_gsm8k_500.jsonl")
    p.add_argument("--historical-holdout", default="data/holdout_500.jsonl")
    p.add_argument("--gsm8k-limit", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--output", default="artifacts/pilot-evaluation")
    args = p.parse_args()
    torch.set_num_threads(2)
    data_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    tok_hash = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    tokenizer = BPETokenizer(args.tokenizer)

    gsm_all = read_jsonl(args.gsm8k)
    gsm = random.Random(42).sample(gsm_all, min(args.gsm8k_limit, len(gsm_all)))
    historical_raw = read_jsonl(args.historical_holdout)
    historical_by_prompt = OrderedDict()
    for row in historical_raw:
        key = norm(row.get("user", ""))
        if key and key not in historical_by_prompt:
            historical_by_prompt[key] = row
    historical = list(historical_by_prompt.values())
    tasks = (
        [("gsm8k", "Question: " + row["user"] + "\nAnswer:", row["hecho_esperado"], row) for row in gsm]
        + [("historical", row["user"], row["hecho_esperado"], row) for row in historical]
    )
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"data_sha256": data_hash, "tokenizer_sha256": tok_hash,
               "gsm8k_test_questions": len(gsm), "historical_unique_prompts": len(historical),
               "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
               "gsm8k_sample_seed": 42, "gsm8k_selected_ids": [r["id"] for r in gsm],
               "models": {}}
    all_predictions = []
    for kind, checkpoint in (("lumen", args.lumen_ckpt), ("transformer", args.transformer_ckpt)):
        model, state = load_model(kind, checkpoint, args.tokenizer)
        if state.get("data_sha256") != data_hash:
            raise ValueError(f"{kind}: checkpoint corpus hash mismatch")
        if state.get("tokenizer_sha256") != tok_hash:
            raise ValueError(f"{kind}: checkpoint tokenizer hash mismatch")
        if state.get("step") != 500 or state.get("seed") != 42:
            raise ValueError(f"{kind}: expected the matched 500-step seed-42 checkpoint")
        predictions = []
        for start in range(0, len(tasks), 8):
            batch_tasks = tasks[start:start+8]
            texts = generate_batch(model, [x[1] for x in batch_tasks], tokenizer, kind,
                                   max_new=args.max_new_tokens)
            for (dataset, prompt, expected, raw), response in zip(batch_tasks, texts):
                if dataset == "gsm8k":
                    correct = numeric(response) == numeric(expected)
                else:
                    response_norm, expected_norm = norm(response), norm(expected)
                    correct = bool(expected_norm and expected_norm in response_norm)
                    fail_terms = [norm(x) for x in raw.get("fail_si", [])]
                    if any(term and term in response_norm for term in fail_terms):
                        correct = False
                record = {
                    "model": kind, "dataset": dataset, "id": raw.get("id"),
                    "prompt": prompt, "expected": expected,
                    "response": response, "correct": correct,
                }
                predictions.append(record)
                all_predictions.append(record)
        by_dataset = {}
        for dataset in ("gsm8k", "historical"):
            subset = [r for r in predictions if r["dataset"] == dataset]
            if subset:
                by_dataset[dataset] = {
                    "n": len(subset), "correct": sum(r["correct"] for r in subset),
                    "accuracy": round(sum(r["correct"] for r in subset) / len(subset), 4),
                }
        summary["models"][kind] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_steps": state.get("step"),
            "tokens_seen": state.get("tokens_seen"),
            "metrics": by_dataset,
        }
        del model
    lumen_tokens = summary["models"]["lumen"].get("tokens_seen")
    transformer_tokens = summary["models"]["transformer"].get("tokens_seen")
    if lumen_tokens != transformer_tokens:
        raise ValueError(f"Token budgets differ: LUMEN={lumen_tokens}, Transformer={transformer_tokens}")
    lumen_params = summary["models"]["lumen"]["parameters"]
    transformer_params = summary["models"]["transformer"]["parameters"]
    summary["parameter_gap_percent"] = round(
        abs(lumen_params - transformer_params) / lumen_params * 100, 4
    )
    if summary["parameter_gap_percent"] >= 1.0:
        raise ValueError("Parameter counts are not within the 1% matched-size tolerance")
    summary["matched_token_budget"] = lumen_tokens
    with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in all_predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
