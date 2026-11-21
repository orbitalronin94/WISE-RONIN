#!/usr/bin/env python3
"""Greedy, matched-budget evaluation of LUMEN and the parameter-matched Transformer."""
from __future__ import annotations
import argparse, json, re, hashlib, unicodedata, random
from collections import OrderedDict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.config import CONFIG
from model.lumen import RoninLumen
from train.dataset import BPETokenizer
from train.transformer_baseline import SmallTransformerLM
from data.audit_holdout import read_jsonl

def norm(s):
    return " ".join(re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", str(s or "")).casefold()).split())

def generation_metrics(text, eos_terminated):
    tokens = str(text or "").split()
    bigrams = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    first_loop = len(tokens)
    for index in range(len(tokens) - 2):
        if tokens[index] == tokens[index + 1] == tokens[index + 2]:
            first_loop = index
            break
    return {
        "rep_bigram": 1.0 - len(set(bigrams)) / max(1, len(bigrams)),
        "rep_trigram": 1.0 - len(set(trigrams)) / max(1, len(trigrams)),
        "first_loop_at": first_loop,
        "diversity": len(set(tokens)) / max(1, len(tokens)),
        "has_eos": int(bool(eos_terminated)),
        "len_tokens": len(tokens),
        "text": str(text or ""),
    }


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

def fit_prompt(ids, max_context=128):
    """Keep both the question opening and its final answer cue within context."""
    budget = max_context - 1  # reserve one position for BOS
    if len(ids) <= budget:
        return list(ids)
    head = budget // 2
    tail = budget - head
    return list(ids[:head]) + list(ids[-tail:])


def generate_batch(model, prompts, tokenizer, kind, max_new=64, max_context=128):
    # Reset transient recurrent activity at every prompt boundary. State remains
    # active only while decoding tokens belonging to this prompt.
    if kind == "lumen":
        reset = getattr(model, "reset_state", None)
        if reset is None:
            reset = getattr(model, "reset_states", None)
        if reset is not None:
            reset()
    # LUMEN carries recurrent state per batch row; sequential prompts avoid
    # right-padding contaminating a shorter prompt's recurrent prefill state.
    if kind == "lumen" and len(prompts) > 1:
        return [
            item
            for prompt in prompts
            for item in generate_batch(model, [prompt], tokenizer, kind, max_new, max_context)
        ]
    bos = tokenizer.tokenizer.token_to_id("[BOS]")
    eos = tokenizer.tokenizer.token_to_id("[EOS]")
    pad = tokenizer.tokenizer.token_to_id("[PAD]")
    sequences = []
    for prompt in prompts:
        ids = tokenizer.tokenizer.encode(str(prompt), add_special_tokens=False).ids
        sequences.append([bos] + fit_prompt(ids, max_context=max_context))
    initial_lengths = [len(s) for s in sequences]
    finished = [False] * len(sequences)
    for generation_step in range(max_new):
        # Transformer recomputes a sliding context. LUMEN prefills once, then
        # consumes one token at a time so its recurrent Delta state is retained.
        if kind == "lumen" and generation_step > 0:
            contexts = [[seq[-1]] for seq in sequences]
        else:
            contexts = [seq[-max_context:] for seq in sequences]
        width = max(len(seq) for seq in contexts)
        batch = torch.full((len(sequences), width), pad, dtype=torch.long)
        lengths = []
        for i, seq in enumerate(contexts):
            lengths.append(len(seq))
            batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        with torch.no_grad():
            out = model(batch, reset_state=(generation_step == 0)) if kind == "lumen" else model(batch)
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
        outputs.append({
            "text": tokenizer.tokenizer.decode(generated, skip_special_tokens=True),
            "eos_terminated": bool(finished[len(outputs)]),
            "generated_tokens": len(generated),
        })
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
    p.add_argument("--expected-step", type=int, default=500, help="Required global checkpoint step for evaluated checkpoints")
    p.add_argument("--lumen-only", action="store_true", help="Evaluate only LUMEN for a non-paired longitudinal checkpoint")
    p.add_argument("--output", default="artifacts/pilot-evaluation")
    args = p.parse_args()
    if args.expected_step < 1:
        p.error("--expected-step must be positive")
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
    summary = {"evaluation_mode": "lumen_only_longitudinal" if args.lumen_only else "matched_pair",
               "data_sha256": data_hash, "tokenizer_sha256": tok_hash,
               "gsm8k_test_questions": len(gsm), "historical_unique_prompts": len(historical),
               "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
               "max_context_tokens": 128,
               "prompt_truncation": "retain first and last tokens within 127-token prompt budget; Transformer uses sliding 128-token context",
               "lumen_decoding": "128-token prompt prefill then one-token stateful recurrent decoding",
               "gsm8k_sample_seed": 42, "gsm8k_selected_ids": [r["id"] for r in gsm],
               "models": {}}
    all_predictions = []
    models_to_eval = (("lumen", args.lumen_ckpt),) if args.lumen_only else (("lumen", args.lumen_ckpt), ("transformer", args.transformer_ckpt))
    for kind, checkpoint in models_to_eval:
        model, state = load_model(kind, checkpoint, args.tokenizer)
        if state.get("data_sha256") != data_hash:
            raise ValueError(f"{kind}: checkpoint corpus hash mismatch")
        if state.get("tokenizer_sha256") != tok_hash:
            raise ValueError(f"{kind}: checkpoint tokenizer hash mismatch")
        if state.get("step") != args.expected_step or state.get("seed") != 42:
            raise ValueError(f"{kind}: expected matched step-{args.expected_step} seed-42 checkpoint; got step={state.get('step')} seed={state.get('seed')}")
        predictions = []
        for start in range(0, len(tasks), 8):
            batch_tasks = tasks[start:start+8]
            generations = generate_batch(model, [x[1] for x in batch_tasks], tokenizer, kind,
                                          max_new=args.max_new_tokens)
            for (dataset, prompt, expected, raw), generation in zip(batch_tasks, generations):
                response = generation["text"]
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
                    "eos_terminated": generation["eos_terminated"],
                    "generated_tokens": generation["generated_tokens"],
                    "generation_metrics": generation_metrics(response, generation["eos_terminated"]),
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
            "eos_terminated": sum(bool(row["eos_terminated"]) for row in predictions),
            "eos_termination_rate": round(
                sum(bool(row["eos_terminated"]) for row in predictions) / max(1, len(predictions)), 4
            ),
            "unique_responses": len({norm(row["response"]) for row in predictions}),
            "unique_response_ratio": round(
                len({norm(row["response"]) for row in predictions}) / max(1, len(predictions)), 4
            ),
            "mean_generated_tokens": round(
                sum(row["generated_tokens"] for row in predictions) / max(1, len(predictions)), 2
            ),
            "generation_diagnostics": {
                key: round(
                    sum(row["generation_metrics"][key] for row in predictions) / max(1, len(predictions)), 4
                )
                for key in ("rep_bigram", "rep_trigram", "first_loop_at", "diversity", "has_eos", "len_tokens")
            },
        }
        del model
    lumen_tokens = summary["models"]["lumen"].get("tokens_seen")
    if not args.lumen_only:
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
    else:
        summary["tokens_seen"] = lumen_tokens
    with (outdir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in all_predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
