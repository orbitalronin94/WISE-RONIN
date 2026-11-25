"""Unprompted free-form generation on a fixed set of 100 Spanish questions.

The multiple-choice options and gold answer are removed before either model sees
the prompt. Both models receive identical prompts and greedy decoding settings.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import CONFIG, LumenConfig
from model.lumen import RoninLumen
from train.transformer_baseline import SmallTransformerLM


def load_rows(path: str):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def freeform_prompt(raw: str) -> str:
    # Pass each authored open question verbatim: no choices, answer key,
    # answer label or extra cue is appended.
    return str(raw).strip()


def generate(model, tokenizer, prompt: str, *, is_lumen: bool,
             max_len: int, max_new_tokens: int, bos_id: int, eos_id: int, pad_id: int):
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not prefix_ids:
        return "", [], "empty_prompt"
    prefix_ids = [bos_id] + prefix_ids
    generated: list[int] = []
    stop_reason = "max_new_tokens"

    for _ in range(max_new_tokens):
        # Recompute the same bounded context each step. Lumen's transient
        # recurrent state is reset for each full-prefix pass; this avoids
        # accidentally feeding the prefix repeatedly into its Delta memory.
        context = (prefix_ids + generated)[-(max_len - 1):]
        x = torch.tensor([context], dtype=torch.long)
        if is_lumen:
            output = model(x, reset_state=True, collect_diagnostics=False)
            logits = output["logits"][0, -1]
        else:
            logits = model(x)[0, -1]

        next_id = int(torch.argmax(logits).item())
        if next_id == eos_id:
            stop_reason = "eos"
            break
        if next_id == pad_id:
            stop_reason = "pad"
            break
        generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated, stop_reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lumen", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--expected-step", type=int, default=31000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.sample_size < 1 or args.max_new_tokens < 1:
        parser.error("sample-size and max-new-tokens must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab = tokenizer.get_vocab()
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if bos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define [BOS], [EOS] and [PAD]")

    rows = load_rows(args.holdout)
    eligible = [
        row for row in rows
        if isinstance(row.get("prompt"), str) and row["prompt"].strip()
    ]
    if len(eligible) < args.sample_size:
        raise ValueError(f"Only {len(eligible)} eligible prompts for sample size {args.sample_size}")
    sample = eligible[:args.sample_size] if args.sample_size == len(eligible) else random.Random(args.seed).sample(eligible, args.sample_size)

    lumen_state = torch.load(args.lumen, map_location="cpu", weights_only=False)
    if int(lumen_state.get("step", -1)) != args.expected_step:
        raise ValueError(f"Lumen checkpoint step mismatch: {lumen_state.get('step')}")
    lumen_config = LumenConfig(**lumen_state["config"]) if isinstance(lumen_state.get("config"), dict) else CONFIG
    lumen = RoninLumen(config=lumen_config, v32_ze=bool(lumen_state.get("v32_ze", False))).cpu()
    if lumen_state.get("params") is not None and sum(x.numel() for x in lumen.parameters()) != int(lumen_state["params"]):
        raise ValueError("Lumen checkpoint parameter count does not match reconstructed architecture")
    lumen.load_state_dict(lumen_state["model"])
    lumen.eval()

    transformer_state = torch.load(args.transformer, map_location="cpu", weights_only=False)
    if int(transformer_state.get("step", -1)) != args.expected_step:
        raise ValueError(f"Transformer checkpoint step mismatch: {transformer_state.get('step')}")
    spec = transformer_state.get("model_spec", {})
    transformer = SmallTransformerLM(
        CONFIG.vocab_size, args.max_len,
        int(spec.get("model_dim", 192)), int(spec.get("heads", 4)),
        int(spec.get("ff_width", int(spec.get("model_dim", 192)) * 3)),
        int(spec.get("residual_groups", 0)),
    ).cpu()
    if transformer_state.get("params") is not None and sum(x.numel() for x in transformer.parameters()) != int(transformer_state["params"]):
        raise ValueError("Transformer checkpoint parameter count does not match reconstructed architecture")
    transformer.load_state_dict(transformer_state["model"])
    transformer.eval()

    results = []
    with torch.inference_mode():
        for index, row in enumerate(sample, 1):
            prompt = freeform_prompt(row["prompt"])
            lumen_text, lumen_ids, lumen_stop = generate(
                lumen, tokenizer, prompt, is_lumen=True,
                max_len=args.max_len, max_new_tokens=args.max_new_tokens,
                bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            )
            transformer_text, transformer_ids, transformer_stop = generate(
                transformer, tokenizer, prompt, is_lumen=False,
                max_len=args.max_len, max_new_tokens=args.max_new_tokens,
                bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            )
            results.append({
                "sample_index": index,
                "source_id": row.get("id"),
                "source": row.get("source"),
                "prompt": prompt,
                "lumen": {"text": lumen_text, "token_ids": lumen_ids, "stop": lumen_stop},
                "transformer": {"text": transformer_text, "token_ids": transformer_ids, "stop": transformer_stop},
            })
            if index % 10 == 0:
                print(f"freeform_progress {index}/{args.sample_size}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": args.expected_step,
        "sample_size": len(results),
        "sample_seed": args.seed,
        "decoding": "greedy argmax; no repetition penalty; max_new_tokens=%d; max_len=%d" % (args.max_new_tokens, args.max_len),
        "prompt_policy": "fixed 100-question Spanish identity/general capability set; prompts supplied directly without answer keys",
        "results": results,
    }
    stem = f"step{args.expected_step}_freeform_100"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# LUMEN vs Transformer — 100 respuestas libres ({args.expected_step:,})",
        "",
        f"- Muestra reproducible: {len(results)} preguntas; semilla {args.seed}.",
        "- Preguntas abiertas entregadas literalmente, sin opciones, respuestas de referencia ni pistas añadidas.",
        f"- Decodificación greedy, sin penalización de repetición; máximo {args.max_new_tokens} tokens por respuesta.",
        "- Evaluación informal: no es una métrica automática de exactitud ni sustituye al holdout oficial.",
        "",
    ]
    for item in results:
        lines.extend([
            f"## {item['sample_index']}. {item['source_id']}",
            "",
            item["prompt"],
            "",
            "**LUMEN**",
            "",
            item["lumen"]["text"] or "∅",
            "",
            "**Transformer**",
            "",
            item["transformer"]["text"] or "∅",
            "",
            "---",
            "",
        ])
    (out_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(results)} paired free-form generations to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
