"""Unprompted free-form generation on a fixed random Spanish question sample.

The multiple-choice options and gold answer are removed before either model sees
the prompt. Both models receive identical prompts and greedy decoding settings.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import CONFIG
from model.lumen import RoninLumen
from train.transformer_baseline import SmallTransformerLM


def load_rows(path: str):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def freeform_prompt(raw: str) -> str:
    # Strip all multiple-choice leakage, then turn the remaining item into a
    # natural, explicit open question. In particular, "¿Qué describe mejor a X?"
    # becomes "¿Quién es X?", which tests identity/description rather than
    # recognition among supplied options.
    text = str(raw).strip()
    text = re.split(r"\n\s*Opciones\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"\n\s*Respuesta\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    match = re.search(r"\n\s*Pregunta\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        prefix = text[:match.start()].strip()
        question = match.group(1).strip()
    else:
        prefix, question = "", text.strip()

    question = re.sub(
        r"^¿?\s*Qué\s+describe\s+mejor\s+a\s+(.+?)\s*\?$",
        r"¿Quién es \1? Descríbelo según el contexto disponible.",
        question,
        flags=re.IGNORECASE,
    )
    if not question.startswith("¿") and not question.endswith("?"):
        question = question.rstrip(".") + "?"
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append("Pregunta: " + question)
    parts.append("Responde con tus propias palabras. No hay opciones de respuesta; utiliza el contexto disponible.")
    return "\n".join(parts) + "\nRespuesta:"


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
    sample = random.Random(args.seed).sample(eligible, args.sample_size)

    lumen_state = torch.load(args.lumen, map_location="cpu", weights_only=False)
    if int(lumen_state.get("step", -1)) != args.expected_step:
        raise ValueError(f"Lumen checkpoint step mismatch: {lumen_state.get('step')}")
    lumen = RoninLumen(v32_ze=bool(lumen_state.get("v32_ze", False))).cpu()
    lumen.load_state_dict(lumen_state["model"])
    lumen.eval()

    transformer_state = torch.load(args.transformer, map_location="cpu", weights_only=False)
    if int(transformer_state.get("step", -1)) != args.expected_step:
        raise ValueError(f"Transformer checkpoint step mismatch: {transformer_state.get('step')}")
    transformer = SmallTransformerLM(CONFIG.vocab_size, args.max_len).cpu()
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
        "prompt_policy": "random sample from Spanish multiple-choice holdout; all choices and gold answer removed; only context/question retained",
        "results": results,
    }
    (out_dir / "step31000_freeform_100.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# LUMEN vs Transformer — 100 respuestas libres (31k)",
        "",
        f"- Muestra aleatoria reproducible: {len(results)} preguntas; semilla {args.seed}.",
        "- Se han eliminado las opciones y la respuesta correcta antes de generar.",
        "- Decodificación greedy, sin penalización de repetición; máximo 64 tokens por respuesta.",
        "- Evaluación informal: no es una métrica de exactitud ni sustituye al holdout de elección múltiple.",
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
    (out_dir / "step31000_freeform_100.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(results)} paired free-form generations to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
