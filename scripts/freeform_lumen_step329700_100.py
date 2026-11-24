"""100-question free-form generation probe for LUMEN step 329700."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model.lumen import RoninLumen

CHECKPOINT_PATH = Path("checkpoint-artifact/checkpoint/step_329700.pt")
STEP = 329700
SOURCE_RUN = 45273897576

def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.manual_seed(STEP)
    tokenizer = Tokenizer.from_file("prepared/artifacts/spanish-v2/tokenizer.json")
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")
    pad = tokenizer.token_to_id("[PAD]")
    if None in (bos, eos, pad):
        raise ValueError("Tokenizer lacks BOS/EOS/PAD")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != STEP:
        raise ValueError(f"Expected LUMEN step {STEP}, got {checkpoint.get('step')}")
    if checkpoint.get("architecture_id") != "ronin_lumen_v3_3_neuroplasticity":
        raise ValueError(f"Unexpected architecture: {checkpoint.get('architecture_id')}")

    model = RoninLumen(v32_ze=bool(checkpoint.get("v32_ze", False))).cpu()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = [json.loads(x) for x in Path("data/freeform_identity_general_100.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 100 or len({r["id"] for r in rows}) != 100:
        raise ValueError("Expected 100 unique fixed prompts")
    if rows[0]["prompt"] != "¿Quién eres?":
        raise ValueError("The first identity prompt must be ¿Quién eres?")

    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            prompt = row["prompt"].strip() + "\nRespuesta:"
            ids = [bos] + tokenizer.encode(prompt, add_special_tokens=False).ids
            generated = []
            stop = "max_new_tokens"
            for _ in range(96):
                context = (ids + generated)[-127:]
                x = torch.tensor([context], dtype=torch.long)
                out = model(x, reset_state=True, collect_diagnostics=False)
                nxt = int(torch.argmax(out["logits"][0, -1]).item())
                if nxt == eos:
                    stop = "eos"
                    break
                if nxt == pad:
                    stop = "pad"
                    break
                generated.append(nxt)
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({
                "index": i, "id": row["id"], "category": row["source"],
                "question": row["prompt"], "answer": answer,
                "generated_tokens": len(generated), "stop": stop
            })
            if i % 10 == 0:
                print(f"progress {i}/100", flush=True)

    Path("results").mkdir(exist_ok=True)
    payload = {
        "model": "LUMEN", "checkpoint_step": STEP, "checkpoint_source_run": SOURCE_RUN,
        "questions": 100,
        "decoding": "greedy argmax; max_new_tokens=96; context=128; CPU; no repetition penalty",
        "note": "Qualitative free-form probe; not an accuracy score or a substitute for the frozen multiple-choice holdout.",
        "results": results
    }
    Path("results/lumen_step329700_100_questions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# LUMEN 329.700 — 100 preguntas abiertas", "",
        "- Checkpoint: step 329.700.", "- Generación greedy; máximo 96 tokens por respuesta; contexto 128.",
        "- Prueba cualitativa: no es una puntuación automática de exactitud.", ""
    ]
    for r in results:
        lines += [
            f"## {r['index']}. {r['id']} — {r['category']}", "",
            f"**Pregunta:** {r['question']}", "", "**Respuesta de LUMEN:**", "",
            r["answer"] or "∅", "", "---", ""
        ]
    Path("results/lumen_step329700_100_questions.md").write_text("\n".join(lines), encoding="utf-8")
    print("completed 100 questions", flush=True)

if __name__ == "__main__":
    main()
