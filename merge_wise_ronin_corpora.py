#!/usr/bin/env python3
"""Build a provenance-preserving WISE-RONIN master SFT corpus."""
from __future__ import annotations
import argparse, collections, json, re
from pathlib import Path

QA_REVIEW = {
    "¿Qué metal es segundo mejor conductor?": "La respuesta aportada dice plata; la plata es el mejor conductor eléctrico entre los metales, mientras que el cobre suele citarse como segundo. Revisión humana necesaria.",
    "¿Cuál es la capital de Israel?": "Pregunta geopolíticamente disputada; la respuesta depende del reconocimiento y del marco institucional. Reescribir con contexto o excluir.",
}

# IDs are used when the same factual claim appears under differently worded prompts.
QA_REVIEW_RECORDS = {
    38: "El Nilo suele figurar como el río más largo, pero la comparación con el Amazonas depende de cómo se definan y midan las fuentes y desembocaduras. Añadir la controversia o evitar el superlativo absoluto.",
    39: "La respuesta confunde el gas necesario con la mezcla que respiramos: respiramos aire, compuesto principalmente por nitrógeno y oxígeno (aprox. 21%).",
    99: "La respuesta confunde el gas necesario con la mezcla que respiramos: respiramos aire, compuesto principalmente por nitrógeno y oxígeno (aprox. 21%).",
    334: "Almaty dejó de ser la capital de Kazajistán en 1997; la capital actual es Astaná. Corregir la respuesta.",
    340: "La capitalidad de Indonesia está en transición: la ley establece Nusantara como nueva capital, pero el traslado oficial de la sede requiere un decreto presidencial que seguía pendiente en 2026. No presentar Yakarta como respuesta simple sin contexto.",
    345: "Colombo no es la capital legislativa oficial de Sri Lanka; esa función corresponde a Sri Jayawardenepura Kotte. Colombo es capital comercial y sede del presidente y del Gobierno. Precisar el tipo de capital.",
    371: "Bolivia tiene capital constitucional en Sucre y sede de Gobierno en La Paz. La pregunta genérica requiere distinguir ambas funciones.",
    423: "El Nilo suele figurar como el río más largo, pero la comparación con el Amazonas depende de cómo se definan y midan las fuentes y desembocaduras. Añadir la controversia o evitar el superlativo absoluto.",
    424: "La respuesta confunde el gas necesario con la mezcla que respiramos: respiramos aire, compuesto principalmente por nitrógeno y oxígeno (aprox. 21%).",
    476: "La teoría de evolución por selección natural fue presentada independientemente por Charles Darwin y Alfred Russel Wallace; atribuirla solo a Darwin simplifica la historia.",
    477: "Einstein formuló la teoría de la relatividad; no la 'escribió' como si fuese una obra única. Reformular la pregunta y la respuesta con precisión.",
    485: "La impresión y los tipos móviles existían en Asia oriental antes de Gutenberg. Su contribución fue desarrollar la imprenta europea de tipos móviles metálicos. Especificar el ámbito europeo.",
    486: "La atribución de la primera señal de radio depende de qué se considere señal/transmisión y de los experimentos previos. La pregunta absoluta requiere contexto histórico.",
    490: "Leeuwenhoek no fue quien usó el primer microscopio; los primeros microscopios compuestos se atribuyen a desarrollos anteriores. Leeuwenhoek destacó por sus microscopios de una sola lente y observaciones.",
    492: "Galileo desarrolló un termoscopio y contribuyó a la termometría, pero no puede afirmarse sin matices que inventó el termómetro. La historia incluye a Santorio y otros investigadores.",
    507: "La pregunta es sobre Cuba, pero la respuesta cambia el sujeto a España. La abolición de la esclavitud en Cuba se produjo en 1886; corregir el texto para que coincida con el país preguntado.",
    644: "El Nilo suele figurar como el río más largo, pero la comparación con el Amazonas depende de cómo se definan y midan las fuentes y desembocaduras. Añadir la controversia o evitar el superlativo absoluto.",
    645: "La respuesta confunde el gas necesario con la mezcla que respiramos: respiramos aire, compuesto principalmente por nitrógeno y oxígeno (aprox. 21%).",
    659: "La teoría de evolución por selección natural fue presentada independientemente por Charles Darwin y Alfred Russel Wallace; atribuirla solo a Darwin simplifica la historia.",
    660: "Einstein formuló la teoría de la relatividad; no la 'escribió' como si fuese una obra única. Reformular la pregunta y la respuesta con precisión.",
    663: "La impresión y los tipos móviles existían en Asia oriental antes de Gutenberg. Su contribución fue desarrollar la imprenta europea de tipos móviles metálicos. Especificar el ámbito europeo.",
    670: "La pregunta no especifica qué Constitución. La respuesta presupone la Constitución española de 1978; incluir 'española' en el prompt.",
    833: "El Nilo suele figurar como el río más largo, pero la comparación con el Amazonas depende de cómo se definan y midan las fuentes y desembocaduras. Añadir la controversia o evitar el superlativo absoluto.",
    834: "La respuesta confunde el gas necesario con la mezcla que respiramos: respiramos aire, compuesto principalmente por nitrógeno y oxígeno (aprox. 21%).",
    866: "La teoría de evolución por selección natural fue presentada independientemente por Charles Darwin y Alfred Russel Wallace; atribuirla solo a Darwin simplifica la historia.",
    867: "Einstein formuló la teoría de la relatividad; no la 'escribió' como si fuese una obra única. Reformular la pregunta y la respuesta con precisión.",
    868: "La pregunta y la respuesta son demasiado vagas para enseñar una atribución histórica precisa. Reformular como '¿Quién descubrió la penicilina en 1928?' y responder con Fleming.",
    869: "La impresión y los tipos móviles existían en Asia oriental antes de Gutenberg. Su contribución fue desarrollar la imprenta europea de tipos móviles metálicos. Especificar el ámbito europeo.",
    873: "La pregunta no especifica qué Constitución. La respuesta presupone la Constitución española de 1978; incluir 'española' en el prompt.",
    960: "El Nilo suele figurar como el río más largo, pero la comparación con el Amazonas depende de cómo se definan y midan las fuentes y desembocaduras. Añadir la controversia o evitar el superlativo absoluto.",
    961: "La respuesta confunde el gas necesario con la mezcla que respiramos: respiramos aire, compuesto principalmente por nitrógeno y oxígeno (aprox. 21%).",
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

            if source == "qa_ronin":
                reason = QA_REVIEW_RECORDS.get(item.get("source_record")) or QA_REVIEW.get(item["user"])
                if reason:
                    review.append({**item, "review_reason": reason})
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
        "qa_manual_review_records": {str(k): v for k, v in QA_REVIEW_RECORDS.items()},
        "policy": "Deduplicate only identical normalized prompt+completion pairs. Preserve multiple completions for the same prompt and report them for inspection. Quarantine factual, ambiguous, or historically oversimplified QA records by source_record ID or exact prompt. Original source files remain untouched. No training or holdout modification."
    }
    p = Path(args.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
