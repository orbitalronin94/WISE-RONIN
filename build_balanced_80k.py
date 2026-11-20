#!/usr/bin/env python3
"""Build a deterministic 80k category-balanced WISE-RONIN SFT set.

The source corpus is never modified. Unselected source rows are written to a
separate reserve JSONL. Factual paraphrases retain their original fact_id;
they do not count as new independent facts.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

TARGETS = {
    "factual_geography": 10_000,
    "factual_chemistry": 15_000,
    "factual_general": 15_000,
    "math_addition": 10_000,
    "math_subtraction": 10_000,
    "math_multiplication": 10_000,
    "math_division": 10_000,
}
EXTRA_SUFFIXES = [
    " Da el dato comprobable, sin adornos.",
    " Responde con precisión y sin inventar.",
    " Una respuesta factual basta.",
    " Comprueba el dato antes de responder.",
    " Da el nombre o la cifra exactos.",
    " No conviertas la respuesta en una moraleja.",
    " Contesta con el hecho y conserva la calma.",
    " Basta con el dato correcto.",
]
PREFIXES = ["", "A ver: ", "Consulta: ", "Sin ceremonia: ", "Comprueba esto: ",
            "Pregunta directa: ", "En limpio: ", "Vamos al dato: ",
            "Te lo planteo así: ", "Una comprobación: "]

def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

def read_jsonl(path: Path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip():
                row=json.loads(line)
                if not row.get("user") or not row.get("assistant") or not row.get("tipo"):
                    raise ValueError(f"{path}:{n}: missing user/assistant/tipo")
                rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source", default="data/expanded/wise_ronin_expanded.jsonl")
    ap.add_argument("--output", default="data/balanced/dataset_reequilibrado_80k.jsonl")
    ap.add_argument("--reserve", default="data/balanced/reserve_unselected.jsonl")
    ap.add_argument("--report", default="data/balanced/balanced_80k_report.json")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    rng=random.Random(args.seed)
    source=read_jsonl(Path(args.source))
    pools=defaultdict(list)
    for i,row in enumerate(source):
        pools[row["tipo"]].append((i,row))

    selected=[]
    used=set()
    counts={}
    # Categories with enough existing rows are sampled without replacement.
    for category,target in TARGETS.items():
        pool=pools[category]
        if category != "math_division":
            if len(pool) < target:
                # The factual bank is expanded by additional prompt formulations,
                # not by pretending paraphrases are new independent facts.
                base_n=len(pool)
                chosen=pool[:]
                rng.shuffle(chosen)
                need=target-base_n
                if category not in ("factual_geography","factual_general"):
                    raise ValueError(f"{category}: only {len(pool)} rows for target {target}")
                for j in range(need):
                    _,base=chosen[j % len(chosen)]
                    clone=dict(base)
                    suffix=EXTRA_SUFFIXES[(j // max(1,len(chosen)) + j) % len(EXTRA_SUFFIXES)]
                    clone["user"]=base["user"]+suffix
                    clone["assistant"]=base["assistant"]+" "+EXTRA_SUFFIXES[(j+3)%len(EXTRA_SUFFIXES)]
                    clone["variant_id"]=int(base.get("variant_id",0))+120+j
                    clone["balance_augmentation"]="extra_prompt_formulation_same_fact_id"
                    selected.append(clone)
                take=pool
            else:
                take=rng.sample(pool,target)
            for i,row in take:
                used.add(i)
                selected.append(dict(row))
            counts[category]=target
        else:
            take=pool[:]
            rng.shuffle(take)
            for i,row in take:
                used.add(i)
            selected.extend(dict(row) for _,row in take)
            need=target-len(take)
            # Generate new exact integer-division problems: 936 distinct base
            # operations × 10 prompt/style formulations = 9,360 rows.
            base_tasks=[]
            for divisor in range(2,42):
                for quotient in range(2,42):
                    dividend=divisor*quotient
                    q=f"¿Cuánto es {dividend} dividido entre {divisor}?"
                    base_tasks.append((q,str(quotient),dividend,divisor))
            if len(base_tasks)*10 < need:
                raise ValueError("Not enough unique division tasks")
            templates=[r for _,r in pool]
            made=0
            for task_i,(question,answer,dividend,divisor) in enumerate(base_tasks):
                if made>=need: break
                fact_id=stable_id("division|"+question+"|"+answer)
                for variant in range(10):
                    if made>=need: break
                    template=templates[(task_i*10+variant)%len(templates)]
                    old=str(template.get("hecho",""))
                    assistant=template["assistant"].replace(old,answer) if old else answer
                    prefix=PREFIXES[(task_i*7+variant*3)%len(PREFIXES)]
                    row=dict(template)
                    row.update({
                        "user":prefix+question+EXTRA_SUFFIXES[variant%len(EXTRA_SUFFIXES)],
                        "assistant":assistant,
                        "tipo":"math_division",
                        "fact_id":fact_id,
                        "variant_id":variant,
                        "hecho":answer,
                        "balance_augmentation":"new_exact_division_task"
                    })
                    selected.append(row); made+=1
            if made!=need:
                raise ValueError(f"Division augmentation made {made}, expected {need}")
            counts[category]=target

    # Guarantee prompt uniqueness. Any collision is a hard failure, not silent loss.
    prompts=[r["user"] for r in selected]
    if len(prompts)!=len(set(prompts)):
        duplicates=len(prompts)-len(set(prompts))
        raise ValueError(f"Balanced set has {duplicates} duplicate prompts")
    if len(selected)!=80_000:
        raise ValueError(f"Expected 80000 rows, got {len(selected)}")
    rng.shuffle(selected)

    reserve=[row for i,row in enumerate(source) if i not in used]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f:
        for row in selected: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    res=Path(args.reserve); res.parent.mkdir(parents=True,exist_ok=True)
    with res.open("w",encoding="utf-8") as f:
        for row in reserve: f.write(json.dumps(row,ensure_ascii=False)+"\n")

    report={
        "rows":len(selected),
        "category_counts":dict(Counter(r["tipo"] for r in selected)),
        "target_counts":TARGETS,
        "unique_prompts":len(set(prompts)),
        "unique_responses":len({r["assistant"] for r in selected}),
        "source_rows":len(source),
        "reserved_unselected_source_rows":len(reserve),
        "source_rows_used":len(used),
        "augmented_same_fact_rows":sum(r.get("balance_augmentation")=="extra_prompt_formulation_same_fact_id" for r in selected),
        "new_exact_division_rows":sum(r.get("balance_augmentation")=="new_exact_division_task" for r in selected),
        "independent_fact_ids":len({r.get("fact_id") for r in selected}),
        "seed":args.seed,
        "note":"Paraphrase augmentation preserves fact_id; it increases prompt coverage, not independent factual knowledge. All unused source rows are preserved in reserve_unselected.jsonl."
    }
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
