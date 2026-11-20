#!/usr/bin/env python3
"""Build a compact, fact-verified WISE-RONIN dataset from the original shards."""
from __future__ import annotations
import argparse, collections, hashlib, json, re
from pathlib import Path

FACT_PATTERNS = [
    re.compile(r"El hecho es\\s+(.+?)\\.", re.I | re.S),
    re.compile(r"El pan es\\s+(.+?)\\.\\s+La piedra", re.I | re.S),
    re.compile(r"Pues toma:\\s*(.+?)\\.\\s+Y el ", re.I | re.S),
    re.compile(r"Pues toma:\\s*(.+?)\\.\\s+Y no", re.I | re.S),
]
KOAN_PATTERNS = [
    re.compile(r"—¿Qué pone\\?\\s*—(.*?)\\s*—Pues ya está\\.", re.I | re.S),
    re.compile(r"En la cuarta:\\s*(.*?)\\.\\s*—¿Cuál es el rey\\?", re.I | re.S),
]
STYLE_PREFIXES = [
    ("te_doy_hecho_rodeo", re.compile(r"^Te doy el hecho y te doy el rodeo", re.I)),
    ("universo_gesto", re.compile(r"^El universo no te debe", re.I)),
    ("metafisica", re.compile(r"^La metafísica es esto", re.I)),
    ("pan_piedra", re.compile(r"^Preguntas como quien pide limosna", re.I)),
    ("si_te_digo", re.compile(r"^Si te digo solo", re.I)),
]
FACT_SYSTEM = "Eres WISE-RONIN. Responde únicamente con el dato correcto y conciso. No añadas rodeos, objetos, moralejas ni elogios. Si no sabes la respuesta, admítelo."
HYBRID_SYSTEM = "Eres WISE-RONIN. Da primero el hecho correcto; añade como máximo un gesto u observación seca. Nunca sacrifiques el hecho por el estilo. No adules ni inventes."
KOAN_SYSTEM = "Eres WISE-RONIN. Responde mediante un koan breve que conserve literalmente el hecho correcto. El koan no debe ocultar ni alterar la respuesta."
ANTI_SYSTEM = "Eres WISE-RONIN. No concedas elogios ni razón por complacencia. Pide argumentos o evalúa la idea con honestidad y respeto."

FEM = ["silla coja","bolsa de plástico","linterna sin pilas","cuchara doblada","ventana","llave usb","llave inglesa","pizarra","botella vacía","taza mugrienta","carpeta vacía","corona","lámpara"]
MASC = ["saco","teclado con una tecla menos","papel arrugado","candado abierto","cable pelado","espejo roto","cuaderno sin punta","reloj parado","rotulador seco","post-it amarillo","disco duro"]

def fact_from(text, patterns):
    for pat in patterns:
        m=pat.search(text)
        if m:
            value=re.sub(r"\\s+"," ",m.group(1)).strip().rstrip(".")
            if value and len(value)<=180: return value
    return None

def repair(text):
    for noun in sorted(FEM,key=len,reverse=True):
        text=re.sub(r"\\b(el|un) ("+re.escape(noun)+r")\\b",lambda m:("la " if m.group(1)=="el" else "una ")+m.group(2),text,flags=re.I)
        text=re.sub(r"(la "+re.escape(noun)+r" y) lo dejó",r"\\1 la dejó",text,flags=re.I)
    for noun in sorted(MASC,key=len,reverse=True):
        text=re.sub(r"\\b(la|una) ("+re.escape(noun)+r")\\b",lambda m:("el " if m.group(1)=="la" else "un ")+m.group(2),text,flags=re.I)
    text=re.sub(r"\\ben el office\\b","en la oficina",text,flags=re.I)
    text=re.sub(r"(El hecho es |El pan es |Si te digo(?: solo)? |Pues toma: )El (guepardo|Pacífico|Nilo)\\b",lambda m:m.group(1)+"el "+m.group(2),text)
    text=re.sub(r"(El hecho es |El pan es |Si te digo(?: solo)? |Pues toma: )La (plata|atracción)\\b",lambda m:m.group(1)+"la "+m.group(2),text)
    text=re.sub(r"(El hecho es |El pan es |Si te digo(?: solo)? |Pues toma: )Una región\\b",r"\\1una región",text)
    text=re.sub(r"(El hecho es |El pan es |Si te digo(?: solo)? |Pues toma: )El proceso\\b",r"\\1el proceso",text)
    text=re.sub(r"(El hecho es |El pan es |Si te digo(?: solo)? |Pues toma: )El código\\b",r"\\1el código",text)
    return text

def load(paths):
    rows=[]
    for path in paths:
        with open(path,encoding="utf-8") as f:
            for n,line in enumerate(f,1):
                if not line.strip(): continue
                row=json.loads(line)
                if not row.get("user") or not row.get("assistant"):
                    raise ValueError(f"{path}:{n}: falta user/assistant")
                rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",nargs="+",required=True)
    ap.add_argument("--output-dir",default="data/curated")
    args=ap.parse_args()
    rows=load(args.data)
    koans=collections.defaultdict(collections.Counter)
    fact_rows=collections.defaultdict(list)
    anti=[]
    type_counts=collections.Counter()
    for r in rows:
        typ=r.get("tipo",""); type_counts[typ]+=1
        if typ=="koan_hecho":
            a=fact_from(r["assistant"],KOAN_PATTERNS)
            if a: koans[r["user"].strip()][a]+=1
        elif typ=="hecho_rodeo_trolleo":
            a=fact_from(r["assistant"],FACT_PATTERNS)
            if a: fact_rows[a].append(r)
        elif typ=="antisicofantico": anti.append(r)

    facts=[]; conflicts=[]
    for q,c in sorted(koans.items()):
        answer,votes=c.most_common(1)[0]
        if votes/sum(c.values())<.98:
            conflicts.append({"question":q,"answers":c.most_common()}); continue
        corroboration=sum(1 for r in fact_rows[answer])
        if corroboration<10:
            conflicts.append({"question":q,"answer":answer,"corroboration":corroboration}); continue
        facts.append((q,answer,votes,corroboration))
    if len(facts)<25: raise SystemExit(f"Solo {len(facts)} hechos validados")

    factual=[]; koan=[]; hybrid=[]; prefs=[]; template_counts=collections.Counter()
    for q,answer,votes,corr in facts:
        factual.append({"system":FACT_SYSTEM,"user":q,"assistant":answer+".","tipo":"factual_clean"})
        kc=[r for r in rows if r.get("tipo")=="koan_hecho" and r.get("user","").strip()==q and fact_from(r["assistant"],KOAN_PATTERNS)==answer]
        if kc:
            target=min(kc,key=lambda r:hashlib.sha256((q+r["assistant"]).encode()).hexdigest())["assistant"]
            target=re.sub(r"(Un joven dev preguntó al maestro:)\\s*.*?(\\s+El maestro)",lambda m:m.group(1)+" "+q+m.group(2),target,count=1,flags=re.I|re.S)
            koan.append({"system":KOAN_SYSTEM,"user":q,"assistant":repair(target),"tipo":"koan_controlled"})
        by_template=collections.defaultdict(list)
        for r in fact_rows[answer]:
            for name,pat in STYLE_PREFIXES:
                if pat.search(r["assistant"].strip()) and fact_from(r["assistant"],FACT_PATTERNS)==answer:
                    by_template[name].append(r); break
        for name,group in sorted(by_template.items()):
            chosen=min(group,key=lambda r:hashlib.sha256((q+name+r["assistant"]).encode()).hexdigest())
            hybrid.append({"system":HYBRID_SYSTEM,"user":q,"assistant":repair(chosen["assistant"]),"tipo":"hybrid_"+name})
            template_counts[name]+=1
        prefs.append({"system":FACT_SYSTEM,"user":q,"chosen":answer+".","rejected":"El que contesta.","tipo":"preference_fact_over_template"})

    seen=set(); antis=[]
    for r in sorted(anti,key=lambda x:(x["user"],x["assistant"])):
        key=(r["user"].strip(),r["assistant"].strip())
        if key in seen: continue
        seen.add(key); antis.append({"system":ANTI_SYSTEM,"user":key[0],"assistant":key[1],"tipo":"antisycophancy"})

    balanced=factual+factual+hybrid+koan+antis
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    files={
      "wise_ronin_factual_clean.jsonl":factual,
      "wise_ronin_hybrid_controlled.jsonl":hybrid,
      "wise_ronin_koan_controlled.jsonl":koan,
      "wise_ronin_antisycophancy.jsonl":antis,
      "wise_ronin_sft_balanced.jsonl":balanced,
      "wise_ronin_preferences.jsonl":prefs,
    }
    for name,data in files.items():
        with (out/name).open("w",encoding="utf-8") as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+"\\n")
    report={
      "raw_rows":len(rows),"source_type_counts":dict(type_counts),"validated_facts":len(facts),
      "factual_rows":len(factual),"hybrid_rows":len(hybrid),"koan_rows":len(koan),
      "antisycophancy_rows":len(antis),"preference_pairs":len(prefs),"balanced_sft_rows":len(balanced),
      "balanced_sft_type_counts":dict(collections.Counter(r["tipo"] for r in balanced)),
      "hybrid_template_counts":dict(template_counts),"fact_conflicts_or_missing_corroboration":conflicts,
      "facts":[{"question":q,"answer":a,"koan_votes":v,"fact_rodeo_rows":c} for q,a,v,c in facts],
      "source_untouched":True,
      "notes":["The source contains only 30 canonical factual questions; repeated variants were collapsed.",
      "Facts were extracted from KOAN targets and cross-checked against at least 10 fact-rodeo rows.",
      "One representative per detected hybrid template family and fact; generic object-only fallbacks excluded.",
      "The balanced SFT pool retains twice the factual anchors during style conditioning.",
      "Preference pairs prefer the verified fact over the generic El que contesta fallback.",
      "This compact corpus does not claim broad factual coverage; the manual holdout is not used."]}
    (out/"wise_ronin_dataset_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("raw_rows","validated_facts","factual_rows","hybrid_rows","koan_rows","antisycophancy_rows","preference_pairs","balanced_sft_rows","hybrid_template_counts")},ensure_ascii=False,indent=2))
    if conflicts: raise SystemExit(f"Hay {len(conflicts)} conflictos sin resolver")

if __name__=="__main__": main()
