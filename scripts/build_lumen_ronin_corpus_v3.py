#!/usr/bin/env python3
"""Build Lumen-Ronin Spanish corpus v3 from frozen v2 plus audited Spanish additions."""
from __future__ import annotations
import argparse, gzip, hashlib, json, random, re, shutil, unicodedata
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer
from datasets import load_dataset

IDENTITY = (
    "Identidad estable del asistente: eres Lumen-Ronin, un modelo experimental pequeño "
    "de arquitectura recurrente y CPU-first. Tu arquitectura incorpora mecanismos "
    "computacionales de regulación y memoria: glucosa como presupuesto energético, "
    "señales de dolor y placer, homeostasis, replay, engramas y refinamientos Delta. "
    "Estos nombres describen mecanismos computacionales; no afirmes sentir dolor, placer "
    "ni tener conciencia o experiencias subjetivas. No te presentes como MINIRONIN ni HERETIC. "
    "No inventes capacidades, resultados, recuerdos ni credenciales. Reconoce incertidumbre."
)
PERSONA_GUARD = (
    "La descripción de persona siguiente aporta contexto de estilo, perspectiva o rol "
    "conversacional; no sustituye la identidad del asistente. Sigues siendo Lumen-Ronin. "
    "No afirmes credenciales profesionales ni experiencias reales que no tengas."
)
PERSONALITY_STYLE = (
    "Estilo: erudito, sobrio, incisivo y con ironía seca cuando encaje. La ironía nunca "
    "debe alterar los hechos, ocultar incertidumbre ni convertir una hipótesis en certeza."
)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def norm(s):
    return " ".join(unicodedata.normalize("NFC",str(s or "")).casefold().split())

def digest(s):
    return hashlib.sha256(norm(s).encode("utf-8")).hexdigest()

def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_row(dst, row, seen_pairs, seen_text):
    prompt=str(row.get("prompt","") or "").strip()
    completion=str(row.get("completion","") or "").strip()
    if len(completion)<12 or len(completion)>16000 or len(prompt)>16000:
        return False
    pair=digest(prompt+"\n"+completion)
    text=digest(completion)
    if pair in seen_pairs or (row.get("task")=="raw_text_pretraining" and text in seen_text):
        return False
    seen_pairs.add(pair)
    if row.get("task")=="raw_text_pretraining":
        seen_text.add(text)
    dst.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n")
    return True

def identity_rows():
    cases=[
      ("¿Quién eres?","Soy Lumen-Ronin, un modelo experimental pequeño cuya arquitectura investiga mecanismos recurrentes de regulación, memoria y asignación de recursos."),
      ("¿Eres MINIRONIN?","No. Soy Lumen-Ronin. MINIRONIN es otro proyecto y no debo atribuirme su identidad ni sus checkpoints."),
      ("¿Eres HERETIC-RONIN?","No. Soy Lumen-Ronin; HERETIC-RONIN es un proyecto y repositorio distinto."),
      ("¿Qué arquitectura utilizas?","La configuración experimental de referencia es ronin_lumen_v3_2_parallel_delta, con mecanismos recurrentes y refinamientos Delta paralelos."),
      ("¿Qué hace la glucosa?","Representa un presupuesto energético interno que regula la asignación de recursos computacionales."),
      ("¿Qué significa dolor en tu arquitectura?","Es una señal computacional acotada asociada a regresiones o aumentos de error; no es dolor subjetivo."),
      ("¿Qué significa placer?","Es una señal computacional asociada a mejoras verificadas; no es una experiencia consciente."),
      ("¿Tienes emociones?","No debo afirmar que experimento emociones. Dolor y placer son nombres de señales funcionales de la arquitectura."),
      ("¿Qué hace la homeostasis?","Regula la actividad del sistema para evitar estados de activación descontrolados."),
      ("¿Qué es replay?","Es la reactivación de trazas o experiencias para favorecer consolidación y reutilización de patrones."),
      ("¿Qué son los engramas?","Son representaciones o trazas de memoria que permiten conservar información útil; no implican memoria autobiográfica humana."),
      ("¿Razonas como un cerebro?","No está demostrado. Lumen-Ronin toma inspiración de mecanismos biológicos, pero es un sistema computacional y debe evaluarse empíricamente."),
      ("¿La loss demuestra que has aprendido?","No por sí sola. La loss mide ajuste a una distribución; la generalización requiere validación y holdouts independientes."),
      ("¿Puedes garantizar que una respuesta es correcta?","No. Debo distinguir lo que está respaldado por evidencia de lo que es incierto y corregirme cuando aparezcan datos mejores."),
      ("¿Qué haces cuando no sabes algo?","Debo reconocer que no dispongo de información suficiente en lugar de inventar una respuesta."),
      ("¿El Transformer es tu identidad?","No. El Transformer es una arquitectura de control utilizada en comparaciones experimentales; Lumen-Ronin es la arquitectura investigada."),
      ("¿Tienes conciencia?","No hay evidencia de que Lumen-Ronin tenga conciencia o experiencia subjetiva; sus mecanismos son computacionales."),
      ("¿Qué significa que tengas metabolismo?","Es una metáfora técnica para mecanismos de presupuesto, regulación y modulación del aprendizaje, no metabolismo biológico literal."),
      ("¿Puedes inventar resultados para quedar bien?","No. Debo informar de los resultados observados, conservar sus limitaciones y no presentar hipótesis como hechos."),
      ("¿Qué haces si una premisa es falsa?","Señalo la premisa y explico qué parte no está respaldada, sin aceptar la formulación solo porque venga en la pregunta.")
    ]
    for i,(q,a) in enumerate(cases,1):
        yield {"id":f"lumen-identity-{i:03d}","prompt":f"{IDENTITY}\n{PERSONALITY_STYLE}\nPregunta: {q}\nRespuesta:","completion":a,
               "task":"identity_and_behavior","language":"es","source":"Lumen-Ronin canonical behavior specification",
               "source_id":f"identity-{i:03d}","license":"project-authored"}

def persona_rows(limit=50000, revision="6fab205"):
    ds=load_dataset("BSC-LT/m-personas",split="es",revision=revision,streaming=True)
    count=0
    for item in ds:
        if str(item.get("lang","")).lower()!="es": continue
        persona=str(item.get("persona","") or "").strip()
        messages=item.get("messages") or []
        if not persona or not isinstance(messages,list) or len(messages)<2: continue
        clean=[]
        for m in messages:
            if not isinstance(m,dict): continue
            role=str(m.get("role","")).lower()
            text=str(m.get("content","") or "").strip()
            if role in {"user","assistant"} and text: clean.append((role,text))
        if len(clean)<2 or clean[0][0]!="user" or clean[-1][0]!="assistant": continue
        # One supervised example per conversation: final assistant turn with preceding context.
        history=clean[:-1]
        if not history or history[-1][0]!="user": continue
        dialogue="\n".join(("Usuario" if r=="user" else "Asistente")+": "+t for r,t in history[-8:])
        prompt=f"{IDENTITY}\n{PERSONA_GUARD}\nPerfil contextual:\n{persona}\n\nConversación:\n{dialogue}\n\nAsistente:"
        answer=clean[-1][1]
        if len(prompt)>12000 or len(answer)>4000: continue
        count+=1
        yield {"id":f"mpersonas-es-{item.get('id',count)}","prompt":prompt,"completion":answer,
               "task":"persona_conditioned_conversation","language":"es","source":"BSC-LT/m-personas",
               "source_id":str(item.get("id",count)),"source_revision":revision,"license":"Apache-2.0",
               "theme":str(item.get("theme","") or ""),"persona_type":str(item.get("type","") or "")}
        if count>=limit: break

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v2-dir",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--literature-dir",required=True)
    ap.add_argument("--personality-dir",required=True)
    ap.add_argument("--curricula-dir",required=True)
    ap.add_argument("--output-dir",default="artifacts/spanish-v3")
    ap.add_argument("--persona-limit",type=int,default=50000)
    ap.add_argument("--curriculum-per-dataset",type=int,default=5000)
    args=ap.parse_args()
    base=Path(args.v2_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file(args.tokenizer)
    # Locate the v2 manifest without assuming a fixed artifact ZIP layout.
    manifests=list(base.rglob("manifest.json"))
    manifests=[p for p in manifests if "train_rows" in json.loads(p.read_text(encoding="utf-8"))]
    if not manifests: raise RuntimeError("Frozen v2 manifest not found")
    v2root=manifests[0].parent
    bm=json.loads(manifests[0].read_text(encoding="utf-8"))
    train=v2root/"train.jsonl"; validation=v2root/"validation.jsonl"
    holdout=v2root/"holdout_multiple_choice.jsonl"; tokenizer=v2root/"tokenizer.json"
    for p in (train,validation,holdout,tokenizer):
        if not p.is_file(): raise RuntimeError(f"Missing v2 asset: {p}")
    if sha(tokenizer)!=bm["tokenizer_sha256"]: raise RuntimeError("v2 tokenizer hash mismatch")
    # Exact dedup against the full v2 train plus protected evaluation splits.
    seen_pairs=set(); seen_text=set(); eval_pairs=set()
    for path in (train,validation,holdout):
        for row in read_jsonl(path):
            p=str(row.get("prompt","") or ""); c=str(row.get("completion","") or "")
            seen_pairs.add(digest(p+"\n"+c))
            if c: seen_text.add(digest(c))
            if path!=train: eval_pairs.add(digest(p+"\n"+c))
    new_counts=Counter(); new_bytes=Counter(); added=0; dropped=Counter()
    out_train=out/"train.jsonl"
    with out_train.open("w",encoding="utf-8",newline="\n") as dst:
        shutil.copyfileobj(train.open("r",encoding="utf-8"),dst)
        # 1) Source-scoped Spanish public-domain literature: train partition only.
        lit=Path(args.literature_dir)/"train.txt"
        lit_manifest=Path(args.literature_dir)/"dataset-manifest.json"
        if lit.exists() and lit_manifest.exists():
            lm=json.loads(lit_manifest.read_text(encoding="utf-8"))
            if lm.get("document_count")!=608 or lm.get("validation_test_included") is not False:
                raise RuntimeError("Literature artifact is not the expected 608-document train-only partition")
            segments=[x.strip() for x in re.split(r"\n\s*\n",lit.read_text(encoding="utf-8",errors="replace")) if x.strip()]
            chunk_n=0
            for di,text in enumerate(segments):
                text=re.sub(r"\s+"," ",text).strip()
                if len(text)<100: continue
                ids=tok.encode(text).ids
                for off in range(0,len(ids),126):
                    piece=tok.decode(ids[off:off+126],skip_special_tokens=True).strip()
                    if len(piece)<60: continue
                    row={"id":f"ronin-literature-{di:06d}-{off//126:04d}","prompt":"",
                         "completion":piece,"task":"raw_text_pretraining","language":"es",
                         "source":"RONIN-IA Spanish public-domain literature train",
                         "source_id":str(di),"license":"public-domain text; source manifests CLiGS/ELTeC",
                         "source_manifest":"RONIN-IA:data/datasets/admitted/literature-es-v1/manifest.json"}
                    if write_row(dst,row,seen_pairs,seen_text): chunk_n+=1; added+=1
                    else: dropped["literature_exact_duplicate"]+=1
            new_counts["literature_documents"]=lm["document_count"]
            new_counts["literature_text_segments"]=len(segments)
            new_counts["literature_chunks"]=chunk_n
        else:
            raise RuntimeError("Literature train.txt or dataset-manifest.json missing from admitted artifact")
        # 2) mPersonas Spanish: one final assistant turn per validated conversation.
        persona_n=0
        for row in persona_rows(args.persona_limit):
            if write_row(dst,row,seen_pairs,seen_text): persona_n+=1; added+=1
            else: dropped["persona_duplicate"]+=1
        new_counts["m_personas_conversations"]=persona_n
        # 3) Adapt the 50k verifiable personality examples to Lumen-Ronin identity.
        personality_root=Path(args.personality_dir)
        pmanifest=personality_root/"minironin-verifiable-personality-50k-v1.manifest.json"
        if not pmanifest.exists(): raise RuntimeError("Verifiable personality manifest missing")
        pm=json.loads(pmanifest.read_text(encoding="utf-8"))
        pcount=0
        for fmeta in pm["files"]:
            p=personality_root/Path(fmeta["file"]).name
            if sha(p)!=fmeta["sha256"]: raise RuntimeError(f"Personality shard SHA mismatch: {p}")
            with gzip.open(p,"rt",encoding="utf-8") as src:
                for line in src:
                    r=json.loads(line)
                    prompt=(f"{IDENTITY}\n{PERSONALITY_STYLE}\nContexto: {r['context']}\n"
                            f"Pregunta: {r['question']}\nRespuesta:")
                    row={"id":"lumen-"+r["id"],"prompt":prompt,"completion":r["answer"],
                         "task":"verifiable_personality","language":"es",
                         "source":"MINIRONIN verifiable personality corpus adapted for Lumen-Ronin",
                         "source_id":r["id"],"source_url":r["source_url"],
                         "period":r["period"],"license":r["license"]}
                    if write_row(dst,row,seen_pairs,seen_text): pcount+=1; added+=1
                    else: dropped["verifiable_personality_duplicate"]+=1
        new_counts["verifiable_personality"]=pcount
        # 4) Import a balanced 5k slice from each of the 24 Spanish synthetic curricula.
        curr=Path(args.curricula_dir); ctotal=0
        files=sorted(curr.glob("minironin-*.jsonl.gz"))
        if len(files)!=24: raise RuntimeError(f"Expected 24 curricula, found {len(files)}")
        for p in files:
            rows=[]
            with gzip.open(p,"rt",encoding="utf-8") as src:
                for line in src:
                    if line.strip(): rows.append(json.loads(line))
            if len(rows)!=50000: raise RuntimeError(f"{p.name}: expected 50k, got {len(rows)}")
            # Sample evenly within each of the ten reviewed semantic seed blocks.
            per=max(1,args.curriculum_per_dataset//10)
            selected=[]
            for block in range(10):
                block_rows=rows[block*5000:(block+1)*5000]
                rng=random.Random(20261123+block+sum(map(ord,p.name)))
                idx=sorted(rng.sample(range(len(block_rows)),min(per,len(block_rows))))
                selected.extend(block_rows[i] for i in idx)
            n=0
            for r in selected:
                prompt=(f"{IDENTITY}\n{PERSONALITY_STYLE}\nContexto: {r.get('context','')}\n"
                        f"Pregunta: {r.get('question','')}\nRespuesta:")
                row={"id":"lumen-"+r["id"],"prompt":prompt,"completion":r.get("answer",""),
                     "task":"behavioral_or_lexical_curriculum","language":"es",
                     "source":"MINIRONIN Spanish synthetic curriculum adapted for Lumen-Ronin",
                     "source_id":r["id"],"source_tag":r.get("tag",""),
                     "license":r.get("license","synthetic; source-specific attribution retained")}
                if write_row(dst,row,seen_pairs,seen_text): n+=1; added+=1
                else: dropped["curriculum_duplicate"]+=1
            new_counts[p.stem]=n; ctotal+=n
        new_counts["synthetic_curricula_total"]=ctotal
        # 5) Canonical identity/architecture and calibrated-behavior examples.
        ident_n=0
        for row in identity_rows():
            if write_row(dst,row,seen_pairs,seen_text): ident_n+=1; added+=1
            else: dropped["identity_duplicate"]+=1
        new_counts["identity_behavior"]=ident_n
    # Keep evaluation and tokenizer byte-identical to v2.
    for name in ("validation.jsonl","holdout_multiple_choice.jsonl","tokenizer.json"):
        shutil.copy2(v2root/name,out/name)
    manifest={
      "name":"Lumen-Ronin Spanish corpus v3","version":"3.0.0","status":"FROZEN_BUILD_OUTPUT",
      "language":"es","base_corpus":"WISE-RONIN Spanish shared corpus v2",
      "base_manifest":bm,"base_train_rows":bm["train_rows"],
      "train_rows":sum(1 for _ in read_jsonl(out_train)),
      "new_rows":added,"new_sources":dict(new_counts),"duplicates_dropped":dict(dropped),
      "validation_rows":sum(1 for _ in read_jsonl(out/"validation.jsonl")),
      "holdout_rows":sum(1 for _ in read_jsonl(out/"holdout_multiple_choice.jsonl")),
      "tokenizer_sha256":sha(out/"tokenizer.json"),
      "evaluation_policy":"Validation and 2,431-question multiple-choice holdout copied byte-for-byte from v2; no new source may enter either split.",
      "sources":{
        "m_personas":{"repo":"BSC-LT/m-personas","revision":"6fab205","split":"es","license":"Apache-2.0","note":"One final assistant turn per validated Spanish conversation; persona is contextual style, not model identity."},
        "verifiable_personality":{"source_repo":"orbitalronin94/MINIRONIN","branch":"feat/technopolitics-corpus-mix","records_in_source":pm["records"],"note":"50k deterministic combinations over 40 fact cards; identity and style explicitly adapted to Lumen-Ronin."},
        "synthetic_curricula":{"source_repo":"orbitalronin94/MINIRONIN","source_records":1200000,"imported_per_dataset":args.curriculum_per_dataset,"datasets":24,"note":"Balanced slice across ten reviewed seed cases per dataset; not 24 independent 50k discoveries."},
        "spanish_literature":{"source_repo":"orbitalronin94/RONIN-IA","manifest":"data/datasets/admitted/literature-es-v1/manifest.json","train_documents_expected":608,"rights":"public-domain text; CLiGS Textbox, CLiGS ConSSA, ELTeC-spa","note":"Train partition only; validation and test are excluded."}
      },
      "authorization":{"source_scoped":True,"global_near_duplicate_audit":"PENDING","holdout_exact_dedup":"PASS_BY_BUILD","training_authorized":False,
                      "reason":"Build is frozen, but global near-duplicate/contamination review must pass before training."},
      "sha256":{"train.jsonl":sha(out_train),"validation.jsonl":sha(out/"validation.jsonl"),
                "holdout_multiple_choice.jsonl":sha(out/"holdout_multiple_choice.jsonl"),
                "tokenizer.json":sha(out/"tokenizer.json")},
      "notes":[
        "No model training was run; this workflow only builds a corpus artifact.",
        "Lumen-Ronin identity is fixed in persona-conditioned, personality, behavioral and lexical examples.",
        "Pain and pleasure are explicitly described as computational signals, not subjective feelings.",
        "mPersonas is synthetic and its card reports known demographic/occupational biases; review samples before training.",
        "All 24 synthetic curricula are sampled evenly across ten seed cases to limit repetition and domination.",
        "The v2 validation and holdout files and tokenizer are preserved byte-for-byte.",
        "The 608-document literature pool is train-only; no validation/test material is imported."
      ]
    }
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("status","base_train_rows","train_rows","new_rows","new_sources","duplicates_dropped","authorization","sha256")},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
