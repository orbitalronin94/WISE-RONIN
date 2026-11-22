#!/usr/bin/env python3
"""Build a provenance-preserving Spanish-only corpus from license-audited HF datasets."""
from __future__ import annotations
import argparse, hashlib, html, json, re, unicodedata
from collections import Counter
from pathlib import Path
from datasets import load_dataset
import langid

SEED = 42
# Component-level audit from BSC-LT/ALIA-2606-SFT card.
# NC, ND, CC-BY-SA and mixed/unclear components are deliberately excluded.
ALIA_LICENSES = {
    "aya-dataset": "Apache-2.0",
    "self-identity": "Apache-2.0 (internal)",
    "eif-augmentation": "Apache-2.0 (internal)",
    "fineweb-edu_contextual-qa": "Apache-2.0 (internal)",
    "fineweb-edu_qa": "Apache-2.0 (internal)",
    "long-context-qa": "Apache-2.0 (internal)",
    "multiturn-augmentation": "Apache-2.0 (internal)",
    "oasst2-selfidentity-rephrase": "Apache-2.0",
    "openr1-math-translated": "Apache-2.0 (internal)",
    "system-prompt_multi-turn_non-verifiable": "Apache-2.0 (internal)",
    "system-prompt-translation": "CC-BY-4.0 (internal)",
}
EXCLUDED_ALIA = ["coqcat-train","databricks-dolly-15k","dolly-ca","flores-dev",
                 "mentor-es","mentor-ca","no-robots","rag-multilingual","tower-blocks",
                 "wildchat-curated-deepseekv3-es-2602"]
SOURCES = {
    "BSC-LT/ALIA-2606-SFT": "CC-BY-4.0; component allowlist enforced",
    "OysterCoreAI/SFT-General-Spanish-50K": "CC-BY-4.0",
    "somosnlp/instruct-legal-refugiados-es": "Apache-2.0",
    "gplsi/uji_parallel_va_es": "CC-BY-4.0 (Spanish ES column only)",
}

def clean(v):
    s = unicodedata.normalize("NFC", html.unescape(str(v or ""))).replace("\u00a0", " ").replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", s).strip()

def spanish(s):
    s = clean(s)
    if len(s) < 25:
        return bool(re.search(r"\b(el|la|los|las|que|de|del|una|para|con|por|qué|cómo|es)\b|[áéíóúñ¿¡]", s, re.I))
    try:
        return langid.classify(s[:5000])[0] == "es"
    except Exception:
        return False

def key(prompt, completion):
    norm = " ".join((prompt + "\n" + completion).casefold().split())
    return hashlib.sha256(norm.encode()).hexdigest()

def make(prompt, completion, source, license_name, task, source_id, component=None):
    prompt, completion = clean(prompt), clean(completion)
    if len(completion) < 12 or len(prompt) > 16000 or len(completion) > 16000:
        return None
    if not spanish(prompt + "\n" + completion):
        return None
    row = {
        "id": hashlib.sha256(f"{source}|{source_id}|{key(prompt, completion)}".encode()).hexdigest()[:20],
        "prompt": prompt, "completion": completion, "task": task, "language": "es",
        "source": source, "source_id": str(source_id), "license": license_name,
    }
    if component:
        row["source_component"] = component
    return row

def chat_pairs(messages, source, license_name, source_id, task):
    if not isinstance(messages, list):
        return []
    history, out = [], []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role, content = clean(msg.get("role","")).lower(), clean(msg.get("content",""))
        if not content:
            continue
        if role == "assistant":
            hist = "\n".join(
                f"{'Sistema' if r=='system' else 'Usuario' if r=='user' else 'Asistente'}: {c}"
                for r,c in history
            )
            row = make(hist + ("\nAsistente:" if hist else "Asistente:"), content,
                       source, license_name, task, source_id)
            if row:
                out.append(row)
        history.append((role, content))
    return out

def stream(repo, split="train", buffer=10000):
    return load_dataset(repo, split=split, streaming=True).shuffle(seed=SEED, buffer_size=buffer)

def build_alia(limit):
    counts, out = Counter(), []
    for item in stream("BSC-LT/ALIA-2606-SFT", buffer=20000):
        if clean(item.get("lang","")).lower() != "es":
            continue
        component = clean(item.get("dataset_name",""))
        license_name = ALIA_LICENSES.get(component)
        if not license_name or counts[component] >= 8000:
            continue
        pairs = chat_pairs(item.get("messages"), "BSC-LT/ALIA-2606-SFT",
                           license_name, item.get("id",len(out)), clean(item.get("task","conversation")))
        for row in pairs:
            row["source_component"] = component
            out.append(row); counts[component] += 1
            if len(out) >= limit:
                return out
    return out

def build_general(limit):
    out = []
    for item in stream("OysterCoreAI/SFT-General-Spanish-50K", buffer=10000):
        for row in chat_pairs(item.get("messages"), "OysterCoreAI/SFT-General-Spanish-50K",
                              "CC-BY-4.0", item.get("id",len(out)), "general_sft"):
            out.append(row)
            if len(out) >= limit:
                return out
    return out

def build_legal(limit):
    out = []
    for item in stream("somosnlp/instruct-legal-refugiados-es", buffer=5000):
        q, ctx, ans = map(clean,(item.get("question"),item.get("context"),item.get("answer")))
        if not q or not ans: continue
        row = make(f"Contexto:\n{ctx}\n\nPregunta: {q}\n\nRespuesta:", ans,
                   "somosnlp/instruct-legal-refugiados-es","Apache-2.0","legal_qa",
                   item.get("id",len(out)))
        if row:
            out.append(row)
            if len(out) >= limit: return out
    return out

def build_uji(limit):
    out = []
    for index, item in enumerate(stream("gplsi/uji_parallel_va_es", buffer=10000)):
        # Keep only the explicit Spanish column; never ingest the Valencian VA column.
        text = clean(item.get("ES", ""))
        row = make("", text, "gplsi/uji_parallel_va_es", "CC-BY-4.0",
                   "lm_text", f"{item.get('source','')}-{index}")
        if row:
            out.append(row)
            if len(out) >= limit:
                return out
    return out

def siqa_choice_label(value):
    """SIQA_es encodes its three answer choices as string labels 1, 2 and 3."""
    label = clean(value).upper()
    return {"1": "A", "2": "B", "3": "C"}.get(label, label if label in {"A", "B", "C"} else None)


def build_eval():
    out = []
    # The Hub exposes the card-described 1,931-row validation set as its only "train" split.
    for item in stream("BSC-LT/SIQA_es", "train", 3000):
        choices = [clean(item.get(k,"")) for k in ("answer_a","answer_b","answer_c")]
        label = siqa_choice_label(item.get("label",""))
        if not all(choices) or label is None: continue
        prompt = (f"Contexto: {clean(item.get('context'))}\nPregunta: {clean(item.get('question'))}\n"
                  f"Opciones: A) {choices[0]} B) {choices[1]} C) {choices[2]}\nRespuesta:")
        out.append({"id":clean(item.get("index",len(out))),"source":"BSC-LT/SIQA_es",
                    "license":"CC-BY-4.0","language":"es","prompt":prompt,
                    "choices":choices,"answer":label,"split":"validation"})
    for item in stream("BSC-LT/openbookqa-es", "test", 1000):
        obj = item.get("choices") or {}
        texts, labels = obj.get("text",[]), obj.get("label",[])
        mapping = {str(k).upper():clean(v) for k,v in zip(labels,texts)}
        answer = clean(item.get("answerKey","")).upper()
        if answer not in mapping: continue
        prompt = f"Pregunta: {clean(item.get('question_stem'))}\nOpciones: " + " ".join(
            f"{k}) {v}" for k,v in mapping.items()) + "\nRespuesta:"
        out.append({"id":clean(item.get("id",len(out))),"source":"BSC-LT/openbookqa-es",
                    "license":"CC-BY-4.0","language":"es","prompt":prompt,
                    "choices":mapping,"answer":answer,"split":"test"})
    return out

def write(path, rows):
    h = hashlib.sha256()
    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n"
            f.write(line); h.update(line.encode())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="artifacts/spanish-v1")
    p.add_argument("--max-alia",type=int,default=40000)
    p.add_argument("--max-general",type=int,default=25000)
    p.add_argument("--max-legal",type=int,default=10000)
    p.add_argument("--max-uji",type=int,default=20000)
    a=p.parse_args(); root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=True)
    raw=build_alia(a.max_alia)+build_general(a.max_general)+build_legal(a.max_legal)+build_uji(a.max_uji)
    unique={}
    for row in raw: unique.setdefault(key(row["prompt"],row["completion"]),row)
    rows=list(unique.values())
    rows.sort(key=lambda r:hashlib.sha256((str(SEED)+r["id"]).encode()).hexdigest())
    train,val=[],[]
    for row in rows:
        bucket=int(hashlib.sha256(row["id"].encode()).hexdigest()[:8],16)%100
        (val if bucket<2 else train).append(row)
    ev=build_eval()
    hashes={"train":write(root/"train.jsonl",train),
            "validation":write(root/"validation.jsonl",val),
            "holdout_multiple_choice":write(root/"holdout_multiple_choice.jsonl",ev)}
    manifest={
        "name":"WISE-RONIN Spanish-only corpus v1","seed":SEED,"language":"es",
        "rows_raw":len(raw),"rows_deduplicated":len(rows),"train_rows":len(train),
        "validation_rows":len(val),"holdout_rows":len(ev),
        "holdout_source_counts":dict(Counter(r["source"] for r in ev)),
        "source_counts":dict(Counter(r["source"] for r in rows)),
        "task_counts":dict(Counter(r["task"] for r in rows)),
        "alia_component_counts":dict(Counter(r.get("source_component","") for r in rows if r.get("source_component"))),
        "sources":SOURCES,"evaluation_sources":{"BSC-LT/SIQA_es":"CC-BY-4.0","BSC-LT/openbookqa-es":"CC-BY-4.0"},
        "alia_component_license_allowlist":ALIA_LICENSES,"alia_components_excluded":EXCLUDED_ALIA,
        "license_policy":"Only Apache-2.0 and CC-BY-4.0 components admitted; NC, ND, CC-BY-SA, unknown and mixed-license components excluded.",
        "cleaning":["Unicode NFC","HTML entity decoding","NUL/whitespace normalization",
                    "Spanish language identification","minimum completion 12 chars",
                    "maximum prompt/completion 16,000 chars","exact prompt+completion dedup"],
        "sha256":hashes,
        "notes":["SIQA_es validation and OpenBookQA-es test remain holdout-only.",
                 "ALIA-2606 is filtered by per-component license; its top-level CC-BY-4.0 is not applied blindly.",
                 "This is an instruction/QA corpus, not a broad web-text pretraining corpus.",
                 "Legal examples remain domain-tagged and are not legal advice."]
    }
    (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"train":len(train),"validation":len(val),"holdout":len(ev),
                      "sources":manifest["source_counts"],"tasks":manifest["task_counts"],
                      "sha256":hashes},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
