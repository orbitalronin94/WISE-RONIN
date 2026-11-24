#!/usr/bin/env python3
"""Build a privacy-minimized row-level review index from a pinned JSONL export."""
import argparse, hashlib, json, re
from pathlib import Path

EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
WORD = re.compile(r"[^\W_]+", re.UNICODE)

def norm(value):
    return " ".join(str(value).split())

def language_label(text):
    words = set(WORD.findall(text.casefold()))
    es = len(words & {"que","de","la","el","en","los","una","para","con","por","como","del","este","esta","son"})
    en = len(words & {"the","of","and","to","in","is","for","with","that","this","from","are"})
    if es >= 3 and es > en: return "es_heuristic"
    if en >= 3 and en > es: return "non_es_heuristic"
    return "language_unresolved"

def classify_task(row):
    keys = {str(k).lower() for k in row}
    if {"question","answer"} <= keys: return "question_answer"
    if "instruction" in keys and ({"response","output","answer"} & keys): return "instruction_response"
    if "document" in keys or "passage" in keys: return "document_or_retrieval"
    if "tokens" in keys or "ner_tags" in keys or "entities" in keys: return "sequence_labeling"
    return "text_or_unknown"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    p.add_argument("--source",required=True); p.add_argument("--revision",required=True)
    p.add_argument("--split",default="train")
    p.add_argument("--text-fields",required=True,help="Explicit comma-separated prose field allowlist")
    p.add_argument("--base-jsonl",help="Optional frozen v3.1 JSONL for exact duplicate references")
    a=p.parse_args()
    fields=tuple(x.strip() for x in a.text_fields.split(",") if x.strip())
    if not fields: raise SystemExit("at least one text field must be allowlisted")
    base=set()
    if a.base_jsonl:
        with Path(a.base_jsonl).open(encoding="utf-8") as f:
            for line in f:
                try: row=json.loads(line)
                except Exception: continue
                chunks=[norm(row[k]) for k in fields if isinstance(row,dict) and isinstance(row.get(k),str) and norm(row[k])]
                if chunks: base.add(hashlib.sha256("\n".join(chunks).encode()).hexdigest())
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    stats={"rows_seen":0,"rows_with_text":0,"empty_text":0,"exact_duplicate_v3_1":0,
           "language_es_heuristic":0,"language_non_es_heuristic":0,"language_unresolved":0,
           "possible_email_in_text":0,"possible_nif_in_text":0,"possible_phone_in_text":0}
    with Path(a.input).open(encoding="utf-8") as src, out.open("w",encoding="utf-8") as dst:
        for index,line in enumerate(src):
            if not line.strip(): continue
            stats["rows_seen"]+=1
            try: row=json.loads(line)
            except Exception: continue
            if not isinstance(row,dict): continue
            chunks=[(field,norm(row[field])) for field in fields if isinstance(row.get(field),str) and norm(row[field])]
            if not chunks: stats["empty_text"]+=1; continue
            stats["rows_with_text"]+=1
            content="\n".join(value for _,value in chunks)
            digest=hashlib.sha256(content.encode("utf-8")).hexdigest()
            language=language_label(content); stats["language_"+language]+=1
            flags=[]
            if any(EMAIL.search(value) for _,value in chunks):
                flags.append("possible_email_in_text"); stats["possible_email_in_text"]+=1
            if any(NIF.search(value) for _,value in chunks):
                flags.append("possible_nif_in_text"); stats["possible_nif_in_text"]+=1
            if any(PHONE.search(value) for _,value in chunks):
                flags.append("possible_phone_in_text"); stats["possible_phone_in_text"]+=1
            if digest in base:
                flags.append("exact_duplicate_v3_1"); stats["exact_duplicate_v3_1"]+=1
            record={"schema":"lumen-corpus-review-index-v1","source":a.source,
                    "source_revision":a.revision,"split":a.split,"row_index":index,
                    "text_fields":[field for field,_ in chunks],"text_sha256":digest,
                    "text_chars":len(content),"whitespace_tokens":len(content.split()),
                    "language":language,"task_type":classify_task(row),"privacy_flags":flags,
                    "review_status":"pending","rights_status":"pending",
                    "privacy_status":"pending","quality_status":"pending","raw_text_exported":False}
            dst.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")
    print(json.dumps({"stats":stats,"output":str(out),"raw_text_exported":False},sort_keys=True))

if __name__=="__main__": main()
