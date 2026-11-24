#!/usr/bin/env python3
"""Deterministic first-pass audit for normalized v3.2 JSONL candidates.

This script does not infer legal permission or semantic quality. It records
exact duplicates, near-duplicate candidates, obvious malformed rows and
high-risk identifier patterns for human review.
"""
from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from collections import defaultdict
from pathlib import Path

EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
NIF = re.compile(r"\b\d{8}[A-Z]\b", re.I)
URL = re.compile(r"https?://\S+", re.I)

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = URL.sub(" URL ", s)
    return " ".join(re.sub(r"[^\w]+", " ", s).split())

def text_of(row: dict) -> str:
    parts=[]
    for k in ("prompt","completion","text","content","question","answer","document"):
        v=row.get(k)
        if isinstance(v,str) and v.strip(): parts.append(v.strip())
    if not parts:
        for v in row.values():
            if isinstance(v,str) and v.strip(): parts.append(v.strip())
    return "\n".join(parts)

def shingles(s: str, n: int=5) -> set[str]:
    w=s.split()
    if not w: return set()
    if len(w)<=n: return {" ".join(w)}
    return {" ".join(w[i:i+n]) for i in range(len(w)-n+1)}

def jaccard(a:set[str],b:set[str])->float:
    if not a or not b: return 0.0
    return len(a&b)/len(a|b)

def read_jsonl(path:Path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if line.strip():
                try: rows.append((line_no,json.loads(line)))
                except json.JSONDecodeError as e:
                    rows.append((line_no,{"__parse_error__":str(e),"raw":line[:500]}))
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",action="append",default=[],help="Existing normalized JSONL; repeatable")
    p.add_argument("--candidate",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--near-threshold",type=float,default=0.88)
    p.add_argument("--max-rows",type=int,default=0,help="0 means all rows")
    a=p.parse_args()
    base_rows=[]
    for path in a.base: base_rows.extend(read_jsonl(Path(path)))
    cand=read_jsonl(Path(a.candidate))
    if a.max_rows: cand=cand[:a.max_rows]
    exact={}
    base_shingles=[]
    for _,r in base_rows:
        s=norm(text_of(r))
        if s:
            exact[hashlib.sha256(s.encode()).hexdigest()]=r.get("id")
            base_shingles.append((shingles(s),r.get("id")))
    seen={}
    out=[]
    stats=defaultdict(int)
    for line_no,row in cand:
        if "__parse_error__" in row:
            decision="reject_quality"; reason="invalid_json"
            raw=""
        else:
            raw=text_of(row); s=norm(raw)
            digest=hashlib.sha256(s.encode()).hexdigest() if s else ""
            if not s:
                decision="reject_quality"; reason="empty_text"
            elif digest in exact or digest in seen:
                decision="reject_duplicate"; reason="exact_duplicate"
            elif EMAIL.search(raw) or NIF.search(raw) or PHONE.search(raw):
                decision="quarantine_review"; reason="possible_personal_identifier"
            else:
                sh=shingles(s)
                best=0.0; match=None
                # Deterministic all-against-base comparison; candidate scale
                # should be chunked by source on very large corpora.
                for other,oid in base_shingles:
                    score=jaccard(sh,other)
                    if score>best: best,match=score,oid
                for other,oid in seen.values():
                    score=jaccard(sh,other)
                    if score>best: best,match=score,oid
                if best>=a.near_threshold:
                    decision="quarantine_review"; reason=f"near_duplicate_jaccard_{best:.4f}"
                else:
                    decision="quarantine_review"; reason="requires_semantic_and_rights_review"
                seen[digest]=(sh,row.get("id"))
        stats[decision]+=1
        out.append({"line":line_no,"id":row.get("id"),"source":row.get("source") or row.get("source_key"),
                    "decision":decision,"reason":reason,"text_sha256":hashlib.sha256(norm(raw).encode()).hexdigest() if raw else None})
    payload={"candidate":a.candidate,"base_files":a.base,"rows_audited":len(cand),
             "near_duplicate_threshold":a.near_threshold,"counts":dict(stats),"training_authorized":False,"rows":out}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k!="rows"},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
