#!/usr/bin/env python3
"""Stream-audit eligible Hugging Face train splits; never authorizes training."""
import argparse, hashlib, json, re, sqlite3, time
from collections import Counter
from pathlib import Path
from typing import Any
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names

ELIGIBLE = {
 "OysterCoreAI/SFT-General-Spanish-50K":"sft_chat",
 "BSC-LT/m-personas":"persona_dialogue",
 "Kukedlc/Personas_Spanish-40k":"persona_profile",
 "StrataSynth/stratasynth-personas-spain":"persona_profile_filtered",
 "somosnlp/instruct-legal-refugiados-es":"contextual_qa",
 "dariolopez/ms-marco-es-500k":"retrieval_pairwise",
 "PlanTL-GOB-ES/pharmaconer":"supervised_extraction",
 "bigbio/distemist":"supervised_extraction"
}
EXCLUDED = {
 "ola-owo/big-five-personality-traits":"psychometric_labels_excluded",
 "somosnlp/LingComp_QA":"741_rows_already_in_v3_1_no_reimport",
 "somosnlp/lenguaje-claro-dataset":"55_rows_already_in_v3_1_no_reimport",
 "PleIAs/common_corpus":"bulk_ingestion_excluded_per_document_rights_and_scale",
 "Alvaro8gb/enfermedades-wiki-marzo-2024":"upstream_article_rights_unverified",
 "bigbio/meddocan":"clinical_privacy_and_upstream_rights_review_required",
 "BARR2":"canonical_source_and_rights_unverified"
}
MANUAL = {
 "GovEasy":"manual_API_export_and_per_record_provenance",
 "ESQAD":"manual_Zenodo_split_and_generated_data_separation",
 "TDX Thesis Spanish Corpus":"manual_document_rights_and_privacy_review",
 "PleIAs/BDH-Books":"manual_item_level_public_domain_OCR_and_overlap_review",
 "Spanish-PD-Books":"manual_item_level_rights_and_cross_dedup_with_BDH"
}
EMAIL=re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
NIF=re.compile(r"\b\d{8}[A-Z]\b",re.I)
PHONE=re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
WORD=re.compile(r"[^\W_]+",re.UNICODE)

def norm(s): return " ".join(WORD.findall(s.casefold()))
def collect(v,out,depth=0):
 if depth>5 or len(out)>200:return
 if isinstance(v,str):
  if len(v.strip())>=20:out.append(v.strip())
 elif isinstance(v,dict):
  for k,x in v.items():
   if str(k).lower() not in {"id","uuid","url","link","license","language","lang","split","label","labels"}:collect(x,out,depth+1)
 elif isinstance(v,(list,tuple)):
  for x in v:collect(x,out,depth+1)
def rowtext(row):
 a=[];collect(row,a);seen=set();b=[]
 for x in a:
  x=" ".join(x.split())
  if x and x not in seen:seen.add(x);b.append(x)
 return "\n".join(b)
def lang(row,s):
 if isinstance(row,dict):
  x=str(row.get("language",row.get("lang",""))).lower()
  if x:return "es_declared" if x in {"es","es-es","spanish","spa"} or x.startswith("es-") else "non_es_declared"
 w=set(norm(s).split()); es=len(w&{"que","de","la","el","en","los","una","para","con","por","como","del","este","esta","son"})
 en=len(w&{"the","of","and","to","in","is","for","with","that","this","from","are"})
 if es>=3 and es>en:return "es_heuristic"
 if en>=3 and en>es:return "non_es_heuristic"
 return "language_unresolved"
def simhash(s):
 w=norm(s).split(); grams=[" ".join(w[i:i+5]) for i in range(max(1,len(w)-4))]
 a=[0]*64
 for g in grams:
  h=int.from_bytes(hashlib.blake2b(g.encode(),digest_size=8).digest(),"big")
  for i in range(64):a[i]+=1 if (h>>i)&1 else -1
 return sum((1<<i) for i,x in enumerate(a) if x>=0)
def audit_split(db,repo,config,split,limit,sample_n):
 stats=Counter(); samples=[]; started=time.time()
 stream=load_dataset(repo,name=None if config=="default" else config,split=split,streaming=True)
 for i,row in enumerate(stream):
  if limit and i>=limit:break
  stats["rows_seen"]+=1;s=rowtext(row)
  if not s:stats["empty_or_nontext"]+=1;continue
  n=norm(s);stats["whitespace_tokens"]+=len(n.split());stats["language_"+lang(row,s)]+=1
  flags=[]
  if EMAIL.search(s):flags.append("possible_email")
  if NIF.search(s):flags.append("possible_spanish_nif")
  if PHONE.search(s):flags.append("possible_phone")
  if len(s)>100000:flags.append("extreme_length")
  if len(s) and sum(c.isalpha() for c in s)/len(s)<.45:flags.append("low_text_density")
  for f in flags:stats["flag_"+f]+=1
  digest=hashlib.sha256(n.encode()).hexdigest()
  old=db.execute("SELECT source FROM exact WHERE digest=?",(digest,)).fetchone()
  if old:
   stats["exact_duplicate"]+=1
   if old[0]!=repo:stats["cross_source_exact_duplicate"]+=1
  else:db.execute("INSERT INTO exact VALUES (?,?)",(digest,repo))
  fp=simhash(n); near=set()
  for band in range(4):
   val=(fp>>(band*16))&65535
   for oldfp,src in db.execute("SELECT fingerprint,source FROM sim WHERE band=? AND value=?",(band,val)):
    if (fp^int(oldfp,16)).bit_count()<=3:near.add(src)
  if near:
   stats["near_duplicate_candidate"]+=1
   if any(x!=repo for x in near):stats["cross_source_near_duplicate_candidate"]+=1
  for band in range(4):
   val=(fp>>(band*16))&65535
   db.execute("INSERT INTO sim VALUES (?,?,?)",(band,val,f"{fp:016x}"))
  if len(samples)<sample_n and (i==0 or (i+1)%(max(1,limit//sample_n) if limit else 1000)==0):
   samples.append({"source":repo,"config":config,"split":split,"row_index":i,"language":lang(row,s),"flags":flags,"excerpt":s[:700]})
  if i and i%10000==0:db.commit();print(f"{repo}/{config}/{split}: {i}",flush=True)
 db.commit();stats["elapsed_seconds"]=round(time.time()-started,2)
 return dict(stats),samples
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--registry",required=True);p.add_argument("--db",required=True);p.add_argument("--output",required=True)
 p.add_argument("--max-rows-per-split",type=int,default=0);p.add_argument("--sample-per-split",type=int,default=100)
 a=p.parse_args();db=sqlite3.connect(a.db);db.execute("PRAGMA journal_mode=WAL")
 db.execute("CREATE TABLE IF NOT EXISTS exact(digest TEXT PRIMARY KEY,source TEXT)")
 db.execute("CREATE TABLE IF NOT EXISTS sim(band INTEGER,value INTEGER,fingerprint TEXT)")
 db.execute("CREATE INDEX IF NOT EXISTS sim_idx ON sim(band,value)")
 report={"training_authorized":False,"note":"All eligible train splits are streamed unless capped. Automated scan is not semantic/legal approval.","sources":{},"samples":[]}
 for repo,route in ELIGIBLE.items():
  ent={"route":route,"configs":{}}
  try:
   configs=get_dataset_config_names(repo) or ["default"]
   for config in configs:
    try:splits=get_dataset_split_names(repo,config_name=None if config=="default" else config)
    except Exception as e:ent["configs"][config]={"error":"split discovery: "+repr(e)};continue
    for split in [x for x in splits if x.lower()=="train"]:
     try:
      st,sm=audit_split(db,repo,config,split,a.max_rows_per_split,a.sample_per_split)
      ent["configs"].setdefault(config,{})[split]=st;report["samples"].extend(sm)
     except Exception as e:ent["configs"].setdefault(config,{})[split]={"error":repr(e)}
  except Exception as e:ent["error"]=repr(e)
  report["sources"][repo]=ent
 for k,v in EXCLUDED.items():report["sources"][k]={"status":"excluded_without_ingestion","reason":v}
 for k,v in MANUAL.items():report["sources"][k]={"status":"manual_adapter_required","reason":v}
 db.close();Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"sources":len(report["sources"]),"training_authorized":False}),flush=True)
if __name__=="__main__":main()
