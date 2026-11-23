#!/usr/bin/env python3
"""Build v3.1 in one pass: remediate flagged records and import available Spanish open datasets."""
import argparse, hashlib, json, re, shutil, unicodedata, collections
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names
from tokenizers import Tokenizer

FLAG = re.compile(r"(?i)(api[_ -]?key|secret|password|BEGIN (?:RSA|OPENSSH) PRIVATE KEY|\\b(?:dni|n[úu]mero de identidad)\\s*[:#])")
def norm(s): return " ".join(unicodedata.normalize("NFC",str(s or "")).casefold().split())
def h(s): return hashlib.sha256(norm(s).encode()).hexdigest()
def sha(p):
 x=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): x.update(b)
 return x.hexdigest()
def read(path):
 with open(path,encoding="utf-8") as f:
  for line in f:
   if line.strip(): yield json.loads(line)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--v2",required=True); ap.add_argument("--v3",required=True); ap.add_argument("--out",required=True)
 a=ap.parse_args(); v2=Path(a.v2); v3=Path(a.v3); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
 tok=Tokenizer.from_file(str(v3/"tokenizer.json"))
 # Build exact indexes over the frozen v2+v3 baseline. Keep source records only when no flag.
 seen=set(); completions=set(); kept=0; removed_flags=collections.Counter(); removed_dupes=collections.Counter()
 base_train=out/"train.jsonl"
 with open(base_train,"w",encoding="utf-8") as dst:
  for row in read(v3/"train.jsonl"):
   p=str(row.get("prompt","") or ""); c=str(row.get("completion","") or "")
   src=str(row.get("source","unknown"))
   if FLAG.search(p+"\n"+c):
    removed_flags[src]+=1; continue
   key=h(p+"\n"+c)
   if key in seen: removed_dupes["baseline_exact"]+=1; continue
   seen.add(key)
   if c: completions.add(h(c))
   dst.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n"); kept+=1
 added=collections.Counter(); skipped=collections.Counter()
 def emit(dst,prompt,completion,source,source_id,license,task,extra=None):
  p=str(prompt or "").strip(); c=str(completion or "").strip()
  if len(c)<30 or len(c)>16000 or len(p)>16000: skipped[source+":length"]+=1; return
  if FLAG.search(p+"\n"+c): removed_flags[source]+=1; return
  key=h(p+"\n"+c)
  if key in seen or h(c) in completions: skipped[source+":duplicate"]+=1; return
  seen.add(key); completions.add(h(c))
  r={"id":source.replace("/","-")+"-"+str(source_id),"prompt":p,"completion":c,"task":task,"language":"es","source":source,"source_id":str(source_id),"license":license}
  if extra: r.update(extra)
  dst.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n"); added[source]+=1
 def ds_try(repo, split, config=None):
  try: return load_dataset(repo,config,split=split) if config else load_dataset(repo,split=split)
  except Exception as e:
   print("SOURCE_UNAVAILABLE",repo,split,repr(e)[:500],flush=True); return None
 with open(base_train,"a",encoding="utf-8") as dst:
  # MGSM8KInstruct: inspect all fields and map Spanish prompt/accepted response.
  d=ds_try("Mathoctopus/GSM8KInstruct_Parallel","train")
  if d is not None:
   print("MGSM_COLUMNS",d.column_names,flush=True)
   for i,r in enumerate(d):
    lang=str(r.get("language",r.get("lang",r.get("locale","")))).lower()
    if lang not in ("es","spa","spanish","español","es-es"): continue
    p=next((r[k] for k in ("prompt","instruction","question","input") if r.get(k)), "")
    c=next((r[k] for k in ("chosen","response","output","answer","completion") if r.get(k)), "")
    emit(dst,p,c,"Mathoctopus/GSM8KInstruct_Parallel",r.get("id",i),"Apache-2.0","math_reasoning")
  # SLR Spanish: only train split; preserve validation program as metadata, not target text.
  d=ds_try("AIML-TUDA/SLR-Bench-Spanish","train","v1-All")
  if d is not None:
   print("SLR_COLUMNS",d.column_names,flush=True)
   for i,r in enumerate(d):
    p=r.get("prompt",""); c=r.get("ground-truth rule",r.get("ground_truth_rule",""))
    emit(dst,p,c,"AIML-TUDA/SLR-Bench-Spanish",r.get("id",i),"CC-BY-4.0","inductive_logic",
         {"validation_program":str(r.get("validation program",r.get("validation_program","")))[:4000],
          "curriculum_level":r.get("curriculum level",r.get("curriculum_level"))})
  # ALIA cultural heritage: raw Spanish text corpus; source-level CC BY-SA 4.0.
  d=ds_try("SINAI/ALIA-es-cultural-heritage","train")
  if d is not None:
   print("ALIA_HERITAGE_COLUMNS",d.column_names,flush=True)
   for i,r in enumerate(d):
    txt=r.get("text","")
    if not txt: continue
    ids=tok.encode(str(txt)).ids
    for j in range(0,len(ids),180):
     chunk=tok.decode(ids[j:j+180],skip_special_tokens=True).strip()
     if len(chunk)>=100:
      emit(dst,"",chunk,"SINAI/ALIA-es-cultural-heritage",f"{r.get('id',i)}:{j//180}","CC-BY-SA-4.0","raw_text_pretraining",
           {"source_id_dataset":str(r.get("source_id",""))})
  # CasiMedicos Spanish MIR SQuAD dataset, train only.
  d=ds_try("HiTZ/casimedicos-squad","train")
  if d is not None:
   print("CASIMEDICOS_COLUMNS",d.column_names,flush=True)
   for i,r in enumerate(d):
    ctx=str(r.get("context","") or "")
    q=str(r.get("question","") or "")
    ans=r.get("answers",{})
    if isinstance(ans,dict):
     texts=ans.get("text",[])
     c=" ".join(str(x) for x in texts if x)
    else: c=str(ans or "")
    emit(dst,(ctx+"\nPregunta: "+q).strip(),c,"HiTZ/casimedicos-squad",r.get("id",i),"CC-BY-4.0","clinical_reasoning")
 # Copy protected files from v3 base, byte identical.
 for name in ("validation.jsonl","holdout_multiple_choice.jsonl","tokenizer.json"):
  shutil.copy2(v3/name,out/name)
 # Verify unchanged against v2.
 for name in ("validation.jsonl","holdout_multiple_choice.jsonl","tokenizer.json"):
  if sha(out/name)!=sha(v2/name): raise RuntimeError("Protected file mismatch: "+name)
 total=sum(1 for _ in read(base_train))
 # Full token count is expensive but this is the one-pass build; count exact BPE IDs.
 token_count=0; lengths=collections.Counter()
 for row in read(base_train):
  text=(str(row.get("prompt",""))+" "+str(row.get("completion",""))).strip()
  n=len(tok.encode(text).ids); token_count+=n; lengths["records"]+=1
 manifest={"name":"Lumen-Ronin Spanish corpus v3.1","version":"3.1.0","status":"BUILT_NEEDS_FINAL_AUDIT",
  "base_version":"3.0.0","base_train_rows":9933007,"train_rows":total,
  "new_rows":dict(added),"rows_removed_flagged":dict(removed_flags),"duplicates_skipped":dict(removed_dupes+skipped),
  "tokenizer_vocab":4096,"tokenizer_sha256":sha(out/"tokenizer.json"),"train_token_count_exact":token_count,
  "validation_rows":sum(1 for _ in read(out/"validation.jsonl")),"holdout_rows":sum(1 for _ in read(out/"holdout_multiple_choice.jsonl")),
  "protected_assets_identical_to_v2":True,
  "sha256":{n:sha(out/n) for n in ("train.jsonl","validation.jsonl","holdout_multiple_choice.jsonl","tokenizer.json")},
  "authorization":{"training_authorized":False,"reason":"One-pass build completed. Exact dedup and alert filtering applied; full near-duplicate/contamination and source-rights audit still required."},
  "sources":{"ALIA_es_cultural_heritage":"CC-BY-SA-4.0; raw Spanish train split only; source_id retained",
  "SLR-Bench-Spanish":"CC-BY-4.0; v1-All train only; validation program retained as metadata",
  "MGSM8KInstruct_Parallel":"Apache-2.0 declared upstream; Spanish train examples only if language field is explicit",
  "HiTZ/casimedicos-squad":"CC-BY-4.0; train split only"}}
 (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__": main()
