#!/usr/bin/env python3
"""Build a frozen Spanish corpus v2 from CSIC and TDX."""
import argparse, hashlib, json, re, urllib.request
from pathlib import Path
from tokenizers import Tokenizer

SOURCES = [
 {"id":"csic_es","url":"https://zenodo.org/records/7313126/files/csic_es.txt?download=1","file":"csic_es.txt","md5":"a7103b0a8c84e5036e4b31759bd7932c","doi":"10.5281/zenodo.7313126"},
 {"id":"tdx_es","url":"https://zenodo.org/records/7313149/files/tdx_es.txt?download=1","file":"tdx_es.txt","md5":"c38ee282c65aee890db1d2491e5a45d7","doi":"10.5281/zenodo.7313149"},
]
def sha(path, alg="sha256"):
 h=hashlib.new(alg)
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
 return h.hexdigest()
def norm(s): return " ".join(s.casefold().split())
def download(url,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.stat().st_size: return
 req=urllib.request.Request(url,headers={"User-Agent":"WISE-RONIN-corpus-builder/2"})
 with urllib.request.urlopen(req,timeout=120) as r,path.open("wb") as w:
  while True:
   b=r.read(4*1024*1024)
   if not b: break
   w.write(b)
def read_rows(path):
 with Path(path).open(encoding="utf-8") as f:
  for line in f:
   if line.strip(): yield json.loads(line)
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--base-dir",required=True); p.add_argument("--tokenizer",required=True)
 p.add_argument("--source-dir",default="artifacts/source-downloads"); p.add_argument("--output-dir",default="artifacts/spanish-v2")
 a=p.parse_args()
 base=Path(a.base_dir); srcdir=Path(a.source_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 tok=Tokenizer.from_file(a.tokenizer); max_chunk=126
 base_manifest=json.loads((base/"manifest.json").read_text(encoding="utf-8"))
 seen=set()
 for row in read_rows(base/"train.jsonl"):
  v=norm(row.get("completion",""))
  if v: seen.add(hashlib.sha256(v.encode()).hexdigest())
 train_hash=hashlib.sha256(); base_n=0; new_n=0; stats={}
 with (out/"train.jsonl").open("w",encoding="utf-8") as dst:
  def put(row):
   nonlocal_dummy=None
   line=json.dumps(row,ensure_ascii=False,sort_keys=True)+"\\n"
   dst.write(line); train_hash.update(line.encode())
  for row in read_rows(base/"train.jsonl"): put(row); base_n+=1
  for s in SOURCES:
   path=srcdir/s["file"]; download(s["url"],path)
   md5=sha(path,"md5")
   if md5!=s["md5"]: raise ValueError(f"MD5 mismatch for {s['id']}: {md5}")
   doc_count=chunk_count=short=dup=token_count=0; doc_seen=set()
   with path.open(encoding="utf-8",errors="replace") as f:
    for raw in f:
     text=re.sub(r"\\s+"," ",raw).strip()
     if len(text)<120: short+=1; continue
     doc_count+=1; dh=hashlib.sha256(norm(text).encode()).hexdigest()
     if dh in doc_seen: dup+=1; continue
     doc_seen.add(dh)
     ids=tok.encode(text).ids
     if ids and ids[0]==tok.token_to_id("[BOS]"): ids=ids[1:]
     if ids and ids[-1]==tok.token_to_id("[EOS]"): ids=ids[:-1]
     token_count+=len(ids)
     for off in range(0,len(ids),max_chunk):
      part=ids[off:off+max_chunk]
      if len(part)<16: continue
      piece=tok.decode(part,skip_special_tokens=True).strip()
      if len(piece)<60: continue
      key=hashlib.sha256(norm(piece).encode()).hexdigest()
      if key in seen: continue
      seen.add(key)
      put({"id":hashlib.sha256((s["id"]+dh+str(off)).encode()).hexdigest()[:20],"prompt":"","completion":piece,"task":"raw_text_pretraining","language":"es","source":s["id"],"source_id":dh,"license":"CC-BY-4.0","source_doi":s["doi"],"chunk_index":off//max_chunk})
      chunk_count+=1
   new_n+=chunk_count
   stats[s["id"]]={"doi":s["doi"],"license":"CC-BY-4.0","url":s["url"],"md5_verified":md5,"sha256":sha(path),"bytes":path.stat().st_size,"documents":doc_count,"duplicates":dup,"short_skipped":short,"bpe_tokens":token_count,"chunks_added":chunk_count}
 for name in ("validation.jsonl","holdout_multiple_choice.jsonl"):
  payload=(base/name).read_bytes(); (out/name).write_bytes(payload)
 manifest={"name":"WISE-RONIN Spanish shared corpus v2","version":"2.0.0","status":"FROZEN_BUILD_OUTPUT","language":"es","shared_by":["Lumen","Transformer"],"tokenizer_sha256":sha(a.tokenizer),"tokenizer_vocab_size":tok.get_vocab_size(),"max_sequence_length":128,"train_rows":base_n+new_n,"base_train_rows":base_n,"new_chunks":new_n,"validation_rows":sum(1 for _ in read_rows(out/"validation.jsonl")),"holdout_rows":sum(1 for _ in read_rows(out/"holdout_multiple_choice.jsonl")),"sources":stats,"excluded":{"CoWeSe":{"doi":"10.5281/zenodo.5513237","status":"QUARANTINED","reason":"Dataset authors state they do not own underlying crawled texts; rights in source documents are not established."}},"sha256":{"train.jsonl":train_hash.hexdigest(),"validation.jsonl":sha(out/"validation.jsonl"),"holdout_multiple_choice.jsonl":sha(out/"holdout_multiple_choice.jsonl")},"notes":["Validation and holdout copied byte-for-byte from v1.","CSIC and TDX verified against published MD5 before ingestion.","Raw documents split into <=126 BPE-token chunks for the 128-token context.","Chunks retain source, DOI, license, source-document hash and chunk index.","Identical artifact for Lumen and Transformer."]}
 (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
 (out/"tokenizer.json").write_bytes(Path(a.tokenizer).read_bytes())
 print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
