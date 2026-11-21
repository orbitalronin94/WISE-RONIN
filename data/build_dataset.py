"""Assemble the final 25k corpus only from approved source artifacts and owner data."""
import argparse, hashlib, json
from pathlib import Path
def read_jsonl(path):
 with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--sources",default="artifacts/lumen-source-candidates"); p.add_argument("--owner",default="data/owner"); p.add_argument("--holdout",default="data/holdout_500.jsonl"); p.add_argument("--output",default="data/train_25k.jsonl"); a=p.parse_args()
 root,owner=Path(a.sources),Path(a.owner); manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8")); rows=[]
 for src in manifest["sources"]:
  part=read_jsonl(root/src["file"])
  if len(part)!=src["quota"]: raise ValueError(f"{src['key']}: expected {src['quota']}, got {len(part)}")
  rows.extend(part)
 code=owner/"python_edu_licensed.jsonl"
 if not code.exists(): raise FileNotFoundError(f"Provide 1,000 per-file-license-audited code rows: {code}")
 code_rows=read_jsonl(code)
 if len(code_rows)!=1000: raise ValueError("Python-Edu licensed set must contain 1,000 rows")
 rows.extend(code_rows)
 curated=read_jsonl(owner/"curated_5000.jsonl")
 if len(curated)!=5000: raise ValueError("Owner-curated corpus must contain exactly 5,000 rows")
 rows.extend(curated)
 if len(rows)!=25000: raise ValueError(f"Expected 25,000 rows, got {len(rows)}")
 def digest(r): return hashlib.sha256((str(r.get("prompt",""))+"\n"+str(r.get("completion",""))).strip().lower().encode()).hexdigest()
 hold={digest(r) for r in read_jsonl(a.holdout)}; seen=set()
 for i,r in enumerate(rows):
  h=digest(r)
  if h in hold: raise ValueError(f"Exact holdout overlap at row {i}")
  if h in seen: raise ValueError(f"Duplicate row {i}")
  seen.add(h)
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("w",encoding="utf-8",newline="\n") as f:
  for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
 print(json.dumps({"rows":len(rows),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"holdout_overlap":0},indent=2))
if __name__=="__main__": main()
