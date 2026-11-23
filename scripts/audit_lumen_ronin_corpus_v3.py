#!/usr/bin/env python3
"""Global admission audit for frozen Lumen-Ronin v3 against frozen v2."""
from __future__ import annotations
import argparse, collections, hashlib, json, re, sqlite3, unicodedata, random
from pathlib import Path

NEW_SOURCES = {
 "RONIN-IA Spanish public-domain literature train":"literature",
 "BSC-LT/m-personas":"m_personas",
 "MINIRONIN verifiable personality corpus adapted for Lumen-Ronin":"verifiable_personality",
 "MINIRONIN Spanish synthetic curriculum adapted for Lumen-Ronin":"synthetic_curricula",
 "Lumen-Ronin canonical behavior specification":"canonical_identity",
 "BSC-LT/ALIA-2606-SFT adapted for Lumen-Ronin":"legacy_identity_adaptation",
}
def norm(s):
    return " ".join(unicodedata.normalize("NFC",str(s or "")).casefold().split())
def h(s):
    return hashlib.sha256(norm(s).encode("utf-8")).hexdigest()
def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        for ln,line in enumerate(f,1):
            if line.strip():
                try: yield ln,json.loads(line)
                except Exception as e: raise RuntimeError(f"{path}:{ln}: invalid JSON: {e}")
def bucket(row):
    src=str(row.get("source",""))
    if src in NEW_SOURCES: return NEW_SOURCES[src]
    if row.get("source_component")=="oasst2-selfidentity-rephrase-generic-cleaned": return "legacy_identity_generic"
    if row.get("task")=="identity_and_behavior" and src=="project-authored": return "canonical_identity"
    return None
def shingles(text):
    toks=re.findall(r"\w+",norm(text),flags=re.UNICODE)
    if len(toks)<12: return set()
    return {" ".join(toks[i:i+3]) for i in range(len(toks)-2)}
def anchors(text):
    toks=re.findall(r"\w+",norm(text),flags=re.UNICODE)
    if len(toks)<12: return []
    last=len(toks)-3
    positions=sorted(set((0,last//3,(2*last)//3,last)))
    return [(int.from_bytes(hashlib.blake2b(" ".join(toks[i:i+3]).encode(),digest_size=8).digest(),"big") & ((1<<63)-1)) for i in positions]
def sha(path):
    x=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): x.update(b)
    return x.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v2",required=True); ap.add_argument("--v3",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--sample-size",type=int,default=80)
    a=ap.parse_args(); v2=Path(a.v2); v3=Path(a.v3); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    m2=json.loads(next(v2.rglob("manifest.json")).read_text(encoding="utf-8"))
    m3=json.loads((v3/"manifest.json").read_text(encoding="utf-8"))
    for name,expected in m3["sha256"].items():
        actual=sha(v3/name)
        if actual!=expected: raise RuntimeError(f"v3 hash mismatch {name}: {actual} != {expected}")
    for name in ("validation.jsonl","holdout_multiple_choice.jsonl","tokenizer.json"):
        if sha(v2/name)!=sha(v3/name): raise RuntimeError(f"Protected asset changed: {name}")
    db=sqlite3.connect(out/"audit-index.sqlite")
    db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF"); db.execute("PRAGMA temp_store=MEMORY"); db.execute("PRAGMA cache_size=-300000")
    db.executescript("""
      CREATE TABLE pair_hash(k TEXT PRIMARY KEY);
      CREATE TABLE completion_hash(k TEXT PRIMARY KEY);
      CREATE TABLE sim_text(id INTEGER PRIMARY KEY, text TEXT NOT NULL);
      CREATE TABLE sim_band(val INTEGER, rid INTEGER);
      CREATE INDEX sim_lookup ON sim_band(val);
    """)
    counts=collections.Counter(); samples=collections.defaultdict(list); dup_pair=[]; dup_completion=[]; near=[]; eval_hits=[]
    # Index the full frozen v2 train: exact pair, exact completion and LSH for long answers.
    batch_pair=[]; batch_comp=[]; batch_text=[]; batch_bands=[]
    base_count=0; base_sig_count=0
    for _,r in rows(v2/"train.jsonl"):
        base_count+=1
        p=str(r.get("prompt","") or ""); c=str(r.get("completion","") or "")
        batch_pair.append((h(p+"\n"+c),))
        if len(norm(c))>=40: batch_comp.append((h(c),))
        anch=anchors(c)
        if anch:
            rid=base_count
            batch_text.append((rid,c))
            batch_bands.extend((val,rid) for val in anch)
            base_sig_count+=1
        if len(batch_pair)>=5000:
            db.executemany("INSERT OR IGNORE INTO pair_hash VALUES(?)",batch_pair)
            db.executemany("INSERT OR IGNORE INTO completion_hash VALUES(?)",batch_comp)
            db.executemany("INSERT INTO sim_text VALUES(?,?)",batch_text)
            db.executemany("INSERT INTO sim_band VALUES(?,?)",batch_bands)
            db.commit(); batch_pair=[]; batch_comp=[]; batch_text=[]; batch_bands=[]
    if batch_pair:
        db.executemany("INSERT OR IGNORE INTO pair_hash VALUES(?)",batch_pair)
        db.executemany("INSERT OR IGNORE INTO completion_hash VALUES(?)",batch_comp)
        db.executemany("INSERT INTO sim_text VALUES(?,?)",batch_text)
        db.executemany("INSERT INTO sim_band VALUES(?,?)",batch_bands); db.commit()
    if base_count!=m2["train_rows"]: raise RuntimeError(f"v2 train rows {base_count} != manifest {m2['train_rows']}")
    # Exact protected-evaluation hashes, including every MC answer option.
    eval_pair=set(); eval_comp=set(); eval_prompts=set()
    for fn in ("validation.jsonl","holdout_multiple_choice.jsonl"):
        for _,r in rows(v2/fn):
            p=str(r.get("prompt",r.get("question","")) or "")
            c=str(r.get("completion","") or "")
            if p: eval_prompts.add(h(p))
            if p or c: eval_pair.add(h(p+"\n"+c))
            if c: eval_comp.add(h(c))
            choices=r.get("choices",[])
            if isinstance(choices,dict): choices=[choices[k] for k in sorted(choices)]
            if isinstance(choices,list):
                for choice in choices:
                    choice=str(choice or "")
                    if choice:
                        eval_pair.add(h(p+"\n"+choice))
                        eval_comp.add(h(choice))
    # Scan only additions, not the copied v2 rows.
    new_seen=set(); add_n=0; bad=[]; task_counts=collections.Counter()
    rng=random.Random(20261123)
    for line_no,r in rows(v3/"train.jsonl"):
        cat=bucket(r)
        if cat is None: continue
        add_n+=1; counts[cat]+=1; task_counts[str(r.get("task",""))]+=1
        p=str(r.get("prompt","") or ""); c=str(r.get("completion","") or "")
        pair=h(p+"\n"+c); ch=h(c) if c else None
        if db.execute("SELECT 1 FROM pair_hash WHERE k=?",(pair,)).fetchone():
            dup_pair.append({"category":cat,"id":r.get("id"),"source_id":r.get("source_id"),"line":line_no})
        if ch and db.execute("SELECT 1 FROM completion_hash WHERE k=?",(ch,)).fetchone():
            dup_completion.append({"category":cat,"id":r.get("id"),"source_id":r.get("source_id"),"line":line_no})
        if pair in eval_pair or (ch and ch in eval_comp) or h(p) in eval_prompts:
            eval_hits.append({"category":cat,"id":r.get("id"),"line":line_no})
        if pair in new_seen: counts["duplicate_within_additions"]+=1
        new_seen.add(pair)
        # LSH near-duplicate search against v2 plus previously indexed additions.
        anch=anchors(c)
        if anch:
            cand=set()
            for val in anch:
                cand.update(x[0] for x in db.execute("SELECT rid FROM sim_band WHERE val=? LIMIT 1001",(val,)))
                if len(cand)>1000: break
            if len(cand)<=1000:
                sh=shingles(c)
                for rid in cand:
                    old=db.execute("SELECT text FROM sim_text WHERE id=?",(rid,)).fetchone()
                    if not old: continue
                    osh=shingles(old[0])
                    if not osh: continue
                    inter=len(sh & osh); jac=inter/max(1,len(sh|osh))
                    ratio=min(len(norm(c)),len(norm(old[0])))/max(1,max(len(norm(c)),len(norm(old[0]))))
                    if jac>=0.88 and ratio>=0.80:
                        near.append({"category":cat,"id":r.get("id"),"matched_base_or_prior_row":rid,"jaccard":round(jac,4),"length_ratio":round(ratio,4),"sample_completion":c[:500],"matched_completion":old[0][:500]})
                        break
            else: counts["near_duplicate_queries_overflow"]+=1
            rid=base_count+add_n
            db.execute("INSERT INTO sim_text VALUES(?,?)",(rid,c))
            db.executemany("INSERT INTO sim_band VALUES(?,?)",[(val,rid) for val in anch])
        # Automated data hygiene checks.
        if not p.strip() and cat!="literature": bad.append({"category":cat,"id":r.get("id"),"issue":"empty_prompt"})
        if not c.strip(): bad.append({"category":cat,"id":r.get("id"),"issue":"empty_completion"})
        if len(c)>16000 or len(p)>16000: bad.append({"category":cat,"id":r.get("id"),"issue":"oversize"})
        if re.search(r"(?i)\b(api[_ -]?key|secret|password|BEGIN (RSA|OPENSSH) PRIVATE KEY)\b",p+"\n"+c):
            bad.append({"category":cat,"id":r.get("id"),"issue":"possible_secret_marker"})
        if len(samples[cat])<a.sample_size:
            samples[cat].append({"id":r.get("id"),"source_id":r.get("source_id"),"task":r.get("task"),"prompt":p[:1200],"completion":c[:1200],"license":r.get("license"),"source_url":r.get("source_url")})
        elif rng.random()<0.015:
            samples[cat][rng.randrange(len(samples[cat]))]={"id":r.get("id"),"source_id":r.get("source_id"),"task":r.get("task"),"prompt":p[:1200],"completion":c[:1200],"license":r.get("license"),"source_url":r.get("source_url")}
        if add_n%10000==0:
            db.commit()
            print(json.dumps({"scanned_additions":add_n,"counts":dict(counts),"exact_pair_duplicates":len(dup_pair),"exact_completion_duplicates":len(dup_completion),"near_duplicates":len(near)}),flush=True)
    db.commit()
    # Persist a reviewable stratified sample and detailed findings.
    sample_file=out/"sample_review.jsonl"
    with sample_file.open("w",encoding="utf-8") as f:
        for cat in sorted(samples):
            for item in samples[cat]: f.write(json.dumps({"category":cat,**item},ensure_ascii=False)+"\n")
    report={
      "audit":"Lumen-Ronin Spanish corpus v3 admission audit","status":"COMPLETED_WITH_GATES",
      "v2":{"rows":base_count,"manifest_rows":m2["train_rows"],"train_sha256":sha(v2/"train.jsonl")},
      "v3":{"rows":m3["train_rows"],"manifest_rows":m3["train_rows"],"train_sha256":sha(v3/"train.jsonl")},
      "protected_assets":{"validation_identical":True,"holdout_identical":True,"tokenizer_identical":True,"validation_rows":m3["validation_rows"],"holdout_rows":m3["holdout_rows"]},
      "additions":{"scanned_rows":add_n,"by_category":dict(counts),"by_task":dict(task_counts)},
      "exact_dedup":{"pair_duplicates_vs_v2":len(dup_pair),"completion_duplicates_vs_v2_len_ge_40":len(dup_completion),"duplicates_within_additions":counts["duplicate_within_additions"],"examples":dup_pair[:100]},
      "evaluation_contamination":{"exact_hits_vs_validation_or_holdout":len(eval_hits),"examples":eval_hits[:100]},
      "near_duplicates":{"method":"Four evenly spaced word-trigram anchors per completion for indexed candidate retrieval; candidate confirmation by full word-trigram Jaccard >=0.88 and normalized length ratio >=0.80. This is a high-precision screening, not exhaustive paraphrase detection.","confirmed":len(near),"queries_over_1000_candidates":counts["near_duplicate_queries_overflow"],"examples":near[:200]},
      "hygiene":{"issues":len(bad),"examples":bad[:200]},
      "samples_file":"sample_review.jsonl",
      "source_provenance":{
        "literature":"Upstream manifest records 608 train documents, immutable source revisions and public-domain text rights; v3 rows currently retain only segment indices, not document IDs/hashes. Provenance linkage must be restored before admission.",
        "m_personas":"Build manifest pins BSC-LT/m-personas revision 6fab205 and declares Apache-2.0; sample review and demographic/occupational bias assessment required.",
        "personality":"Build log pins MINIRONIN feat/technopolitics-corpus-mix at 44d4a5b99d00aaab038c0a63ea740bcd899282d5; source manifest has 50k records, 40 fact cards and seven institutional URLs. It is not 50k independent facts.",
        "synthetic_curricula":"Build log pins MINIRONIN main at bd40e34aca78dfb95cad48b33ae99a804cee3633; generated by project script from 24x10 reviewed seed cases, not independent factual corpus.",
        "legacy_identity":"Adapted responses are project-authored; source lineage is BSC-LT/ALIA-2606-SFT. Verify upstream license and adaptation notice before reuse."
      },
      "authorization":{"training_authorized":False,"reason":"Requires zero unresolved exact duplicates, review of near-duplicate findings and samples, repair of literature document-level provenance, and per-source rights review. Automated checks are not legal or semantic certification."}
    }
    (out/"audit-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    db.close()
if __name__=="__main__": main()
