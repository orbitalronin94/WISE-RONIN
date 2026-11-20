#!/usr/bin/env python3
"""Structural gates for the WISE-RONIN pre-30k correction pack. Never trains."""
import json
from pathlib import Path

root=Path("data/correction")
def rows(name):
    return [json.loads(x) for x in (root/name).read_text(encoding="utf-8").splitlines() if x.strip()]
sft=rows("correction_sft_train.jsonl")
prefs=rows("correction_orpo_train.jsonl")
hold=rows("correction_holdout.jsonl")
report=json.loads((root/"correction_report.json").read_text(encoding="utf-8"))
assert sft and prefs and hold
train_users={r["user"] for r in sft}
hold_users={r["user"] for r in hold}
# A canonical fact held out by group must not leak into SFT or preference training.
train_fact_ids={r.get("fact_id") for r in sft if r.get("fact_id")}
hold_fact_ids={r.get("fact_id") for r in hold if r.get("fact_id")}
assert not (train_fact_ids & hold_fact_ids), "fact_id leakage"
overlap=sorted(train_users & hold_users)
if overlap:
    print("EXACT_PROMPT_OVERLAP:", json.dumps(overlap, ensure_ascii=False))
    raise AssertionError("exact prompt leakage")
assert all(r.get("chosen") and r.get("rejected") and r["chosen"] != r["rejected"] for r in prefs)
assert all(r.get("assistant") and r.get("user") for r in sft)
assert all(r.get("category") for r in hold)
# Antisycophancy/false-premise records must not inherit factual numeric filler.
anti=[r for r in sft if r.get("tipo")=="antisycophancy"]
assert anti and all(not any(ch.isdigit() for ch in r["assistant"]) for r in anti)
# Every ORPO pair is an actual contrast, not the old generic fallback.
assert all(r["rejected"] != "El que contesta." for r in prefs)
assert report["heldout_fact_groups"] >= 6
assert report["holdout_math_rows"] >= 100
assert report["fact_group_overlap"] == 0
print(json.dumps({
 "status":"PASS",
 "sft_rows":len(sft),"orpo_pairs":len(prefs),"holdout_rows":len(hold),
 "heldout_fact_groups":report["heldout_fact_groups"],
 "holdout_math_rows":report["holdout_math_rows"],
 "holdout_behavior_rows":report["holdout_behavior_rows"],
 "training_launched":False
},ensure_ascii=False,indent=2))
