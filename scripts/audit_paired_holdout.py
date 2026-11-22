#!/usr/bin/env python3
"""Audit paired multiple-choice holdout outputs without treating reused IDs as unique questions."""
import argparse, collections, json, math
from pathlib import Path
import numpy as np

def load_metrics(preds):
    acc=np.asarray([p["pred"] == p["gold"] for p in preds], dtype=float)
    nll=np.asarray([p["scores"][p["gold"]] for p in preds], dtype=float)
    brier=[]; confidence=[]; correct=[]
    for p in preds:
        s=np.asarray(p["scores"], dtype=float)
        z=-s; z-=z.max()
        q=np.exp(z); q/=q.sum()
        y=np.zeros(len(q)); y[int(p["gold"])]=1.0
        brier.append(float(np.sum((q-y)**2)))
        confidence.append(float(q.max()))
        correct.append(int(q.argmax()==int(p["gold"])))
    brier=np.asarray(brier); confidence=np.asarray(confidence); correct=np.asarray(correct)
    ece=0.0; bins=[]
    for j in range(10):
        lo=j/10; hi=(j+1)/10
        mask=(confidence>=lo)&((confidence<hi) if j<9 else (confidence<=hi))
        if mask.any():
            ca=float(correct[mask].mean()); cc=float(confidence[mask].mean())
            ece+=float(mask.mean())*abs(ca-cc)
            bins.append({"bin":f"{lo:.1f}-{hi:.1f}","n":int(mask.sum()),"confidence":cc,"accuracy":ca})
    return {"accuracy":float(acc.mean()),"correct":int(acc.sum()),
            "mean_gold_answer_nll":float(nll.mean()),
            "brier_score_softmax_neg_mean_nll":float(brier.mean()),
            "ece10_softmax_neg_mean_nll":float(ece),
            "mean_top_confidence":float(confidence.mean()),"bins":bins}, {
            "acc":acc,"nll":nll,"brier":brier,"confidence":confidence,"correct":correct}

def ece(confidence, correct):
    out=0.0
    for j in range(10):
        lo=j/10; hi=(j+1)/10
        mask=(confidence>=lo)&((confidence<hi) if j<9 else (confidence<=hi))
        if mask.any(): out+=float(mask.mean())*abs(float(correct[mask].mean())-float(confidence[mask].mean()))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--bootstrap",type=int,default=5000)
    args=ap.parse_args()
    raw=json.loads(Path(args.input).read_text(encoding="utf-8"))
    L=raw["results"]["lumen"]["predictions"]; T=raw["results"]["transformer"]["predictions"]
    if len(L)!=len(T): raise ValueError("Prediction arrays have different lengths")
    n=len(L)
    for i,(l,t) in enumerate(zip(L,T)):
        if str(l["id"])!=str(t["id"]) or int(l["gold"])!=int(t["gold"]):
            raise ValueError(f"Unmatched paired row at index {i}")
    ids=collections.defaultdict(list)
    for i,p in enumerate(L): ids[str(p["id"])].append(i)
    groups=list(ids.values())
    conflicting=sum(len({int(L[i]["gold"]) for i in ix})>1 for ix in groups)
    ml,al=load_metrics(L); mt,at=load_metrics(T)
    b=int(np.sum((al["acc"]==1)&(at["acc"]==0)))
    c=int(np.sum((al["acc"]==0)&(at["acc"]==1)))
    # Exact two-sided McNemar/binomial result at row level; clustering caveat is explicit below.
    from scipy.stats import binomtest
    mcnemar=float(binomtest(min(b,c),b+c,0.5,alternative="two-sided").pvalue) if b+c else 1.0
    # ID-cluster bootstrap is a sensitivity interval, not deduplication: all rows in sampled IDs remain.
    rng=np.random.default_rng(20261122); boot={k:[] for k in ("accuracy","nll","brier","ece")}
    for _ in range(args.bootstrap):
        chosen=rng.integers(0,len(groups),size=len(groups))
        ix=np.concatenate([groups[j] for j in chosen])
        boot["accuracy"].append(float(np.mean(al["acc"][ix]-at["acc"][ix])))
        boot["nll"].append(float(np.mean(al["nll"][ix]-at["nll"][ix])))
        boot["brier"].append(float(np.mean(al["brier"][ix]-at["brier"][ix])))
        boot["ece"].append(ece(al["confidence"][ix],al["correct"][ix])-ece(at["confidence"][ix],at["correct"][ix]))
    ci={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in boot.items()}
    report={
      "rows":n,"unique_id_values":len(groups),
      "duplicate_id_groups":sum(len(g)>1 for g in groups),
      "extra_rows_from_reused_ids":sum(len(g)-1 for g in groups),
      "duplicate_id_groups_with_conflicting_gold":int(conflicting),
      "lumen":ml,"transformer":mt,
      "paired_discordance":{"lumen_only_correct":b,"transformer_only_correct":c,"net_lumen_correct":b-c,
                            "mcnemar_exact_two_sided_row_level":mcnemar},
      "id_cluster_bootstrap_95ci_lumen_minus_transformer":ci,
      "method_notes":[
        "Do not deduplicate by id: repeated IDs are not unique question keys; some repeated-ID groups contain conflicting gold labels.",
        "Bootstrap resamples ID groups while retaining all rows; interpret as sensitivity analysis because ID semantics are not guaranteed to define independent clusters.",
        "Brier/ECE are heuristic diagnostics from softmax(-mean conditional answer NLL). The evaluator stores mean token-NLL choice scores, not normalized choice probabilities; these are not definitive probability-calibration measurements."
      ]
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("lumen","transformer")},indent=2))
    print("LUMEN",json.dumps({k:v for k,v in ml.items() if k!="bins"}))
    print("TRANSFORMER",json.dumps({k:v for k,v in mt.items() if k!="bins"}))
if __name__=="__main__": main()
