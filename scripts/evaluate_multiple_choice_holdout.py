"""Matched multiple-choice holdout evaluation for the fresh step-1000 pair."""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model.config import CONFIG, LumenConfig
from model.lumen import RoninLumen
from train.transformer_baseline import SmallTransformerLM

def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def normalize_answer(answer, choices):
    if isinstance(answer, int):
        return answer if 0 <= answer < len(choices) else None
    if isinstance(answer, str):
        s = answer.strip()
        if s.isdigit() and 0 <= int(s) < len(choices):
            return int(s)
        if len(s) == 1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            idx = ord(s.upper()) - 65
            if idx < len(choices): return idx
        for i, c in enumerate(choices):
            if s == str(c).strip(): return i
    return None

def score_candidate(model, tokenizer, prompt, choice, max_len, lumen):
    # Score only the candidate continuation; the prompt supplies causal context.
    prefix = tokenizer.encode(prompt + "\n", add_special_tokens=False).ids
    suffix = tokenizer.encode(str(choice), add_special_tokens=False).ids
    if not suffix: return float("inf")
    if len(suffix) >= max_len: suffix = suffix[:max_len-1]
    prefix = prefix[-(max_len-len(suffix)):]
    ids = prefix + suffix
    if len(ids) < 2: return float("inf")
    x = torch.tensor([ids], dtype=torch.long)
    if lumen:
        model.reset_states()
        logits = model(x, reset_state=True, collect_diagnostics=False)["logits"]
    else:
        logits = model(x)
    start = len(prefix)-1
    pred = logits[0, start:start+len(suffix), :]
    target = torch.tensor(suffix, dtype=torch.long)
    return float(F.cross_entropy(pred, target, reduction="mean").item())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--holdout", required=True); p.add_argument("--tokenizer", required=True)
    p.add_argument("--lumen", required=True); p.add_argument("--transformer")
    p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--expected-step", type=int, default=None, help="Legacy shared expected step")
    p.add_argument("--expected-lumen-step", type=int, default=None)
    p.add_argument("--expected-transformer-step", type=int, default=None)
    a=p.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(2)
    tok=Tokenizer.from_file(a.tokenizer)
    data=rows(a.holdout)
    if a.limit: data=data[:a.limit]
    models={}
    ls=torch.load(a.lumen,map_location="cpu",weights_only=False)
    expected_lumen = a.expected_lumen_step if a.expected_lumen_step is not None else a.expected_step
    if expected_lumen is not None and int(ls.get("step",-1)) != expected_lumen: raise ValueError(f"Lumen checkpoint is not step {expected_lumen}")
    lumen_config = LumenConfig(**ls["config"]) if isinstance(ls.get("config"), dict) else CONFIG
    lm=RoninLumen(config=lumen_config, v32_ze=bool(ls.get("v32_ze",False))).cpu()
    if ls.get("params") is not None and sum(x.numel() for x in lm.parameters()) != int(ls["params"]):
        raise ValueError("Lumen checkpoint parameter count does not match reconstructed architecture")
    lm.load_state_dict(ls["model"]); lm.eval(); models["lumen"]=(lm,True)
    if a.transformer:
        ts=torch.load(a.transformer,map_location="cpu",weights_only=False)
        expected_transformer = a.expected_transformer_step if a.expected_transformer_step is not None else a.expected_step
        if expected_transformer is not None and int(ts.get("step",-1)) != expected_transformer: raise ValueError(f"Transformer checkpoint is not step {expected_transformer}")
        spec=ts.get("model_spec", {})
        tm=SmallTransformerLM(
            CONFIG.vocab_size, a.max_len,
            int(spec.get("model_dim",192)), int(spec.get("heads",4)),
            int(spec.get("ff_width",int(spec.get("model_dim",192))*3)),
            int(spec.get("residual_groups",0)),
        ).cpu()
        if ts.get("params") is not None and sum(x.numel() for x in tm.parameters()) != int(ts["params"]):
            raise ValueError("Transformer checkpoint parameter count does not match reconstructed architecture")
        tm.load_state_dict(ts["model"]); tm.eval(); models["transformer"]=(tm,False)
    results={k:{"correct":0,"total":0,"mean_answer_nll":0.0,"predictions":[]} for k in models}
    sums={k:0.0 for k in models}
    with torch.inference_mode():
        for n,row in enumerate(data,1):
            prompt=str(row.get("prompt","")).strip()
            choices=row.get("choices",[])
            if isinstance(choices,dict): choices=[choices[k] for k in sorted(choices)]
            if not prompt or not isinstance(choices,list) or len(choices)<2: continue
            gold=normalize_answer(row.get("answer"),choices)
            if gold is None: continue
            for name,(model,is_lumen) in models.items():
                scores=[score_candidate(model,tok,prompt,c,a.max_len,is_lumen) for c in choices]
                pred=min(range(len(scores)),key=scores.__getitem__)
                r=results[name]; r["total"]+=1; r["correct"]+=int(pred==gold)
                sums[name]+=scores[gold]
                r["predictions"].append({"id":row.get("id",n),"gold":gold,"pred":pred,"scores":scores})
            if n%100==0: print(f"holdout_progress {n}/{len(data)}",flush=True)
    for name,r in results.items():
        r["accuracy"]=r["correct"]/max(1,r["total"])
        r["mean_answer_nll"]=sums[name]/max(1,r["total"])
    out={
        "holdout_rows":len(data),
        "holdout_sha256":hashlib.sha256(Path(a.holdout).read_bytes()).hexdigest(),
        "tokenizer_sha256":hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest(),
        "method":"mean conditional token NLL over each answer choice; same frozen tokenizer/context; no updates",
        "results":results,
    }
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{x:v for x,v in r.items() if x!="predictions"} for k,r in results.items()},indent=2),flush=True)
if __name__=="__main__": main()
