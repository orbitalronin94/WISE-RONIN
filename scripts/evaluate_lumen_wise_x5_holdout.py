"""Matched multiple-choice holdout evaluation for the fresh step-25000 pair."""
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
    p.add_argument("--lumen-config", default="config/lumen_wise_x5.json")
    p.add_argument("--transformer-model-dim", type=int, default=488)
    p.add_argument("--transformer-heads", type=int, default=8)
    p.add_argument("--transformer-ff-width", type=int, default=1464)
    p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--expected-holdout-sha256", default=None)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--expected-step", type=int, default=None, help="Legacy shared expected step")
    p.add_argument("--expected-lumen-step", type=int, default=None)
    p.add_argument("--expected-transformer-step", type=int, default=None)
    a=p.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(2)
    tokenizer_sha=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    holdout_sha=hashlib.sha256(Path(a.holdout).read_bytes()).hexdigest()
    if a.expected_holdout_sha256 and holdout_sha != a.expected_holdout_sha256:
        raise ValueError(f"Holdout SHA-256 mismatch: {holdout_sha}")
    tok=Tokenizer.from_file(a.tokenizer)
    data=rows(a.holdout)
    if a.limit: data=data[:a.limit]
    models={}
    lumen_config=LumenConfig(**json.loads(Path(a.lumen_config).read_text(encoding="utf-8")))
    lm=RoninLumen(config=lumen_config, v32_ze=True).cpu()
    ls=torch.load(a.lumen,map_location="cpu",weights_only=False)
    expected_lumen = a.expected_lumen_step if a.expected_lumen_step is not None else a.expected_step
    if expected_lumen is not None and int(ls.get("step",-1)) != expected_lumen: raise ValueError(f"Lumen checkpoint is not step {expected_lumen}")
    if ls.get("v32_ze") is not True: raise ValueError("Lumen checkpoint lacks V3.2 ZE")
    if ls.get("architecture_id") != "ronin_lumen_v3_3_neuroplasticity": raise ValueError("Unexpected Lumen architecture ID")
    if ls.get("tokenizer_sha256") != tokenizer_sha: raise ValueError("Lumen tokenizer SHA mismatch")
    if int(ls.get("params",-1)) != sum(p.numel() for p in lm.parameters()): raise ValueError("Lumen checkpoint parameter count mismatch")
    lm.load_state_dict(ls["model"]); lm.eval(); models["lumen"]=(lm,True)
    if a.transformer:
        tm=SmallTransformerLM(CONFIG.vocab_size,a.max_len,a.transformer_model_dim,a.transformer_heads,a.transformer_ff_width).cpu()
        ts=torch.load(a.transformer,map_location="cpu",weights_only=False)
        expected_transformer = a.expected_transformer_step if a.expected_transformer_step is not None else a.expected_step
        if expected_transformer is not None and int(ts.get("step",-1)) != expected_transformer: raise ValueError(f"Transformer checkpoint is not step {expected_transformer}")
        if int(ts.get("params",-1)) != sum(p.numel() for p in tm.parameters()): raise ValueError("Transformer checkpoint parameter count mismatch")
        if ts.get("architecture_id") != f"decoder_transformer_4l_d{a.transformer_model_dim}_h{a.transformer_heads}_ff{a.transformer_ff_width}_tied_v1": raise ValueError("Unexpected Transformer architecture ID")
        if ts.get("tokenizer_sha256") != tokenizer_sha: raise ValueError("Transformer tokenizer SHA mismatch")
        if ts.get("data_sha256") != ls.get("data_sha256"): raise ValueError("Model checkpoints were not trained on the same corpus")
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
    out={"holdout_rows":len(data),"holdout_sha256":holdout_sha,"tokenizer_sha256":tokenizer_sha,"method":"mean conditional token NLL over each answer choice; same frozen tokenizer/context; no updates","results":results}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:{x:v for x,v in r.items() if x!="predictions"} for k,r in results.items()},indent=2),flush=True)
if __name__=="__main__": main()
