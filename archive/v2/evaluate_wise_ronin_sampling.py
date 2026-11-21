#!/usr/bin/env python3
"""Sampling evaluation: factual anchors, response diversity and style-motif repetition."""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from train_wise_ronin import SYSTEM_FIXED, WISERonin
from generate_wise_ronin_expanded import OPENERS, TAILS, OBJECTS, KOAN_FORMULAS

def sample_next(logits, temperature, top_p):
    logits=logits.float()/temperature
    probs=F.softmax(logits,dim=-1)
    sorted_probs, sorted_idx=torch.sort(probs,descending=True)
    cumulative=torch.cumsum(sorted_probs,dim=-1)
    remove=cumulative-sorted_probs>top_p
    sorted_probs[remove]=0
    probs=torch.zeros_like(probs).scatter(-1,sorted_idx,sorted_probs)
    probs=probs/probs.sum(dim=-1,keepdim=True)
    return torch.multinomial(probs,1)

def motif(answer):
    for template in OPENERS:
        prefix=template.split("{a}")[0]
        if answer.startswith(prefix):
            return "opener:"+prefix.strip()
    if answer.startswith("KOAN:"):
        return "opener:KOAN"
    return "opener:"+ " ".join(answer.split()[:4]).casefold()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--holdout",default="holdout_wise_ronin_500.jsonl")
    ap.add_argument("--max-new-tokens",type=int,default=48)
    ap.add_argument("--temperature",type=float,default=0.8)
    ap.add_argument("--top-p",type=float,default=0.9)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--output-dir",default="artifacts")
    args=ap.parse_args()
    if args.temperature<=0 or not 0<args.top_p<=1:
        raise ValueError("temperature must be >0 and top-p in (0,1]")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=WISERonin(ckpt["config"]).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    tok=Tokenizer.from_file(args.tokenizer)
    end_id=tok.token_to_id("<|end|>")
    rows=[json.loads(x) for x in Path(args.holdout).read_text(encoding="utf-8").splitlines() if x.strip()]
    results=[]
    for i,row in enumerate(rows):
        system=row.get("system",SYSTEM_FIXED)
        prompt="<|system|>"+system+"<|user|>"+row["user"]+"<|assistant|>"
        ids=tok.encode(prompt).ids[-(ckpt["config"]["ctx"]-args.max_new_tokens):]
        input_ids=torch.tensor([ids],dtype=torch.long,device=device)
        generated=[]
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                logits=model(input_ids)[0,-1]
                nxt=int(sample_next(logits,args.temperature,args.top_p).item())
                if nxt==end_id: break
                generated.append(nxt)
                input_ids=torch.cat([input_ids,torch.tensor([[nxt]],device=device)],dim=1)
        answer=tok.decode(generated).strip()
        low=answer.casefold()
        expected=str(row.get("hecho_esperado","")).strip()
        generic={"portazo, no palmadita","estructura koan","wise-ronin, no claude, no gpt","negarse a dar moraleja con gesto"}
        factual=bool(expected and expected.casefold() not in generic and expected.casefold() in low)
        results.append({"id":row.get("id"),"categoria":row.get("categoria"),"user":row["user"],
                        "answer":answer,"factual_anchor_match":factual,
                        "opening_motif":motif(answer),
                        "tail_motifs":[x for x in TAILS if x.casefold() in low],
                        "object_motifs":[x for x in OBJECTS if x.casefold() in low],
                        "koan_motifs":[x for x in KOAN_FORMULAS if x.split("{a}")[0].casefold() in low]})
        if (i+1)%25==0 or i+1==len(rows): print(f"[sampling] {i+1}/{len(rows)}",flush=True)

    answers=[re.sub(r"\s+"," ",x["answer"].casefold()).strip() for x in results]
    opening_counts=Counter(x["opening_motif"] for x in results)
    tail_counts=Counter(m for x in results for m in x["tail_motifs"])
    object_counts=Counter(m for x in results for m in x["object_motifs"])
    koan_counts=Counter(m for x in results for m in x["koan_motifs"])
    n=max(1,len(results))
    def top(counter):
        if not counter:return {"motif":None,"count":0,"share":0.0}
        name,count=counter.most_common(1)[0]
        return {"motif":name,"count":count,"share":round(count/n,4)}
    report={
      "checkpoint":args.checkpoint,"evaluated":len(results),
      "unique_question_strings":len({x["user"] for x in results}),
      "unique_normalized_responses":len(set(answers)),
      "diversity_target_unique_responses":300,
      "diversity_pass":len(set(answers))>=300,
      "factual_anchor_matches":sum(x["factual_anchor_match"] for x in results),
      "factual_anchor_rate":round(sum(x["factual_anchor_match"] for x in results)/n,4),
      "max_opening_share":top(opening_counts),
      "max_tail_share":top(tail_counts),
      "max_object_share":top(object_counts),
      "max_koan_share":top(koan_counts),
      "repetition_warning":any(top(c)["share"]>0.10 for c in (opening_counts,tail_counts,object_counts,koan_counts)),
      "sampling":{"temperature":args.temperature,"top_p":args.top_p,"seed":args.seed},
      "note":"Factual matching uses literal hecho_esperado anchors; review generations for semantic matches. Question-string uniqueness does not imply 500 independent underlying facts."
    }
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"sampling_holdout_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"sampling_holdout_generations.jsonl").open("w",encoding="utf-8") as f:
        for x in results:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
