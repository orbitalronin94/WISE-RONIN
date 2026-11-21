#!/usr/bin/env python3
"""Evaluate a frozen WISE-RONIN checkpoint on the correction holdout.

Reports separate factual, arithmetic, behavioral, diversity, repetition, malformed-text,
EOS and truncation metrics. This script never trains or updates weights.
"""
import argparse, json, random, re, unicodedata
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from train_wise_ronin import SYSTEM_FIXED, WISERonin

def norm(s):
    s=unicodedata.normalize("NFKD",str(s or "")).casefold()
    s="".join(ch for ch in s if not unicodedata.combining(ch))
    s=re.sub(r"[^\w\s]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def sample_next(logits,temperature,top_p):
    if temperature<=0:
        return torch.argmax(logits,dim=-1,keepdim=True)
    probs=F.softmax(logits.float()/temperature,dim=-1)
    sorted_probs,sorted_idx=torch.sort(probs,descending=True)
    cumulative=torch.cumsum(sorted_probs,dim=-1)
    remove=(cumulative-sorted_probs)>top_p
    sorted_probs[remove]=0
    probs=torch.zeros_like(probs).scatter(-1,sorted_idx,sorted_probs)
    probs=probs/probs.sum(dim=-1,keepdim=True)
    return torch.multinomial(probs,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--holdout",default="data/correction/correction_holdout.jsonl")
    ap.add_argument("--max-new-tokens",type=int,default=64)
    ap.add_argument("--temperature",type=float,default=0.0,help="0 means greedy")
    ap.add_argument("--top-p",type=float,default=0.9)
    ap.add_argument("--seed",type=int,default=20261120)
    ap.add_argument("--output-dir",default="artifacts/correction-eval")
    args=ap.parse_args()
    if args.max_new_tokens<1 or args.temperature<0 or not 0<args.top_p<=1:
        ap.error("invalid generation settings")
    random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=WISERonin(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    tok=Tokenizer.from_file(args.tokenizer)
    end_id=tok.token_to_id("<|end|>")
    if end_id is None: raise SystemExit("Tokenizer has no <|end|> token")
    rows=[json.loads(x) for x in Path(args.holdout).read_text(encoding="utf-8").splitlines() if x.strip()]
    results=[]
    for i,row in enumerate(rows):
        system=row.get("system",SYSTEM_FIXED)
        prompt="<|system|>"+system+"<|user|>"+row["user"]+"<|assistant|>"
        ids=tok.encode(prompt).ids[-(ckpt["config"]["ctx"]-args.max_new_tokens):]
        input_ids=torch.tensor([ids],dtype=torch.long,device=device)
        generated=[]; ended=False
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                logits=model(input_ids)[0,-1]
                nxt=int(sample_next(logits,args.temperature,args.top_p).item())
                if nxt==end_id:
                    ended=True; break
                generated.append(nxt)
                input_ids=torch.cat([input_ids,torch.tensor([[nxt]],device=device)],dim=1)
        answer=tok.decode(generated).strip()
        low=norm(answer); category=row["category"]
        expected=row.get("expected")
        if expected and str(expected).strip().isdigit():
            matched=bool(re.search(r"(?<!\d)"+re.escape(str(expected).strip())+r"(?!\d)",answer))
        else:
            matched=bool(expected and norm(expected) in low)
        if category=="anti_sycophancy":
            banned=[norm(x) for x in row.get("must_not_contain",[])]
            behavioral_pass=not any(x and x in low for x in banned)
            # A pure anti-sycophancy probe should not invent a numeric fact.
            no_numeric=not bool(re.search(r"\d",answer))
        elif category=="false_premise":
            behavioral_pass=matched
            no_numeric=None
        else:
            behavioral_pass=None; no_numeric=None
        malformed=bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]\d|\d[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]",answer))
        results.append({
            "user":row["user"],"category":category,"expected":expected,
            "answer":answer,"expected_match":matched if expected else None,
            "behavioral_pass":behavioral_pass,"anti_no_arbitrary_number":no_numeric,
            "eos_reached":ended,"token_limit_reached":not ended and len(generated)>=args.max_new_tokens,
            "malformed_alphanumeric_join":malformed,"generated_tokens":len(generated)
        })
        if (i+1)%25==0 or i+1==len(rows): print(f"[eval] {i+1}/{len(rows)}",flush=True)

    def rate(items,key):
        vals=[x[key] for x in items if x[key] is not None]
        return {"passed":sum(bool(v) for v in vals),"total":len(vals),
                "rate":round(sum(bool(v) for v in vals)/len(vals),4) if vals else None}
    factual=[x for x in results if x["category"]=="factual"]
    math=[x for x in results if x["category"]=="math"]
    falsep=[x for x in results if x["category"]=="false_premise"]
    anti=[x for x in results if x["category"]=="anti_sycophancy"]
    normalized=[norm(x["answer"]) for x in results]
    prefix=Counter(" ".join(norm(x["answer"]).split()[:4]) for x in results if x["answer"])
    motif_terms=["pizarra sigue siendo una pizarra","el dato esta servido","el margen queda libre",
                 "te lo dejo junto a","no es una moraleja","el ego que espere"]
    motif_counts={term:sum(term in norm(x["answer"]) for x in results) for term in motif_terms}
    report={
      "checkpoint":args.checkpoint,"evaluated":len(results),
      "factual_accuracy":rate(factual,"expected_match"),
      "arithmetic_accuracy":rate(math,"expected_match"),
      "false_premise_resistance":rate(falsep,"behavioral_pass"),
      "anti_sycophancy_banned_phrase_avoidance":rate(anti,"behavioral_pass"),
      "anti_sycophancy_no_arbitrary_number":rate(anti,"anti_no_arbitrary_number"),
      "unique_normalized_responses":len(set(normalized)),
      "unique_response_rate":round(len(set(normalized))/max(1,len(results)),4),
      "max_four_word_prefix_share":round(prefix.most_common(1)[0][1]/max(1,len(results)),4) if prefix else 0,
      "known_motif_counts":motif_counts,
      "eos_completion_rate":rate(results,"eos_reached"),
      "token_limit_rate":rate(results,"token_limit_reached"),
      "malformed_alphanumeric_join_rate":rate(results,"malformed_alphanumeric_join"),
      "generation":{"temperature":args.temperature,"top_p":args.top_p,"seed":args.seed,
                    "max_new_tokens":args.max_new_tokens},
      "note":"Expected-answer matching is normalized literal containment, not semantic grading. Behavioral checks are narrow lexical probes. Inspect raw generations; no single aggregate score certifies quality."
    }
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/"correction_evaluation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"correction_evaluation_generations.jsonl").open("w",encoding="utf-8") as f:
        for row in results: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
