#!/usr/bin/env python3
"""Audit matched step-8250 parents and prepare baseline + LR-stress continuations."""
from __future__ import annotations
import argparse, copy, hashlib, json, math
from pathlib import Path
import torch

PARENT, TARGET = 8250, 9250
EXPECTED = {
    "lumen": ("ronin_lumen_v3_1_residual_homeostasis", 2_281_841, 2e-5, 0.5),
    "transformer": ("decoder_transformer_4l_d192_h4_ff576_tied_v1", 2_295_168, 5e-5, 1.0),
}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def hash_tree(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous()
            h.update(b"tensor"+str(t.dtype).encode()+str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b"dict")
            for k in sorted(x,key=lambda z:str(z)): visit(str(k)); visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for y in x: visit(y)
        else: h.update(type(x).__name__.encode()+repr(x).encode())
    visit(value); return h.hexdigest()
def make_variant(kind, source, data_hash, tok_hash, out, *, target, multiplier):
    arch, params, base_lr, clip=EXPECTED[kind]
    state=torch.load(source,map_location="cpu",weights_only=False)
    if int(state.get("step",-1))!=PARENT: raise ValueError(f"{kind}: expected parent {PARENT}, got {state.get('step')}")
    if state.get("architecture_id")!=arch or int(state.get("params",-1))!=params: raise ValueError(f"{kind}: architecture/parameter mismatch")
    if int(state.get("seed",-1))!=42: raise ValueError(f"{kind}: seed mismatch")
    if state.get("data_sha256")!=data_hash or state.get("tokenizer_sha256")!=tok_hash: raise ValueError(f"{kind}: corpus/tokenizer mismatch")
    if not state.get("optimizer",{}).get("state") or not state.get("rng"): raise ValueError(f"{kind}: optimizer/RNG missing")
    cfg=state.get("training_config",{})
    expected={"batch":8,"max_len":128,"lr":base_lr,"warmup":100,"total_steps":9000,"weight_decay":0.01,
      "scheduler_mode":"cosine_transition_v1","transition_start":8000,"transition_end":9000,"warmup_transition":1,"grad_clip":clip}
    if cfg!=expected: raise ValueError(f"{kind}: unexpected parent training config: {cfg}")
    old=state.get("scheduler_transition")
    if not old or old.get("kind")!="cosine_transition_v1" or int(old.get("current_step",-1))!=PARENT:
        raise ValueError(f"{kind}: scheduler is not at parent step {PARENT}")
    if state.get("scheduler")!=old: raise ValueError(f"{kind}: scheduler metadata mismatch")
    if kind=="lumen" and (state.get("phase")!=1 or state.get("pilot_all_rows") is not True or state.get("v32_ze",False)):
        raise ValueError("LUMEN parent must be phase-1 all-rows V3.1")
    if kind=="lumen" and state.get("architecture",{}).get("no_jepa",False): raise ValueError("Unexpected JEPA-disabled parent")
    if kind=="lumen" and state.get("architecture",{}).get("no_rep",False): raise ValueError("Unexpected repetition-loss-disabled parent")
    hashes={k:hash_tree(state[k]) for k in ("model","optimizer","rng")}
    tokens=int(state.get("tokens_seen",0))
    if tokens<=0: raise ValueError(f"{kind}: invalid token counter")
    current_lr=float(state["optimizer"]["param_groups"][0]["lr"])
    if not math.isfinite(current_lr) or current_lr<=0: raise ValueError(f"{kind}: invalid saved LR")
    lr=base_lr if multiplier==1.0 else base_lr*multiplier
    warmup=1 if multiplier==1.0 else 10
    schedule={"kind":"cosine_transition_v1","start_step":PARENT,"end_step":target,
      "old_lr":current_lr,"new_lr":lr,"warmup_steps":warmup,"current_step":PARENT}
    result=copy.deepcopy(state)
    result["training_config"]={**cfg,"lr":lr,"total_steps":target,"transition_start":PARENT,
      "transition_end":target,"warmup_transition":warmup}
    result["scheduler_transition"]=schedule
    result["scheduler"]=copy.deepcopy(schedule)
    result["continuation_parent"]={"source_step":PARENT,"target_step":target,
      "source_checkpoint_sha256":sha(source),"weights_sha256":hashes["model"],
      "optimizer_state_sha256":hashes["optimizer"],"rng_state_sha256":hashes["rng"],
      "tokens_seen":tokens,"saved_parent_lr":current_lr,"target_lr":lr,
      "lr_multiplier":multiplier,"policy":"preserve weights, AdamW moments, RNG and token count"}
    for k in hashes:
        if hash_tree(result[k])!=hashes[k]: raise AssertionError(f"{kind}: {k} changed during preparation")
    out.parent.mkdir(parents=True,exist_ok=True); torch.save(result,out)
    return {"architecture_id":arch,"params":params,"source_sha256":sha(source),
      "prepared_sha256":sha(out),"weights_sha256":hashes["model"],
      "optimizer_sha256":hashes["optimizer"],"rng_sha256":hashes["rng"],
      "tokens_seen":tokens,"saved_parent_lr":current_lr,"target_lr":lr,
      "lr_multiplier":multiplier}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lumen",required=True); p.add_argument("--transformer",required=True)
    p.add_argument("--data",required=True); p.add_argument("--tokenizer",required=True)
    p.add_argument("--output-dir",required=True); a=p.parse_args()
    dh,th=sha(a.data),sha(a.tokenizer); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    records={}
    for kind,src in (("lumen",a.lumen),("transformer",a.transformer)):
        records[kind]={}
        records[kind]["baseline"]=make_variant(kind,src,dh,th,out/f"{kind}_step_8250_continuation.pt",target=TARGET,multiplier=1.0)
        for mult,label in ((1.5,"1p5x"),(2.0,"2x")):
            records[kind][label]=make_variant(kind,src,dh,th,out/"stress"/f"{kind}_{label}.pt",target=PARENT+100,multiplier=mult)
    if records["lumen"]["baseline"]["tokens_seen"]!=records["transformer"]["baseline"]["tokens_seen"]:
        raise ValueError("Matched baseline token counters differ")
    manifest={"schema_version":1,"experiment":"RONIN-LUMEN-vs-Transformer-8250-to-9250",
      "parent_step":PARENT,"target_step":TARGET,"data_sha256":dh,"tokenizer_sha256":th,
      "tokens_seen_at_parent":records["lumen"]["baseline"]["tokens_seen"],"models":records}
    (out/"parent_validation.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
