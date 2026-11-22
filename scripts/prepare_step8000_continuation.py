#!/usr/bin/env python3
"""Audit and prepare exact matched step-8000 continuations through step 9000.

Preserves model weights, Adam moments, RNG state and token counters. The prior
cosine schedule reached zero at step 8000, so a fresh 1000-step cosine segment
is initialized from the saved schedule's peak/new_lr for a controlled restart.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math
from pathlib import Path
import torch

EXPECTED = {
    "lumen": ("ronin_lumen_v3_1_residual_homeostasis", 2_281_841, 2e-5, 0.5),
    "transformer": ("decoder_transformer_4l_d192_h4_ff576_tied_v1", 2_295_168, 5e-5, 1.0),
}
PARENT, TARGET = 8000, 9000

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

def prepare(kind, source, data_hash, tok_hash, out):
    arch, params, base_lr, clip=EXPECTED[kind]
    state=torch.load(source,map_location="cpu",weights_only=False)
    if state.get("step")!=PARENT: raise ValueError(f"{kind}: expected step {PARENT}, got {state.get('step')}")
    if state.get("architecture_id")!=arch or state.get("params")!=params: raise ValueError(f"{kind}: architecture/parameter mismatch")
    if state.get("seed")!=42: raise ValueError(f"{kind}: seed mismatch")
    if state.get("data_sha256")!=data_hash or state.get("tokenizer_sha256")!=tok_hash: raise ValueError(f"{kind}: corpus/tokenizer hash mismatch")
    cfg=state.get("training_config",{})
    expected={"batch":8,"max_len":128,"lr":base_lr,"warmup":100,"total_steps":8000,"weight_decay":0.01,
              "scheduler_mode":"cosine_transition_v1","transition_start":3000,"transition_end":8000,
              "warmup_transition":1,"grad_clip":clip}
    if cfg!=expected: raise ValueError(f"{kind}: unexpected training config {cfg}")
    if not state.get("optimizer",{}).get("state"): raise ValueError(f"{kind}: missing optimizer moments")
    if not state.get("rng"): raise ValueError(f"{kind}: missing RNG state")
    old=state.get("scheduler_transition")
    if not old or old.get("kind")!="cosine_transition_v1" or old.get("current_step")!=PARENT or old.get("end_step")!=PARENT:
        raise ValueError(f"{kind}: checkpoint is not at the end of the audited cosine schedule")
    if state.get("scheduler")!=old: raise ValueError(f"{kind}: scheduler state mismatch")
    if kind=="lumen" and (state.get("phase")!=1 or state.get("pilot_all_rows") is not True or state.get("v32_ze",False)):
        raise ValueError("LUMEN must be phase-1 all-rows V3.1")
    weights_hash=hash_tree(state["model"]); optim_hash=hash_tree(state["optimizer"]); rng_hash=hash_tree(state["rng"])
    tokens=int(state.get("tokens_seen",0))
    if tokens<=0: raise ValueError(f"{kind}: invalid token counter")
    # The old cosine LR is zero at 8000. Restart from its saved peak/new_lr,
    # retaining the full optimizer state and using one flat warmup step.
    restart_lr=float(old["new_lr"])
    if not math.isfinite(restart_lr) or restart_lr<=0: raise ValueError(f"{kind}: invalid restart LR")
    schedule={"kind":"cosine_transition_v1","start_step":PARENT,"end_step":TARGET,
              "old_lr":restart_lr,"new_lr":restart_lr,"warmup_steps":1,"current_step":PARENT}
    result=copy.deepcopy(state)
    result["training_config"]={**cfg,"total_steps":TARGET,"transition_start":PARENT,
                               "transition_end":TARGET,"warmup_transition":1}
    result["scheduler_transition"]=schedule
    result["scheduler"]=copy.deepcopy(schedule)
    result["continuation_parent"]={"source_step":PARENT,"target_step":TARGET,
        "source_checkpoint_sha256":sha(source),"weights_sha256":weights_hash,
        "optimizer_state_sha256":optim_hash,"rng_state_sha256":rng_hash,
        "tokens_seen":tokens,"restart_lr":restart_lr,
        "policy":"preserve weights/Adam/RNG/tokens; restart cosine from saved schedule peak because LR was zero at step 8000"}
    if hash_tree(result["model"])!=weights_hash or hash_tree(result["optimizer"])!=optim_hash or hash_tree(result["rng"])!=rng_hash:
        raise AssertionError(f"{kind}: preparation mutated training state")
    torch.save(result,out)
    return {"architecture_id":arch,"parameters":params,"source_sha256":sha(source),
        "prepared_sha256":sha(out),"weights_sha256":weights_hash,"optimizer_sha256":optim_hash,
        "rng_sha256":rng_hash,"tokens_seen":tokens,"restart_lr":restart_lr,
        "weights_preserved":True,"optimizer_preserved":True,"rng_preserved":True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lumen",required=True); p.add_argument("--transformer",required=True)
    p.add_argument("--data",required=True); p.add_argument("--tokenizer",required=True)
    p.add_argument("--output-dir",required=True); a=p.parse_args()
    dh,th=sha(a.data),sha(a.tokenizer); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    models={}
    for kind,src in (("lumen",a.lumen),("transformer",a.transformer)):
        models[kind]=prepare(kind,src,dh,th,out/f"{kind}_step_8000_continuation.pt")
    if models["lumen"]["tokens_seen"]!=models["transformer"]["tokens_seen"]:
        raise ValueError("Matched token counters differ")
    manifest={"schema_version":1,"experiment":"RONIN-LUMEN-vs-Transformer-8000-to-9000",
        "parent_step":PARENT,"target_step":TARGET,"data_sha256":dh,"tokenizer_sha256":th,
        "tokens_seen_at_parent":models["lumen"]["tokens_seen"],"models":models}
    (out/"parent_validation.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
