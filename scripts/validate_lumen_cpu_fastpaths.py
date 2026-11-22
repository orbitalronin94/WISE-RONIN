#!/usr/bin/env python3
"""Numerical forward/loss/gradient parity gate for LUMEN CPU fast paths."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Direct script execution places scripts/ on sys.path, not the repository root.
# Add the root so model/ and train/ resolve on clean CI runners.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from model.config import CONFIG
from model.lumen import RoninLumen
from train.loss import perdida_total

def run(model, tokens):
    model.zero_grad(set_to_none=True)
    output=model(tokens,reset_state=True,collect_diagnostics=False)
    loss,_=perdida_total(output,tokens,model,use_jepa=True,use_rep=True,
                         use_free_energy=False,beta=0.01,pad_id=0)
    grads=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
    return output,loss,grads

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--rtol",type=float,default=2e-4)
    p.add_argument("--atol",type=float,default=2e-5)
    a=p.parse_args()
    torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if state.get("architecture_id")!="ronin_lumen_v3_1_residual_homeostasis":
        raise ValueError("Fast-path parity gate requires the frozen LUMEN V3.1 parent")
    torch.manual_seed(20261122)
    reference=RoninLumen(v32_ze=False).cpu()
    optimized=RoninLumen(v32_ze=False).cpu()
    reference.load_state_dict(state["model"])
    optimized.load_state_dict(state["model"])
    reference.eval(); optimized.eval()
    for layer in reference.delta_layers: layer.pre_normalize_sequence=False
    for organ in reference.homeo: organ.fused_scatter_add=False
    for layer in optimized.delta_layers: layer.pre_normalize_sequence=True
    for organ in optimized.homeo: organ.fused_scatter_add=True
    tokens=torch.randint(4,CONFIG.vocab_size,(2,16),dtype=torch.long)
    out_a,loss_a,grad_a=run(reference,tokens)
    out_b,loss_b,grad_b=run(optimized,tokens)
    torch.testing.assert_close(loss_a,loss_b,rtol=a.rtol,atol=a.atol)
    keys=("logits","activations","hidden","router","surprise","energy","free_energy_complexity")
    max_output_diff=0.0
    for key in keys:
        torch.testing.assert_close(out_a[key],out_b[key],rtol=a.rtol,atol=a.atol,msg=f"Output mismatch in {key}")
        if out_a[key].numel():
            max_output_diff=max(max_output_diff,float((out_a[key]-out_b[key]).abs().max()))
    if len(grad_a)!=len(grad_b):
        raise AssertionError("Parameter-gradient list length changed")
    max_grad_diff=0.0
    for index,(ga,gb) in enumerate(zip(grad_a,grad_b)):
        if ga is None or gb is None:
            if ga is not None or gb is not None: raise AssertionError(f"Gradient presence mismatch at parameter {index}")
            continue
        torch.testing.assert_close(ga,gb,rtol=a.rtol,atol=a.atol,
                                   msg=f"Gradient mismatch at parameter {index}")
        if ga.numel(): max_grad_diff=max(max_grad_diff,float((ga-gb).abs().max()))
    print({"parity":"PASS","loss":float(loss_a.detach()),
           "max_output_abs_diff":max_output_diff,
           "max_gradient_abs_diff":max_grad_diff,
           "delta_sequence_prenorm":True,"homeostasis_fused_scatter":True,
           "jepa_and_repetition_loss_included":True})
if __name__=="__main__": main()
