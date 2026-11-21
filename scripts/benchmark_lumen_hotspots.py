"""Matched end-to-end CPU benchmark for LUMEN hot paths."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
from model.config import CONFIG
from model.lumen import RoninLumen
from train.dataset import load_jsonl, load_token_cache, pack_token_cache, collate_cached_indices
from train.loss import perdida_total

def profile(threads=2, interop_threads=1, mkldnn_enabled=True, foreach=True,
            cpu_affinity=False, delta_pre_normalize=False, homeostasis_fused_scatter=False):
 return dict(threads=threads, interop_threads=interop_threads, mkldnn_enabled=mkldnn_enabled,
             foreach=foreach, cpu_affinity=cpu_affinity, delta_pre_normalize=delta_pre_normalize,
             homeostasis_fused_scatter=homeostasis_fused_scatter)

# Orthogonal, interpretable CPU profiles. Each changes one or a small number of
# factors so a win can be attributed and reproduced rather than guessed.
CANDIDATES = {
 "torch_default_2t": profile(),
 "one_thread": profile(threads=1),
 "three_threads": profile(threads=3),
 "four_threads": profile(threads=4),
 "interop_2_only": profile(interop_threads=2),
 "mkldnn_off_only": profile(mkldnn_enabled=False),
 "foreach_off_only": profile(foreach=False),
 "affinity_only": profile(cpu_affinity=True),
 "mkldnn_foreach_off": profile(mkldnn_enabled=False, foreach=False),
 "mkldnn_affinity": profile(mkldnn_enabled=False, cpu_affinity=True),
 "foreach_affinity": profile(foreach=False, cpu_affinity=True),
 "interop_mkldnn_off": profile(interop_threads=2, mkldnn_enabled=False),
 "interop_foreach_off": profile(interop_threads=2, foreach=False),
 "current_combined_cpu": profile(interop_threads=2, mkldnn_enabled=False, foreach=False, cpu_affinity=True),
 "delta_batched_norm": profile(interop_threads=2, mkldnn_enabled=False, foreach=False, cpu_affinity=True, delta_pre_normalize=True),
 "homeostasis_fused_scatter": profile(interop_threads=2, mkldnn_enabled=False, foreach=False, cpu_affinity=True, homeostasis_fused_scatter=True),
 "both_exact_fastpaths": profile(interop_threads=2, mkldnn_enabled=False, foreach=False, cpu_affinity=True, delta_pre_normalize=True, homeostasis_fused_scatter=True),
 "fastpaths_backend_default": profile(delta_pre_normalize=True, homeostasis_fused_scatter=True),
 "fastpaths_mkldnn_off": profile(mkldnn_enabled=False, delta_pre_normalize=True, homeostasis_fused_scatter=True),
 "threads1_combined": profile(threads=1, interop_threads=1, mkldnn_enabled=False, foreach=False, cpu_affinity=True),
 "threads4_combined": profile(threads=4, interop_threads=2, mkldnn_enabled=False, foreach=False, cpu_affinity=True),
}

def configure(p):
 torch.set_num_threads(int(p["threads"]))
 torch.set_num_interop_threads(int(p["interop_threads"]))
 torch.backends.mkldnn.enabled=bool(p["mkldnn_enabled"])
 affinity=None
 if p["cpu_affinity"] and hasattr(os,"sched_getaffinity"):
  available=sorted(os.sched_getaffinity(0)); affinity=available[:int(p["threads"])]
  os.sched_setaffinity(0,set(affinity))
 return affinity

def load_batch(a):
 rows,data_hash=load_jsonl(a.data)
 tokenizer_hash=hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
 encoded=load_token_cache(a.token_cache,data_hash,tokenizer_hash,a.max_len,len(rows))
 packed,lengths=pack_token_cache(encoded,a.max_len)
 import random
 rng=random.Random(42)
 valid=[i for i,n in enumerate(lengths.tolist()) if n>=2]
 return collate_cached_indices(packed,lengths,rng.choices(valid,k=a.batch))

def make_model(ckpt,p):
 model=RoninLumen().cpu()
 model.load_state_dict(ckpt["model"],strict=True)
 for layer in model.delta_layers: layer.pre_normalize_sequence=bool(p["delta_pre_normalize"])
 for organ in model.homeo: organ.fused_scatter_add=bool(p["homeostasis_fused_scatter"])
 model.train()
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=CONFIG.weight_decay,foreach=bool(p["foreach"]))
 if "optimizer" in ckpt:
  opt.load_state_dict(ckpt["optimizer"])
  for group in opt.param_groups: group["foreach"]=bool(p["foreach"])
 return model,opt

def update(model,opt,tokens,use_rep=True,collect_diagnostics=False):
 opt.zero_grad(set_to_none=True)
 output=model(tokens,reset_state=True,collect_diagnostics=collect_diagnostics)
 loss,_=perdida_total(output,tokens,model,use_jepa=True,use_rep=use_rep)
 if not torch.isfinite(loss): raise FloatingPointError("Non-finite benchmark loss")
 loss.backward()
 clip_grad_norm_(model.parameters(),0.5,foreach=opt.param_groups[0]["foreach"])
 opt.step()
 return float(loss.detach())

def trial(ckpt,tokens,p,warmup,steps,repeats,use_rep=True,collect_diagnostics=False):
 samples=[]
 for repeat in range(repeats):
  torch.manual_seed(4200+repeat)
  model,opt=make_model(ckpt,p)
  for _ in range(warmup): update(model,opt,tokens,use_rep,collect_diagnostics)
  started=time.perf_counter(); losses=[]
  for _ in range(steps): losses.append(update(model,opt,tokens,use_rep,collect_diagnostics))
  elapsed=time.perf_counter()-started
  samples.append(dict(elapsed_seconds=elapsed,steps_per_second=steps/elapsed,
                      tokens_per_second=steps*int(tokens.ne(0).sum())/elapsed,
                      mean_loss=statistics.mean(losses),last_loss=losses[-1]))
  del model,opt
 return dict(runs=samples,median_steps_per_second=statistics.median(x["steps_per_second"] for x in samples),
             median_tokens_per_second=statistics.median(x["tokens_per_second"] for x in samples),
             median_elapsed_seconds=statistics.median(x["elapsed_seconds"] for x in samples),
             median_mean_loss=statistics.median(x["mean_loss"] for x in samples))

def checkpoint_save_cost(ckpt,p):
 model,opt=make_model(ckpt,p)
 payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"step":int(ckpt["step"])}
 samples=[]
 with tempfile.TemporaryDirectory(prefix="lumen-ckpt-cost-") as tmp:
  target=Path(tmp)/"checkpoint.pt"
  for _ in range(2):
   started=time.perf_counter(); torch.save(payload,target); samples.append(time.perf_counter()-started)
  file_bytes=target.stat().st_size
 return {"median_seconds":statistics.median(samples),"file_bytes":file_bytes,"repeats":len(samples)}

def profiler_table(ckpt,tokens,p):
 from torch.profiler import ProfilerActivity, profile as torch_profile
 model,opt=make_model(ckpt,p)
 with torch_profile(activities=[ProfilerActivity.CPU],record_shapes=True) as prof: update(model,opt,tokens,True)
 return prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=30)

def worker(a):
 p=json.loads(a.worker_profile); affinity=configure(p)
 torch.manual_seed(4200)
 ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False); tokens=load_batch(a)
 result=dict(profile=p,affinity_cpus=affinity,full_objective=trial(ckpt,tokens,p,a.warmup,a.steps,a.repeats,True),
  diagnostic_collection_cost=trial(ckpt,tokens,p,max(2,a.warmup//2),max(8,a.steps//2),1,True,True),
  checkpoint_save_cost=checkpoint_save_cost(ckpt,p))
 if a.measure_ablation:
  result["repetition_loss_ablation"]=trial(ckpt,tokens,p,max(2,a.warmup//2),max(8,a.steps//2),1,False)
  result["repetition_loss_ablation"]["promotion_eligible"]=False
  result["repetition_loss_ablation"]["reason"]="Changes the objective; speed alone cannot authorize promotion."
 if a.profile_ops: result["top_cpu_ops"]=profiler_table(ckpt,tokens,p)
 Path(a.output).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")

def orchestrate(a):
 current=json.loads(Path(a.current_profile).read_text(encoding="utf-8"))
 profiles={"current":current,**CANDIDATES}; results={}
 with tempfile.TemporaryDirectory(prefix="lumen-hotspot-") as tmp:
  for name,p in profiles.items():
   out=str(Path(tmp)/(name+".json"))
   cmd=[sys.executable,str(Path(__file__).resolve()),"--worker-profile",json.dumps(p),
        "--checkpoint",a.checkpoint,"--data",a.data,"--tokenizer",a.tokenizer,
        "--token-cache",a.token_cache,"--output",out,"--max-len",str(a.max_len),
        "--batch",str(a.batch),"--warmup",str(a.warmup),"--steps",str(a.steps),"--repeats",str(a.repeats)]
   if name=="current": cmd += ["--measure-ablation","--profile-ops"]
   subprocess.run(cmd,check=True)
   results[name]=json.loads(Path(out).read_text(encoding="utf-8"))
 baseline=results["current"]["full_objective"]["median_steps_per_second"]
 baseline_loss=results["current"]["full_objective"]["median_mean_loss"]
 loss_limit=baseline_loss+a.max_loss_regression
 evaluated=[]
 for name,r in results.items():
  speed=r["full_objective"]["median_steps_per_second"]
  loss=r["full_objective"]["median_mean_loss"]
  evaluated.append(dict(name=name,median_steps_per_second=speed,speedup_vs_current=speed/baseline,
                        median_mean_loss=loss,loss_delta_vs_current=loss-baseline_loss,
                        loss_gate_passed=(loss<=loss_limit)))
 eligible=[x for x in evaluated if x["name"]=="current" or x["loss_gate_passed"]]
 ranked=sorted(eligible,key=lambda x:x["median_steps_per_second"],reverse=True)
 winner=ranked[0]["name"]; speed=ranked[0]["median_steps_per_second"]; gain=speed/baseline
 selected=winner if winner!="current" and gain>=a.minimum_speedup else "current"
 report=dict(benchmark="lumen_hotspot_cpu_v2_loss_gated",checkpoint_step=int(torch.load(a.checkpoint,map_location="cpu",weights_only=False)["step"]),
  batch=a.batch,sequence_length=a.max_len,warmup_steps=a.warmup,measured_steps=a.steps,repeats=a.repeats,
  baseline_steps_per_second=baseline,baseline_median_mean_loss=baseline_loss,
  max_loss_regression=a.max_loss_regression,loss_gate_limit=loss_limit,
  ranked_profiles=sorted(evaluated,key=lambda x:x["median_steps_per_second"],reverse=True),
  eligible_profiles=[x["name"] for x in eligible],selected_profile=selected,
  selected_steps_per_second=results[selected]["full_objective"]["median_steps_per_second"],
  best_eligible_speedup=gain,minimum_speedup_required=a.minimum_speedup,
  promoted=selected!="current",promotion_rule=">=5% median full-objective steps/s gain AND median training loss no more than configured regression tolerance above current profile",
  profiles=results)
 Path(a.report).parent.mkdir(parents=True,exist_ok=True)
 Path(a.report).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
 if selected!="current":
  p=profiles[selected].copy()
  p["benchmark_metadata"]=dict(selected_profile=selected,median_steps_per_second=results[selected]["full_objective"]["median_steps_per_second"],
   baseline_steps_per_second=baseline,speedup_vs_current=gain,checkpoint_step=report["checkpoint_step"],
   median_mean_loss=results[selected]["full_objective"]["median_mean_loss"],baseline_median_mean_loss=baseline_loss,
   max_loss_regression=a.max_loss_regression,objective_unchanged=True,
   numerical_parity_tests="passed in workflow before benchmark")
  Path(a.current_profile).write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(dict(selected_profile=selected,promoted=selected!="current",baseline_steps_per_second=baseline,
  baseline_median_mean_loss=baseline_loss,best_eligible_speedup=gain,report=a.report),indent=2))

def main():
 p=argparse.ArgumentParser()
 for name in ("checkpoint","data","tokenizer","token_cache","output"): p.add_argument("--"+name.replace("_","-"))
 p.add_argument("--current-profile",default="config/lumen_cpu_profile.json")
 p.add_argument("--report",default="artifacts/lumen_hotspot_benchmark.json")
 p.add_argument("--worker-profile",help=argparse.SUPPRESS)
 p.add_argument("--max-len",type=int,default=128); p.add_argument("--batch",type=int,default=8)
 p.add_argument("--warmup",type=int,default=6); p.add_argument("--steps",type=int,default=25)
 p.add_argument("--repeats",type=int,default=3); p.add_argument("--minimum-speedup",type=float,default=1.05)
 p.add_argument("--max-loss-regression",type=float,default=0.01)
 p.add_argument("--measure-ablation",action="store_true"); p.add_argument("--profile-ops",action="store_true")
 a=p.parse_args()
 if a.worker_profile: worker(a)
 else:
  for field in ("checkpoint","data","tokenizer","token_cache"):
   if not getattr(a,field): p.error("--"+field.replace("_","-")+" is required")
  orchestrate(a)
if __name__=="__main__": main()
