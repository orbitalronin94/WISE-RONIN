# train_wise_ronin.py - WISE-RONIN 24MB - CON BPE REAL
# Uso: python train_wise_ronin.py --data dataset_enano_70mb.jsonl --epochs 3 --lr 3e-4 --orpo True
import json, argparse, math, random, os
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False
try:
    from tokenizers import ByteLevelBPETokenizer
    HAS_TOKENIZERS=True
except ImportError:
    HAS_TOKENIZERS=False
CONFIG={"vocab":6500,"dim":256,"layers":6,"q_heads":8,"kv_heads":2,"head_dim":32,"hidden":640,"ctx":1024}
SYSTEM_FIJO="Eres WISE-RONIN. Respondon y antisicofantico. Hechos ciertos con rodeo y trolleo. Nunca lames. Nunca das moraleja. Nunca dices como IA. Nunca dices que eres Claude o GPT."
def get_or_train_tokenizer(data_paths, vocab_size=6500, save_path="tokenizer-wise-ronin.json"):
    if os.path.exists(save_path) and HAS_TOKENIZERS:
        from tokenizers import Tokenizer
        print(f"[WISE-RONIN] Cargando tokenizer BPE: {save_path}")
        return Tokenizer.from_file(save_path)
    print(f"[WISE-RONIN] Entrenando BPE real desde {data_paths} vocab={vocab_size}")
    tmp_txt="tmp_bpe_train.txt"
    with open(tmp_txt,"w",encoding="utf-8") as out:
        for path in data_paths:
            with open(path,encoding="utf-8") as f:
                for line in f:
                    try:
                        r=json.loads(line)
                    except:
                        continue
                    out.write(f"{r.get('system',SYSTEM_FIJO)}\n{r.get('user','')}\n{r.get('assistant','')}\n")
    tok=ByteLevelBPETokenizer()
    tok.train(files=[tmp_txt],vocab_size=vocab_size,min_frequency=2,special_tokens=["<|system|>","<|user|>","<|assistant|>","<|end|>","<|pad|>"])
    tok.save(save_path)
    print(f"[WISE-RONIN] Tokenizer guardado {save_path} vocab={tok.get_vocab_size()}")
    if os.path.exists(tmp_txt):
        os.remove(tmp_txt)
    from tokenizers import Tokenizer
    return Tokenizer.from_file(save_path)
def encode_example(tokenizer, rec, max_len=1024):
    sys_ids=tokenizer.encode("<|system|>"+rec.get("system",SYSTEM_FIJO)).ids
    user_ids=tokenizer.encode("<|user|>"+rec.get("user","")).ids
    ast_ids=tokenizer.encode("<|assistant|>"+rec.get("assistant","")+"<|end|>").ids
    input_ids=sys_ids+user_ids+ast_ids
    labels=[-100]*(len(sys_ids)+len(user_ids))+ast_ids
    if len(input_ids)>max_len:
        input_ids=input_ids[:max_len]; labels=labels[:max_len]
    pad_id=tokenizer.encode("<|pad|>").ids[0] if tokenizer.encode("<|pad|>").ids else 0
    pad_len=max_len-len(input_ids)
    if pad_len>0:
        input_ids=input_ids+[pad_id]*pad_len; labels=labels+[-100]*pad_len
    return input_ids, labels
class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):
        rms=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        return (x/rms)*self.weight
def rope_freqs(head_dim,max_len,device,theta=10000.0):
    freqs=1.0/(theta**(torch.arange(0,head_dim,2,device=device).float()/head_dim))
    pos=torch.arange(max_len,device=device).float()
    return torch.cos(torch.outer(pos,freqs)), torch.sin(torch.outer(pos,freqs))
def apply_rope(q,k,cos,sin):
    def rot(x,cos,sin):
        x1=x[...,0::2]; x2=x[...,1::2]
        cos=cos.unsqueeze(0).unsqueeze(2); sin=sin.unsqueeze(0).unsqueeze(2)
        xr1=x1*cos-x2*sin; xr2=x1*sin+x2*cos
        out=torch.empty_like(x); out[...,0::2]=xr1; out[...,1::2]=xr2
        return out
    return rot(q,cos,sin), rot(k,cos,sin)
class GQAAttn(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.q_heads=cfg["q_heads"]; self.kv_heads=cfg["kv_heads"]; self.head_dim=cfg["head_dim"]
        self.q=nn.Linear(cfg["dim"],self.q_heads*self.head_dim,bias=False)
        self.k=nn.Linear(cfg["dim"],self.kv_heads*self.head_dim,bias=False)
        self.v=nn.Linear(cfg["dim"],self.kv_heads*self.head_dim,bias=False)
        self.o=nn.Linear(self.q_heads*self.head_dim,cfg["dim"],bias=False)
    def forward(self,x,cos,sin,mask=None):
        B,T,_=x.shape
        q=self.q(x).view(B,T,self.q_heads,self.head_dim)
        k=self.k(x).view(B,T,self.kv_heads,self.head_dim)
        v=self.v(x).view(B,T,self.kv_heads,self.head_dim)
        q,k=apply_rope(q,k,cos[:T],sin[:T])
        rep=self.q_heads//self.kv_heads
        k_rep=k.repeat_interleave(rep,dim=2); v_rep=v.repeat_interleave(rep,dim=2)
        scores=torch.einsum('bthd,bshd->bhts',q,k_rep)/math.sqrt(self.head_dim)
        if mask is not None: scores=scores+mask
        attn=F.softmax(scores,dim=-1)
        out=torch.einsum('bhts,bshd->bthd',attn,v_rep).reshape(B,T,self.q_heads*self.head_dim)
        return self.o(out)
class SwiGLU(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.gate=nn.Linear(cfg["dim"],cfg["hidden"],bias=False); self.up=nn.Linear(cfg["dim"],cfg["hidden"],bias=False); self.down=nn.Linear(cfg["hidden"],cfg["dim"],bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))
class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.rms1=RMSNorm(cfg["dim"]); self.attn=GQAAttn(cfg); self.rms2=RMSNorm(cfg["dim"]); self.ffn=SwiGLU(cfg)
    def forward(self,x,cos,sin,mask):
        x=x+self.attn(self.rms1(x),cos,sin,mask); x=x+self.ffn(self.rms2(x)); return x
class WISE_RONIN_V2(nn.Module):
    def __init__(self,cfg=CONFIG):
        super().__init__(); self.cfg=cfg; self.tok_emb=nn.Embedding(cfg["vocab"],cfg["dim"]); self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg["layers"])]); self.rms_f=RMSNorm(cfg["dim"])
        cos,sin=rope_freqs(cfg["head_dim"],cfg["ctx"],device="cpu"); self.register_buffer("cos",cos,persistent=False); self.register_buffer("sin",sin,persistent=False)
    def forward(self,ids):
        B,T=ids.shape; x=self.tok_emb(ids); mask=torch.triu(torch.ones(T,T,device=ids.device)*-1e9,diagonal=1)
        for blk in self.blocks: x=blk(x,self.cos,self.sin,mask)
        x=self.rms_f(x); return F.linear(x,self.tok_emb.weight)
def sft_loss(logits,labels):
    return F.cross_entropy(logits[:,:-1,:].contiguous().view(-1,logits.size(-1)),labels[:,1:].contiguous().view(-1),ignore_index=-100)
def orpo_loss(logits_c,labels_c,logits_r,labels_r,beta=0.1):
    def seq_logprob(logits,labels):
        logp=F.log_softmax(logits[:,:-1,:],dim=-1); gathered=torch.gather(logp,2,labels[:,1:].clamp(min=0).unsqueeze(-1)).squeeze(-1); mask=(labels[:,1:]!=-100).float(); return (gathered*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
    logp_c=seq_logprob(logits_c,labels_c); logp_r=seq_logprob(logits_r,labels_r)
    odds_c=torch.log(torch.sigmoid(logp_c)+1e-9)-torch.log(1-torch.sigmoid(logp_c)+1e-9)
    odds_r=torch.log(torch.sigmoid(logp_r)+1e-9)-torch.log(1-torch.sigmoid(logp_r)+1e-9)
    orpo=-F.logsigmoid(beta*(odds_c-odds_r)).mean(); nll=-logp_c.mean(); return nll+beta*orpo, nll.item(), orpo.item()
def load_jsonl(path):
    out=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out
def train(args):
    if not HAS_TORCH or not HAS_TOKENIZERS:
        print("Instala: pip install torch tokenizers"); return
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"[WISE-RONIN] device={device} config={CONFIG}")
    tokenizer=get_or_train_tokenizer([args.data],vocab_size=CONFIG["vocab"],save_path="tokenizer-wise-ronin.json")
    actual=tokenizer.get_vocab_size(); print(f"[WISE-RONIN] tokenizer vocab={actual}")
    if actual!=CONFIG["vocab"]:
        CONFIG["vocab"]=actual; print(f"[WISE-RONIN] Ajustando config vocab a {actual}")
    data=load_jsonl(args.data); print(f"[WISE-RONIN] cargados {len(data)}")
    seen=set(); deduped=[]
    for r in data:
        k=(r.get("user",""),r.get("assistant",""))
        if k not in seen: seen.add(k); deduped.append(r)
    print(f"[WISE-RONIN] dedup {len(data)} -> {len(deduped)}")
    data=deduped; random.shuffle(data)
    print("[WISE-RONIN] Tokenizando con BPE real + mascara assistant-only...")
    tokenized=[]
    for r in data:
        ids,labels=encode_example(tokenizer,r,max_len=CONFIG["ctx"]); tokenized.append((ids,labels,r))
    print(f"[WISE-RONIN] tokenizados {len(tokenized)}")
    model=WISE_RONIN_V2(CONFIG).to(device); n_params=sum(p.numel() for p in model.parameters()); print(f"[WISE-RONIN] params={n_params} -> {n_params*4/1024/1024:.2f} MB")
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.1,betas=(0.9,0.95),eps=1e-8)
    total_steps=(len(tokenized)//args.batch)*args.epochs; warmup=int(total_steps*0.05)
    print(f"[WISE-RONIN] total_steps={total_steps} warmup={warmup} epochs={args.epochs} lr={args.lr} orpo={args.orpo}")
    def lr_at(step):
        if step<warmup: return args.lr*(step+1)/warmup
        prog=(step-warmup)/max(1,total_steps-warmup); return args.lr*(0.1+0.9*0.5*(1+math.cos(math.pi*prog)))
    model.train(); step=0
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ==="); random.shuffle(tokenized)
        for i in range(0,len(tokenized),args.batch):
            batch=tokenized[i:i+args.batch]
            ids=torch.tensor([ids for ids,_,_ in batch],dtype=torch.long,device=device)
            labels=torch.tensor([lb for _,lb,_ in batch],dtype=torch.long,device=device)
            lr=lr_at(step)
            for g in opt.param_groups: g["lr"]=lr
            opt.zero_grad(); logits=model(ids); loss=sft_loss(logits,labels); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            if step%100==0: print(f"step {step}/{total_steps} lr={lr:.2e} loss={loss.item():.4f}")
            step+=1
            if step>=total_steps: break
        ckpt=f"wise-ronin_v2_epoch{epoch+1}.pt"; torch.save({"model":model.state_dict(),"config":CONFIG,"epoch":epoch+1,"tokenizer":"tokenizer-wise-ronin.json"},ckpt); print(f"[WISE-RONIN] checkpoint: {ckpt}")
    print("\n[WISE-RONIN] Train terminado. Evalua con holdout fijo, NUNCA entrenes con holdout.")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="dataset_enano_70mb.jsonl"); ap.add_argument("--holdout",default="holdout_wise_ronin_500.jsonl"); ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--batch",type=int,default=8); ap.add_argument("--orpo",type=bool,default=True); ap.add_argument("--beta",type=float,default=0.1); args=ap.parse_args(); train(args)
