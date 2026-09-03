"""Paper-comparison experiments for B-role Longformer/Memformer.

The runner deliberately separates paper-reported facts, method-aligned
mechanism checks, and causal paired comparisons. It does not label synthetic
data or attention-only adapters as strict paper reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from b_role_experiments import (  # noqa: E402
    LongformerSelfAttention,
    MemformerSegmentAttention,
    StandardAttention,
    config_hash,
    env_meta,
    seed_all,
)


def timed(model: nn.Module, x: torch.Tensor, runs: int = 3, warmup: int = 2) -> dict[str, Any]:
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(warmup): _ = model(x)
            if x.device.type == "cuda":
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(runs):
                if x.device.type == "cuda":
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); _ = model(x); b.record(); b.synchronize(); samples.append(a.elapsed_time(b)/1000.0)
                else:
                    t=time.perf_counter(); _=model(x); samples.append(time.perf_counter()-t)
            if x.device.type == "cuda":
                alloc=torch.cuda.max_memory_allocated()/2**20; reserved=torch.cuda.max_memory_reserved()/2**20
            else: alloc=reserved=float("nan")
        med=float(torch.tensor(samples).median())
        return {"status":"ok","median_latency_s":med,"mean_latency_s":sum(samples)/len(samples),
                "tokens_per_s":x.shape[1]/med,"peak_allocated_mb":alloc,
                "peak_reserved_mb":reserved,"timed_samples_s":samples}
    except torch.cuda.OutOfMemoryError as exc:
        return {"status":"oom","error":repr(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status":"error","error":repr(exc)}


class PaperMemformerAttention(nn.Module):
    """One-layer official MemBartEncoderAttention with MemBART-base dimensions."""
    def __init__(self, dim: int = 768, heads: int = 12, memory_len: int = 64):
        super().__init__()
        path = ROOT / "memformers" / "memformers" / "models" / "membart" / "membart_attention.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("paper_memformer_attention", path)
        if spec is None or spec.loader is None: raise ImportError(path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.attention = mod.MemBartEncoderAttention(embed_dim=dim, num_heads=heads, dropout=0.0)
        self.memory_len = memory_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = torch.zeros(x.shape[0], self.memory_len, x.shape[-1], device=x.device, dtype=x.dtype)
        out, _, _ = self.attention(x, memory_states=memory)
        return out


class StandardPaperAttention(nn.Module):
    def __init__(self, dim: int, heads: int, causal: bool):
        super().__init__(); self.inner=StandardAttention(dim, heads, causal=causal)
    def forward(self, x): return self.inner(x)


def run_longformer_paper(device: torch.device, dtype: torch.dtype, lengths: list[int], runs: int) -> dict[str, Any]:
    """Longformer text8/enwik8 mechanism/length sanity check.

    The original character datasets are not installed; random character-like
    embeddings preserve sequence shape only. Therefore this is method-aligned,
    not a paper metric reproduction.
    """
    rows=[]
    cfg_base={"layers":12,"hidden_size":512,"heads":8,"dropout":0.2,
              "optimizer":"AdamW","weight_decay":0.01,"grad_clip":0.25,
              "position":"Transformer-XL relative + sinusoidal","dataset":"text8/enwik8 (unavailable locally)"}
    for n in lengths:
        seed_all(17); x=torch.randn(1,n,512,device=device,dtype=dtype)
        variants={
            "paper_full_reference": StandardPaperAttention(512,8,causal=True).to(device,dtype),
            "paper_local_window": LongformerSelfAttention(512,n,heads=8,window_size=513,global_tokens=0,causal=True).to(device,dtype),
            "paper_local_global": LongformerSelfAttention(512,n,heads=8,window_size=513,global_tokens=1,causal=True).to(device,dtype),
        }
        for name, model in variants.items():
            c={**cfg_base,"seq_len":n,"variant":name,"window":513 if "local" in name else None,
               "global_tokens":1 if name.endswith("global") else 0,"semantic":"causal"}
            rows.append({"run_id":f"paper-longformer-{name}-{n}-s17","method":"Longformer" if name!="paper_full_reference" else "Standard",
                         "variant":name,"label":"method_aligned","seed":17,"config_hash":config_hash(c),**c,
                         **timed(model,x,runs=runs)})
            del model
            if device.type=="cuda": torch.cuda.empty_cache()
    return {"paper_reference": cfg_base, "records": rows,
            "note":"Original text8/enwik8 and 5-phase training were unavailable; this measures the paper's local/global mechanism and length trend at d=512, heads=8."}


def run_memformer_paper(device: torch.device, dtype: torch.dtype, lengths: list[int], runs: int) -> dict[str, Any]:
    rows=[]
    cfg_base={"backbone":"BART-base","vocab_size":50265,"d_model":768,"encoder_layers":6,
              "decoder_layers":6,"heads":12,"ffn_dim":3072,"max_position":1024,
              "memory_length":64,"activation":"GELU","dropout":0.0,"semantic":"bidirectional encoder"}
    for n in lengths:
        seed_all(17); x=torch.randn(1,n,768,device=device,dtype=dtype)
        variants={"paper_full_reference":StandardPaperAttention(768,12,causal=False).to(device,dtype),
                  "paper_memformer_attention":PaperMemformerAttention(768,12,64).to(device,dtype)}
        for name,model in variants.items():
            c={**cfg_base,"seq_len":n,"variant":name}
            rows.append({"run_id":f"paper-memformer-{name}-{n}-s17","method":"Memformer" if "memformer" in name else "Standard",
                         "variant":name,"label":"method_aligned","seed":17,"config_hash":config_hash(c),**c,
                         **timed(model,x,runs=runs)})
            del model
            if device.type=="cuda": torch.cuda.empty_cache()
    # State capacity and update sanity with the project recurrent adapter.
    rec=MemformerSegmentAttention(768,12,memory_slots=64,causal=False,detach_memory=False).to(device,dtype)
    x1=torch.randn(1,128,768,device=device,dtype=dtype); x2=torch.randn(1,128,768,device=device,dtype=dtype)
    with torch.inference_mode(): _,m1=rec(x1,segment_length=128); _,m2=rec(x2,segment_length=128,memory=m1)
    return {"paper_reference":cfg_base,"records":rows,
            "state_check":{"memory_shape":list(m1.shape),"state_delta_between_segments":float((m2-m1).abs().mean()),
                            "state_elements":64*768,"state_bytes_bf16":64*768*2},
            "note":"Official MemBart attention is used for the one-layer paper-config timing; full BART training/checkpoint is not run."}


class PairTinyLM(nn.Module):
    def __init__(self, method: str, dim: int, layers: int, heads: int, seq_len: int, bidirectional: bool = False):
        super().__init__(); self.method=method; self.vocab=64; self.seq_len=seq_len
        self.tok=nn.Embedding(self.vocab,dim); self.pos=nn.Embedding(seq_len,dim); self.blocks=nn.ModuleList()
        for _ in range(layers):
            if method=="Standard": attn=StandardAttention(dim,heads,causal=not bidirectional)
            elif method=="Longformer": attn=LongformerSelfAttention(dim,seq_len,heads=heads,window_size=65,global_tokens=1,causal=True)
            else: attn=MemformerSegmentAttention(dim,heads,memory_slots=32,causal=True,detach_memory=False)
            self.blocks.append(nn.ModuleList([nn.LayerNorm(dim),attn,nn.LayerNorm(dim),nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))]))
        self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,self.vocab,bias=False)
    def forward(self,idx):
        n=idx.shape[1]; x=self.tok(idx)+self.pos(torch.arange(n,device=idx.device))[None]
        for ln,attn,ln2,mlp in self.blocks:
            a,_=attn(ln(x),segment_length=min(64,n)) if self.method=="Memformer" else (attn(ln(x)),None)
            x=x+a; x=x+mlp(ln2(x))
        return self.head(self.norm(x))


def run_causal_pair(device: torch.device, dtype: torch.dtype, seeds: list[int], steps: int) -> dict[str, Any]:
    rows=[]; dim=256; layers=4; heads=8; n=128; batch=4
    for seed in seeds:
        seed_all(seed); gen=torch.Generator(device="cpu").manual_seed(seed)
        base=torch.randint(64,(batch,n+1),generator=gen); base[:,1:]=(base[:,:-1]+3)%64
        idx,target=base[:,:-1].to(device),base[:,1:].to(device)
        for variant, method, bidi in (("C1_bidirectional","Standard",True),("C2_causal","Standard",False),("C3_longformer_causal","Longformer",False),("C3_memformer_causal","Memformer",False)):
            seed_all(seed); model=PairTinyLM(method,dim,layers,heads,n,bidirectional=bidi).to(device,dtype)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4); model.train(); losses=[]
            for _ in range(steps):
                logits=model(idx); loss=F.cross_entropy(logits.float().reshape(-1,64),target.reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.item()))
            model.eval()
            with torch.inference_mode(): ev=float(F.cross_entropy(model(idx).float().reshape(-1,64),target.reshape(-1)).item())
            c={"variant":variant,"seed":seed,"dim":dim,"layers":layers,"heads":heads,"seq_len":n,"steps":steps,"dataset":"synthetic_copy_stream_fallback"}
            rows.append({"run_id":f"causal-{variant}-s{seed}","method":method,"variant":variant,"label":"method_aligned","mask":"bidirectional" if bidi else "causal","eval_nll":ev,"eval_ppl":math.exp(ev),"train_loss_last":losses[-1],"config_hash":config_hash(c),**c})
            del model
            if device.type=="cuda": torch.cuda.empty_cache()
    # Aggregate mean/std and C2-C1, C3-C2 differences.
    agg=[]
    for variant in ("C1_bidirectional","C2_causal","C3_longformer_causal","C3_memformer_causal"):
        vals=[r["eval_ppl"] for r in rows if r["variant"]==variant]; mean=sum(vals)/len(vals); sd=(sum((v-mean)**2 for v in vals)/max(1,len(vals)-1))**0.5
        # 95% t interval for n=3 (df=2); report explicitly rather than using
        # a misleading normal approximation.
        half_width = 4.303 * sd / math.sqrt(len(vals))
        agg.append({"variant":variant,"mean_ppl":mean,"std_ppl":sd,"n":len(vals),
                    "ci95_low":mean-half_width,"ci95_high":mean+half_width})
    lookup={a["variant"]:a["mean_ppl"] for a in agg}
    return {"records":rows,"aggregate":agg,"differences":{"causal_minus_bidirectional_ppl":lookup["C2_causal"]-lookup["C1_bidirectional"],"longformer_structure_minus_causal_ppl":lookup["C3_longformer_causal"]-lookup["C2_causal"],"memformer_structure_minus_causal_ppl":lookup["C3_memformer_causal"]-lookup["C2_causal"]},"note":"C1/C2/C3 use identical synthetic data, optimizer, backbone size and three seeds; only mask/structure changes."}


def make_figures(out: Path, lf: dict[str,Any], mf: dict[str,Any], cp: dict[str,Any]) -> None:
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for method in ("Standard","Longformer"):
        rs=[r for r in lf["records"] if r["method"]==method and r["status"]=="ok"]; rs.sort(key=lambda r:r["seq_len"])
        if rs: ax[0].plot([r["seq_len"] for r in rs],[r["median_latency_s"]*1000 for r in rs],"o-",label=method+"/Longformer paper")
    for method in ("Standard","Memformer"):
        rs=[r for r in mf["records"] if r["method"]==method and r["status"]=="ok"]; rs.sort(key=lambda r:r["seq_len"])
        if rs: ax[1].plot([r["seq_len"] for r in rs],[r["median_latency_s"]*1000 for r in rs],"o-",label=method+"/Memformer paper")
    ax[0].set(xscale="log",yscale="log",xlabel="Sequence length",ylabel="Median latency (ms)",title="Longformer paper-config sanity")
    ax[1].set(xlabel="Sequence length",ylabel="Median latency (ms)",title="Memformer paper-config sanity")
    for a in ax:a.grid(alpha=.25);a.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"paper_alignment_latency.png",dpi=180,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); ag=cp["aggregate"]; ax.bar([a["variant"].replace("_","\n") for a in ag],[a["mean_ppl"] for a in ag],yerr=[a["std_ppl"] for a in ag],capsize=4,color=["#7aa6d8","#4c78a8","#f58518","#54a24b"]); ax.set_ylabel("PPL (mean ± SD)"); ax.set_title("Causal paired comparison (3 seeds)"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(out/"causal_pair_ppl.png",dpi=180,bbox_inches="tight");plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--out-dir",default=str(ROOT/"results"/"paper_comparison_b")); p.add_argument("--runs",type=int,default=3); p.add_argument("--steps",type=int,default=8); p.add_argument("--longformer-lengths",type=int,nargs="+",default=[2048,4096]); p.add_argument("--memformer-lengths",type=int,nargs="+",default=[128,256,512]); args=p.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dtype=torch.bfloat16 if device.type=="cuda" else torch.float32
    lf=run_longformer_paper(device,dtype,args.longformer_lengths,args.runs); mf=run_memformer_paper(device,dtype,args.memformer_lengths,args.runs); cp=run_causal_pair(device,dtype,[17,29,43],args.steps)
    payloads={"longformer_paper_aligned":lf,"memformer_paper_aligned":mf,"causal_pair":cp}
    for name,payload in payloads.items(): (out/(name+".json")).write_text(json.dumps({"environment":env_meta(device,dtype),**payload},indent=2),encoding="utf-8")
    make_figures(out,lf,mf,cp); print(json.dumps({"out_dir":str(out),"longformer_records":len(lf["records"]),"memformer_records":len(mf["records"]),"causal_records":len(cp["records"]),"causal_aggregate":cp["aggregate"],"differences":cp["differences"]},indent=2))


if __name__=="__main__": main()
