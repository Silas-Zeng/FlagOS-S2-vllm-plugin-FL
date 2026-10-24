"""Replay fixed failed norm fixtures with explicit reduction trees, no model hook.

Changes the reduction association only. Does not promote a layer/shape because
it happened to pass a saved fixture; any new choice needs a full fresh matrix.
"""

import argparse
import hashlib
import importlib
import json
import socket
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from biv150_bounded_norm_quant import norm_quant


@triton.jit
def chunked_norm(X,R,W,Y,R_OUT,M:tl.constexpr,EPS:tl.constexpr,
                 CHUNK:tl.constexpr,XBLOCK:tl.constexpr):
    rows=tl.program_id(0)*XBLOCK+tl.arange(0,XBLOCK)
    lane=tl.arange(0,CHUNK)
    # Match native IR's serial accumulation of chunks before its lane reduction.
    accum=tl.zeros((XBLOCK,CHUNK),tl.float32)
    for start in range(0,2048,CHUNK):
        cols=start+lane
        x=tl.load(X+rows[:,None]*2048+cols[None,:],rows[:,None]<M,other=0).to(tl.float32)
        r=tl.load(R+rows[:,None]*2048+cols[None,:],rows[:,None]<M,other=0).to(tl.float32)
        value=x+r
        accum=accum+value*value
    total=tl.sum(accum,1)
    rrms=libdevice.rsqrt(total/2048.0+EPS)
    for start in range(0,2048,CHUNK):
        cols=start+lane
        x=tl.load(X+rows[:,None]*2048+cols[None,:],rows[:,None]<M,other=0).to(tl.float32)
        r=tl.load(R+rows[:,None]*2048+cols[None,:],rows[:,None]<M,other=0).to(tl.float32)
        w=tl.load(W+cols)
        value=x+r
        y=(value*rrms[:,None])*w[None,:].to(tl.float32)
        tl.store(Y+rows[:,None]*2048+cols[None,:],y,rows[:,None]<M)
        tl.store(R_OUT+rows[:,None]*2048+cols[None,:],value,rows[:,None]<M)


def compare(got,gold):
    g=got.cpu();h=gold.cpu();delta=(g.float()-h.float()).abs()
    return dict(bit_differences=int((g!=h).sum()),
        violations=int((delta>.005+.005*h.float().abs()).sum()),max_abs=float(delta.max()))


def main(args):
    with socket.socket() as port:assert port.connect_ex(("127.0.0.1",9031))!=0
    assert "No running processes found" in subprocess.check_output(["/usr/local/corex/bin/ixsmi"],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    from vllm.ir.ops.layernorm import fused_add_rms_norm
    pin=Path(importlib.import_module("vllm.ir.ops.layernorm").__file__)
    assert hashlib.sha256(pin.read_bytes()).hexdigest()=="d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910"
    compiled=torch.compile(fused_add_rms_norm.impls["native"].impl_fn,fullgraph=True,backend="inductor")
    old=json.loads((args.probe/"result.json").read_text())
    result=dict(diagnostic_only=True,service_installed=False,model_accuracy_verified=False,
        fixture_replay_only=True,configs=[],records=[],compiled_native=[])
    # Chunk size changes addition order. XBLOCK reflects grouping seen in native
    # Inductor IR; this is a bounded structural diagnostic, not performance tuning.
    configs=[(chunk,xblock) for chunk in (128,256,512,1024,2048) for xblock in (1,4)]
    result["configs"]=[list(c) for c in configs]
    for record in old["records"]:
        if record["passed"]:continue
        path=args.probe/record["fixture"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record["fixture_sha256"]
        data=torch.load(path,map_location="cpu",weights_only=True)
        x=data["x"].to("cuda");r=data["residual"].to("cuda");w=data["weight"].to("cuda")
        reference,res=compiled(x,r,w,data["epsilon"])
        _,_,current_r,current,_=norm_quant(x,r,w,data["epsilon"],write_norm=True)
        row=dict(m=record["m"],layer=record["layer"],pattern=data["pattern"],
                 fixture=record["fixture"],fixture_sha256=record["fixture_sha256"],
                 current_norm=compare(current,reference),current_residual=compare(current_r,res),trees=[])
        for chunk,xblock in configs:
            y=torch.empty_like(x);rr=torch.empty_like(r)
            chunked_norm[(triton.cdiv(x.shape[0],xblock),)](x,r,w,y,rr,x.shape[0],data["epsilon"],chunk,xblock,num_warps=4)
            row["trees"].append(dict(chunk=chunk,xblock=xblock,
                                      norm=compare(y,reference),residual=compare(rr,res)))
        result["records"].append(row)
        (args.out/"result.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(m=row["m"],layer=row["layer"],current=row["current_norm"],
            exact_trees=[(a["chunk"],a["xblock"]) for a in row["trees"] if a["norm"]["bit_differences"]==0])),flush=True)
    result["summary"]=[dict(chunk=c,xblock=xb,
        exact=sum(next(t for t in r["trees"] if (t["chunk"],t["xblock"])==(c,xb))["norm"]["bit_differences"]==0 for r in result["records"]),
        total=len(result["records"])) for c,xb in configs]
    # Persist numeric evidence before optional metadata inspection. The deployed
    # PyCodeCache has `modules`, not the older `cache` dictionary.
    result["numeric_diagnostic_completed"]=True
    (args.out/"result.json").write_text(json.dumps(result,indent=2))
    from torch._inductor.codecache import PyCodeCache
    for module in PyCodeCache.modules:
        for name,value in vars(module).items():
            if not type(value).__name__.endswith("CachingAutotuner"):continue
            configs=getattr(value,"configs",[])
            result["compiled_native"].append(dict(module_file=getattr(module,"__file__",None),
                name=name,type=type(value).__name__,configs=[dict(kwargs=c.kwargs,num_warps=c.num_warps,num_stages=c.num_stages) for c in configs],
                best_config=str(getattr(value,"best_config",None))))
    result["diagnostic_completed"]=True
    (args.out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["summary"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
