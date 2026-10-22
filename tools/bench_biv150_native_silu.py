"""Compare fused int8 GEMM epilogue/SwiGLU with the installed FlagGems activation."""
import argparse
import importlib.util
import json
import socket
from pathlib import Path

import torch
import torch.nn.functional as F
from flag_gems.modules.activation import gems_silu_and_mul as silu_and_mul

from biv150_benchmark_utils import graph_time
import triton
import triton.language as tl



@triton.jit
def compiled_reference(x,out,count:tl.constexpr,stride:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=index//6144
    col=index%6144
    gate=tl.load(x+row*stride+col,index<count,other=0).to(tl.float32)
    up=tl.load(x+row*stride+6144+col,index<count,other=0).to(tl.float32)
    value=(gate*tl.sigmoid(gate))*up
    tl.store(out+index,value,index<count)

def compiled_activation(x,block=1024,warps=4):
    out=torch.empty((x.shape[0],6144),dtype=x.dtype,device=x.device)
    compiled_reference[(triton.cdiv(out.numel(),block),)](
        x,out,out.numel(),x.stride(0),BLOCK=block,num_warps=warps)
    return out

def main(args):
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1",9031)) != 0, "A service is still running"
    torch.ops.load_library(str(args.library))
    @torch.library.register_fake("biv150_native::gate_silu")
    def fake(x,wq,scales,original,min_rows):
        return torch.empty((x.shape[0],wq.shape[1]//2),dtype=x.dtype,device=x.device)

    torch.manual_seed(20261023)
    w=torch.randn(12288,2048,device="cuda",dtype=torch.bfloat16)*.02
    sw=w.float().abs().amax(1).clamp_min(1.e-6)/127
    wq=torch.round(w.float()/sw[:,None]).clamp(-127,127).to(torch.int8).t().contiguous()
    marker=torch.zeros((),device="cuda",dtype=w.dtype).expand_as(w)
    for m in [1,8,16,32,64,2048,8192]:
        x=torch.randn(m,2048,device="cuda",dtype=w.dtype)
        def old():
            y=torch.ops.biv150_native.linear(x,wq,sw,marker,None,17)
            return silu_and_mul(y[:,:6144],y[:,6144:])
        def bf16():
            y=F.linear(x,w)
            return silu_and_mul(y[:,:6144],y[:,6144:])
        def compiled_bf16():
            return compiled_activation(F.linear(x,w))
        fn=lambda:torch.ops.biv150_native.gate_silu(x,wq,sw,marker,17)
        ref=old()
        got=fn()
        torch.testing.assert_close(got,ref,atol=.005,rtol=.005)
        diff=(got.float()-ref.float()).abs().max().item()
        qdq=torch.ops.biv150_native.linear(x,wq,sw,marker,None,17)
        compiled_ref=compiled_activation(qdq)
        torch.testing.assert_close(got,compiled_ref,atol=.005,rtol=.005)
        compiled_diff=(got.float()-compiled_ref.float()).abs().max().item()
        a=graph_time(old,repeats=9,unroll=4)[0]
        b=graph_time(fn,repeats=9,unroll=4)[0]
        baseline=graph_time(bf16,repeats=9,unroll=4)[0]
        compiled_baseline=min(graph_time(lambda:compiled_activation(F.linear(x,w),block,warps),repeats=9,unroll=4)[0]
                              for block,warps in [(256,4),(512,4),(1024,4),(2048,4),(4096,4),(2048,8)])
        compiled=torch.compile(fn,fullgraph=True,dynamic=True)
        torch.testing.assert_close(compiled(),got,atol=0,rtol=0)
        print(json.dumps(dict(m=m,old_ms=a,bf16_ms=baseline,fused_ms=b,speedup=a/b,
                             compiled_bf16_ms=compiled_baseline,speedup_vs_compiled_bf16=compiled_baseline/b,
                             max_abs_diff_vs_compiled_activation=compiled_diff,
                             speedup_vs_bf16=baseline/b,max_abs_diff=diff,fullgraph="passed")),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--library",type=Path,required=True)
    main(p.parse_args())
