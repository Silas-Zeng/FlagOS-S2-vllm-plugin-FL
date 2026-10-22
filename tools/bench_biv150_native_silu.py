"""Compare fused int8 GEMM epilogue/SwiGLU with the installed FlagGems activation."""
import argparse
import importlib.util
import json
import socket
from pathlib import Path

import torch
import torch.nn.functional as F
from flag_gems.modules.activation import gems_silu_and_mul as silu_and_mul

from bench_biv150_fastpaths import graph_time


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
        fn=lambda:torch.ops.biv150_native.gate_silu(x,wq,sw,marker,17)
        ref=old()
        got=fn()
        torch.testing.assert_close(got,ref,atol=.005,rtol=.005)
        diff=(got.float()-ref.float()).abs().max().item()
        a=graph_time(old,repeats=9,unroll=4)[0]
        b=graph_time(fn,repeats=9,unroll=4)[0]
        baseline=graph_time(bf16,repeats=9,unroll=4)[0]
        compiled=torch.compile(fn,fullgraph=True,dynamic=True)
        torch.testing.assert_close(compiled(),got,atol=0,rtol=0)
        print(json.dumps(dict(m=m,old_ms=a,bf16_ms=baseline,fused_ms=b,speedup=a/b,
                             speedup_vs_bf16=baseline/b,max_abs_diff=diff,fullgraph="passed")),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--library",type=Path,required=True)
    main(p.parse_args())
