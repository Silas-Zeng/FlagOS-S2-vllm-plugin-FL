"""Verify the isolated BF16 reference against actual FlagGems serving dispatch."""
import argparse
import json
from pathlib import Path
import socket
import torch
import torch.nn.functional as F
import flag_gems
from biv150_benchmark_utils import graph_time


def names(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn();torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA})


def main(args):
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261023);rows=[]
    for m in [32,64,2048]:
        for n,k in [(2560,2048),(2048,2048),(12288,2048),(2048,6144)]:
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
            vendor=lambda:F.linear(x,w)
            gems=lambda:flag_gems.mm(x,w.t())
            torch.testing.assert_close(gems(),vendor(),atol=.005,rtol=.005)
            row=dict(m=m,n=n,k=k,vendor_ms=graph_time(vendor,repeats=9,unroll=4)[0],gems_ms=graph_time(gems,repeats=9,unroll=4)[0])
            if m==32 and n==12288:row.update(vendor_kernels=names(vendor),gems_kernels=names(gems))
            rows.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(rows,indent=2))
            del x,w
            torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
