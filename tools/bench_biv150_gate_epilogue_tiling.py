"""Numerical and warmed complete-gate checks for probe-only epilogue tiling."""
import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import graph_time


def cpu_epilogue(acc, rows, scales, m):
    cols=torch.tensor([0,1,31,127,1023,6143])
    both=torch.cat((cols,cols+6144))
    a=acc[:m,both].cpu().float()
    rs=rows.cpu()
    ws=scales[both.to(scales.device)].cpu()
    if m<=16:
        b=acc[16:16+m,both].cpu().float()
        residual=b*rs[16:16+m,None]
        linear=(a.double()*rs[:m,None].double()+residual.double()).float()*ws
    else:
        linear=a*(rs[:m,None]*ws)
    gate,up=linear.to(torch.bfloat16).float().chunk(2,-1)
    return cols,(torch.nn.functional.silu(gate)*up).to(torch.bfloat16)


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['BIV150_NATIVE_W8A8_LIBRARY']=str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library,quantize_weight
    load_library()
    torch.manual_seed(20261024)
    weight=torch.randn(12288,2048,device='cuda',dtype=torch.bfloat16)*0.02
    q,scales=quantize_weight(weight)
    marker=torch.zeros((),device='cuda',dtype=weight.dtype).expand_as(weight)
    result=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                service_installed=False,records=[])
    for m in (1,8,16,17,32,40,64,256,2048):
        padded=max(32,m)
        acc=torch.randint(-8000000,8000001,(padded,12288),device='cuda',dtype=torch.int32)
        rs=torch.rand(padded,device='cuda')*0.01+1e-4
        ws=torch.rand(12288,device='cuda')*0.001+1e-5
        iso_old=partial(torch.ops.biv150_native.epilogue_tiling_probe,acc,rs,ws,m,m<=16,0,1)
        cols,gold=cpu_epilogue(acc,rs,ws,m)
        torch.testing.assert_close(iso_old()[:,cols.to('cuda')].cpu(),gold,atol=0.005,rtol=0.005)
        x=torch.randn(m,2048,device='cuda',dtype=weight.dtype)
        old=partial(torch.ops.biv150_native.gate_silu_auto,x,q,scales,marker,17)
        for threads in (64,128,256):
            for items in (1,2,4,8):
                new=partial(torch.ops.biv150_native.gate_epilogue_tiling_probe,x,q,scales,threads,items)
                iso_new=partial(torch.ops.biv150_native.epilogue_tiling_probe,acc,rs,ws,m,m<=16,threads,items)
                row=dict(m=m,threads=threads,items=items,patterns=[])
                try:
                    torch.testing.assert_close(iso_new(),iso_old(),atol=0,rtol=0)
                    for pattern in ('zero','tiny','normal','large','outlier'):
                        x.normal_()
                        if pattern=='zero':x.zero_()
                        elif pattern=='tiny':x.mul_(1e-9)
                        elif pattern=='large':x.mul_(100)
                        elif pattern=='outlier':x[:,0]=32
                        torch.testing.assert_close(new(),old(),atol=0,rtol=0)
                        row['patterns'].append(pattern)
                    x.normal_()
                    rounds=[]
                    for i in range(3):
                        calls=[('baseline_ms',old),('candidate_ms',new)]
                        if i%2:calls.reverse()
                        times={name:graph_time(fn,repeats=11,unroll=4)[0] for name,fn in calls}
                        times['speedup']=times['baseline_ms']/times['candidate_ms']
                        rounds.append(times)
                    isolated=[]
                    for i in range(3):
                        calls=[('baseline_ms',iso_old),('candidate_ms',iso_new)]
                        if i%2:calls.reverse()
                        times={name:graph_time(fn,repeats=11,unroll=4)[0] for name,fn in calls}
                        times['speedup']=times['baseline_ms']/times['candidate_ms']
                        isolated.append(times)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=new()
                    for factor in (0.0,0.5,4.0):
                        x.normal_().mul_(factor)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(output,old(),atol=0,rtol=0)
                    del graph,output
                    row.update(passed=True,full_gate_bit_exact=True,isolated_bit_exact=True,
                               independent_cpu_epilogue=True,changed_input_graph=True,
                               rounds=rounds,isolated_rounds=isolated,
                               median_speedup=statistics.median(t['speedup'] for t in rounds),
                               isolated_median_speedup=statistics.median(t['speedup'] for t in isolated))
                except Exception as error:
                    row.update(passed=False,error=repr(error))
                    fixture=args.out/f'failed-m{m}-t{threads}-i{items}.pt'
                    torch.save(dict(x=x.cpu(),acc=acc.cpu(),row_scales=rs.cpu(),weight_scales=ws.cpu()),fixture)
                    row.update(fixture=fixture.name,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
                result['records'].append(row)
                (args.out/'result.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(row),flush=True)
    if len(result['records'])!=108 or not all(r['passed'] for r in result['records']):
        raise RuntimeError('Incomplete numeric gates; no production or service claim')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
