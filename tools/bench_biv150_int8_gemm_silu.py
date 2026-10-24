"""F09 complete gate component comparison against unchanged team main.

No service hook; quantizer/GEMM-output/output allocation are included in timing.
"""

import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import traceback
from pathlib import Path

import torch

from biv150_benchmark_utils import graph_time
from biv150_int8_gemm_silu import CONFIGS,full_gate,from_quantized


def main(args):
    with socket.socket() as port:assert port.connect_ex(("127.0.0.1",9031))!=0
    assert "No running processes found" in subprocess.check_output(["/usr/local/corex/bin/ixsmi"],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"]=str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library,quantize_weight
    load_library()
    torch.manual_seed(20261025)
    weight=torch.randn(12288,2048,device="cuda",dtype=torch.bfloat16)*.02
    wq,sw=quantize_weight(weight)
    marker=torch.zeros((),device="cuda",dtype=torch.bfloat16).expand_as(weight)
    del weight
    columns=torch.tensor([0,1,31,127,1023,3071,6143],device="cuda")
    both=torch.cat((columns,columns+6144))
    host_w=wq[:,both].cpu().long();host_s=sw[both].cpu()
    result=dict(component_only=True,service_installed=False,rows=list(args.rows),
                no_int32_output_tensor=True,weight_repacked=False,
                commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),records=[])
    save=lambda:(args.out/"result.json").write_text(json.dumps(result,indent=2))
    for rows in args.rows:
        x=torch.empty(rows,2048,device="cuda",dtype=torch.bfloat16)
        records=[dict(m=rows,config=list(c),patterns=[],passed=True) for c in CONFIGS]
        for pattern in ("zero","tiny","normal","large","outlier"):
            x.normal_()
            if pattern=="zero":x.zero_()
            elif pattern=="tiny":x.mul_(1e-9)
            elif pattern=="large":x.mul_(100)
            elif pattern=="outlier":x[:,0]=32
            original=torch.ops.biv150_native.gate_silu_auto(x,wq,sw,marker,17)
            aq,sa=torch.ops.biv150_native.quantize_rows_probe(x,256)
            host_acc=aq[:rows].cpu().long()@host_w
            assert bool((host_acc.abs()<=2048*127*127).all())
            host_linear=(host_acc.float()*(sa[:rows].cpu()[:,None]*host_s[None,:])).to(torch.bfloat16).float()
            gate,up=host_linear.chunk(2,dim=1)
            gold=((gate/(1+torch.exp(-gate)))*up).to(torch.bfloat16)
            torch.testing.assert_close(original[:,columns].cpu(),gold,atol=.005,rtol=.005)
            for config,row in zip(CONFIGS,records):
                if not row["passed"]:continue
                try:
                    output=from_quantized(aq,sa,wq,sw,rows,config)
                    torch.testing.assert_close(output,original,atol=.005,rtol=.005)
                    torch.testing.assert_close(output[:,columns].cpu(),gold,atol=.005,rtol=.005)
                    row["patterns"].append(dict(name=pattern,full_native_comparison=True,cpu_int64_dot=True))
                except Exception as error:
                    row.update(passed=False,failed_pattern=pattern,error=repr(error),traceback=traceback.format_exc())
            del original,aq,sa
        for config,row in zip(CONFIGS,records):
            if not row["passed"]:continue
            try:
                x.normal_()
                def baseline():return torch.ops.biv150_native.gate_silu_auto(x,wq,sw,marker,17)
                def candidate():return full_gate(x,wq,sw,config)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):observed=candidate()
                for factor in (0.0,.5,4.0):
                    x.normal_().mul_(factor);graph.replay();torch.cuda.synchronize()
                    torch.testing.assert_close(observed,baseline(),atol=.005,rtol=.005)
                del graph,observed
                timings=[]
                for iteration in range(3):
                    calls=[("baseline_ms",baseline),("candidate_ms",candidate)]
                    if iteration%2:calls.reverse()
                    t={name:graph_time(fn,repeats=11,unroll=8)[0] for name,fn in calls}
                    t["speedup"]=t["baseline_ms"]/t["candidate_ms"];timings.append(t)
                row.update(rounds=timings,changed_input_graph=True,
                    median_speedup=statistics.median(t["speedup"] for t in timings),
                    min_round_speedup=min(t["speedup"] for t in timings),
                    avoided_int32_global_bytes=max(32,rows)*12288*4,
                    full_call_includes_quantizer=True,full_call_includes_output_allocation=True)
            except Exception as error:row.update(passed=False,failed_stage="graph_or_timing",error=repr(error),traceback=traceback.format_exc())
        result["records"].extend(records);save()
        print(json.dumps(dict(m=rows,passed=sum(a['passed'] for a in records),total=len(records))),flush=True)
        if not any(a["passed"] for a in records) and rows==args.rows[0]:
            result["fatal_error"]="No initial configuration passed; stop before remaining matrix";save()
            raise RuntimeError(result["fatal_error"])
    result["summary"]=dict(records=len(result['records']),passed=sum(a['passed'] for a in result['records']))
    result["diagnostic_completed"]=True;save()
    print(json.dumps(result['summary']),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--rows",type=int,nargs="+",default=[17,32,40,64,256,2048])
    main(parser.parse_args())
