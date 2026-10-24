"""F09 complete gate component comparison against unchanged team main.

No service hook; quantizer/GEMM-output/output allocation are included in timing.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import traceback
from pathlib import Path

import torch

from biv150_benchmark_utils import graph_time
from biv150_int8_gemm_silu import CONFIGS,full_gate,from_quantized,fused_gate



RESOURCE_CONFIGS=((32,64,128,4),(64,32,128,4),(64,64,128,4),(64,64,128,8))


def tiled_from_quantized(aq,sa,wq,sw,rows,config,resource=None,artifact_dir=None):
    assert 17<=rows<=2048 and config in RESOURCE_CONFIGS
    assert aq.shape==(max(32,rows),2048) and aq.dtype==torch.int8 and aq.is_contiguous()
    assert wq.shape==(2048,12288) and wq.dtype==torch.int8 and wq.is_contiguous()
    assert sa.shape==(max(32,rows),) and sw.shape==(12288,)
    assert sa.dtype==sw.dtype==torch.float32 and sa.is_contiguous() and sw.is_contiguous()
    assert aq.is_cuda and aq.device==sa.device==wq.device==sw.device
    output=torch.empty((rows,6144),device=aq.device,dtype=torch.bfloat16)
    bm,bn,bk,warps=config
    kernel=fused_gate[((rows+bm-1)//bm,(6144+bn-1)//bn)](
        aq,wq,sa,sw,output,rows,bm,bn,bk,num_warps=warps,num_stages=1)
    if resource is not None and not resource:
        try:
            metadata=kernel.metadata
            resource.update(grid=[(rows+bm-1)//bm,(6144+bn-1)//bn],
                n_regs_api=getattr(kernel,'n_regs',None),n_spills_api=getattr(kernel,'n_spills',None),
                shared_bytes=getattr(metadata,'shared',None),warp_size=getattr(metadata,'warp_size',None),
                num_warps=warps,explicit_accumulator_elements=2*bm*bn,artifacts={})
            artifact_dir.mkdir(parents=True,exist_ok=True)
            for name in ('ttir','ttgir','llir','asm','ptx'):
                value=kernel.asm.get(name)
                if not isinstance(value,str):continue
                path=artifact_dir/(name+'.txt');path.write_text(value)
                resource['artifacts'][name]=dict(path=str(path),sha256=hashlib.sha256(value.encode()).hexdigest())
                if name in ('asm','ptx'):
                    for key in ('vgpr_count','vgpr_spill_count','sgpr_spill_count','private_segment_fixed_size'):
                        found=re.search(r'\.'+key+r':\s*(\d+)',value)
                        if found:resource[key]=int(found.group(1))
        except Exception as error:
            resource.update(observation_error=repr(error),traceback=traceback.format_exc())
    return output


def main(args):
    with socket.socket() as port:assert port.connect_ex(("127.0.0.1",9031))!=0
    assert "No running processes found" in subprocess.check_output(["/usr/local/corex/bin/ixsmi"],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"]=str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library,quantize_weight
    load_library()
    policy=None
    if args.resource_tiles:
        import flag_gems
        from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
        assert use_flaggems()
        whitelist,blacklist=get_flag_gems_whitelist_blacklist()
        assert not whitelist and not blacklist
        before=torch._C._dispatch_dump_table('aten::_int_mm')
        flag_gems.enable(record=False,once=True)
        policy=dict(before_int_mm=before,after_int_mm=torch._C._dispatch_dump_table('aten::_int_mm'),
                    whitelist=whitelist,blacklist=blacklist,diagnostic_record=False,worker_record=True)
    torch.manual_seed(20261025)
    weight_identity=[]
    if args.weight_layer is None:
        weight=torch.randn(12288,2048,device="cuda",dtype=torch.bfloat16)*.02
    else:
        from safetensors import safe_open
        assert 4<=args.weight_layer<=37
        locations=json.loads((args.model/'model.safetensors.index.json').read_text())['weight_map']
        parts=[]
        for name in ('gate','up'):
            key=f'model.layers.{args.weight_layer}.mlp.{name}_proj.weight'
            with safe_open(args.model/locations[key],framework='pt',device='cpu') as handle:part=handle.get_tensor(key)
            weight_identity.append(dict(key=key,sha256=hashlib.sha256(part.view(torch.uint8).numpy().tobytes()).hexdigest()))
            parts.append(part)
        weight=torch.cat(parts).to('cuda')
        del parts,part
    configs=RESOURCE_CONFIGS if args.resource_tiles else CONFIGS
    wq,sw=quantize_weight(weight)
    marker=torch.zeros((),device="cuda",dtype=torch.bfloat16).expand_as(weight)
    del weight
    columns=torch.tensor([0,1,31,127,1023,3071,6143],device="cuda")
    both=torch.cat((columns,columns+6144))
    host_w=wq[:,both].cpu().long();host_s=sw[both].cpu()
    result=dict(component_only=True,service_installed=False,rows=list(args.rows),
                weight_identity=weight_identity,resource_tiles=args.resource_tiles,worker_policy=policy,
                inference_mode=torch.is_inference_mode_enabled(),
                no_int32_output_tensor=True,weight_repacked=False,
                commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),records=[])
    save=lambda:(args.out/"result.json").write_text(json.dumps(result,indent=2))
    for rows in args.rows:
        x=torch.empty(rows,2048,device="cuda",dtype=torch.bfloat16)
        records=[dict(m=rows,config=list(c),patterns=[],passed=True) for c in configs]
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
            for config,row in zip(configs,records):
                if not row["passed"]:continue
                try:
                    if args.resource_tiles:
                        output=tiled_from_quantized(aq,sa,wq,sw,rows,config,
                            row.setdefault('resources',{}),args.out/'compiled'/(f'm{rows}-'+'-'.join(map(str,config))))
                    else:output=from_quantized(aq,sa,wq,sw,rows,config)
                    torch.testing.assert_close(output,original,atol=.005,rtol=.005)
                    torch.testing.assert_close(output[:,columns].cpu(),gold,atol=.005,rtol=.005)
                    row["patterns"].append(dict(name=pattern,full_native_comparison=True,cpu_int64_dot=True))
                except Exception as error:
                    row.update(passed=False,failed_pattern=pattern,error=repr(error),traceback=traceback.format_exc())
            del original,aq,sa
        for config,row in zip(configs,records):
            if not row["passed"]:continue
            try:
                x.normal_()
                def baseline():return torch.ops.biv150_native.gate_silu_auto(x,wq,sw,marker,17)
                def candidate():
                    if not args.resource_tiles:return full_gate(x,wq,sw,config)
                    q,scale=torch.ops.biv150_native.quantize_rows_probe(x,256)
                    return tiled_from_quantized(q,scale,wq,sw,rows,config)
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
    parser.add_argument('--resource-tiles',action='store_true')
    parser.add_argument('--weight-layer',type=int)
    parser.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'))
    args=parser.parse_args()
    if args.resource_tiles:
        assert args.weight_layer is not None
        with torch.inference_mode():main(args)
    else:main(args)
