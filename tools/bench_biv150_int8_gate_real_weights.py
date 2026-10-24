"""Validate M32 or the measured M64 tile across all34 real gate/up weights."""

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
from safetensors import safe_open

from biv150_benchmark_utils import graph_time
from biv150_int8_gemm_silu import full_gate

CONFIG=(32,64,128,4)


def learned_weights(model):
    locations={}
    for path in sorted(model.glob('*.safetensors')):
        with safe_open(str(path),framework='pt',device='cpu') as h:
            for key in h.keys():
                if key.endswith(('.mlp.gate_proj.weight','.mlp.up_proj.weight')):
                    locations[key]=path
    for layer in range(4,38):
        tensors=[];identities=[]
        for projection in ('gate','up'):
            key=f'model.layers.{layer}.mlp.{projection}_proj.weight'
            with safe_open(str(locations[key]),framework='pt',device='cpu') as h:t=h.get_tensor(key)
            assert t.shape==(6144,2048) and t.dtype==torch.bfloat16
            identities.append(dict(key=key,sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()))
            tensors.append(t)
        yield layer,torch.cat(tensors),identities


def main(args):
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['BIV150_NATIVE_W8A8_LIBRARY']=str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library,quantize_weight
    load_library();torch.manual_seed(20261025)
    rows=64 if args.m64_resource_tile else 32
    config=(64,64,128,8) if args.m64_resource_tile else CONFIG
    policy=None
    if args.m64_resource_tile:
        import flag_gems
        from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
        from bench_biv150_int8_gemm_silu import tiled_from_quantized
        assert use_flaggems()
        whitelist,blacklist=get_flag_gems_whitelist_blacklist()
        assert not whitelist and not blacklist
        before=torch._C._dispatch_dump_table('aten::_int_mm')
        flag_gems.enable(record=False,once=True)
        policy=dict(before_int_mm=before,after_int_mm=torch._C._dispatch_dump_table('aten::_int_mm'),
                    whitelist=whitelist,blacklist=blacklist,diagnostic_record=False,worker_record=True)
        torch.set_num_threads(2)
    def evaluate(x,wq,sw):
        if not args.m64_resource_tile:return full_gate(x,wq,sw,CONFIG)
        q,scale=torch.ops.biv150_native.quantize_rows_probe(x,256)
        return tiled_from_quantized(q,scale,wq,sw,rows,config)
    result=dict(component_only=True,service_installed=False,real_projection_weights=True,
        synthetic_activations=True,model_accuracy_verified=False,rows=rows,config=list(config),
        worker_policy=policy,inference_mode=torch.is_inference_mode_enabled(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),records=[])
    save=lambda:(args.out/'result.json').write_text(json.dumps(result,indent=2))
    for layer,host_weight,identities in learned_weights(args.model):
        weight=host_weight.to('cuda');wq,sw=quantize_weight(weight)
        marker=torch.zeros((),device='cuda',dtype=torch.bfloat16).expand_as(weight)
        del weight,host_weight
        cols=torch.tensor([0,1,31,127,1023,3071,6143],device='cuda')
        both=torch.cat((cols,cols+6144));wc=wq[:,both].cpu().long();sc=sw[both].cpu()
        x=torch.empty(rows,2048,device='cuda',dtype=torch.bfloat16)
        row=dict(layer=layer,weights=identities,patterns=[])
        stage='patterns';pattern=None
        try:
            for pattern in ('zero','tiny','normal','large','outlier'):
                x.normal_()
                if pattern=='zero':x.zero_()
                elif pattern=='tiny':x.mul_(1e-9)
                elif pattern=='large':x.mul_(100)
                elif pattern=='outlier':x[:,0]=32
                expected=torch.ops.biv150_native.gate_silu_auto(x,wq,sw,marker,17)
                actual=evaluate(x,wq,sw)
                torch.testing.assert_close(actual,expected,atol=.005,rtol=.005)
                aq,sa=torch.ops.biv150_native.quantize_rows_probe(x,256)
                acc=aq.cpu().long()@wc
                linear=(acc.float()*(sa.cpu()[:,None]*sc[None,:])).to(torch.bfloat16).float()
                gate,up=linear.chunk(2,dim=1)
                gold=((gate/(1+torch.exp(-gate)))*up).to(torch.bfloat16)
                for output in (actual,expected):torch.testing.assert_close(output[:,cols].cpu(),gold,atol=.005,rtol=.005)
                row['patterns'].append(dict(name=pattern,cpu_int64_dot=True,full_comparison=True,
                                            bit_differences=int((actual!=expected).sum())))
            stage='changed_graph'
            def baseline():return torch.ops.biv150_native.gate_silu_auto(x,wq,sw,marker,17)
            def candidate():return evaluate(x,wq,sw)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=candidate()
            for factor in (0.,.5,4.):
                x.normal_().mul_(factor);graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(output,baseline(),atol=.005,rtol=.005)
            del graph,output
            stage='timing';rounds=[]
            for index in range(3):
                calls=[('baseline_ms',baseline),('candidate_ms',candidate)]
                if index%2:calls.reverse()
                timing={name:graph_time(fn,repeats=11,unroll=8)[0] for name,fn in calls}
                timing['speedup']=timing['baseline_ms']/timing['candidate_ms'];rounds.append(timing)
            row.update(passed=True,changed_input_graph=True,rounds=rounds,
                       median_speedup=statistics.median(t['speedup'] for t in rounds),
                       min_round_speedup=min(t['speedup'] for t in rounds))
        except Exception as error:
            row.update(passed=False,stage=stage,failed_pattern=pattern,error=repr(error),traceback=traceback.format_exc())
            path=args.out/f'failed-layer{layer}.pt';torch.save(dict(x=x.cpu(),layer=layer,weights=identities,stage=stage,pattern=pattern),path)
            row['fixture_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        result['records'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ('weights','patterns','traceback','rounds')}),flush=True)
    passed=[r for r in result['records'] if r['passed']]
    result['summary']=dict(layers=len(result['records']),passed=len(passed),expected=34,
        min_median_speedup=min((r['median_speedup'] for r in passed),default=None),
        max_median_speedup=max((r['median_speedup'] for r in passed),default=None))
    result['diagnostic_completed']=True;save()
    if len(passed)!=34:raise RuntimeError('Actual-weight component failure: no promotion')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'))
    parser.add_argument('--m64-resource-tile',action='store_true')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.m64_resource_tile:
        with torch.inference_mode():main(args)
    else:main(args)
