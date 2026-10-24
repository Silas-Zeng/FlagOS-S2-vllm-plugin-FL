"""Small-M real projections with complete CPU FP64 output oracles.

Unlike the first synthetic benchmark, every output element is checked on CPU.
Native and candidate are assessed separately; disagreements are not relabeled
as passes. These are components with synthetic activations, not model accuracy.
"""

import argparse
import hashlib
import inspect
import json
import socket
import statistics
import subprocess
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

from biv150_benchmark_utils import graph_time
from biv150_bf16_small_splitk import allocate,matmul,run_into

CHOICES=((16,128,64,1,4),(16,128,64,2,4),(16,128,64,4,4))


def weights(model,layers=(4,20,37)):
    locations={}
    for path in sorted(model.glob('*.safetensors')):
        with safe_open(str(path),framework='pt',device='cpu') as h:
            for key in h.keys():locations[key]=path
    for layer in layers:
        for name,projections in (('qkv',('q_proj','k_proj','v_proj')),
                                 ('o',('o_proj',)),('down',('down_proj',))):
            parts=[];identities=[]
            for proj in projections:
                section='mlp' if proj=='down_proj' else 'self_attn'
                key=f'model.layers.{layer}.{section}.{proj}.weight'
                with safe_open(str(locations[key]),framework='pt',device='cpu') as h:w=h.get_tensor(key)
                assert w.dtype==torch.bfloat16 and w.ndim==2
                parts.append(w);identities.append(dict(key=key,sha256=hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()))
            weight=torch.cat(parts) if len(parts)>1 else parts[0]
            expected={'qkv':(2560,2048),'o':(2048,2048),'down':(2048,6144)}[name]
            assert tuple(weight.shape)==expected
            yield layer,name,weight,identities


def check(a,b):
    a=a.cpu().float();b=b.cpu().float();diff=(a-b).abs()
    bad=(diff>.005+.005*b.abs()) | ~torch.isfinite(a)
    return dict(passed=not bool(bad.any()),violations=int(bad.sum()),count=bad.numel(),
                max_abs=float(diff.max()),bit_differences=int((a!=b).sum()))


def main(args):
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    import flag_gems
    path=Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm)))
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
    assert Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm_out)))==path
    dispatch_policy=None
    dispatch_tables={}
    if args.baseline=='out':
        from vllm_fl.utils import use_flaggems,get_flag_gems_whitelist_blacklist
        assert use_flaggems(), 'Expected current worker default FlagGems policy'
        whitelist,blacklist=get_flag_gems_whitelist_blacklist()
        assert not whitelist and not blacklist, 'Revalidate policy against current service before using a different op list'
        dispatch_tables['before']=torch._C._dispatch_dump_table('aten::mm.out')
        flag_gems.enable(record=False,once=True)
        dispatch_tables['after']=torch._C._dispatch_dump_table('aten::mm.out')
        dispatch_policy=dict(use_flaggems=True,whitelist=whitelist,blacklist=blacklist,
                             diagnostic_record=False,worker_record=True)
    layers=list(range(42)) if args.all_layers else [4,20,37]
    torch.set_num_threads(2);torch.manual_seed(20261025)
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(component_only=True,service_installed=False,full_cpu_fp64_output_oracle=True,
        synthetic_activations=True,layers=layers,rows=[1,8,16],baseline=args.baseline,pitches='contiguous',
        inference_mode=torch.is_inference_mode_enabled(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        native_mm_file=str(path) if args.baseline=='returning' else None,
        flaggems_mm_candidate_file=str(path),flaggems_mm_candidate_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_function='torch.mm(out=caller_buffer) under current worker FlagGems policy' if args.baseline=='out' else 'flag_gems.mm',
        dispatch_policy=dispatch_policy,mm_out_dispatch_tables=dispatch_tables,records=[])
    save=lambda:(args.out/'result.json').write_text(json.dumps(result,indent=2))
    for layer,name,host,identities in weights(args.model,layers):
        w=host.to('cuda');host64=host.double();n,k=host.shape
        for m in (1,8,16):
            x=torch.empty(m,k,device='cuda',dtype=torch.bfloat16)
            weight_view=torch.as_strided(w,(k,n),(1,k))
            assert weight_view.untyped_storage().data_ptr()==w.untyped_storage().data_ptr()
            native_output=(torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                           if args.baseline=='out' else None)
            def native_mm():
                if args.baseline=='out':return torch.mm(x,weight_view,out=native_output)
                return flag_gems.mm(x,w.t())
            records=[dict(layer=layer,geometry=name,m=m,n=n,k=k,weight_identity=identities,
                config=list(c),patterns=[],candidate_cpu_all_passed=True,native_cpu_all_passed=True,
                direct_native_all_passed=True) for c in CHOICES]
            for pattern in ('zero','tiny','normal','large','outlier'):
                x.normal_()
                if pattern=='zero':x.zero_()
                elif pattern=='tiny':x.mul_(1e-4)
                elif pattern=='large':x.mul_(100)
                elif pattern=='outlier':x[:,k//2]=32
                gold=(x.cpu().double()@host64.t()).to(torch.bfloat16)
                native=native_mm();native_check=check(native,gold)
                for config,row in zip(CHOICES,records):
                    try:
                        output,scratch=allocate(x,w,config)
                        if config[3]>1:scratch.fill_(float('nan'))
                        run_into(x,w,output,scratch,config)
                        cpu=check(output,gold);direct=check(output,native)
                        row['patterns'].append(dict(name=pattern,candidate_cpu=cpu,native_cpu=native_check,direct_native=direct))
                        row['candidate_cpu_all_passed'] &= cpu['passed']
                        row['native_cpu_all_passed'] &= native_check['passed']
                        row['direct_native_all_passed'] &= direct['passed']
                        if (not cpu['passed'] or not native_check['passed'] or not direct['passed']) and 'fixture' not in row:
                            p=args.out/f'failed-l{layer}-{name}-m{m}-s{config[3]}.pt'
                            torch.save(dict(x=x.cpu(),weight_identity=identities,cpu=gold,
                                candidate=output.cpu(),native=native.cpu(),pattern=pattern),p)
                            row.update(fixture=p.name,fixture_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    except Exception as error:
                        row.update(candidate_cpu_all_passed=False,error=repr(error),traceback=traceback.format_exc())
            for config,row in zip(CHOICES,records):
                if not row['candidate_cpu_all_passed']:continue
                try:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=matmul(x,w,config)
                    row['graph_cpu_passed']=True
                    for factor in (0.,.5,4.):
                        x.normal_().mul_(factor);graph.replay();torch.cuda.synchronize()
                        gold=(x.cpu().double()@host64.t()).to(torch.bfloat16)
                        checked=check(output,gold);row['graph_cpu_passed'] &= checked['passed']
                    del graph,output
                    if not row['graph_cpu_passed']:continue
                    x.normal_()
                    timing_gold=(x.cpu().double()@host64.t()).to(torch.bfloat16)
                    row['timing_candidate_cpu']=check(matmul(x,w,config),timing_gold)
                    row['timing_native_cpu']=check(native_mm(),timing_gold)
                    if not row['timing_candidate_cpu']['passed']:continue
                    def baseline():return native_mm()
                    def candidate():return matmul(x,w,config)
                    rounds=[]
                    for i in range(3):
                        calls=[('baseline_ms',baseline),('candidate_ms',candidate)]
                        if i%2:calls.reverse()
                        t={name:graph_time(fn,repeats=9,unroll=4)[0] for name,fn in calls}
                        t['speedup']=t['baseline_ms']/t['candidate_ms'];rounds.append(t)
                    row.update(rounds=rounds,median_speedup=statistics.median(t['speedup'] for t in rounds),
                        min_round_speedup=min(t['speedup'] for t in rounds),
                        extra_fp32_workspace_bytes=(config[3]*m*n*4 if config[3]>1 else 0))
                except Exception as error:row.update(error=repr(error),traceback=traceback.format_exc(),graph_cpu_passed=False)
            result['records'].extend(records);save()
            print(json.dumps(dict(layer=layer,geometry=name,m=m,candidate_cpu_passed=sum(a['candidate_cpu_all_passed'] for a in records),
                native_cpu_passed=records[0]['native_cpu_all_passed'],graph_cpu_passed=sum(a.get('graph_cpu_passed',False) for a in records))),flush=True)
        del w,host,host64
    result['summary']=dict(records=len(result['records']),expected=len(layers)*3*3*len(CHOICES),
        candidate_full_cpu_passed=sum(a['candidate_cpu_all_passed'] for a in result['records']),
        candidate_graph_cpu_passed=sum(a.get('graph_cpu_passed',False) for a in result['records']),
        native_full_cpu_passed_records=sum(a['native_cpu_all_passed'] for a in result['records']))
    result['diagnostic_completed']=len(result['records'])==result['summary']['expected'];save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'))
    p.add_argument('--all-layers',action='store_true')
    p.add_argument('--baseline',choices=('returning','out'),default='returning')
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.baseline=='out':
        with torch.inference_mode():main(args)
    else:main(args)
