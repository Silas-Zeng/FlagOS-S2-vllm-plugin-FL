"""Opt-in real-weight, large-working-set component provider comparison.

No production installation, model score or official benchmark modification.
All timings follow independent FP64/stream/changed-input Graph checks.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess

MM_SHA='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
TIMED_ROWS=(32,64,2048)
AWKWARD_ROWS=(9,16,24,31,33,40,48)
GROUPS=('o','qkv','down','gate')
ROUTES=('mm_out','functional_linear','provided_linear')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    if not a.run_gpu:
        print(json.dumps(dict(gpu_executed=False,timed_rows=TIMED_ROWS,awkward_rows=AWKWARD_ROWS,
                              weight_layers=dict(o=42,qkv=42,down=42,gate=8),routes=ROUTES,
                              gates='CPU FP64;default/side;changed-input poisoned Graph before timing')));return
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.commit
    assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True),'GPU busy'
    a.out.mkdir(parents=True,exist_ok=False)
    env=dict(TRITON_CACHE_DIR=str(a.out/'triton-cache'),FLAGGEMS_CACHE_DIR=str(a.out/'flaggems-cache'))
    os.environ.update(env);target=a.out/'flaggems-cache/config_cache';target.mkdir(parents=True)
    records=json.loads((a.tuning_snapshot/'manifest.json').read_text())['databases']
    for record in records:
        p=a.tuning_snapshot/record['name'];assert Path(record['name']).name==record['name'] and sha(p)==record['sha256'];shutil.copy2(p,target/p.name)
    import sys
    sys.path.insert(0,str(a.source.resolve()))
    import torch
    import flag_gems
    from safetensors import safe_open
    import biv150_provided_linear as provided
    torch.set_num_threads(2);flag_gems.enable(record=False,once=True)
    installation=provided.initialize()
    mm=inspect.unwrap(flag_gems.mm_out);assert sha(inspect.getsourcefile(mm))==MM_SHA,'MM provider changed'
    result=dict(complete=False,passed=False,source=str(a.source),commit=a.commit,
                script_sha256=sha(__file__),helper_sha256=sha(Path(__file__).with_name('biv150_provided_linear.py')),
                provider=installation,mm_sha256=MM_SHA,env=env,tuning_snapshot=records,
                model_score_claim=False,groups=[],errors=[])
    def save():
        p=a.out/'result.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(a.out/'result.json')
    index_path=a.model/'model.safetensors.index.json'
    index=json.loads(index_path.read_text())['weight_map'] if index_path.exists() else None
    def tensor(key):
        paths=[a.model/index[key]] if index is not None else sorted(a.model.glob('*.safetensors'))
        for path in paths:
            with safe_open(str(path),framework='pt',device='cpu') as file:
                if key in file.keys():return file.get_tensor(key)
        raise KeyError(key)
    def weight(group,layer):
        names=dict(o=['self_attn.o_proj.weight'],qkv=['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight'],
                   down=['mlp.down_proj.weight'],gate=['mlp.gate_proj.weight','mlp.up_proj.weight'])[group]
        parts=[tensor(f'model.layers.{layer}.'+name) for name in names]
        return (parts[0] if len(parts)==1 else torch.cat(parts,dim=0)).contiguous()
    def compare(value,reference):
        # CPU first: GPU FP64 conversion is not a valid reference on this stack.
        v=value.cpu().double();diff=(v-reference).abs();bad=~torch.isfinite(v)|(diff>0.005+0.005*reference.abs())
        return dict(passed=not bool(bad.any()),violations=int(bad.sum()),
                    max_abs_error=float(diff.max()) if bool(torch.isfinite(v).all()) else None)
    def make(route,x,w,out):
        if route=='mm_out':return lambda:flag_gems.mm_out(x,w.t(),out=out)
        if route=='functional_linear':return lambda:flag_gems.linear(x,w)
        if route=='provided_linear':return lambda:provided.linear_into(x,w,out)
        raise ValueError(route)
    save()
    try:
        with torch.inference_mode():
            for group in a.group or GROUPS:
                layers=list(range(42)) if group!='gate' else [0,1,2,3,38,39,40,41]
                selected=sorted({0,len(layers)//2,len(layers)-1})
                gpu=[];host_checks={};weights=[]
                for i,layer in enumerate(layers):
                    host=weight(group,layer);gpu.append(host.cuda())
                    if i in selected:host_checks[i]=host
                    weights.append(dict(layer=layer,shape=list(host.shape),sha256=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()))
                torch.cuda.synchronize();n,k=gpu[0].shape
                row_group=dict(projection=group,weights=weights,working_set_bytes=sum(w.numel()*w.element_size() for w in gpu),
                               checks=[],timings=[],passed=False);result['groups'].append(row_group);save()
                for rows in (*TIMED_ROWS,*AWKWARD_ROWS):
                    generator=torch.Generator().manual_seed(20261026+rows+k)
                    h0=torch.randn((rows,k),generator=generator).to(torch.bfloat16)
                    hosts=[h0,(-0.75*h0.float()+0.125).to(torch.bfloat16)]
                    x=torch.empty_like(h0,device='cuda');default=torch.cuda.default_stream();side=torch.cuda.Stream();side.wait_stream(default)
                    for i in selected:
                        oracles=[h.double()@host_checks[i].double().t() for h in hosts]
                        for route in ROUTES:
                            out=torch.empty((rows,n),device='cuda',dtype=torch.bfloat16);side.wait_stream(default)
                            fn=make(route,x,gpu[i],out);checks=[]
                            for stream_name,stream in [('default',default),('side',side)]:
                                for pattern,h in enumerate(hosts):
                                    with torch.cuda.stream(stream):x.copy_(h);v=fn()
                                    stream.synchronize();checks.append(dict(stage='eager',stream=stream_name,pattern=pattern,**compare(v,oracles[pattern])))
                            with torch.cuda.stream(side):
                                for _ in range(3):fn()
                                graph=torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph,stream=side):graph_out=fn()
                            side.synchronize()
                            for stream_name,stream in [('default',default),('side',side)]:
                                for pattern,h in enumerate(hosts):
                                    with torch.cuda.stream(stream):x.copy_(h);graph_out.fill_(float('nan'));graph.replay()
                                    stream.synchronize();checks.append(dict(stage='graph',stream=stream_name,pattern=pattern,**compare(graph_out,oracles[pattern])))
                            default.wait_stream(side)
                            item=dict(rows=rows,layer=layers[i],route=route,checks=checks,passed=all(c['passed'] for c in checks));row_group['checks'].append(item);save()
                            if not item['passed']:raise RuntimeError(f'{group}/{rows}/{layers[i]}/{route} numerical failure')
                            del graph,graph_out,fn,out,v
                    if rows in TIMED_ROWS:
                        graphs={};held=[];x.copy_(hosts[0]);torch.cuda.synchronize()
                        for route in ROUTES:
                            out=torch.empty((rows,n),device='cuda',dtype=torch.bfloat16);held.append(out);side.wait_stream(default)
                            funcs=[make(route,x,w,out) for w in gpu]
                            with torch.cuda.stream(side):
                                for _ in range(3):
                                    for fn in funcs:fn()
                                graph=torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph,stream=side):
                                    for fn in funcs:final=fn()
                            side.synchronize();graphs[route]=graph;held.append(final)
                            with torch.cuda.stream(side):final.fill_(float('nan'));graph.replay()
                            side.synchronize();assert compare(final,hosts[0].double()@host_checks[len(layers)-1].double().t())['passed']
                        rounds=[]
                        for order in (ROUTES,ROUTES[1:]+ROUTES[:1],ROUTES[2:]+ROUTES[:2]):
                            timing=dict(order=order)
                            for route in order:
                                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                with torch.cuda.stream(side):
                                    graphs[route].replay();graphs[route].replay();begin.record()
                                    for _ in range(16):graphs[route].replay()
                                    end.record()
                                end.synchronize();timing[route+'_chain_ms']=begin.elapsed_time(end)/16
                            rounds.append(timing)
                        ratios={route:statistics.median(v['mm_out_chain_ms']/v[route+'_chain_ms'] for v in rounds) for route in ROUTES}
                        row_group['timings'].append(dict(rows=rows,provider_calls_per_replay=len(gpu),rounds=rounds,speedup_vs_mm=ratios));save()
                        print(json.dumps(dict(group=group,rows=rows,working_set_bytes=row_group['working_set_bytes'],speedup_vs_mm=ratios)),flush=True)
                        del graphs,held,funcs,final,fn,out,graph
                    del x
                row_group['passed']=True;save();del gpu,host_checks
                torch.cuda.synchronize();torch.cuda.empty_cache()
            result.update(complete=True,passed=all(g['passed'] for g in result['groups']));save()
    except BaseException as error:
        import traceback
        result['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()));save();raise
    print(json.dumps(dict(complete=result['complete'],passed=result['passed'],groups=len(result['groups']))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--source',type=Path);p.add_argument('--commit');p.add_argument('--out',type=Path);p.add_argument('--tuning-snapshot',type=Path)
    p.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'));p.add_argument('--group',action='append',choices=GROUPS)
    a=p.parse_args()
    if a.run_gpu and any(getattr(a,v) is None for v in ('source','commit','out','tuning_snapshot')):p.error('GPU execution requires source/commit/out/tuning-snapshot')
    main(a)
