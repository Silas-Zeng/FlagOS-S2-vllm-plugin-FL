"""Opt-in real compiler/FP64/stream/Graph check; no model performance claim."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    if not a.run_gpu:
        print(json.dumps({'gpu_executed':False,'rows':[1,2,4,8,16,32,40,64,2048],
            'weights':'real first-layer O,QKV,down,gate/up','oracle':'CPU FP64',
            'checks':'actual dynamic Torch compile, generated out calls, default/side streams, changed-input poisoned Graph'}));return
    assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.commit
    assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0
    a.out.mkdir(parents=True,exist_ok=False)
    env={'TORCHINDUCTOR_CACHE_DIR':str(a.out/'inductor-cache'),'TRITON_CACHE_DIR':str(a.out/'triton-cache'),'FLAGGEMS_CACHE_DIR':str(a.out/'flaggems-cache')};os.environ.update(env)
    target=a.out/'flaggems-cache/config_cache';target.mkdir(parents=True)
    for record in json.loads((a.tuning_snapshot/'manifest.json').read_text())['databases']:
        assert Path(record['name']).name==record['name'];p=a.tuning_snapshot/record['name'];assert sha(p)==record['sha256'];shutil.copy2(p,target/p.name)
    import sys
    sys.path.insert(0,str(a.source.resolve()))
    import torch
    import torch._dynamo
    import torch._inductor.config
    import flag_gems
    from safetensors import safe_open
    from vllm_fl.patches import biv150_stable_small_o as small
    from vllm_fl.patches import biv150_small_out_lowering as lowering
    torch.set_num_threads(2);flag_gems.enable(record=False,once=True)
    torch._inductor.config.fx_graph_cache=False
    small._register();installation=lowering.install()
    result={'complete':False,'passed':False,'source_commit':a.commit,'script_sha256':sha(__file__),
            'installation':installation,'env':env,'records':[],'errors':[],'model_score_claim':False}
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
    def weight(name):
        prefix='model.layers.0.'
        keys={'o':['self_attn.o_proj.weight'],'qkv':['self_attn.q_proj.weight','self_attn.k_proj.weight','self_attn.v_proj.weight'],
              'down':['mlp.down_proj.weight'],'gate':['mlp.gate_proj.weight','mlp.up_proj.weight']}[name]
        parts=[tensor(prefix+key) for key in keys];return (parts[0] if len(parts)==1 else torch.cat(parts,dim=0)).contiguous()
    def compare(actual,reference):
        v=actual.double();d=(v-reference).abs();bad=(d>0.005+0.005*reference.abs())|~torch.isfinite(v)
        return {'passed':not bool(bad.any()),'violations':int(bad.sum()),'max_abs_error':float(d.max())}
    def operation(x,w):return torch.ops.biv150_stable_small_o.linear(x,w)
    save()
    try:
        with torch.inference_mode():
            for name in ['o','qkv','down','gate']:
                host_w=weight(name);w=host_w.cuda();torch._dynamo.mark_static(w)
                compiled=torch.compile(operation,fullgraph=True,dynamic=True)
                for rows in [1,2,4,8,16,32,40,64,2048]:
                    generator=torch.Generator().manual_seed(4100+rows+host_w.shape[1]);x0=torch.randn((rows,host_w.shape[1]),generator=generator).to(torch.bfloat16);x1=(-0.75*x0.float()+0.125).to(torch.bfloat16)
                    hosts=[x0,x1];oracles=[x.double()@host_w.double().t() for x in hosts]
                    x=torch.empty_like(x0,device='cuda');torch._dynamo.mark_static(x,1)
                    if rows>1:torch._dynamo.mark_dynamic(x,0,min=2,max=2048)
                    x.copy_(x0);compiled(x,w);compiled(x,w);torch.cuda.synchronize()
                    default=torch.cuda.default_stream();side=torch.cuda.Stream();capture_stream=torch.cuda.Stream();side.wait_stream(default);capture_stream.wait_stream(default)
                    checks=[]
                    for stream_name,stream in [('default',default),('side',side)]:
                        for pattern,host in enumerate(hosts):
                            with torch.cuda.stream(stream):x.copy_(host);value=compiled(x,w)
                            stream.synchronize();checks.append({'stage':'compiled','stream':stream_name,'input':pattern,**compare(value.cpu(),oracles[pattern])})
                        stream.synchronize();capture_stream.wait_stream(stream)
                        with torch.cuda.stream(capture_stream):
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=capture_stream):graph_out=compiled(x,w)
                        capture_stream.synchronize();stream.wait_stream(capture_stream)
                        for pattern,host in enumerate(hosts):
                            for repeat in range(3):
                                with torch.cuda.stream(stream):x.copy_(host);graph_out.fill_(float('nan'));graph.replay()
                                stream.synchronize();checks.append({'stage':'graph','stream':stream_name,'input':pattern,'repeat':repeat,**compare(graph_out.cpu(),oracles[pattern])})
                        if stream_name=='side':default.wait_stream(side)
                        del graph,graph_out
                    result['records'].append({'projection':name,'rows':rows,'weight_sha256':hashlib.sha256(host_w.view(torch.uint8).numpy().tobytes()).hexdigest(),'checks':checks,'passed':all(c['passed'] for c in checks)});save()
                del compiled,w,host_w
                torch._dynamo.reset()
            generated=[]
            for p in (a.out/'inductor-cache').rglob('*.py'):
                text=p.read_text(errors='replace')
                if 'async_compile.triton' not in text and 'biv150_small_linear_out' not in text:continue
                calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call)]
                count=sum(n=='extern_kernels.biv150_small_linear_out' for n in calls)
                if count:generated.append({'file':str(p.relative_to(a.out)),'sha256':sha(p),'out_calls':count})
            assert generated,'Actual Inductor out-buffer lowering was not selected'
            result.update(generated_out_modules=generated,complete=True,passed=all(r['passed'] for r in result['records']));save()
    except BaseException as error:
        import traceback
        result['errors'].append({'error':repr(error),'traceback':traceback.format_exc()});save();raise
    print(json.dumps({'complete':result['complete'],'passed':result['passed'],'projection_row_cases':len(result['records']),'generated_out_modules':len(generated)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true');p.add_argument('--source',type=Path);p.add_argument('--commit');p.add_argument('--out',type=Path);p.add_argument('--tuning-snapshot',type=Path);p.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'));a=p.parse_args()
    if a.run_gpu and any(getattr(a,n) is None for n in ['source','commit','out','tuning_snapshot']):p.error('source,commit,out,tuning-snapshot required for GPU run')
    main(a)
