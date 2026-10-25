"""GPU-only standalone norm precision diagnostic, never a production hook.

Runs only on an idle device. Reference is the exact generated service kernel,
forced to its one saved config; CPU quantization is an independent RN/away
rounding oracle. Every case and every mismatch is retained. L3 is not evaluated.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from biv150_original_topology_norm_quant import generated_source,load_file,REFERENCE,REFERENCE_SHA256,ROOT,NAME


def main(args):
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(args.out/'inductor-cache')
    os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
    import torch
    import triton
    from safetensors import safe_open
    torch.set_num_threads(1)
    ref_module=load_file(args.out/'original_service_norm.py',REFERENCE.read_text(),'precision_reference_norm')
    reference=getattr(ref_module,NAME)
    reference.configs=[triton.Config(dict(XBLOCK=2,R0_BLOCK=2048),num_warps=16,num_stages=1)]
    reference.inductor_meta['dynamic_scale_rblock']=False
    repaired=load_file(args.out/'repaired_norm_quant.py',generated_source(),'precision_repaired_norm_quant')
    # The current production NQ functions are copied unchanged, without loading
    # their unrelated model hooks or custom operator registration.
    current_source=ast.parse((ROOT/'vllm_fl/patches/biv150_live_carry_gate.py').read_text())
    current_funcs=[n for n in current_source.body if isinstance(n,ast.FunctionDef) and n.name in ('_norm_quant','norm_quant')]
    assert len(current_funcs)==2
    header='import torch\nimport triton\nimport triton.language as tl\nfrom torch._inductor.runtime.triton_helpers import libdevice\nSELECTED_ROWS=(64,)\n'
    old=load_file(args.out/'current_norm_quant.py',header+ast.unparse(ast.Module(body=current_funcs,type_ignores=[]))+'\n','precision_current_norm_quant')
    stream=torch.cuda.current_stream().cuda_stream
    weights={}
    for f in sorted(args.model.glob('*.safetensors')):
        with safe_open(str(f),framework='pt',device='cpu') as handle:
            for key in handle.keys():
                if key.startswith('model.layers.') and key.endswith('.post_attention_layernorm.weight'):
                    layer=int(key.split('.')[2]);assert layer not in weights
                    weights[layer]=handle.get_tensor(key)
    assert set(weights)==set(range(42))
    result=dict(diagnostic_only=True,production_hook=False,model_accuracy_verified=False,
        reference_source_sha256=REFERENCE_SHA256,reference_single_config_only=True,reference_config=dict(XBLOCK=2,R0_BLOCK=2048,num_warps=16,num_stages=1),
        generated_source_sha256=hashlib.sha256(generated_source().encode()).hexdigest(),records=[],timings=[],
        runtime_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
    def save():
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def compare(a,b):
        a=a.detach().cpu();b=b.detach().cpu();delta=(a.float()-b.float()).abs()
        return dict(bit_exact=torch.equal(a,b),finite=bool(torch.isfinite(a).all()),
            differences=int((a!=b).sum()),violations=int((delta>.005+.005*b.float().abs()).sum()),max_abs=float(delta.max()))
    def cpu_quant(y):
        y=y.cpu().float();scale=y.abs().amax(dim=1).clamp(min=1e-6)/127.0
        values=y/scale[:,None]
        q=torch.where(values>=0,torch.floor(values+.5),torch.ceil(values-.5)).clamp(-127,127).to(torch.int8)
        return q,scale
    def ref_norm(x,r,w,out):
        reference.run(x,r,w,out,64,2048,grid=lambda meta:(triton.cdiv(64,meta['XBLOCK']),1,1),stream=stream)
        assert len(reference.launchers)==1,'reference must retain exactly the saved config'
    def make_case(layer,pattern):
        g=torch.Generator().manual_seed(37000+layer*17+patterns.index(pattern))
        x=torch.randn((64,2048),generator=g);r=torch.randn((64,2048),generator=g)
        if pattern=='small':x*=.001;r*=.001
        if pattern=='large':x*=64;r*=64
        if pattern=='cancellation':r=-x+r*.015625
        if pattern=='column-range':
            scale=2.0**torch.linspace(-8,8,2048);x*=scale;r*=scale
        if pattern=='zero':x.zero_();r.zero_()
        return x.bfloat16().cuda(),r.bfloat16().cuda()
    patterns=('normal','small','large','cancellation','column-range','zero')
    passed=True
    for layer,weight in sorted(weights.items()):
        w=weight.cuda();weight_sha=hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
        for pattern in patterns:
            x,r=make_case(layer,pattern);snapshots=[v.cpu().clone() for v in (x,r,w)]
            yref=torch.empty_like(x);ref_norm(x,r,w,yref)
            oq,oscale,_,oy,_=old.norm_quant(x,r,w,1e-6,write_norm=True,materialize_residual=False)
            nq,ns,ny,kernel=repaired.norm_quant(x,r,w,write_norm=True)
            fq,fs,_,fkernel=repaired.norm_quant(x,r,w,write_norm=False)
            cq,cs=cpu_quant(yref)
            row=dict(layer=layer,pattern=pattern,m=64,weight_sha256=weight_sha,
                current_norm=compare(oy,yref),current_quant=compare(oq,cq),current_scale=compare(oscale,cs),
                repaired_norm=compare(ny,yref),repaired_quant=compare(nq,cq),repaired_scale=compare(ns,cs),
                no_materialization_quant=compare(fq,cq),no_materialization_scale=compare(fs,cs),
                inputs_unchanged=all(torch.equal(value.cpu(),before) for value,before in zip((x,r,w),snapshots)))
            row['passed']=row['inputs_unchanged'] and all(row[k]['bit_exact'] and row[k]['finite'] for k in
                ('repaired_norm','repaired_quant','repaired_scale','no_materialization_quant','no_materialization_scale'))
            if not row['passed']:
                fixture=f'failed-layer{layer}-{pattern}.pt'
                torch.save(dict(x=snapshots[0],residual=snapshots[1],weight=snapshots[2],reference=yref.cpu(),repaired=ny.cpu(),
                    q=nq.cpu(),scale=ns.cpu(),reference_q=cq,reference_scale=cs),args.out/fixture)
                row['fixture']=fixture
            result['records'].append(row);passed &= row['passed'];save()
            print(json.dumps(dict(layer=layer,pattern=pattern,passed=row['passed'],
                current_norm_diffs=row['current_norm']['differences'],repaired_norm_diffs=row['repaired_norm']['differences'])),flush=True)
        if layer in (4,20,37):
            # Fixed representative normal inputs; graph timings compare only NQ
            # components. They do not predict original full-service throughput.
            x,r=make_case(layer,'normal');graphs={};outs={}
            for name,fn in [('current',lambda:old.norm_quant(x,r,w,1e-6,materialize_residual=False)[:2]),
                            ('repaired',lambda:repaired.norm_quant(x,r,w)[:2])]:
                for _ in range(5):fn()
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outs[name]=fn()
                graphs[name]=graph
                for _ in range(10):graph.replay()
            rounds=[]
            for round_id in range(3):
                timing={}
                for name in (('current','repaired') if round_id%2==0 else ('repaired','current')):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(200):graphs[name].replay()
                    end.record();end.synchronize();timing[name]=start.elapsed_time(end)*1000/200
                rounds.append(timing)
            # Replay after changing inputs proves capture did not freeze values.
            with torch.no_grad():x.mul_(.5);r.mul_(2)
            ref_norm(x,r,w,yref);cq,cs=cpu_quant(yref)
            for graph in graphs.values():graph.replay()
            graph_check={name:dict(q=compare(values[0],cq),s=compare(values[1],cs)) for name,values in outs.items()}
            graph_pass=all(graph_check['repaired'][key]['bit_exact'] for key in ('q','s'))
            passed &= graph_pass
            result['timings'].append(dict(layer=layer,rounds_us=rounds,graph_replay=graph_check,
                repaired_graph_exact=graph_pass,resources=[dict(n_regs=getattr(k,'n_regs',None),n_spills=getattr(k,'n_spills',None),
                    shared=getattr(getattr(k,'metadata',None),'shared',None)) for k in (kernel,fkernel)]))
            save()
    result['completed']=True;result['passed']=passed;save()
    assert passed,'topology is diagnostic only: inspect all retained precision mismatches'


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=Path('/workspace/MiniCPM5-2B'))
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
