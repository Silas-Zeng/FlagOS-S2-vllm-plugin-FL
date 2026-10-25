"""Opt-in exact INT64 oracle for this SDK's signed INT8 K32 dot primitive."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main(a):
    if not a.run_gpu:
        print(json.dumps({'gpu_executed':False,'shape':[8,128,32],'oracle':'CPU INT64','streams':['default','side'],'changed_graph':True}));return
    root=Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True)
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0
    a.out.mkdir(parents=True,exist_ok=False)
    import torch
    import biv150_benchmark_utils as bench
    module=bench.load_module(root/'tools/kernels/biv150_int8_pv.py','i8_dot_hardware_probe')
    torch.set_num_threads(2);torch.manual_seed(20261026)
    patterns=[(torch.randint(-127,128,(8,32),dtype=torch.int8),torch.randint(-127,128,(32,128),dtype=torch.int8)),
              (torch.full((8,32),127,dtype=torch.int8),torch.full((32,128),-127,dtype=torch.int8)),
              (torch.zeros(8,32,dtype=torch.int8),torch.full((32,128),127,dtype=torch.int8))]
    p=torch.empty(8,32,dtype=torch.int8,device='cuda');v=torch.empty(32,128,dtype=torch.int8,device='cuda')
    storage=torch.full((8*128+16,),123456789,dtype=torch.int32,device='cuda');out=storage[8:-8].reshape(8,128)
    result={'passed':False,'gpu_executed':True,'warps':a.warps,'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            'operator_sha256':hashlib.sha256((root/'tools/kernels/biv150_int8_pv.py').read_bytes()).hexdigest(),'checks':[]}
    def save():(a.out/'result.json').write_text(json.dumps(result,indent=2))
    save();default=torch.cuda.current_stream()
    for name in ['default','side']:
        stream=default if name=='default' else torch.cuda.Stream();stream.wait_stream(default)
        def call():return module.int8_dot_probe[(1,)](p,v,out,num_warps=a.warps,num_stages=1)
        def check(index,mode):
            stream.synchronize();host=out.cpu();expected=(patterns[index][0].long()@patterns[index][1].long()).to(torch.int32)
            assert torch.equal(host,expected),'signed INT8 dot differs from independent CPU INT64'
            assert torch.equal(storage[:8].cpu(),torch.full((8,),123456789,dtype=torch.int32))
            assert torch.equal(storage[-8:].cpu(),torch.full((8,),123456789,dtype=torch.int32))
            result['checks'].append({'stream':name,'pattern':index,'mode':mode,'exact':True,'canaries':True});save()
        with torch.cuda.stream(stream):
            for index,(hp,hv) in enumerate(patterns):
                p.copy_(hp);v.copy_(hv);out.fill_(-999);compiled=call();check(index,'eager')
            capture=torch.cuda.Stream();capture.wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=capture):call()
            stream.wait_stream(capture)
            for index in [0,1,2,0,1,2]:
                p.copy_(patterns[index][0]);v.copy_(patterns[index][1]);out.fill_(-999);graph.replay();check(index,'changed-poisoned-graph')
        default.wait_stream(stream)
    result['resources']={'hash':str(compiled.hash),'n_regs':compiled.n_regs,'n_spills':compiled.n_spills,'use_sme':getattr(compiled.metadata,'use_sme',None)}
    result['passed']=True;save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-gpu',action='store_true');p.add_argument('--out',type=Path);p.add_argument('--warps',type=int,choices=(1,2),default=1)
    a=p.parse_args()
    if a.run_gpu and a.out is None:p.error('--out required')
    main(a)
