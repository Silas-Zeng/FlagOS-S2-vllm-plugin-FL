#!/usr/bin/env python3
"""CPU source/cost plan by default; explicit compile-only frontier, no numeric/timing runner yet."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from biv150_konly_decode_factory import MAIN,NAME,build,validate_binding
from biv150_konly_decode_ir import audit
from biv150_konly_decode_quant_reference import FORMAT
from biv150_konly_decode_reference import cost_plan
from biv150_konly_decode_sdk import INSTALLED_MATH,MATH_SHA,check_frontend

ROOT=Path(__file__).resolve().parents[1]
JIT=Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/runtime/jit.py')
JIT_SHA='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
CASES={'4k':(64,4096),'16k':(32,16384)}
BASE_HASHES={64:'641478e6ddcd3d9c11192fbd0c6d4d472bf8ad3e3bb2fd9fa27b09f18e4dc9ff',
             32:'e29a0c84d6b99f1ad44189fbdb74434b2d7da5675af2e2e2422941739916adb9'}


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);loaded=importlib.util.module_from_spec(spec)
    sys.modules[name]=loaded;spec.loader.exec_module(loaded);return loaded


def preserve(kernel,out,label):
    target=out/label/str(kernel.hash);target.mkdir(parents=True,exist_ok=True);files={};strings={}
    for name,value in kernel.asm.items():
        if not isinstance(value,(str,bytes)): continue
        raw=value.encode() if isinstance(value,str) else value;path=target/('actual.'+name);path.write_bytes(raw)
        files[name]=dict(path=str(path.relative_to(out)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(value,str): strings[name]=value
    md=kernel.metadata
    return dict(phase='pre_launch',hash=str(kernel.hash),n_regs=getattr(kernel,'n_regs',None),
                n_spills=getattr(kernel,'n_spills',None),loaded_resources_known=False,
                metadata={k:getattr(md,k,None) for k in ('shared','num_warps','num_stages','warp_size','use_sme')},
                files=files),strings


class Warmup:
    def __init__(self,kernel,**extras): self.kernel,self.extras=kernel,extras
    def __getitem__(self,grid):
        def launch(**kwargs): return self.kernel.warmup(**kwargs,**self.extras,grid=grid)
        return launch


def compile_only(args,generated,proof):
    assert args.commit and args.out,'explicit compile requires frozen --commit and fresh --out'
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==args.commit
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code',MAIN,'--','vllm_fl','csrc'],check=True)
    assert hashlib.sha256(JIT.read_bytes()).hexdigest()==JIT_SHA
    assert hashlib.sha256(INSTALLED_MATH.read_bytes()).hexdigest()==MATH_SHA
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True),'GPU busy; do not queue'
    with socket.socket() as s: assert s.connect_ex(('127.0.0.1',9031))!=0,'service9031 active'
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(source_commit=args.commit,case=args.case,mode=args.mode,scope='private_decode_compile_only',compile_passed=False,
                device_numeric_passed=False,complete_cost_measured=False,performance_verified=False,
                precision_gate=False,service_verified=False,capacity_reclaimed=False,
                proof=proof,cost_plan=cost_plan(args.mode),records=[],errors=[])
    atomic(args.out/'result.json',result)
    try:
        import os
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        import triton
        from biv150_benchmark_utils import main_call,reduce_call
        target=triton.runtime.driver.active.get_current_target()
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        base=module(ROOT/'vllm_fl/patches/biv150_gqa_decode.py','cjj_konly_decode_base')
        assert base.TARGET_PROGRAMS==256,'Unexpected main dispatch policy'
        support=module(ROOT/'tools/kernels/biv150_konly_decode_support.py','cjj_konly_decode_support')
        path=args.out/'generated-producer.py';path.write_text(generated)
        support_source=(ROOT/'tools/kernels/biv150_konly_decode_support.py').read_bytes()
        assert hashlib.sha256(support_source).hexdigest()==proof['support_sha256']
        (args.out/'support-source.py.txt').write_bytes(support_source)
        own=module(path,'cjj_konly_decode_producer')
        batch,length=CASES[args.case];pages=batch*((length+15)//16)
        segments=base._pick_segments(batch,2,16)
        result['geometry']=dict(batch=batch,kv=length,pages=pages,segments=segments,table_pitch=8192,physical_page_bytes=16384)
        # Empty metadata buffers only: warmup compiles, never launches encoders/random init/producers.
        old=torch.empty((pages,2,16,2,128),device='cuda',dtype=torch.bfloat16);key,value=old.unbind(1)
        raw=torch.empty(pages*16384,device='cuda',dtype=torch.uint8)
        code=torch.as_strided(raw.view(torch.int8),(pages,16,2,128),(16384,256,128,1))
        packed_value=torch.as_strided(raw.view(torch.bfloat16),(pages,16,2,128),(8192,256,128,1),4096)
        scale=raw.view(torch.float32)
        query=torch.empty((batch,16,128),device='cuda',dtype=torch.bfloat16)
        qcode=torch.empty_like(query,dtype=torch.int8);qscale=torch.empty((batch,16),device='cuda',dtype=torch.float32)
        scratch=torch.empty((batch,16,16,128),device='cuda',dtype=torch.float32).narrow(2,0,segments)
        maxima=torch.empty((batch,16,16),device='cuda',dtype=torch.float32).narrow(2,0,segments)
        table=torch.empty((batch,8192),device='cuda',dtype=torch.int32)[:,:((length+15)//16)]
        c=dict(q=query,k=key,v=value,table=table,lengths=torch.empty(batch,device='cuda',dtype=torch.int32),
               scratch=scratch,maxima=maxima,sums=None,out=torch.empty_like(query))
        # Preserve original narrowed scratch strides for all scalar buffers, including sums.
        c['sums']=torch.empty((batch,16,16),device='cuda',dtype=torch.float32).narrow(2,0,segments)
        slots=torch.empty(batch,device='cuda',dtype=torch.int64)
        new_k=torch.empty((batch,2,128),device='cuda',dtype=torch.bfloat16);new_v=torch.empty_like(new_k)
        validate_binding(mode=args.mode,k_strides=code.stride(),v_strides=packed_value.stride(),
                         scale_strides=(4096,2,1),q_strides=query.stride(),source_storage=old.data_ptr(),
                         destination_storage=raw.data_ptr(),format_id=FORMAT)
        candidate=dict(c,k=code,v=packed_value)
        base_proxy=SimpleNamespace(LOG2E=base.LOG2E,_gqa_decode_attention=Warmup(base._gqa_decode_attention))
        new_proxy=SimpleNamespace(LOG2E=base.LOG2E,_gqa_decode_attention=Warmup(getattr(own,NAME),
                                  query_code_ptr=qcode,query_scale_ptr=qscale,key_scale_ptr=scale))
        kernels=[('qprep',support.prepare_q.warmup(query,qcode,qscale,batch,grid=(batch,2),num_warps=1,num_stages=1)),
                 ('initial_encoder',support.encode_initial.warmup(key,value,code,scale,packed_value,pages,
                                      grid=(pages*32,),num_warps=1,num_stages=1)),
                 ('baseline_append',support.append_kv.warmup(new_k,new_v,slots,key,scale,value,pages*16,False,
                                      grid=(batch,2),num_warps=1,num_stages=1)),
                 ('candidate_append',support.append_kv.warmup(new_k,new_v,slots,code,scale,packed_value,pages*16,True,
                                      grid=(batch,2),num_warps=1,num_stages=1)),
                 ('baseline_producer',main_call(base_proxy,c,segments)),
                 ('candidate_producer',main_call(new_proxy,candidate,segments)),
                 ('original_reducer',reduce_call(SimpleNamespace(_reduce_partials=Warmup(base._reduce_partials)),c,segments,4))]
        for label,kernel in kernels:
            record,strings=preserve(kernel,args.out,label)
            record['label']=label
            if label=='baseline_producer': assert str(kernel.hash)==BASE_HASHES[batch],'Control specialization differs from current main'
            if label in ('baseline_producer','candidate_producer'):
                assert kernel.metadata.num_warps==1 and kernel.metadata.num_stages==1
            if label=='original_reducer': assert kernel.metadata.num_warps==4
            if label=='candidate_producer': record['actual_decoder_ir']=audit(strings,generated)
            result['records'].append(record);atomic(args.out/'result.json',result)
        result['compile_passed']=all(r.get('actual_decoder_ir',dict(passed=True))['passed'] for r in result['records'])
        result['status']='compile_only_not_numeric_or_performance_qualified'
        atomic(args.out/'result.json',result)
        assert result['compile_passed'],'Unknown or changed actual decoder mechanism'
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)))
        atomic(args.out/'result.json',result);raise
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('inline','preq'),default='inline')
    parser.add_argument('--case',choices=CASES,default='4k')
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--commit');parser.add_argument('--out',type=Path)
    parser.add_argument('--emit',type=Path)
    args=parser.parse_args();frontend=check_frontend(ROOT)
    generated,proof=build((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),args.mode)
    if args.emit:
        with args.emit.open('x') as f: f.write(generated)
    result=compile_only(args,generated,proof) if args.compile_only else dict(cpu_only=True,proof=proof,
           frontend=frontend,cost_plan=cost_plan(args.mode),numeric_and_timing_runner_prepared=False,
           gpu_executed=False,torch_imported='torch' in sys.modules,triton_imported='triton' in sys.modules)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
