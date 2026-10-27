#!/usr/bin/env python3
"""Default CPU proposal; explicit fresh service-source warm compiles, 0 launches."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
sys.dont_write_bytecode=True
import biv150_konly_service_q32_cache as cache
import biv150_konly_service_q32_hoist as h
from biv150_konly_service_q32_arguments import kwargs
ROOT=Path(__file__).resolve().parents[1]
MAIN='dac1b8fb48e7983e48283032efeee9b592c5fbe5'


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def freeze(value):
    if value is None or type(value) in (str,int,float,bool):return value
    if isinstance(value,dict):return {type(k).__name__+':'+repr(k):freeze(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [freeze(v) for v in value]
    return str(value)


def save(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def allocate(torch):
    p=dict(profile='q32',pages=773,total=2048,sequences=1,segments=1,
        config=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,IS_3D=False))
    def empty(shape,dtype):return torch.empty(shape,device='cuda',dtype=dtype)
    raw=empty((773*16384,),torch.uint8)
    t=dict(q=empty((2048,16,128),torch.bfloat16),out=empty((2048,16,128),torch.bfloat16),
        k=torch.as_strided(raw.view(torch.int8),(773,16,2,128),(16384,256,128,1),0),
        v=torch.as_strided(raw.view(torch.bfloat16),(773,16,2,128),(8192,256,128,1),4096),kscale=raw.view(torch.float32),
        table=empty((1,8192),torch.int32),lengths=empty((1,),torch.int32),cu_q=empty((2,),torch.int32),
        k_descale=empty((),torch.float32),v_descale=empty((),torch.float32))
    return p,t,raw


def preserve(kernel,folder):
    folder.mkdir();files={}
    for ext,value in kernel.asm.items():
        if isinstance(value,(str,bytes)):
            b=value.encode() if isinstance(value,str) else value;(folder/('actual.'+ext)).write_bytes(b)
            files[ext]=dict(bytes=len(b),sha256=h.sha(b))
    h.need({'ttir','ttgir','llir','asm','cubin'}<=set(files),'all actual compiler artifacts')
    fn=kernel.src.fn;src=fn.src;(folder/'actual.jit_python_source.py').write_text(src)
    md=kernel.metadata
    result=dict(hash=str(kernel.hash),signature=freeze(kernel.src.signature),constants=freeze(kernel.src.constants),
        attrs=freeze(kernel.src.attrs),metadata=freeze(md._asdict() if hasattr(md,'_asdict') else vars(md)),
        source_digest=h.sha(src),files=files,n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),
        phase='prelaunch_compile_only',loaded_resources_verified=False)
    save(folder/'record.json',result);return result


def run(args,payload):
    files,meta,override,proof,source=payload
    h.need(args.commit and args.out and args.service_source and not args.out.exists(),'exact commit/source/fresh output')
    h.need(git('rev-parse','HEAD')==args.commit and not git('status','--porcelain'),'exact clean frontend freeze')
    h.need(not git('diff',MAIN,'--','vllm_fl','csrc'),'production differs from DAC')
    h.need(args.service_source.resolve()==Path(proof['provenance']['actual_source_path']),'original actual service source path required')
    h.need(args.service_source.read_bytes()==source.encode(),'actual359 private source bytes differ')
    h.need(not {'torch','triton','numpy'}.intersection(sys.modules),'cold admission')
    args.out.mkdir(parents=True);result=dict(commit=args.commit,completed=False,compile_passed=False,errors=[],records=[],
        kernel_launches=0,encoder_launches=0,initialized_tensor_elements=0,source_provenance=proof['provenance'],
        JIT_warmups_attempted=0,
        actual_I8_hoist_verified=False,native_synchronization_verified=False,loaded_resources_verified=False,
        numerical_verified=False,precision_gate=False,service_verified=False,performance_measured=False)
    def checkpoint():save(args.out/'fresh-compile-summary.json',result)
    checkpoint()
    try:
        base=Path('/usr/local/lib/python3.12/site-packages')
        for name,path in [('sdk',base/'vllm/v1/attention/ops/triton_unified_attention.py'),
                          ('helpers',base/'vllm/v1/attention/ops/triton_attention_helpers.py'),
                          ('jit',base/'triton/spec/iluvatar/triton/runtime/jit.py')]:
            h.need(h.sha(path.read_bytes())==h.PINS[name],'installed source pin '+name)
        smi=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True);(args.out/'prelaunch-ixsmi.txt').write_text(smi)
        h.need('No running processes found' in smi,'GPU must be idle')
        with socket.socket() as sock:h.need(sock.connect_ex(('127.0.0.1',9031))!=0,'service9031 must be closed')
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target()
        h.need((target.backend,target.arch,target.warp_size)==('corex',71,64),'actual target')
        spec=importlib.util.spec_from_file_location('cjj_actual359_private_q32',args.service_source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        fn=getattr(module,meta['function_name']);p,t,owner=allocate(torch);args_raw=kwargs(p,t,True)
        result['reconstructed_tensor_descriptors']={k:dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),
            storage_offset=v.storage_offset(),storage_bytes=v.untyped_storage().nbytes()) for k,v in t.items()}
        result['JIT_warmups_attempted']+=1;checkpoint()
        first=fn.warmup(grid=tuple(meta['grid']),**args_raw,**meta['launch_options'])
        before=preserve(first,args.out/'unhoisted');result['records'].append(before);checkpoint()
        h.need(before['hash']==meta['kernel_hash'] and before['source_digest']==meta['source_digest'],'source/cache hash reconstruction differs')
        for ext in ('ttgir','llir'):
            h.need((args.out/'unhoisted'/('actual.'+ext)).read_bytes()==files['kernel_cjj_konly_q32.'+ext],
                'actual service '+ext+' reconstruction differs; retain artifacts, no launch')
        result['reconstructed_unhoisted_matches_runtime_cache']=True
        path=args.out/'fresh-service-override.ttgir';path.write_text(override);save(args.out/'proposal.json',proof)
        result['JIT_warmups_attempted']+=1;checkpoint()
        second=fn.warmup(grid=tuple(meta['grid']),**args_raw,**meta['launch_options'],ir_override=str(path))
        after=preserve(second,args.out/'proposed-hoist');result['records'].append(after);checkpoint()
        h.need(all(before[k]==after[k] for k in ('signature','constants','attrs','source_digest')),'actual typed ABI/source changed beyond override')
        md=second.metadata;h.need((md.num_warps,md.num_stages,md.warp_size)==(16,1,64),'actual override geometry')
        result.update(completed=True,compile_passed=True,actual_live_signature_constants_attrs_now_saved=True,
            actual_native_audit_pending=True,bitwise_numeric_not_started=True)
        checkpoint();return result
    except Exception as exc:
        result['errors'].append(dict(type=type(exc).__name__,message=str(exc)));checkpoint();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compile-only',action='store_true')
    p.add_argument('--commit');p.add_argument('--service-source',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    payload=cache.original()
    if a.compile_only:r=run(a,payload)
    else:
        h.need(not any((a.commit,a.service_source,a.out)),'device options require explicit --compile-only')
        r=dict(CPU_only=True,provenance=payload[3]['provenance'],proposal=payload[3]['proposal'],
            GPU_executed=False,JIT_executed=False,kernel_launches=0,actual_I8_hoist_verified=False)
    print(json.dumps(r,indent=2,sort_keys=True))


if __name__=='__main__':main()
