#!/usr/bin/env python3
"""Default CPU plan; explicit private mixed-reader compilation, no service or timing."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

from biv150_konly_mixed_factory import MAIN, NAMES, build, span
from biv150_konly_mixed_ir import mechanism, typed_abi

ROOT = Path(__file__).resolve().parents[1]
CASES = dict(mixed4k=([1]*63+[1985], 4096), mixed16k=([1]*31+[2017], 16384), tail=([17,1,0,1,3], 129))


def atomic(path, value):
    temporary = path.with_suffix('.tmp'); temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path); loaded=importlib.util.module_from_spec(spec)
    sys.modules[name]=loaded; spec.loader.exec_module(loaded); return loaded


def preserve(kernel, out, label, phase):
    folder=out/label/str(kernel.hash);folder.mkdir(parents=True,exist_ok=True);files={};strings={}
    for name,value in kernel.asm.items():
        if not isinstance(value,(str,bytes)): continue
        data=value.encode() if isinstance(value,str) else value;path=folder/('actual.'+name);path.write_bytes(data)
        files[name]=dict(path=str(path.relative_to(out)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if isinstance(value,str): strings[name]=value
    metadata={k:getattr(kernel.metadata,k,None) for k in ('shared','num_warps','num_stages','warp_size','use_sme')}
    record=dict(hash=str(kernel.hash),phase=phase,n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),metadata=metadata,files=files)
    record['loaded_resources_known']=phase=='post_launch' and all(type(record[k]) is int for k in ('n_regs','n_spills')) and type(metadata['shared']) is int
    return record,strings


def pick(main_source, n):
    namespace=dict(TARGET_PROGRAMS=256)
    exec(compile(span(main_source,'_pick_segments'),'<pinned-policy-cpu>','exec'),namespace)
    return namespace['_pick_segments'](n,2,16)


def producer_arguments(io, mode, n, capacity, segments):
    out=io['scratch'];maxima=io['maxima']
    result=dict(segm_output_ptr=out,segm_max_ptr=maxima,segm_expsum_ptr=io['sums'],query_ptr=io['q'],
        query_start_len_ptr=io['cu_q'],NUM_SEQS=n,TOTAL_QUERY_TOKENS=capacity,
        key_cache_ptr=io['key'],value_cache_ptr=io['value'],seq_lens_ptr=io['lengths'],block_table_ptr=io['table'],
        softmax_scale_log2e=128**-.5*1.4426950408889634,queries_per_kv=8,
        block_table_stride=io['table'].stride(0),query_stride_token=io['q'].stride(0),query_stride_head=128,
        k_stride_block=8192 if mode=='bf16' else 16384,k_stride_slot=256,k_stride_head=128,
        v_stride_block=8192,v_stride_slot=256,v_stride_head=128,
        segment_stride_token=out.stride(0),segment_stride_head=out.stride(1),segment_stride_segment=out.stride(2),
        max_stride_token=maxima.stride(0),max_stride_head=maxima.stride(1),HEAD_SIZE=128,PAGE_SIZE=16,ROWS=8,SEGMENTS=segments)
    if mode=='k8': result.update(query_code_ptr=io['qcode_dummy'],query_scale_ptr=io['qscale_dummy'],key_scale_ptr=io['key_scale'])
    return result


def reducer_arguments(io,n,capacity,segments):
    out=io['scratch'];maxima=io['maxima'];final=io['out']
    return dict(output_ptr=final,segm_output_ptr=out,segm_max_ptr=maxima,segm_expsum_ptr=io['sums'],seq_lens_ptr=io['lengths'],
        query_start_len_ptr=io['cu_q'],NUM_SEQS=n,TOTAL_QUERY_TOKENS=capacity,
        segment_stride_token=out.stride(0),segment_stride_head=out.stride(1),segment_stride_segment=out.stride(2),segment_stride_dim=1,
        max_stride_token=maxima.stride(0),max_stride_head=maxima.stride(1),max_stride_segment=1,
        output_stride_token=final.stride(0),output_stride_head=128,page_size=16,num_segments=segments,HEAD_SIZE=128)


def preflight(args):
    if not args.commit or not args.out: raise ValueError('Explicit compilation requires exact commit and fresh output')
    assert subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==args.commit
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet',MAIN,'--','vllm_fl','csrc'],check=True)
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True),'GPU busy, do not queue'
    with socket.socket() as sock: assert sock.connect_ex(('127.0.0.1',9031))!=0,'Service9031 active'
    args.out.mkdir(parents=True,exist_ok=False)


def compile_device(args,sources,proof):
    preflight(args)
    result=dict(source_commit=args.commit,main=MAIN,case=args.case,source_proof=proof,completed=False,compile_passed=False,compile_only=True,
        device_compilation_executed=True,kernel_launch_executed=False,gpu_executed=False,
        gpu_numeric_verified=False,numeric_ready=False,service_installed=False,timing_attempted=False,precision_gate=False,resources={},errors=[])
    def save(): atomic(args.out/'result.json',result)
    save()
    try:
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        import triton
        target=triton.runtime.driver.active.get_current_target()
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        result['target']=[target.backend,target.arch,target.warp_size]
        qlens,length=CASES[args.case];n=len(qlens);capacity=sum(qlens)+3;segments=pick((ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text(),n)
        result.update(num_seqs=n,query_capacity=capacity,segments=segments)
        q=torch.empty((capacity,16,128),device='cuda',dtype=torch.bfloat16)
        stock=torch.empty((n*((length+15)//16),2,16,2,128),device='cuda',dtype=torch.bfloat16)
        key,value=stock.unbind(1);packed=torch.empty((stock.shape[0],16384),device='cuda',dtype=torch.uint8)
        k8v=packed.view(torch.bfloat16).reshape(-1,2,16,2,128)[:,1]
        common=dict(q=q,cu_q=torch.empty(n+1,device='cuda',dtype=torch.int32),lengths=torch.empty(n,device='cuda',dtype=torch.int32),
            table=torch.empty((n,8192),device='cuda',dtype=torch.int32)[:,:((length+15)//16)],
            qcode_dummy=torch.empty(1,device='cuda',dtype=torch.int8),qscale_dummy=torch.empty(1,device='cuda',dtype=torch.float32))
        module(ROOT/'tools/kernels/biv150_konly_decode_support.py','cjj_konly_decode_support')
        modes=('bf16','k8') if args.arm=='both' else (args.arm,)
        for mode in modes:
            path=args.out/('generated-'+mode+'.py');path.write_text(sources[mode]);compiled=module(path,'cjj_mixed_'+mode)
            io=dict(common,key=key if mode=='bf16' else packed.view(torch.int8),value=value if mode=='bf16' else k8v,key_scale=packed.view(torch.float32),
                scratch=torch.empty((n,16,16,128),device='cuda',dtype=torch.float32).narrow(2,0,segments),
                maxima=torch.empty((n,16,16),device='cuda',dtype=torch.float32).narrow(2,0,segments),
                sums=torch.empty((n,16,16),device='cuda',dtype=torch.float32).narrow(2,0,segments),out=torch.empty_like(q))
            for role,fn,kwargs,grid,warps in (
                ('producer',getattr(compiled,NAMES[mode]),producer_arguments(io,mode,n,capacity,segments),(n,2,segments),1),
                ('reducer',compiled._reduce_mixed_partials,reducer_arguments(io,n,capacity,segments),(n,16),4)):
                label=mode+'-'+role;kernel=fn.warmup(**kwargs,grid=grid,num_warps=warps,num_stages=1)
                resource,arts=preserve(kernel,args.out,label,'pre_launch');result['resources'][label]=resource;save()
                resource['actual_typed_abi']=typed_abi(arts,sources[mode],NAMES[mode] if role=='producer' else '_reduce_mixed_partials',mode)
                if role=='producer':
                    resource['actual_mechanism']=mechanism(arts,sources[mode],mode);save()
                    assert resource['actual_mechanism']['passed'],'Actual source/IR PV mechanism rejected; inspect raw artifacts'
                save()
        result.update(completed=True,compile_passed=True,status='compile_only_no_numeric_or_resource_equivalence')
        save()
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)))
        (args.out/'error-traceback.txt').write_text(traceback.format_exc());save();raise
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-only',action='store_true');parser.add_argument('--case',choices=CASES,default='mixed4k')
    parser.add_argument('--arm',choices=('both','bf16','k8'),default='both');parser.add_argument('--commit');parser.add_argument('--out',type=Path)
    args=parser.parse_args();text=(ROOT/'vllm_fl/patches/biv150_gqa_decode.py').read_text();sources,proof=build(text)
    if args.compile_only: print(json.dumps(compile_device(args,sources,proof),indent=2));return
    print(json.dumps(dict(cpu_only=True,cases={k:dict(query_lengths=v[0],kv=v[1],segments=pick(text,len(v[0]))) for k,v in CASES.items()},
        factory=proof,compile_only_ready=True,numeric_ready=False,gpu_executed=False,service_installed=False,timing_prepared=False,
        torch_imported='torch' in sys.modules,triton_imported='triton' in sys.modules,numpy_imported='numpy' in sys.modules),indent=2))


if __name__=='__main__': main()
