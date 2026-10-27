#!/usr/bin/env python3
"""CPU plan by default; explicit --compile-only warm-compiles, never launches."""
import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
import bench_biv150_konly_prefill_compile as legacy
import biv150_konly_2d128_compile_contract as contract

ROOT=Path(__file__).resolve().parents[1]


def source_bundle():
    manifest=ROOT/'docs/evidence/konly-prefill-2d128-20261027/borrowed80.json'
    if legacy.factory.sha(manifest.read_bytes())!='c3996588984acfccd02e543830730a5d02cbda06ad68d9d41ce7a03becf8e0be':
        raise ValueError('Frozen80 provenance manifest changed')
    for name,row in json.loads(manifest.read_text()).items():
        data=(ROOT/name).read_bytes()
        if len(data)!=row['bytes'] or legacy.factory.sha(data)!=row['sha256']:
            raise ValueError('Frozen80 copied file changed: '+name)
    return legacy.source_bundle(contract.PROFILE)


def compile_only(args,p,bundle):
    if not args.commit or args.out is None: raise ValueError('Exact --commit and fresh --out required')
    if legacy.git('rev-parse','HEAD')!=args.commit or legacy.git('status','--porcelain'):
        raise ValueError('Clean exact new component freeze required')
    if legacy.git('diff',contract.MAIN,'--','vllm_fl','csrc'): raise ValueError('Production differs from parentDAC')
    if {'torch','triton'}.intersection(sys.modules): raise ValueError('Cold admission before framework import required')
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(completed=False,compile_passed=False,compile_only=True,commit=args.commit,plan=p,
                records=[],errors=[],kernel_launches=0,JIT_warmups_attempted=0,
                initialized_tensor_elements=0,actual_I8_operand_hoist_verified=False,
                mechanism_status='unknown_not_audited_for_BM128',borrowed_Q32_actual_proof=False,
                loaded_resources_verified=False,device_numeric_verified=False,precision_gate=False,
                service_installed=False,performance_measured=False,ready_for_numeric_or_timing=False,
                baseline_is_current_main_policy=False)
    def save(): legacy.save(args.out/'compile-summary.json',result)
    save()
    try:
        sources,baseline,candidate,proof=bundle;result['source_proof']=proof
        folder=args.out/'sources';folder.mkdir()
        for name,text in [('baseline.py',baseline),('candidate.py',candidate),('sdk.py.txt',sources['sdk.py']),('helpers.py.txt',sources['helpers.py'])]:
            (folder/name).write_text(text)
        result['installed_sources']=legacy.installed_sources()
        result['admission']=legacy.idle_before_cuda()
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target()
        if (target.backend,target.arch,target.warp_size)!=('corex',71,64): raise ValueError('Unknown target')
        result['target']=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
        result['versions']={name:getattr(module,'__version__',None) for name,module in [('torch',torch),('triton',triton)]}
        for name,pin in [('vllm.v1.attention.ops.triton_unified_attention',legacy.factory.PINS['sdk.py']),
                         ('vllm.v1.attention.ops.triton_attention_helpers',legacy.factory.PINS['helpers.py'])]:
            module=importlib.import_module(name)
            if legacy.factory.sha(Path(module.__file__).read_bytes())!=pin: raise ValueError('Loaded SDK/helper identity')
        tensors,owners=contract.allocate(torch,p)
        desc=contract.descriptors(tensors);result['tensor_descriptors']=desc
        result['descriptor_contract']=contract.check_descriptors(p,desc)
        for label,text,is_candidate in [('baseline',baseline,False),('candidate',candidate,True)]:
            module=legacy.module(folder/(label+'.py'),'cjj_2d128_compile_'+label)
            name=legacy.factory.name(contract.PROFILE) if is_candidate else 'kernel_biv150_prefill_stride'
            schema=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
            names=[a.arg for a in schema.args.args]
            wrapper=contract.WarmupOnly(getattr(module,name))
            values=contract.kwargs(p,tensors,is_candidate)
            result['JIT_warmups_attempted']+=1
            compiled=wrapper.warmup(grid=p['grid'],**values,**p['requested_options'])
            record=legacy.preserve(compiled,args.out,label,p['requested_options'],p['grid'],values,tensors,desc)
            result['records'].append(record);save()
            record['typed_cache_pointer_proof']=contract.typed_cache_proof(dict(compiled.src.signature),names,is_candidate)
            md=compiled.metadata
            if (md.num_warps,md.num_stages,md.warp_size)!=(8,1,64): raise ValueError('Actual BM128 options differ')
            if not all(k in record['files'] and record['files'][k]['bytes']>0
                       for k in ('source','ttir','ttgir','llir','asm','cubin')):
                raise ValueError('Required actual source/TTIR/TTGIR/LLVM/ASM/cubin missing')
            record.update(phase='pre_launch_compile_only',loaded_resources_verified=False,
                          resource_counts_are_prelaunch_only=True,
                          mechanism_status='unknown_not_audited_for_BM128',kernel_launches=0,
                          forbidden_launch_attempts=wrapper.forbidden_launch_attempts)
            if wrapper.warmups!=1 or wrapper.forbidden_launch_attempts: raise ValueError('Compile-only boundary changed')
            save()
        result.update(completed=True,compile_passed=True)
        save();return result
    except Exception as exc:
        result['completed']=False;result['errors'].append(dict(type=type(exc).__name__,message=str(exc)));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--commit');parser.add_argument('--out',type=Path);args=parser.parse_args()
    p=contract.plan();bundle=source_bundle()
    if args.compile_only: result=compile_only(args,p,bundle)
    else:
        if args.commit: parser.error('--commit belongs to explicit compile-only')
        result=dict(cpu_only=True,plan=p,source_proof=bundle[3],kernel_launches=0,JIT_executed=False,
                    torch_imported='torch' in sys.modules,triton_imported='triton' in sys.modules,
                    compiled_typed_cache_pointers_verified=False,actual_I8_operand_hoist_verified=False,
                    mechanism_status='unknown_not_audited_for_BM128',loaded_resources_verified=False,
                    numerical_verified=False,precision_gate=False,service_installed=False,performance_measured=False)
        if args.out:
            args.out.mkdir(parents=True,exist_ok=False);legacy.save(args.out/'cpu-plan.json',result)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__': main()
