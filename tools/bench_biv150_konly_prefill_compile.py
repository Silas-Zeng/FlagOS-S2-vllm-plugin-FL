#!/usr/bin/env python3
"""CPU plan by default. Explicit root opt-in compiles empty-descriptor readers; never launches."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import biv150_konly_prefill_factory as factory
import biv150_konly_prefill_compile_contract as contract

ROOT=Path(__file__).resolve().parents[1]
READER_FREEZE='24c741cc152fc7a53f0b8ce27c1e988898fc59ec'
SITE=Path('/usr/local/lib/python3.12/site-packages')
SDK=SITE/'vllm/v1/attention/ops/triton_unified_attention.py'
HELPERS=SITE/'vllm/v1/attention/ops/triton_attention_helpers.py'
JIT=SITE/'triton/spec/iluvatar/triton/runtime/jit.py'
JIT_SHA='66a168ae959e719186771e66815e807f1e449e66c5283ef596ec3eb0e35cb775'
NATIVE_QUERY=['/usr/local/corex/bin/ixsmi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits']


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def source_bundle(profile):
    sources=factory.inputs();candidate,proof=factory.build(profile,sources)
    recorded=json.loads((ROOT/'docs/evidence/konly-prefill-reader-20261027/cpu.json').read_text())
    if proof['generated_sha256']!=recorded['source_proofs'][profile]['generated_sha256']:
        raise ValueError('Reader is not the independently replayed24c source')
    baseline=factory.IMPORTS+factory.canonical(sources,profile)
    # No accepted BF16 IR-hoist transform is borrowed or applied to either role.
    return sources,baseline,candidate,proof


def installed_sources():
    record={}
    for name,path,expected in [('sdk',SDK,factory.PINS['sdk.py']),('helpers',HELPERS,factory.PINS['helpers.py']),('jit',JIT,JIT_SHA)]:
        raw=path.read_bytes();actual=factory.sha(raw)
        if actual!=expected:
            raise ValueError('Unaudited actual installed source: '+str(path))
        record[name]=dict(path=str(path),bytes=len(raw),sha256=actual)
    return record


def idle_before_cuda():
    # No waiting/queuing or stopping another process. This is a compile admission,
    # not the native-PID transition contract required by a future numeric runner.
    query=subprocess.run(NATIVE_QUERY,text=True,capture_output=True,timeout=15,check=True)
    full=subprocess.run(['/usr/local/corex/bin/ixsmi'],text=True,capture_output=True,timeout=15,check=True)
    if query.stdout.strip() or 'No running processes found' not in full.stdout:
        raise RuntimeError('GPU not explicitly idle before CUDA; compile job refuses, never queues')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',9031))==0:
            raise RuntimeError('Service9031 is active; compile job refuses')
    return dict(native_query=NATIVE_QUERY,native_stdout=query.stdout,native_stderr=query.stderr,
                full_stdout=full.stdout,full_stderr=full.stderr,port9031_closed=True,
                before_any_framework_import=True,whole_job_numeric_admission=False)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    loaded=importlib.util.module_from_spec(spec);sys.modules[name]=loaded;spec.loader.exec_module(loaded)
    return loaded


def freeze(value):
    if value is None or type(value) in (str,int,float,bool):
        return value
    if isinstance(value,dict):
        return {str(k):freeze(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [freeze(v) for v in value]
    return str(value)


def typed(value):
    """Keep tuple/dict-key types as well as a readable ABI rendering."""
    if value is None or type(value) in (str,int,float,bool):
        return dict(type=type(value).__name__,value=value)
    if isinstance(value,dict):
        return dict(type='dict',items=[[typed(k),typed(v)] for k,v in value.items()])
    if isinstance(value,(tuple,list)):
        return dict(type=type(value).__name__,items=[typed(x) for x in value])
    return dict(type=type(value).__module__+'.'+type(value).__name__,unrecognized_repr=repr(value))


def observed_resources(kernel):
    md=kernel.metadata
    result=dict(n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),
                **{k:getattr(md,k,None) for k in ('shared','num_warps','num_stages','warp_size','use_sme')})
    known=(type(result['n_regs']) is int and result['n_regs']>0 and type(result['n_spills']) is int
           and result['n_spills']>=0 and all(type(result[k]) is int and result[k]>=0 for k in ('shared','use_sme'))
           and all(type(result[k]) is int and result[k]>0 for k in ('num_warps','num_stages','warp_size')))
    result['loaded_resource_fields_known']=known
    result['zero_spill_observed']=known and result['n_spills']==0
    result['handles_explicitly_initialized']=False
    return result


def preserve(kernel,directory,label,requested,grid,kwargs,tensors,descriptors):
    destination=directory/'ir'/label;destination.mkdir(parents=True)
    files={}
    def write(kind,value):
        raw=value.encode() if isinstance(value,str) else value
        path=destination/('actual.'+kind);path.write_bytes(raw)
        files[kind]=dict(path=str(path.relative_to(directory)),bytes=len(raw),sha256=factory.sha(raw))
    for kind,value in kernel.asm.items():
        if isinstance(value,(str,bytes)):
            write(kind,value)
    fn=getattr(kernel.src,'fn',None);actual_source=getattr(fn,'src',None)
    if isinstance(actual_source,str):
        write('source',actual_source)
    roles={id(v):k for k,v in tensors.items()}
    arguments={k:dict(tensor=roles[id(v)],descriptor=descriptors[roles[id(v)]]) if id(v) in roles else freeze(v)
               for k,v in kwargs.items()}
    options=getattr(kernel,'metadata',None)
    try:
        metadata=options._asdict() if hasattr(options,'_asdict') else vars(options)
    except TypeError:
        metadata=dict(unrecognized_metadata_repr=repr(options))
    record=dict(label=label,hash=str(kernel.hash),files=files,grid=grid,requested_options=requested,
                actual_options=freeze(metadata),actual_options_source='CompiledKernel.metadata; requested options separately preserved',
                jit_version=freeze(getattr(fn,'version',None)),
                jit_do_not_specialize=freeze(getattr(fn,'do_not_specialize',None)),
                signature=freeze(getattr(kernel.src,'signature',None)),constants=freeze(getattr(kernel.src,'constants',None)),
                attrs=freeze(getattr(kernel.src,'attrs',None)),arguments=arguments,
                typed_ABI={k:typed(getattr(kernel.src,k,None)) for k in ('signature','constants','attrs')},
                resources=observed_resources(kernel),
                complete_source_and_ir={'source','ttgir','llir','asm','cubin'}.issubset(files),
                actual_I8_hoist=dict(auditor_supported=False,verified=False,
                                    reason='No audited I8 operand SSA/loop grammar; preserve original IR for review'),
                runtime_binding_qualified=False,ready_for_numeric_or_timing=False)
    save(destination/'record.json',record)
    return record


def compile_only(args,p,bundle):
    if not args.commit or args.out is None:
        raise ValueError('Explicit --compile-only requires exact --commit and fresh --out')
    if git('rev-parse','HEAD')!=args.commit or git('status','--porcelain'):
        raise ValueError('Clean exact driver freeze required')
    if git('diff',factory.MAIN,'--','vllm_fl','csrc'):
        raise ValueError('Production source changed')
    if 'torch' in sys.modules or 'triton' in sys.modules:
        raise ValueError('Compile admission must precede framework import/CUDA initialization')
    args.out.mkdir(parents=True,exist_ok=False)
    sources,baseline,candidate,proof=bundle
    result=dict(completed=False,compile_only=True,driver_commit=args.commit,reader_source_commit=READER_FREEZE,
                plan=p,proof=proof,records=[],errors=[],kernel_launches=0,initialized_tensor_elements=0,
                framework_imported=False,CUDA_API_entered=False,JIT_warmups_attempted=0,
                actual_I8_operand_hoist_verified=False,runtime_binding_qualified=False,
                accuracy_verified=False,precision_gate=False,performance_verified=False,service_installed=False,
                capacity_reclaimed=False,ready_for_numeric_or_timing=False,
                baseline_is_accepted_service_lowering=False,baseline_note='Canonical BF16 source without borrowed BF16 IR hoist')
    save(args.out/'compile-summary.json',result)
    try:
        source_dir=args.out/'sources';source_dir.mkdir()
        for name,text in [('baseline.py',baseline),('candidate.py',candidate),('sdk.py.txt',sources['sdk.py']),('helpers.py.txt',sources['helpers.py'])]:
            (source_dir/name).write_text(text)
        result['source_files']={str(path.relative_to(args.out)):dict(bytes=path.stat().st_size,sha256=factory.sha(path.read_bytes()))
                                for path in source_dir.iterdir()}
        result['installed_sources']=installed_sources()
        result['admission']=idle_before_cuda()
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        from vllm.triton_utils import triton
        result['framework_imported']=True
        result['runtime_versions']=dict(torch=getattr(torch,'__version__',None),triton=getattr(triton,'__version__',None))
        result['CUDA_API_entered']=True
        target=triton.runtime.driver.active.get_current_target()
        if (target.backend,target.arch,target.warp_size)!=('corex',71,64):
            raise ValueError('Unvalidated compiler target')
        result['target']=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size)
        sdk=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
        helper=importlib.import_module('vllm.v1.attention.ops.triton_attention_helpers')
        for actual,expected in ((sdk,factory.PINS['sdk.py']),(helper,factory.PINS['helpers.py'])):
            if factory.sha(Path(actual.__file__).read_bytes())!=expected:
                raise ValueError('Imported SDK does not match checked actual source')
        tensors,owners=contract.allocate(torch,p)
        desc=contract.descriptors(tensors);result['tensor_descriptors']=desc
        result['CPU_descriptor_contract']=contract.check_descriptors(p,desc)
        base=module(source_dir/'baseline.py','cjj_konly_compile_baseline')
        own=module(source_dir/'candidate.py','cjj_konly_compile_candidate')
        original_name='kernel_unified_attention' if p['profile']=='raw3d' else 'kernel_biv150_prefill_stride'
        for label,kernel,is_candidate in [('baseline',getattr(base,original_name),False),('candidate',getattr(own,factory.name(p['profile'])),True)]:
            values=contract.kwargs(p,tensors,is_candidate)
            result['JIT_warmups_attempted']+=1
            compiled=kernel.warmup(grid=p['grid'],**values,**p['requested_options'])
            result['records'].append(preserve(compiled,args.out,label,p['requested_options'],p['grid'],values,tensors,desc))
            save(args.out/'compile-summary.json',result)
        if p['profile']=='raw3d':
            values=contract.reducer_kwargs(p,tensors);shape=(p['total'],16)
            result['JIT_warmups_attempted']+=1
            compiled=sdk.reduce_segments.warmup(grid=shape,**values)
            result['records'].append(preserve(compiled,args.out,'original_reducer',{},shape,values,tensors,desc))
        result['completed']=all(r['complete_source_and_ir'] for r in result['records'])
        if not result['completed']:
            raise RuntimeError('Required actual source/TTGIR/LLVM/ASM/cubin missing; raw exports preserved')
        save(args.out/'compile-summary.json',result)
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)))
        result['completed']=False;save(args.out/'compile-summary.json',result)
        raise
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=contract.PROFILES_SUPPORTED,default='q32')
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--commit');parser.add_argument('--out',type=Path)
    args=parser.parse_args();p=contract.plan(args.profile);bundle=source_bundle(args.profile)
    if args.compile_only:
        result=compile_only(args,p,bundle)
    else:
        if args.commit or args.out:
            parser.error('--commit/--out belong only to explicit compile-only; CPU plan writes stdout')
        result=dict(cpu_only=True,plan=p,proof=bundle[3],reader_source_commit=READER_FREEZE,
                    kernel_launches=0,JIT_executed=False,torch_imported='torch' in sys.modules,
                    triton_imported='triton' in sys.modules,actual_I8_operand_hoist_verified=False,
                    runtime_binding_qualified=False,accuracy_verified=False,performance_verified=False,
                    precision_gate=False,service_installed=False,capacity_reclaimed=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
