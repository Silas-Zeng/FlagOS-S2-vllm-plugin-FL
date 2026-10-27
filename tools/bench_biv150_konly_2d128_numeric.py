#!/usr/bin/env python3
"""Default CPU plan; explicit two-pattern eager BM128 consumer. No timing/service."""
import argparse
import gzip
import importlib
import json
import os
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import bench_biv150_konly_2d128_compile as frontend
import biv150_konly_2d128_actual as actual
import biv150_konly_2d128_compile_contract as descriptors
import biv150_konly_2d128_numeric_contract as numeric
import biv150_konly_q32_numeric_contract as oracle
from biv150_konly_2d128_numeric_admission import Admission
parent=frontend.legacy


def guarded_cpu_snapshot(buffer,torch):
    """Read the entire allocation before selecting guard rows on the CPU.

    The isolated 4d probe found GPU advanced[0,-1] incorrect without attention.
    This keeps both original 71 guards and preserves the actual whole backing.
    """
    full=buffer.cpu()
    actual.need(full.device.type=='cpu' and tuple(full.shape)==(2050,16,128),'full guarded CPU snapshot')
    return full,torch.stack((full[0],full[2049]))


def preflight():
    files,proof,summary=actual.original();frontend.source_bundle()
    roots={}
    for role in ('baseline','candidate'):
        text=files['out/ir/'+role+'/actual.ttir'].decode()
        roots[role]=str(Path(re.search(r'#loc = loc\("([^\"]+)"',text)[1]).parent)
    actual.need(len(set(roots.values()))==1,'same original source directory')
    return dict(plan=numeric.plan(),original_compile_commit=actual.COMMIT,source_root=roots['candidate'],
        actual_compile_mechanism_reviewed=True,actual_I8_operand_hoist_verified=False,
        baseline_is_current_main_policy=False,loaded_resources_verified=False,
        eager_numeric_passed=False,precision_gate=False,Graph_verified=False,
        side_stream_verified=False,model_or_service_verified=False,performance_measured=False),files,summary


def verify_compilation(record,expected,role,observed,reviewed):
    for key in ('hash','typed_ABI','actual_options','jit_version','jit_do_not_specialize'):
        actual.need(record[key]==expected[key],role+': unknown '+key)
    for ext in ('source','ttir','ttgir','llir'):
        actual.need(observed[ext]==reviewed[ext] and actual.sha(observed[ext])==record['files'][ext]['sha256'],role+': unknown '+ext)
    for ext in ('asm','cubin'):
        actual.need(actual.sha(observed[ext])==record['files'][ext]['sha256'],'saved actual artifact SHA')
    actual.need(len(observed['cubin'])>64 and observed['cubin'][:6]==b'\x7fELF\x02\x01','actual target ELF64 cubin')
    assembly=observed['asm'].decode()
    if role=='candidate':proof=actual.mechanism(*(observed[e].decode() for e in ('ttgir','llir','asm')),known=False)
    else:
        actual.need(assembly.count('ml_matrix_mad_f32_bf16_m16n16k16')==64
                    and 'ml_matrix_mad_i32_i8_' not in assembly,'baseline native BF16 paths')
        proof=dict(BF16_matrix_instructions=64)
    def controls(data):
        return [s.strip() for s in data.decode().splitlines() if s.strip().startswith(('sl_cbr','sl_jump','sl_barrier','sl_wait'))]
    actual.need(controls(observed['asm'])==controls(reviewed['asm']),'actual ASM control/synchronization differs')
    return dict(source_typed_ABI_options_TTIR_TTGIR_LLVM_exact=True,actual_mechanism=proof,
                ASM_text_exact=observed['asm']==reviewed['asm'],cubin_bytes_exact=observed['cubin']==reviewed['cubin'],
                cubin_equality_required=False,backend_binary_equivalence_claimed=False,numerical_validation_required=True)


def run(args,payload):
    pre,files,summary=payload
    actual.need(args.commit and args.out and args.source_root,'exact commit/original source-root/fresh output')
    actual.need(not args.out.exists() and not args.out.resolve().is_relative_to(parent.ROOT),'fresh external output')
    actual.need(args.source_root.resolve()==Path(pre['source_root']),'original compile debug/source directory required')
    actual.need(not {'torch','triton'}.intersection(sys.modules),'cold admission before framework')
    def binding():
        actual.need(parent.git('rev-parse','HEAD')==args.commit and not parent.git('status','--porcelain'),'exact clean frontend freeze')
        actual.need(not parent.git('diff',descriptors.MAIN,'--','vllm_fl','csrc'),'production changed from DAC')
        hashes={}
        for role in ('baseline','candidate'):
            path=args.source_root/(role+'.py');data=path.read_bytes()
            actual.need(data==files['out/sources/'+role+'.py'],'original source path SHA changed')
            hashes[role]=dict(path=str(path),sha256=actual.sha(data))
        return dict(frontend_commit=args.commit,source_files=hashes,production_parent=descriptors.MAIN)
    args.out.mkdir(parents=True)
    result=dict(**pre,commit=args.commit,completed=False,errors=[],records=[],component_only=True,
                reader_launches=0,encoder_launches=0,actual_binding_verified=False,
                full_numerical_matrix_verified=False,timing_attempted=False)
    admission=Admission(lambda:(binding(),parent.installed_sources(),parent.idle_before_cuda()),binding)
    def save():result['admission']=admission.receipt();parent.save(args.out/'numeric-summary.json',result)
    save()
    try:
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        token=admission.begin();save()
        import numpy as np
        import torch
        from vllm.triton_utils import triton
        import biv150_prefill_probe_fixtures as fixture
        fixture.torch=torch;torch.set_num_threads(4);torch.set_flush_denormal(False)
        result['loaded_modules']={}
        for module_name,pin in [('vllm.v1.attention.ops.triton_unified_attention',parent.factory.PINS['sdk.py']),
                                ('vllm.v1.attention.ops.triton_attention_helpers',parent.factory.PINS['helpers.py'])]:
            module=importlib.import_module(module_name);path=Path(module.__file__)
            actual.need(actual.sha(path.read_bytes())==pin,'actual loaded SDK/helper identity')
            result['loaded_modules'][module_name]=dict(path=str(path),sha256=pin)
        result['versions']={name:getattr(module,'__version__',None) for name,module in [('torch',torch),('triton',triton)]}
        actual.need(sys.byteorder=='little','little-endian CPU oracle')
        target=triton.runtime.driver.active.get_current_target()
        actual.need((target.backend,target.arch,target.warp_size)==('corex',71,64),'actual target')
        p=numeric.plan();t,owners=descriptors.allocate(torch,p)
        guarded={role:torch.empty((2050,16,128),device='cuda',dtype=torch.bfloat16) for role in ('baseline','candidate')}
        functions={role:getattr(parent.module(args.source_root/(role+'.py'),'cjj_2d128_numeric_'+role),
                    'kernel_biv150_prefill_stride' if role=='baseline' else parent.factory.name(descriptors.PROFILE))
                    for role in guarded}
        expected={r['label']:r for r in summary['records']};calls={};compiled={};records={}
        admission.check('compile',token)
        for role in guarded:
            current=dict(t,out=guarded[role][1:-1]);d=descriptors.descriptors(current);descriptors.check_descriptors(p,d)
            kw=descriptors.kwargs(p,current,role=='candidate');calls[role]=(current,kw,d)
            kernel=functions[role].warmup(grid=p['grid'],**kw,**p['requested_options']);compiled[role]=kernel
            record=parent.preserve(kernel,args.out,role,p['requested_options'],p['grid'],kw,current,d)
            records[role]=record
        result['actual_records']=records;save()
        for role,record in records.items():
            folder=args.out/'ir'/role
            observed={ext:(folder/('actual.'+ext)).read_bytes() for ext in ('source','ttir','ttgir','llir','asm','cubin')}
            reviewed={ext:files['out/ir/'+role+'/actual.'+ext] for ext in observed}
            proof=verify_compilation(record,expected[role],role,observed,reviewed)
            parent.save(folder/'actual-compilation-binding.json',proof)
            kernel=compiled[role];kernel._init_handles();resources=parent.observed_resources(kernel)
            resources['handles_explicitly_initialized']=True
            actual.need(resources['loaded_resource_fields_known'] and resources['zero_spill_observed'],'unknown/spilled loaded resources')
            record['loaded_resources']=resources;parent.save(folder/'record.json',record)
        result['loaded_resources_verified']=True;result['actual_binding_verified']=True;save()
        def host_bytes(tensor):return tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        def fingerprints():
            return {name:actual.sha(host_bytes(v)) for name,v in dict(q=t['q'],packed=owners[0],encoded=owners[1],
                table=t['table'],lengths=t['lengths'],cu_q=t['cu_q'],k_descale=t['k_descale'],v_descale=t['v_descale']).items()}
        for pattern in p['patterns']:
            cpu=fixture.cpu_inputs(p['qlens'],p['klens'],p['seed'],pattern)
            raw,codes,scales,codec=oracle.encoded_bytes(cpu['packed'][:,0],cpu['packed'][:,1],torch,np)
            reference=oracle.references(cpu,codes,scales,torch,np)
            owners[0].copy_(cpu['packed']);owners[1].copy_(torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()))
            for name,source in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]:t[name].copy_(cpu[source])
            torch.cuda.synchronize();read_cache=host_bytes(owners[1]);codec_ok=read_cache==raw
            actual.need(codec_ok,'CPU-coded cache upload/readback differs')
            before=fingerprints();outputs={};canaries={};readonly=True;canary=True
            artifact=dict(original_CPU=cpu,actual_encoded_cache_bytes=torch.from_numpy(np.frombuffer(read_cache,dtype=np.uint8).copy()),
                          expected_encoded_sha256=actual.sha(raw),reference=reference,outputs=outputs,output_canaries=canaries,
                          read_only_before=before,read_only_after={},guarded_before={},guarded_after={},
                          guard_readback='entire guarded.cpu(), then CPU positive rows0/2049')
            artifact_path=args.out/('pattern-'+str(pattern)+'.pt.gz')
            def dump():
                with gzip.open(artifact_path,'wb',compresslevel=3) as gz:torch.save(artifact,gz)
            try:
                for role in guarded:
                    guarded[role].fill_(71);current,kw,d=calls[role];current['out'].fill_(float('nan'))
                    full_before,guard_before=guarded_cpu_snapshot(guarded[role],torch)
                    artifact['guarded_before'][role]=full_before
                    actual.need(bool((guard_before==71).all()),'output guard setup before attention')
                    admission.check(role+str(pattern),token)
                    actual.need(descriptors.descriptors(current)==d,'live descriptor changed')
                    launched=functions[role][p['grid']](**kw,**p['requested_options']);result['reader_launches']+=1
                    actual.need(str(launched.hash)==records[role]['hash'],'launched specialization differs')
                    torch.cuda.synchronize();outputs[role]=current['out'].cpu()
                    full_after,canaries[role]=guarded_cpu_snapshot(guarded[role],torch)
                    artifact['guarded_after'][role]=full_after
                    actual.need(host_bytes(full_after[1:2049])==host_bytes(outputs[role]),'output view/full-storage CPU copies differ')
                    canary=canary and bool((canaries[role]==71).all())
                    after=fingerprints();artifact['read_only_after'][role]=after;readonly=readonly and after==before
                    actual.need(canary and readonly,'output canary/readonly hard gate')
                check=numeric.gates(outputs['baseline'][reference['rows']].double().reshape(-1).tolist(),
                    outputs['candidate'][reference['rows']].double().reshape(-1).tolist(),
                    reference['original'].reshape(-1).tolist(),reference['quantized'].reshape(-1).tolist())
                # Pattern0 also checks every query/head/channel against an FP64
                # cumulative mean rounded to BF16 (the existing causal fixture).
                uniform=None
                if pattern==0:
                    rows,ref=fixture.reference(cpu,p['qlens'],p['klens'],True,False)
                    uniform={role:fixture.compare(value[rows],ref,.005,.005) for role,value in outputs.items()}
                hard=dict(finite=all(bool(torch.isfinite(v).all()) for v in [*outputs.values(),reference['original'],reference['quantized']]),
                    baseline_reference=check['baseline_hard_passed'] and (uniform is None or uniform['baseline']['passed']),
                    CPU_codec_cache_bytes=codec_ok,output_canaries=canary,read_only_inputs=readonly,actual_IR=True,
                    quantized_implementation=check['quantized_implementation_passed'] and (uniform is None or uniform['candidate']['passed']))
                item=dict(pattern=pattern,hard_checks=hard,numeric_checks=check,uniform_all_rows=uniform,codec_CPU=codec,
                          read_only_before=before,read_only_after=artifact['read_only_after'])
                result['records'].append(item);save()
                item['policy']=numeric.policy(hard,check['original_BF16_quality_passed'],args.numeric_diagnostic)
            finally:
                dump()
                if result['records'] and result['records'][-1]['pattern']==pattern:
                    result['records'][-1]['artifact']=dict(file=artifact_path.name,bytes=artifact_path.stat().st_size,
                        sha256=actual.sha(artifact_path.read_bytes()),format='lossless gzip(torch.save) CPU inputs/cache/refs/full outputs/canaries')
                save()
        admission.check('complete',token)
        result['completed']=True;result['eager_numeric_passed']=all(r['numeric_checks']['candidate_numeric_passed'] for r in result['records'])
        return result
    except Exception as exc:
        admission.state='failed';admission.errors.append(repr(exc));result['errors'].append(dict(type=type(exc).__name__,message=str(exc)));raise
    finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-gpu',action='store_true')
    p.add_argument('--numeric-diagnostic',action='store_true');p.add_argument('--commit');p.add_argument('--source-root',type=Path);p.add_argument('--out',type=Path);args=p.parse_args()
    payload=preflight()
    if args.run_gpu:result=run(args,payload)
    else:
        if any((args.commit,args.out,args.source_root,args.numeric_diagnostic)):p.error('device-only options require --run-gpu')
        result=dict(**payload[0],CPU_only=True,JIT_executed=False,GPU_executed=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
