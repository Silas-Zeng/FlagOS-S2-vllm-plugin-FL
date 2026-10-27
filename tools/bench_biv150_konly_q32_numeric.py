#!/usr/bin/env python3
"""Default CPU; explicit Root-only first eager Q32 comparison. No timing/service hook."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sys

# Preserve the original debug/source root and installed package caches.
sys.dont_write_bytecode=True

import bench_biv150_konly_prefill_compile as parent
import bench_biv150_konly_q32_hoist_compile as frontend
import biv150_konly_prefill_compile_contract as descriptors
import biv150_konly_q32_actual_hoist as actual
import biv150_konly_q32_numeric_contract as numeric
from biv150_konly_numeric_admission import Admission


def preflight():
    payload=frontend.preflight();files,summary,record=actual.original()
    proof=actual.audit(*(files['out/ir/candidate/actual.'+e].decode() for e in ('ttgir','llir','asm')))
    source=actual.old.original()[0]['out/ir/candidate/actual.ttgir'].decode()
    source_root=Path(re.search(r'#loc = loc\("([^\"]+)/candidate.py"',source)[1])
    return dict(plan=numeric.plan(),source_root=str(source_root),override=record['actual_options']['ir_override'],
                override_sha256=frontend.OVERRIDE_SHA,actual_mechanism_verified=proof['actual_I8_operand_hoist_verified'],
                baseline_is_accepted_service_main=False,Graph_verified=False,side_stream_verified=False,
                actual_binding_verified=False,eager_numeric_passed=False,precision_gate=False,
                performance_verified=False,service_verified=False,capacity_reclaimed=False),payload,files,summary


def differences(record,expected):
    changed={key:dict(expected=expected[key],actual=record[key]) for key in
             ('hash','typed_ABI','actual_options','jit_version','jit_do_not_specialize') if record[key]!=expected[key]}
    for ext in ('source','ttir','ttgir','llir','asm','cubin'):
        before=expected['files'].get(ext,{}).get('sha256');after=record['files'].get(ext,{}).get('sha256')
        if before is None or before!=after: changed[ext]=dict(expected=before,actual=after)
    return changed


def bind_record(record,expected,role):
    changed=differences(record,expected)
    if changed: raise ValueError(role+': unknown actual specialization/ABI/options/IR; retained diff='+json.dumps(changed,sort_keys=True))


def run(args,payload):
    pre,_,files,summary=payload
    if not args.commit or args.out is None or args.source_root is None: raise ValueError('Exact commit, fresh out and original source-root required')
    if args.out.exists() or args.out.resolve().is_relative_to(parent.ROOT): raise ValueError('Fresh output outside source checkout required')
    if 'torch' in sys.modules or 'triton' in sys.modules: raise ValueError('Admission must precede framework import')
    expected_root=Path(pre['source_root'])
    if args.source_root.resolve()!=expected_root: raise ValueError('Original d6 debug/source-root must be retained; no checkout impersonation')
    original=actual.old.original()[0]
    def binding():
        if parent.git('rev-parse','HEAD')!=args.commit or parent.git('status','--porcelain'):
            raise ValueError('Exact clean numeric frontend freeze required')
        if parent.git('diff',parent.factory.MAIN,'--','vllm_fl','csrc'): raise ValueError('Production changed')
        hashes={name:actual.old.sha((expected_root/name).read_bytes()) for name in ('baseline.py','candidate.py')}
        if any(hashes[name]!=actual.old.sha(original['out/sources/'+name]) for name in hashes): raise ValueError('Original source-root bytes changed')
        if actual.old.sha(Path(pre['override']).read_bytes())!=pre['override_sha256']: raise ValueError('Original fixed override missing/changed')
        return dict(frontend_root=str(parent.ROOT),frontend_commit=args.commit,old_debug_source_root=str(expected_root),
                    source_hashes=hashes,override=pre['override'],override_sha256=pre['override_sha256'])
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(**pre,frontend_commit=args.commit,completed=False,errors=[],records=[],
                reader_launches=0,encoder_launches=0,Graph_executed=False,side_stream_executed=False,
                timing_attempted=False,timings=None,full_validation_passed=False)
    admission=Admission(lambda:(binding(),parent.installed_sources(),parent.idle_before_cuda()),binding)
    def save():
        result['admission']=admission.receipt();parent.save(args.out/'numeric-summary.json',result)
    save()
    try:
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        token=admission.begin();save()
        import numpy as np
        import torch
        from vllm.triton_utils import triton
        import biv150_prefill_probe_fixtures as fixture
        fixture.torch=torch;torch.set_num_threads(4);torch.set_flush_denormal(False)
        if sys.byteorder!='little': raise ValueError('Packed byte oracle requires the witnessed little-endian host')
        target=triton.runtime.driver.active.get_current_target()
        if (target.backend,target.arch,target.warp_size)!=('corex',71,64): raise ValueError('Unknown compiler target')
        p=numeric.plan();t,owners=descriptors.allocate(torch,p)
        # One live prefix/suffix token guards every BF16 output view.
        guarded={role:torch.empty((2050,16,128),device='cuda',dtype=torch.bfloat16) for role in ('baseline','unhoisted','hoisted')}
        modules={role:parent.module(expected_root/('baseline.py' if role=='baseline' else 'candidate.py'),
                    'cjj_konly_numeric_'+role) for role in ('baseline','unhoisted','hoisted')}
        support_path=args.out/'codec-support-ae00.py';support_path.write_text(parent.factory.inputs()['codec-support-ae00.py'])
        support=parent.module(support_path,'cjj_konly_numeric_codec_ae00')
        options={role:dict(p['requested_options'],**({'ir_override':pre['override']} if role=='hoisted' else {})) for role in modules}
        functions={role:getattr(module,'kernel_biv150_prefill_stride' if role=='baseline' else parent.factory.name('q32')) for role,module in modules.items()}
        expected={role:next(r for r in summary['records'] if r['label']==('baseline' if role=='baseline' else 'candidate')) for role in modules}
        expected['unhoisted']=actual.old.original()[2]
        calls={};compiled={};actual_records={}
        admission.check('compile',token)
        for role in modules:
            current=dict(t,out=guarded[role][1:-1]);d=descriptors.descriptors(current);descriptors.check_descriptors(p,d)
            kw=descriptors.kwargs(p,current,role!='baseline');calls[role]=(current,kw,d)
            kernel=functions[role].warmup(grid=p['grid'],**kw,**options[role]);compiled[role]=kernel
            record=parent.preserve(kernel,args.out,role,options[role],p['grid'],kw,current,d)
            parent.save(args.out/'ir'/role/'expected-reviewed-record.json',expected[role])
            parent.save(args.out/'ir'/role/'binding-difference.json',dict(differences=differences(record,expected[role]),
                        actual_geometry=p,actual_tensor_descriptors=d,expected_geometry_is_original4096=True))
            bind_record(record,expected[role],role)
            if role=='hoisted': actual.audit(*(kernel.asm[e] for e in ('ttgir','llir','asm')))
            kernel._init_handles();resources=parent.observed_resources(kernel);resources['handles_explicitly_initialized']=True
            if not resources['loaded_resource_fields_known'] or not resources['zero_spill_observed']:
                raise ValueError(role+': loaded register/spill/resource fields unknown or spilled')
            record['loaded_resources']=resources;parent.save(args.out/'ir'/role/'record.json',record)
            actual_records[role]=record
        result['actual_records']=actual_records;result['actual_binding_verified']=True;save()
        def save_artifact(path,value):
            # Lossless outer compression; all CPU inputs/raw bytes/references and
            # all three independently read full outputs remain in the artifact.
            with gzip.open(path,'wb',compresslevel=3) as stream: torch.save(value,stream)
        def host_bytes(tensor): return tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        def fingerprints():
            return {name:actual.old.sha(host_bytes(value)) for name,value in dict(q=t['q'],packed=owners[0],encoded=owners[1],
                table=t['table'],lengths=t['lengths'],cu_q=t['cu_q'],k_descale=t['k_descale'],v_descale=t['v_descale']).items()}
        for pattern in p['patterns']:
            cpu=fixture.cpu_inputs(p['qlens'],p['klens'],p['seed'],pattern)
            raw,codes,scales,codec=numeric.encoded_bytes(cpu['packed'][:,0],cpu['packed'][:,1],torch,np)
            reference=numeric.references(cpu,codes,scales,torch,np)
            artifact=dict(original_CPU=cpu,expected_encoded_bytes=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()),
                          expected_codes=codes,expected_scales=scales,reference=reference,outputs={})
            artifact_path=args.out/('pattern-'+str(pattern)+'.pt.gz');save_artifact(artifact_path,artifact)
            owners[0].copy_(cpu['packed']);owners[1].fill_(0xa5)
            for name,source in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]:
                t[name].copy_(cpu[source])
            before_encode=fingerprints()
            admission.check('encode'+str(pattern),token)
            support.encode_initial[(p['pages']*32,)](t['baseline_k'],t['baseline_v'],t['k'],t['kscale'],t['v'],p['pages'])
            result['encoder_launches']+=1;torch.cuda.synchronize()
            actual_raw=host_bytes(owners[1]);artifact['actual_encoded_bytes']=torch.from_numpy(np.frombuffer(actual_raw,dtype=np.uint8).copy())
            save_artifact(artifact_path,artifact)
            codec_passed=actual_raw==raw
            if not codec_passed: raise ValueError('Full CPU-independent encoded cache/scale/value/gap bytes differ')
            before=fingerprints()
            if any(before[name]!=value for name,value in before_encode.items() if name!='encoded'):
                raise ValueError('Encoder modified a read-only original input')
            if host_bytes(owners[0])!=host_bytes(cpu['packed']): raise ValueError('Original BF16 cache differs from frozen CPU source')
            outputs={};canary=True;readonly=True
            for role in modules:
                guarded[role].fill_(71);calls[role][0]['out'].fill_(float('nan'))
                admission.check(role+str(pattern),token)
                current,kw,d=calls[role]
                if descriptors.descriptors(current)!=d: raise ValueError('Runtime tensor descriptor changed after compile')
                # Same immutable Tensor objects, scalars and options as guarded warmup.
                launched=functions[role][p['grid']](**kw,**options[role]);result['reader_launches']+=1
                if str(launched.hash)!=expected[role]['hash']: raise ValueError('Unexpected actual launched specialization')
                torch.cuda.synchronize();outputs[role]=current['out'].cpu()
                canary=canary and bool((guarded[role][0]==71).all() and (guarded[role][-1]==71).all())
                readonly=readonly and fingerprints()==before
                artifact['outputs'][role]=outputs[role];save_artifact(artifact_path,artifact)
                if not canary or not readonly: raise ValueError('Output canary/read-only input guard failed')
            check=numeric.gates(outputs['baseline'][reference['rows']].double().reshape(-1).tolist(),
                outputs['hoisted'][reference['rows']].double().reshape(-1).tolist(),
                reference['original'].reshape(-1).tolist(),reference['quantized'].reshape(-1).tolist())
            hard=dict(finite=all(bool(torch.isfinite(v).all()) for v in [*outputs.values(),reference['original'],reference['quantized']]),
                baseline_reference=check['baseline_hard_passed'],codec=codec_passed,output_canaries=canary,
                read_only_inputs=readonly,actual_IR=True,
                hoisted_unhoisted_bitwise=host_bytes(outputs['hoisted'])==host_bytes(outputs['unhoisted']))
            item=dict(pattern=pattern,hard_checks=hard,numeric_checks=check,codec_CPU=codec,
                      artifact=dict(path=str(artifact_path),format='lossless gzip(torch.save)',bytes=artifact_path.stat().st_size,sha256=actual.old.sha(artifact_path.read_bytes())))
            result['records'].append(item);save()
            item['policy']=numeric.policy(hard,check['candidate_numeric_passed'],args.numeric_diagnostic);save()
        admission.check('complete',token)
        result['completed']=True;result['eager_numeric_passed']=all(r['numeric_checks']['candidate_numeric_passed'] for r in result['records'])
        return result
    except Exception as error:
        admission.state='failed';admission.errors.append(repr(error))
        result['errors'].append(dict(type=type(error).__name__,message=str(error)));raise
    finally: save()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--numeric-diagnostic',action='store_true');parser.add_argument('--commit')
    parser.add_argument('--source-root',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    payload=preflight()
    if args.run_gpu: result=run(args,payload)
    else:
        if any((args.numeric_diagnostic,args.commit,args.source_root,args.out)): parser.error('Device-only flags require explicit --run-gpu')
        result=dict(**payload[0],cpu_only=True,JIT_executed=False,GPU_executed=False)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__': main()
