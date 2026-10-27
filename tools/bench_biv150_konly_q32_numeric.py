#!/usr/bin/env python3
"""Default CPU; explicit Root-only first eager Q32 comparison. No timing/service hook."""
import argparse
import difflib
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
    original=actual.old.original()[0]
    source=original['out/ir/candidate/actual.ttgir'].decode()
    source_root=Path(re.search(r'#loc = loc\("([^\"]+)/candidate.py"',source)[1])
    roots={}
    for role in ('baseline','unhoisted','hoisted'):
        origin,label=(files,'candidate') if role=='hoisted' else (original,'baseline' if role=='baseline' else 'candidate')
        # TTIR describes the actual Python source root; hoisted TTGIR locations
        # instead come from the separately fixed original-d6 override.
        roots[role]=str(Path(re.search(r'#loc = loc\("([^\"]+)"',origin['out/ir/'+label+'/actual.ttir'].decode())[1]).parent)
    return dict(plan=numeric.plan(),source_root=str(source_root),override=record['actual_options']['ir_override'],
                role_source_roots=roots,
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


def verify_exact_machine(record,expected,role,observed,reviewed):
    """Limited three-role Q32 contract: exact source/IR/cubin, audited ASM sidecar.

    A printer-only ASM difference cannot license different actual machine bytes.
    Every observed/exported byte is still preserved and hashed independently.
    """
    if role not in ('baseline','unhoisted','hoisted'): raise ValueError('Unknown exact-machine role')
    changed=differences(record,expected)
    if set(changed)-{'asm'}: raise ValueError(role+': source/ABI/options/IR/cubin differs; '+json.dumps(changed,sort_keys=True))
    for ext in ('source','ttir','ttgir','llir','asm','cubin'):
        if actual.old.sha(observed[ext])!=record['files'][ext]['sha256'] or actual.old.sha(reviewed[ext])!=expected['files'][ext]['sha256']:
            raise ValueError('Exported/reviewed '+ext+' is not its recorded original')
    assembly=observed['asm'].decode();old_assembly=reviewed['asm'].decode()
    def instructions(text):
        return [s.strip() for s in text.splitlines() if s.strip().startswith(('ml_','sl_'))]
    current_ops,old_ops=instructions(assembly),instructions(old_assembly)
    if [s.split()[0] for s in current_ops]!=[s.split()[0] for s in old_ops]:
        raise ValueError('Unknown ASM instruction sequence, despite identical cubin')
    controls=lambda ops:[s for s in ops if s.startswith(('sl_cbr','sl_jump','sl_barrier','sl_wait'))]
    if controls(current_ops)!=controls(old_ops): raise ValueError('ASM control/synchronization text changed')
    if role=='hoisted':
        audit=actual.audit(*(observed[e].decode() for e in ('ttgir','llir','asm')),known_hash=False)
    elif role=='unhoisted':
        grammar=actual.old.grammar(observed['ttgir'].decode())
        llvm=actual.old.llvm_observation(observed['llir'].decode())
        if len(llvm['Q_shared_reads_inside_KV'])!=16: raise ValueError('Unhoisted control placement changed')
        audit=dict(TTGIR=grammar,LLVM=llvm,actual_I8_operand_hoist_verified=False)
    else:
        count=assembly.count('ml_matrix_mad_f32_bf16_m16n16k16')
        if count!=64 or 'ml_matrix_mad_i32_i8_' in assembly: raise ValueError('BF16 control native matrix grammar changed')
        audit=dict(BF16_matrix_instructions=count,source_IR_and_cubin_exact=True,actual_I8_operand_hoist_verified=False)
    diff='\n'.join(difflib.unified_diff(old_assembly.splitlines(),assembly.splitlines(),fromfile='reviewed.asm',tofile='actual.asm',n=2))
    if len(diff.encode())>65536: raise ValueError('ASM sidecar exceeds this bounded review scope')
    return dict(contract='three recorded Q32 roles: exact ABI/source/TTIR/TTGIR/LLVM/cubin plus strict actual audit',
                actual_cubin_exact=True,typed_ABI_options_exact=True,source_and_IR_exact=True,
                ASM_text_exact=observed['asm']==reviewed['asm'],ASM_diff=diff,actual_strict_audit=audit,
                original_numeric_gates_unchanged=True,precision_gate=False)


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
        hashes={}
        for role,root in pre['role_source_roots'].items():
            name='baseline.py' if role=='baseline' else 'candidate.py'
            hashes[role]=dict(path=str(Path(root)/name),sha256=actual.old.sha((Path(root)/name).read_bytes()))
            if hashes[role]['sha256']!=actual.old.sha(original['out/sources/'+name]): raise ValueError('Original role source-root bytes changed')
        if actual.old.sha(Path(pre['override']).read_bytes())!=pre['override_sha256']: raise ValueError('Original fixed override missing/changed')
        return dict(frontend_root=str(parent.ROOT),frontend_commit=args.commit,old_debug_source_root=str(expected_root),
                    role_source_roots=pre['role_source_roots'],source_hashes=hashes,
                    override=pre['override'],override_sha256=pre['override_sha256'])
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
        modules={role:parent.module(Path(pre['role_source_roots'][role])/('baseline.py' if role=='baseline' else 'candidate.py'),
                    'cjj_konly_numeric_'+role) for role in ('baseline','unhoisted','hoisted')}
        support_path=args.out/'codec-support-ae00.py';support_path.write_text(parent.factory.inputs()['codec-support-ae00.py'])
        support=parent.module(support_path,'cjj_konly_numeric_codec_ae00')
        options={role:dict(p['requested_options'],**({'ir_override':pre['override']} if role=='hoisted' else {})) for role in modules}
        functions={role:getattr(module,'kernel_biv150_prefill_stride' if role=='baseline' else parent.factory.name('q32')) for role,module in modules.items()}
        expected={role:next(r for r in summary['records'] if r['label']==('baseline' if role=='baseline' else 'candidate')) for role in modules}
        expected['unhoisted']=actual.old.original()[2]
        expected['baseline']=next(r for r in actual.old.original()[1]['records'] if r['label']=='baseline')
        reviewed={role:{ext:(files if role=='hoisted' else original)['out/ir/'+('baseline' if role=='baseline' else 'candidate')+'/actual.'+ext]
                         for ext in ('source','ttir','ttgir','llir','asm','cubin')} for role in modules}
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
            # Warmup/export all roles before refusing any binding. This keeps
            # the first unknown specialization useful without any reader launch.
            actual_records[role]=record
        result['actual_records']=actual_records;save()
        for role,record in actual_records.items():
            directory=args.out/'ir'/role
            for ext in ('source','ttir','ttgir','llir','asm','cubin'):
                (directory/('reviewed.'+ext)).write_bytes(reviewed[role][ext])
            observed={ext:(directory/('actual.'+ext)).read_bytes() for ext in reviewed[role]}
            audit=verify_exact_machine(record,expected[role],role,observed,reviewed[role])
            parent.save(directory/'exact-machine-binding.json',audit)
            (directory/'actual-vs-reviewed-ASM.diff').write_text(audit['ASM_diff'])
            kernel=compiled[role]
            kernel._init_handles();resources=parent.observed_resources(kernel);resources['handles_explicitly_initialized']=True
            if not resources['loaded_resource_fields_known'] or not resources['zero_spill_observed']:
                raise ValueError(role+': loaded register/spill/resource fields unknown or spilled')
            record['loaded_resources']=resources;parent.save(args.out/'ir'/role/'record.json',record)
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
