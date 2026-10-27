#!/usr/bin/env python3
"""Default CPU; explicit Root-only M2/raw3D eager numerics, no timing or hook."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True

import biv150_konly_allreader_numeric_contract as contract
import bench_biv150_konly_q32_attention_perf as ownership

parent=contract.parent


def run(args):
    files,reviewed,source_root=contract.original(args.profile)
    if not args.commit or args.out is None or args.source_root is None: raise ValueError('Exact commit/fresh output/original source root required')
    if args.source_root.resolve()!=Path(source_root) or args.out.exists() or args.out.resolve().is_relative_to(parent.ROOT): raise ValueError('Unknown source root or nonfresh output')
    if any(n in sys.modules for n in ('torch','triton')): raise ValueError('Admission precedes framework import')
    def binding():
        if parent.git('rev-parse','HEAD')!=args.commit or parent.git('status','--porcelain'): raise ValueError('Exact clean frontend required')
        if parent.git('diff',parent.factory.MAIN,'--','vllm_fl','csrc'): raise ValueError('Production changed')
        sources={}
        for role in ('baseline','candidate'):
            path=Path(source_root)/(role+'.py');raw=path.read_bytes()
            if raw!=files['out/sources/'+role+'.py']: raise ValueError('Original role source path/bytes changed')
            sources[role]=dict(path=str(path),sha256=parent.factory.sha(raw))
        return dict(frontend_root=str(parent.ROOT),frontend_commit=args.commit,profile=args.profile,original_sources=sources,
                    original_compile_archive_sha256=contract.PINS[args.profile][0])
    binding();args.out.mkdir(parents=True,exist_ok=False)
    result=dict(completed=False,errors=[],profile=args.profile,frontend_commit=args.commit,records=[],
                reader_launches=0,reducer_launches=0,encoder_launches=0,actual_binding_verified=False,
                actual_I8_operand_hoist_verified=False,baseline_is_accepted_service_main=False,
                Graph_verified=False,side_stream_verified=False,timing_attempted=False,writer_ms=None,fullcost_ms=None,
                precision_gate=False,performance_verified=False,service_verified=False,capacity_reclaimed=False)
    owned=ownership.Owned(lambda:(binding(),parent.installed_sources(),parent.idle_before_cuda()),binding)
    def save(): result['admission']=owned.receipt();parent.save(args.out/'allreader-numeric-summary.json',result)
    save()
    try:
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache');owned.begin();save()
        import numpy as np
        import torch
        from vllm.triton_utils import triton
        from vllm.v1.attention.ops.triton_unified_attention import reduce_segments
        import biv150_prefill_probe_fixtures as fixture
        fixture.torch=torch;torch.set_num_threads(4);torch.set_flush_denormal(False)
        if sys.byteorder!='little': raise ValueError('Witnessed packed byte format is little-endian')
        target=triton.runtime.driver.active.get_current_target()
        if (target.backend,target.arch,target.warp_size)!=('corex',71,64): raise ValueError('Unknown target')
        props=torch.cuda.get_device_properties(0)
        if 'BI-V150' not in props.name or props.multi_processor_count!=16: raise ValueError('Unknown GPU')
        support_path=args.out/'codec-support-ae00.py';support_path.write_text(parent.factory.inputs()['codec-support-ae00.py'])
        support=parent.module(support_path,'cjj_allreader_codec')
        functions={role:getattr(parent.module(Path(source_root)/(role+'.py'),'cjj_allreader_'+role),
                   parent.factory.name(args.profile) if role=='candidate' else ('kernel_unified_attention' if args.profile=='raw3d' else 'kernel_biv150_prefill_stride')) for role in ('baseline','candidate')}
        def byte(t): return t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        def resource(k): return dict(n_regs=getattr(k,'n_regs',None),n_spills=getattr(k,'n_spills',None),
            actual_metadata=parent.freeze(k.metadata._asdict() if hasattr(k.metadata,'_asdict') else vars(k.metadata)),after_actual_launch=True,zero_spill_required=False)
        for case in (0,1):
            p=contract.plan(args.profile,case);directory=args.out/('case-'+str(case));directory.mkdir()
            t,owners=contract.descriptors.allocate(torch,p)
            outguards={r:torch.empty((p['total']+2,16,128),dtype=torch.bfloat16,device='cuda') for r in functions}
            live={r:dict(t,out=g[1:-1]) for r,g in outguards.items()};workspace={};workguards={}
            if args.profile=='raw3d':
                for role in functions:
                    workguards[role]={}
                    for name,tail in [('scratch',(16,p['segments'],128)),('maxima',(16,p['segments'])),('sums',(16,p['segments']))]:
                        owner=torch.empty((p['total']+1,*tail),dtype=torch.float32,device='cuda')
                        workguards[role][name]=owner;live[role][name]=owner[:-1]
            compiled={};calls={};records=[]
            owned.check('compile-case'+str(case))
            for role in functions:
                kw=contract.descriptors.kwargs(p,live[role],role=='candidate');opts=p['requested_options']
                desc=contract.descriptors.descriptors(live[role]);contract.descriptors.check_descriptors(p,desc)
                k=functions[role].warmup(grid=p['grid'],**kw,**opts)
                records.append(parent.preserve(k,directory,role,opts,p['grid'],kw,live[role],desc));compiled[role]=k;calls[role]=(kw,opts)
            if args.profile=='raw3d':
                kw=contract.descriptors.reducer_kwargs(p,live['baseline']);k=reduce_segments.warmup(grid=(p['total'],16),**kw)
                records.append(parent.preserve(k,directory,'original_reducer',{},(p['total'],16),kw,live['baseline'],contract.descriptors.descriptors(live['baseline'])))
                compiled['original_reducer']=k
            # Preserve every actual role before comparing; never replace old IR.
            parent.save(directory/'actual-records.json',records)
            audits={}
            for record in records:
                role=record['label'];expected=next(r for r in reviewed['records'] if r['label']==role)
                observed={e:(directory/record['files'][e]['path']).read_bytes() for e in ('source','ttir','ttgir','llir','asm','cubin')}
                before={e:files['out/'+expected['files'][e]['path']] for e in observed}
                parent.save(directory/(role+'-binding-differences.json'),contract.frozen.differences(record,expected))
                audits[role]=contract.bind_actual(args.profile,record,expected,observed,before)
            parent.save(directory/'actual-binding.json',audits)
            def fingerprint():
                return {name:parent.factory.sha(byte(v)) for name,v in dict(q=t['q'],packed=owners[0],encoded=owners[1],table=t['table'],lengths=t['lengths'],cu_q=t['cu_q'],kd=t['k_descale'],vd=t['v_descale']).items()}
            for pattern in (0,1):
                cpu=fixture.cpu_inputs(p['qlens'],p['klens'],20261025,pattern)
                raw,codes,scales,codec=contract.frozen.numeric.encoded_bytes(cpu['packed'][:,0],cpu['packed'][:,1],torch,np)
                ref=contract.references(cpu,codes,scales,p['qlens'],p['klens'],pattern==0,torch,np)
                artifact=dict(original_CPU=cpu,expected_codes=codes,expected_scales=scales,expected_raw=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()),
                              reference=ref,outputs={},workspace={},guard_rows={})
                artifact_path=directory/('pattern-'+str(pattern)+'.pt.gz')
                def store():
                    with gzip.open(artifact_path,'wb',compresslevel=3) as stream: torch.save(artifact,stream)
                store();owners[0].copy_(cpu['packed']);owners[1].fill_(0xa5)
                for name,key in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]: t[name].copy_(cpu[key])
                before=fingerprint();owned.check('encode-'+str(case)+'-'+str(pattern))
                support.encode_initial[(p['pages']*32,)](t['baseline_k'],t['baseline_v'],t['k'],t['kscale'],t['v'],p['pages']);result['encoder_launches']+=1;torch.cuda.synchronize()
                actual=byte(owners[1]);artifact['actual_raw']=torch.from_numpy(np.frombuffer(actual,dtype=np.uint8).copy());store()
                if actual!=raw: raise ValueError('Independent full codec bytes differ')
                frozen_inputs=fingerprint()
                if any(frozen_inputs[k]!=v for k,v in before.items() if k!='encoded'): raise ValueError('Encoder mutated read-only input')
                canaries=readonly=workspace_ok=True;loaded={}
                for role,fn in functions.items():
                    outguards[role].fill_(71);live[role]['out'].fill_(float('nan'))
                    if args.profile=='raw3d':
                        for name,owner in workguards[role].items(): owner.fill_(43);live[role][name].fill_(float('nan'))
                    owned.check(role+'-'+str(case)+'-'+str(pattern));kw,opts=calls[role]
                    k=fn[p['grid']](**kw,**opts);result['reader_launches']+=1;torch.cuda.synchronize();loaded[role]=resource(k)
                    if str(k.hash)!=str(compiled[role].hash): raise ValueError('Reader specialization changed at launch')
                    if args.profile=='raw3d':
                        artifact['workspace'][role]={name:live[role][name].cpu() for name in workguards[role]};store()
                        active=torch.arange(p['segments'],device='cuda')[None,:] < torch.tensor([(n+15)//16 for n in p['klens']],device='cuda')[:,None]
                        mask=active[:,None,:].expand(-1,16,-1)
                        for name in workguards[role]:
                            value=live[role][name];selected=value[mask]
                            workspace_ok=workspace_ok and bool(torch.isfinite(selected).all()) and bool(torch.isnan(value[~mask]).all())
                            if name=='sums': workspace_ok=workspace_ok and bool((selected>0).all())
                        owned.check('reduce-'+role+'-'+str(case)+'-'+str(pattern))
                        k=reduce_segments[(p['total'],16)](**contract.descriptors.reducer_kwargs(p,live[role]));result['reducer_launches']+=1;torch.cuda.synchronize();loaded[role+'_reducer']=resource(k)
                        if str(k.hash)!=str(compiled['original_reducer'].hash): raise ValueError('Reducer specialization changed at launch')
                        workspace_ok=workspace_ok and all(bool((v[-1]==43).all()) for v in workguards[role].values())
                    artifact['outputs'][role]=live[role]['out'].cpu();store()
                    artifact['guard_rows'][role]=dict(output=torch.stack((outguards[role][0].cpu(),outguards[role][-1].cpu())),
                        workspace_suffix={name:v[-1].cpu() for name,v in workguards.get(role,{}).items()});store()
                    canaries=canaries and bool((outguards[role][0]==71).all() and (outguards[role][-1]==71).all())
                    readonly=readonly and fingerprint()==frozen_inputs
                arrays=artifact['outputs'];rows=ref['rows']
                checks=contract.frozen.numeric.gates(arrays['baseline'][rows].double().reshape(-1).tolist(),arrays['candidate'][rows].double().reshape(-1).tolist(),
                    ref['original'].reshape(-1).tolist(),ref['quantized'].reshape(-1).tolist())
                hard=dict(codec=True,finite=all(bool(torch.isfinite(a).all()) for a in [*arrays.values(),ref['original'],ref['quantized']]),
                    baseline_reference=checks['baseline_hard_passed'],output_canaries=canaries,readonly_inputs=readonly,actual_IR=True,workspace_guards=workspace_ok)
                row=dict(case=case,pattern=pattern,geometry=p,hard_checks=hard,numeric_checks=checks,codec_CPU=codec,actual_postlaunch_resources=loaded,
                    readonly_input_digests_before=frozen_inputs,readonly_input_digests_after=fingerprint(),
                    artifact=dict(path=str(artifact_path.relative_to(args.out)),bytes=artifact_path.stat().st_size,sha256=parent.factory.sha(artifact_path.read_bytes())))
                result['records'].append(row);save();row['policy']=contract.policy(hard,checks,args.numeric_diagnostic);save()
        owned.check('complete');owned.a.state='completed';result['completed']=True;result['actual_binding_verified']=True
        result['eager_numeric_passed']=all(r['numeric_checks']['candidate_numeric_passed'] for r in result['records']);return result
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)));owned.a.state='failed';raise
    finally: save()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',choices=('multi2','raw3d'),default='multi2')
    parser.add_argument('--run-gpu',action='store_true');parser.add_argument('--numeric-diagnostic',action='store_true')
    parser.add_argument('--commit');parser.add_argument('--out',type=Path);parser.add_argument('--source-root',type=Path)
    args=parser.parse_args();files,summary,root=contract.original(args.profile)
    if args.run_gpu: result=run(args)
    else:
        if any((args.numeric_diagnostic,args.commit,args.out,args.source_root)): parser.error('GPU-only flags require explicit --run-gpu')
        result=dict(cpu_only=True,profile=args.profile,plans=[contract.plan(args.profile,i) for i in (0,1)],original_source_root=root,
            original_kernel_hashes={r['label']:r['hash'] for r in summary['records']},actual_binding_verified=False,
            actual_I8_operand_hoist_verified=False,precision_gate=False,Graph_verified=False,side_stream_verified=False,
            timing_attempted=False,writer_ms=None,fullcost_ms=None,performance_verified=False,service_verified=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__': main()
