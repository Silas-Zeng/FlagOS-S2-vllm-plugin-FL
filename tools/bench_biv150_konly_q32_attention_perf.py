#!/usr/bin/env python3
"""CPU-default DAC1-selected BF16 Q-hoist vs I8 attention-only diagnostic."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True

import bench_biv150_konly_q32_numeric as frozen
import bench_biv150_konly_q32_graph_side as graph_source
import biv150_konly_accepted_main as main_control
from biv150_konly_numeric_admission import Admission

HARD={'actual_main_selected','actual_I8_IR','codec','finite','main_raw_BF16_bitwise',
      'I8_hoisted_unhoisted_bitwise','output_canaries','readonly_inputs','baseline_reference'}


def plan(length):
    if type(length) is not int or length not in (4096,6144,16384): raise ValueError('One explicit KV length required')
    p=frozen.numeric.plan()
    # The frozen eager plan's stage-status labels are not geometry. This new
    # entry records actual capture/timing status in its own result instead.
    for name in ('Graph_verified','side_stream_verified','context16k_tested','timing_enabled'): p.pop(name)
    p.update(klens=[length],context=length-2048,pages=2*((length+15)//16)+5)
    return p


def timing_gate(hard,checks,diagnostic):
    if set(hard)!=HARD or any(v is not True for v in hard.values()): raise ValueError('Implementation hard failure forbids performance')
    if not checks['baseline_hard_passed'] or not checks['quantized_operand']['finite'] or not checks['quantized_operand']['pointwise_passed']:
        raise ValueError('This actual length lacks passing baseline/quantized-implementation numerics')
    if not checks['candidate_numeric_passed'] and not diagnostic: raise ValueError('Original BF16 proxy failed; explicit numeric diagnostic required')
    return dict(implementation_gate=True,original_numeric_passed=checks['candidate_numeric_passed'],precision_gate=False,
                attention_only=True,writer_ms=None,fullcost_ms=None,service_performance=False)


class Owned:
    """Fresh original empty→native PID transition, then exact owned phase checks."""
    def __init__(self,preflight,binding): self.a=Admission(preflight,binding);self.token=None;self.checks=[]
    def begin(self): self.token=self.a.begin()
    def check(self,name):
        a=self.a
        if a.state!='active' or self.token!=a.token or a.identity()!=a.owner or a.binding()!=a.roots or not a.closed() or a.query()!=[a.native]:
            raise ValueError('Performance phase lost its owned native PID/root/token')
        self.checks.append(dict(phase=name,native_host_pid=a.native,token_sha256=frozen.actual.old.sha(self.token)))
    def receipt(self): return dict(original_transition=self.a.receipt(),actual_phase_checks=self.checks)


def run(args):
    if not args.commit or args.out is None or args.source_root is None or args.main_root is None: raise ValueError('Exact commit, fresh output, two source roots required')
    if args.out.exists() or args.out.resolve().is_relative_to(frozen.parent.ROOT): raise ValueError('Fresh output outside source checkout required')
    if not args.prepare_only and not args.attention_only_diagnostic: raise ValueError('Explicit attention-only diagnostic timing opt-in required')
    if 'torch' in sys.modules or 'triton' in sys.modules: raise ValueError('Owned admission must precede framework import')
    p=plan(args.kv_length);pre=frozen.preflight();main_control.source_contract()
    if args.source_root.resolve()!=Path(pre[0]['source_root']): raise ValueError('Original candidate source-root required')
    original=frozen.actual.old.original()[0];new=frozen.actual.original()[0]
    def binding():
        if frozen.parent.git('rev-parse','HEAD')!=args.commit or frozen.parent.git('status','--porcelain'): raise ValueError('Exact clean frontend freeze required')
        if frozen.parent.git('diff',frozen.parent.factory.MAIN,'--','vllm_fl','csrc'): raise ValueError('Candidate production changed')
        main=main_control.root_contract(args.main_root)
        for role,root in pre[0]['role_source_roots'].items():
            name='baseline.py' if role=='baseline' else 'candidate.py'
            if frozen.actual.old.sha((Path(root)/name).read_bytes())!=frozen.actual.old.sha(original['out/sources/'+name]): raise ValueError('Frozen role source changed')
        if frozen.actual.old.sha(Path(pre[0]['override']).read_bytes())!=pre[0]['override_sha256']: raise ValueError('Candidate override changed')
        return dict(frontend_root=str(frozen.parent.ROOT),frontend_commit=args.commit,accepted_main=main,
                    candidate_role_roots=pre[0]['role_source_roots'],candidate_override=pre[0]['override'],candidate_source_commit='2d79d9546a8a240447fee05479c5f6caa00aa296')
    binding();args.out.mkdir(parents=True,exist_ok=False)
    result=dict(completed=False,errors=[],actual_geometry=p,accepted_main_commit=main_control.MAIN,records=[],
        precision_gate=False,performance_verified=False,service_verified=False,capacity_reclaimed=False,
        timing_scope='attention-only warmed fixed-fixture Graph replay; no SDK host-dispatch timing',
        writer_ms=None,fullcost_ms=None,model105_verified=False,timing_attempted=False,
        encoder_is_all_page_cold_setup=True,encoder_is_not_a_measured_incremental_writer=True)
    owned=Owned(lambda:(binding(),frozen.parent.installed_sources(),frozen.parent.idle_before_cuda()),binding)
    def save(): result['admission']=owned.receipt();frozen.parent.save(args.out/'attention-perf-summary.json',result)
    save()
    try:
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache');owned.begin();save()
        sys.path.insert(0,str(args.main_root.resolve()))
        import numpy as np
        import torch
        from vllm.triton_utils import triton
        import biv150_prefill_probe_fixtures as fixture
        fixture.torch=torch;torch.set_num_threads(4);torch.set_flush_denormal(False)
        target=triton.runtime.driver.active.get_current_target()
        if (target.backend,target.arch,target.warp_size)!=('corex',71,64): raise ValueError('Unknown actual compiler target')
        props=torch.cuda.get_device_properties(0)
        if 'BI-V150' not in props.name or props.multi_processor_count!=16: raise ValueError('Unvalidated actual GPU')
        t,owners=frozen.descriptors.allocate(torch,p)
        guards={role:torch.empty((2050,16,128),dtype=torch.bfloat16,device='cuda') for role in ('raw_bf16','unhoisted','hoisted','main')}
        outputs={role:storage[1:-1] for role,storage in guards.items()}
        main=main_control.Materializer(args.main_root,torch,fixture,args.out)
        functions={};calls={};compiled={};exports={}
        rolemap={'raw_bf16':'baseline','unhoisted':'candidate','hoisted':'candidate'}
        previous=frozen.actual.old.original()[1]
        expected={role:next(r for r in (pre[3] if role=='hoisted' else previous)['records'] if r['label']==label) for role,label in rolemap.items()}
        owned.check('compile-frozen-I8-controls')
        for role,label in rolemap.items():
            origin_role='baseline' if role=='raw_bf16' else role
            path=Path(pre[0]['role_source_roots'][origin_role])/('baseline.py' if role=='raw_bf16' else 'candidate.py')
            module=frozen.parent.module(path,'cjj_konly_perf_'+role)
            fn=getattr(module,'kernel_biv150_prefill_stride' if role=='raw_bf16' else frozen.parent.factory.name('q32'))
            current=dict(t,out=outputs[role]);desc=frozen.descriptors.descriptors(current);frozen.descriptors.check_descriptors(p,desc)
            kw=frozen.descriptors.kwargs(p,current,role!='raw_bf16');options=dict(p['requested_options'])
            if role=='hoisted': options['ir_override']=pre[0]['override']
            kernel=fn.warmup(grid=p['grid'],**kw,**options)
            record=frozen.parent.preserve(kernel,args.out,role,options,p['grid'],kw,current,desc)
            reviewed={(ext):(new if role=='hoisted' else original)['out/ir/'+label+'/actual.'+ext] for ext in ('source','ttir','ttgir','llir','asm','cubin')}
            observed={ext:(args.out/'ir'/role/('actual.'+ext)).read_bytes() for ext in reviewed}
            audit=frozen.verify_actual_compilation(record,expected[role],origin_role,observed,reviewed)
            frozen.parent.save(args.out/'ir'/role/'actual-compilation-binding.json',audit)
            functions[role]=fn;calls[role]=(kw,options);compiled[role]=kernel;exports[role]=record
        support_path=args.out/'codec-support-ae00.py';support_path.write_text(frozen.parent.factory.inputs()['codec-support-ae00.py'])
        support=frozen.parent.module(support_path,'cjj_konly_perf_codec_ae00')
        namespace=frozen.numeric.__dict__.copy();exec(compile(graph_source.transforms()[1],'<CPU per-length attention reference>','exec'),namespace)
        def byte(tensor): return tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        def fingerprint():
            return {name:frozen.actual.old.sha(byte(v)) for name,v in dict(q=t['q'],packed=owners[0],encoded=owners[1],table=t['table'],lengths=t['lengths'],cu_q=t['cu_q'],kd=t['k_descale'],vd=t['v_descale']).items()}
        def post(kernel):
            return dict(n_regs=getattr(kernel,'n_regs',None),n_spills=getattr(kernel,'n_spills',None),
                        metadata=frozen.parent.freeze(kernel.metadata._asdict() if hasattr(kernel.metadata,'_asdict') else vars(kernel.metadata)),
                        after_actual_launch=True,zero_spill_required=False)
        def export_main():
            kw=main.kwargs;registry={k:v for k,v in kw.items() if hasattr(v,'data_ptr')};desc=frozen.descriptors.descriptors(registry)
            if not (args.out/'ir/accepted_main').exists():
                result['actual_main_record']=frozen.parent.preserve(main.compiled,args.out,'accepted_main',dict(num_warps=16,num_stages=1),main.grid,
                    {k:v for k,v in kw.items() if k not in ('num_warps','num_stages')},registry,desc)
        for pattern in (0,1):
            cpu=fixture.cpu_inputs(p['qlens'],p['klens'],20261025,pattern)
            raw,codes,scales,codec=frozen.numeric.encoded_bytes(cpu['packed'][:,0],cpu['packed'][:,1],torch,np)
            ref=namespace['references'](cpu,codes,scales,torch,np)
            artifact=dict(original_CPU=cpu,expected_codes=codes,expected_scales=scales,expected_raw=torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).copy()),reference=ref,outputs={})
            path=args.out/('pattern-'+str(pattern)+'.pt.gz')
            def artifact_save():
                with gzip.open(path,'wb',compresslevel=3) as stream: torch.save(artifact,stream)
            artifact_save()
            owners[0].copy_(cpu['packed']);owners[1].fill_(0xa5)
            for name,key in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]: t[name].copy_(cpu[key])
            before_encode=fingerprint();owned.check('encode'+str(pattern))
            support.encode_initial[(p['pages']*32,)](t['baseline_k'],t['baseline_v'],t['k'],t['kscale'],t['v'],p['pages']);torch.cuda.synchronize()
            actual_raw=byte(owners[1]);artifact['actual_raw']=torch.from_numpy(np.frombuffer(actual_raw,dtype=np.uint8).copy());artifact_save()
            assert actual_raw==raw,'Independent full codec bytes differ'
            frozen_inputs=fingerprint();assert all(frozen_inputs[k]==v for k,v in before_encode.items() if k!='encoded'),'Encoder changed read-only input'
            canary=True;readonly=True
            for role in ('raw_bf16','unhoisted','hoisted','main'):
                guards[role].fill_(71);outputs[role].fill_(float('nan'));owned.check(role+str(pattern))
                if role=='main':
                    try: materialized=main.call(t,outputs[role],args.kv_length)
                    finally:
                        if main.compiled is not None: export_main()
                    kernel=main.compiled
                    frozen.parent.save(args.out/'main-materialization.json',materialized)
                else:
                    kw,options=calls[role];kernel=functions[role][p['grid']](**kw,**options)
                    if str(kernel.hash)!=expected[role]['hash']: raise ValueError('Frozen I8/control specialization changed at launch')
                torch.cuda.synchronize();artifact['outputs'][role]=outputs[role].cpu();artifact_save()
                canary=canary and bool((guards[role][0]==71).all() and (guards[role][-1]==71).all())
                readonly=readonly and fingerprint()==frozen_inputs
                result.setdefault('actual_postlaunch_resources',[]).append(dict(pattern=pattern,role=role,**post(kernel)))
                if not canary or not readonly: raise ValueError('Actual numerical buffer/read-only guard failed')
            array=artifact['outputs'];rows=ref['rows']
            checks=frozen.numeric.gates(array['main'][rows].double().reshape(-1).tolist(),array['hoisted'][rows].double().reshape(-1).tolist(),
                ref['original'].reshape(-1).tolist(),ref['quantized'].reshape(-1).tolist())
            hard=dict(actual_main_selected=bool(materialized['actual_BF16_Q_hoist_verified']),actual_I8_IR=True,codec=True,
                finite=all(bool(torch.isfinite(v).all()) for v in [*array.values(),ref['original'],ref['quantized']]),
                main_raw_BF16_bitwise=byte(array['main'])==byte(array['raw_bf16']),I8_hoisted_unhoisted_bitwise=byte(array['hoisted'])==byte(array['unhoisted']),
                output_canaries=canary,readonly_inputs=readonly,baseline_reference=checks['baseline_hard_passed'])
            row=dict(pattern=pattern,hard_checks=hard,numeric_checks=checks,codec_CPU=codec,
                artifact=dict(path=path.name,bytes=path.stat().st_size,sha256=frozen.actual.old.sha(path.read_bytes())))
            result['records'].append(row);save();row['timing_policy']=timing_gate(hard,checks,args.numeric_diagnostic);save()
        # The accepted main source/IR was already exported before numeric gates.
        result['fixture_cache']=dict(pages=p['pages'],table_pitch=8192,baseline_allocated_bytes=owners[0].untyped_storage().nbytes(),
            encoded_allocated_bytes=owners[1].untyped_storage().nbytes(),query_bytes=t['q'].numel()*2,
            baseline_logical_KV_payload_once=args.kv_length*1024,encoded_logical_KV_payload_once=args.kv_length*776,
            logical_payload_is_not_measured_traffic=True,l2_cache_size=getattr(props,'l2_cache_size',None),
            fixture_replayed_without_flush=True,HBM_or_service_extrapolation_allowed=False)
        graphs={}
        for role in ('main','hoisted'):
            owned.check('capture-'+role);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel=main.warmed_call() if role=='main' else functions[role][p['grid']](**calls[role][0],**calls[role][1])
            graphs[role]=graph;outputs[role].fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            if byte(outputs[role])!=byte(artifact['outputs'][role]): raise ValueError('Actual timed Graph differs from fresh per-length eager')
            if not bool((guards[role][0]==71).all() and (guards[role][-1]==71).all()) or fingerprint()!=frozen_inputs:
                raise ValueError('Timed Graph canary/read-only input guard failed')
        result['per_length_implementation_passed']=True;result['original_numeric_passed']=all(r['numeric_checks']['candidate_numeric_passed'] for r in result['records']);save()
        if not args.prepare_only:
            owned.check('one-fixed-warmup')
            for _ in range(4):
                for graph in graphs.values(): graph.replay()
            torch.cuda.synchronize();timings=[];result['timing_attempted']=True;result['attention_only_diagnostic']=timings;save()
            for index,role in enumerate(('main','hoisted','hoisted','main')):
                owned.check('attention-only-'+str(index)+'-'+role);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(64): graphs[role].replay()
                end.record();end.synchronize();timings.append(dict(order=index,role=role,event_ms_per_replay=start.elapsed_time(end)/64,replays=64));save()
            main_ms=sum(x['event_ms_per_replay'] for x in timings if x['role']=='main')/2
            candidate_ms=sum(x['event_ms_per_replay'] for x in timings if x['role']=='hoisted')/2
            result['main_over_I8_attention_only_ratio']=main_ms/candidate_ms
            for role in graphs:
                if byte(outputs[role])!=byte(artifact['outputs'][role]) or not bool((guards[role][0]==71).all() and (guards[role][-1]==71).all()) or fingerprint()!=frozen_inputs:
                    raise ValueError('Post-timing numeric/canary/read-only guard failed')
        owned.check('complete');owned.a.state='completed';result['completed']=True;return result
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)));owned.a.state='failed';raise
    finally: save()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--attention-only-diagnostic',action='store_true')
    parser.add_argument('--numeric-diagnostic',action='store_true');parser.add_argument('--kv-length',type=int,choices=(4096,6144,16384),default=6144)
    parser.add_argument('--commit');parser.add_argument('--out',type=Path);parser.add_argument('--source-root',type=Path);parser.add_argument('--main-root',type=Path)
    args=parser.parse_args();p=plan(args.kv_length);main=main_control.source_contract();graph_source.transforms()
    if args.run_gpu: result=run(args)
    else:
        if any((args.prepare_only,args.attention_only_diagnostic,args.numeric_diagnostic,args.commit,args.out,args.source_root,args.main_root)): parser.error('Device-only flags require explicit --run-gpu')
        result=dict(cpu_only=True,actual_geometry=p,accepted_main=main,actual_materialization=False,per_length_implementation_passed=False,
            attention_timing_attempted=False,writer_ms=None,fullcost_ms=None,precision_gate=False,service_verified=False,
            performance_verified=False,cache_policy='fixed fixture replay; no flush tuning; resident data not service/HBM')
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__': main()
