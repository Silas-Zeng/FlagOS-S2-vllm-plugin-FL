#!/usr/bin/env python3
"""Default CPU; private changed-input BM128 Graph/side over unchanged f04 gates."""
import argparse
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import bench_biv150_konly_2d128_numeric as frozen
import biv150_konly_2d128_graph_contract as contract
import biv150_prefill_probe_fixtures as fixture

PINS={'bench_biv150_konly_2d128_numeric.py':'b6371f8da0769c4a4c14b182a86b2f8578d05b0776037a37577ff1d8ec5ced62',
      'biv150_konly_2d128_numeric_contract.py':'0acecf155cfc0f192c17be44e9cbfba4de35a5f31f64363cdfa5e72c57ad94c7',
      'biv150_konly_2d128_numeric_admission.py':'22f534bf0db43aa873244f1287c954a4eb56385ce99e7526dda0dadd9c2fbdcb',
      'biv150_prefill_probe_fixtures.py':'289afc781f17e80465626eef045cd4bfb3db5ed7c516ce662d8e75c06a698132'}
ROLES=('baseline','candidate')


def pins():
    for name,digest in PINS.items():
        contract.need(frozen.actual.sha((frozen.parent.ROOT/'tools'/name).read_bytes())==digest,'unchanged f04 host input '+name)
    return dict(frozen_eager_commit=contract.FROZEN,source_pins=PINS,kernel_codec_gates_resource_guard_unchanged=True,
        private_scoped_kernel_wrapper=True,CPU_reference_runtime_length_only=True,no_host_source_rewrite=True)


class Kernel:
    def __init__(self,kernel,scope,role):self.kernel,self.scope,self.role=kernel,scope,role
    def warmup(self,*args,**kwargs):
        k=self.kernel.warmup(*args,**kwargs);self.scope.warmups[self.role]=k;return k
    def __getitem__(self,grid):
        launch=self.kernel[grid]
        def call(**kwargs):
            s=self.scope;torch=fixture.torch;role=self.role;out=kwargs['output_ptr']
            baseline_inputs=s.inputs_fingerprint(kwargs);s.pointer_contract(role,kwargs)
            controls=contract.control_fingerprints(baseline_inputs,s.controls[str(s.pattern)])
            s.device_controls.setdefault(role,{})[str(s.pattern)]=controls
            if s.pattern==0:
                s.guard(role,'eager0');k=launch(**kwargs);torch.cuda.synchronize()
                s.observe(role,'eager0',k,kwargs,baseline_inputs)
                s.guard(role,'capture0');graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=launch(**kwargs)
                s.graphs[role]=(graph,captured);s.counts['capture_calls']+=1;s.save();return k
            contract.need(s.pattern==1 and role in s.graphs,'capture must precede changed data')
            contract.need(set(s.controls)=={'0','1'} and all(s.controls['0'][n]!=s.controls['1'][n] for n in ('q','table','lengths')),'Q/table/runtime lengths must actually change')
            contract.need(all(s.device_controls[role]['0'][n]!=controls[n] for n in ('q','table','lengths')),'actual device Q/table/runtime lengths unchanged')
            s.guard(role,'graph1');graph,captured=s.graphs[role];graph.replay();s.counts['replays']+=1
            torch.cuda.synchronize();s.observe(role,'graph1',captured,kwargs,baseline_inputs)
            out.fill_(float('nan'));s.guard(role,'eager1');k=launch(**kwargs);torch.cuda.synchronize()
            s.observe(role,'eager1',k,kwargs,baseline_inputs)
            contract.need(s.bytes(s.outputs[role]['graph1'])==s.bytes(s.outputs[role]['eager1']),role+' full Graph/eager bitwise')
            stream=torch.cuda.Stream();current=torch.cuda.current_stream();stream.wait_stream(current)
            s.guard(role,'side1')
            with torch.cuda.stream(stream):
                out.fill_(float('nan'));k=launch(**kwargs)
                for value in kwargs.values():
                    if isinstance(value,torch.Tensor):value.record_stream(stream)
            current.wait_stream(stream);torch.cuda.synchronize();s.counts['side_calls']+=1
            s.observe(role,'side1',k,kwargs,baseline_inputs)
            contract.need(s.bytes(s.outputs[role]['side1'])==s.bytes(s.outputs[role]['eager1']),role+' full side/eager bitwise')
            s.coherent.append(role);s.save();return k
        return call


class Scope:
    def __init__(self,out):
        self.out=out;self.pattern=None;self.admission=None;self.seal=None;self.owned=False
        self.warmups={};self.graphs={};self.outputs={};self.canaries={};self.full_outputs={};self.controls={};self.coherent=[]
        self.checks=[];self.resources=[];self.errors=[];self.mode_order={r:[] for r in ROLES};self.device_controls={}
        self.counts=dict(eager_calls=0,capture_calls=0,replays=0,side_calls=0)
    def bytes(self,tensor):
        return tensor.detach().cpu().contiguous().view(-1).view(fixture.torch.uint8).numpy().tobytes()
    def inputs_fingerprint(self,kwargs):
        torch=fixture.torch;seen={};result={}
        for name,value in kwargs.items():
            if not isinstance(value,torch.Tensor) or name in ('output_ptr','segm_output_ptr','segm_max_ptr','segm_expsum_ptr'):continue
            storage=value.untyped_storage();base=int(storage.data_ptr())
            if base in seen:continue
            # First make a metadata-only full-storage typed vector. This also
            # admits the SDK's genuine 0D FP32 descales; scalar.view(uint8)
            # itself is invalid, and reshape of a pitched cache could copy it.
            nbytes=int(storage.nbytes());width=value.element_size()
            contract.need(nbytes%width==0,'whole typed storage byte extent')
            flat=torch.as_strided(value,(nbytes//width,),(1,),0)
            raw=flat.view(torch.uint8)
            result[name]=frozen.actual.sha(self.bytes(raw));seen[base]=name
        return result
    def pointer_contract(self,role,kwargs):
        torch=fixture.torch;k=kwargs['key_cache_ptr'];v=kwargs['value_cache_ptr'];out=kwargs['output_ptr']
        contract.need(kwargs['KV_QUANT_MODE']==0 and kwargs['BLOCK_M']==128 and kwargs['BLOCK_Q']==16
                      and kwargs['TILE_SIZE']==64 and kwargs['num_seqs']==1,'exact BM128 private mode0')
        contract.need(k.dtype==(torch.int8 if role=='candidate' else torch.bfloat16) and v.dtype==torch.bfloat16,'role typed K/V')
        contract.need(tuple(k.stride())==((16384,256,128,1) if role=='candidate' else (8192,256,128,1))
                      and k.storage_offset()==0 and tuple(v.stride())==(8192,256,128,1) and v.storage_offset()==4096,'role pointer units/offset')
        base=k.untyped_storage().data_ptr();contract.need(base==v.untyped_storage().data_ptr(),'one physical K/V cache allocation')
        if role=='candidate':
            scale=kwargs['key_code_scale_ptr'];contract.need(scale.dtype==torch.float32 and scale.storage_offset()==0
                and scale.untyped_storage().data_ptr()==base,'full-base FP32 scale')
        contract.need(out.storage_offset()==2048 and out.untyped_storage().nbytes()==2050*16*128*2,'two live output token guards')
    def guard(self,role,mode):
        expected=('eager0','capture0','graph1','eager1','side1');order=self.mode_order[role]
        contract.need(len(order)<len(expected) and expected[len(order)]==mode,'Graph-first exact mode order')
        a=self.admission
        contract.need(a is not None and a.state=='active' and a.token and a.identity()==a.owner
            and a.binding()==a.roots and a.closed() and a.query()==[a.native],'Graph phase ownership/source/native PID')
        identity=json.dumps((a.owner,a.roots,a.native,frozen.actual.sha(a.token)),sort_keys=True)
        if self.seal is None:self.seal=identity
        contract.need(identity==self.seal,'Graph identity/token seal')
        order.append(mode);self.checks.append(dict(role=role,mode=mode,native_host_pid=a.native))
    def observe(self,role,mode,kernel,kwargs,before):
        torch=fixture.torch;out=kwargs['output_ptr'];allout=torch.as_strided(out,(2050,16,128),(2048,128,1),0)
        full,canaries=frozen.guarded_cpu_snapshot(allout,torch);output=out.cpu()
        contract.need(self.bytes(full[1:2049])==self.bytes(output),'full-buffer/output CPU copy agreement')
        contract.need(bool(torch.isfinite(output).all()) and bool((canaries==71).all()),'full output finite/canary')
        after=self.inputs_fingerprint(kwargs);contract.need(before==after,'readonly full-backing bytes')
        resource=frozen.parent.observed_resources(kernel)
        contract.need(resource['loaded_resource_fields_known'] and resource['zero_spill_observed'],'strict postlaunch resource gate')
        contract.need(str(kernel.hash)==str(self.warmups[role].hash),'actual returned/captured kernel hash')
        self.resources.append(dict(role=role,mode=mode,hash=str(kernel.hash),resources=resource,
            same_object_as_warmup=kernel is self.warmups[role],observed_after_real_attention=True))
        self.outputs.setdefault(role,{})[mode]=output;self.canaries.setdefault(role,{})[mode]=canaries
        self.full_outputs.setdefault(role,{})[mode]=full
        if mode.startswith('eager'):self.counts['eager_calls']+=1
        self.checks[-1].update(readonly_before=before,readonly_after=after,full_output_finite=True,output_canary_passed=True)
        self.save()
    def inputs(self,original,qlens,klens,seed,pattern):
        cpu=original(qlens,klens,seed,pattern);self.pattern=pattern
        if pattern==1:
            table,length=contract.changed_metadata(cpu['table'][0].tolist(),int(cpu['lengths'][0]))
            cpu['table'][0]=fixture.torch.tensor(table,dtype=fixture.torch.int32)
            cpu['lengths'][0]=length;cpu['cu_k'][-1]=length
        self.controls[str(pattern)]={n:frozen.actual.sha(self.bytes(cpu[n])) for n in ('q','table','lengths')}
        self.controls[str(pattern)]['KV_length']=int(cpu['lengths'][0]);self.save();return cpu
    def save(self):
        if not self.owned or not self.out.is_dir():return
        report=dict(completed=False,coherent_roles=self.coherent,counts=self.counts,changed_controls=self.controls,
            actual_device_controls=self.device_controls,mode_order=self.mode_order,native_phase_checks=self.checks,postlaunch_resources=self.resources,errors=self.errors,
            Graph_verified=False,side_stream_verified=False,precision_gate=False,performance_measured=False,
            service_verified=False,capture_calls_are_not_claimed_attention_executions=True,resource_guard_unchanged=True)
        frozen.parent.save(self.out/'graph-side-summary.json',report)
        if self.outputs:
            with gzip.open(self.out/'graph-side-full-outputs.pt.gz','wb',compresslevel=3) as gz:
                fixture.torch.save(dict(outputs=self.outputs,canaries=self.canaries,full_guarded_outputs=self.full_outputs,controls=self.controls,checks=self.checks),gz)
    @contextmanager
    def adapt(self):
        loader=frozen.parent.module;inputs=fixture.cpu_inputs;reference=frozen.oracle.references;admit=frozen.Admission
        def module(path,name):
            result=loader(path,name)
            for role in ROLES:
                if name=='cjj_2d128_numeric_'+role:
                    key='kernel_biv150_prefill_stride' if role=='baseline' else frozen.parent.factory.name(frozen.descriptors.PROFILE)
                    setattr(result,key,Kernel(getattr(result,key),self,role))
            return result
        def admission(*args,**kwargs):
            self.admission=admit(*args,**kwargs);self.owned=True;return self.admission
        frozen.parent.module=module;fixture.cpu_inputs=lambda *args:self.inputs(inputs,*args)
        frozen.oracle.references=contract.references;frozen.Admission=admission
        try:yield
        finally:
            frozen.parent.module=loader;fixture.cpu_inputs=inputs;frozen.oracle.references=reference;frozen.Admission=admit


def run(args):
    contract.need(args.out is not None and not args.out.exists(),'fresh output required')
    contract.need(args.eager_summary is not None,'actual f04 eager summary required')
    raw=args.eager_summary.read_bytes();gate=contract.eager_admission(json.loads(raw));scope=Scope(args.out)
    try:
        with scope.adapt():result=frozen.run(args,frozen.preflight())
        contract.need(set(scope.coherent)==set(ROLES),'both role Graph/side coherence')
        contract.need(all(scope.mode_order[r]==['eager0','capture0','graph1','eager1','side1'] for r in ROLES),'complete mode order')
        result['legacy_wrapper_entries']=result.pop('reader_launches');result['Graph_verified']=True;result['side_stream_verified']=True
        result['plan']['legacy_eager_frontend_entries_planned']=result['plan'].pop('attention_launches_planned')
        result['Graph_executed']=True;result['side_stream_executed']=True;result['actual_I8_operand_hoist_verified']=False
        result['attention_calls_by_mode']=scope.counts;result['unchanged_ef4_pins']=pins()
        result['eager_precondition']=dict(path=str(args.eager_summary),sha256=frozen.actual.sha(raw),**gate)
        frozen.parent.save(args.out/'numeric-summary.json',result);scope.save()
        report=json.loads((args.out/'graph-side-summary.json').read_text());report.update(completed=True,Graph_verified=True,side_stream_verified=True)
        frozen.parent.save(args.out/'graph-side-summary.json',report);return result
    except Exception as exc:
        scope.errors.append(repr(exc));scope.save()
        if scope.owned and (args.out/'numeric-summary.json').exists():
            result=json.loads((args.out/'numeric-summary.json').read_text());result['completed']=False
            result['errors'].append(dict(type=type(exc).__name__,message=str(exc)));frozen.parent.save(args.out/'numeric-summary.json',result)
        raise
    finally:
        if scope.owned and args.out.is_dir():(args.out/'required-f04-eager-summary.json').write_bytes(raw)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--numeric-diagnostic',action='store_true');parser.add_argument('--commit');parser.add_argument('--source-root',type=Path)
    parser.add_argument('--eager-summary',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args();proof=pins()
    if args.run_gpu:result=run(args)
    else:
        if any((args.numeric_diagnostic,args.commit,args.source_root,args.eager_summary,args.out)):parser.error('device-only flags require --run-gpu')
        result=dict(CPU_only=True,preflight=frozen.preflight()[0],source_contract=proof,runtime_KV_lengths=[6144,6143],
            requested_operations=dict(eager_calls=4,capture_calls=2,replays=2,side_calls=2),
            Q_table_runtime_length_change_required=True,eager_before_Graph_for_changed_data_forbidden=True,
            strict_f04_resource_and_numeric_gates_preserved=True,actual_Graph_verified=False,actual_side_verified=False,
            eager_record_precondition_verified=False,GPU_executed=False,JIT_executed=False,precision_gate=False,performance_measured=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
