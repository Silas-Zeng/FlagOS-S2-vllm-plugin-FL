#!/usr/bin/env python3
"""Private CPU-default Graph/side host adapter over frozen2d79; no timer/service hook."""
import argparse
from contextlib import contextmanager
import gzip
import inspect
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode=True
import bench_biv150_konly_q32_numeric as frozen
import biv150_konly_q32_numeric_contract as numeric
import biv150_prefill_probe_fixtures as fixture
from biv150_konly_numeric_admission import Admission

PINS={'bench_biv150_konly_q32_numeric.py':'2ee5d031aca8007211205c6ed0796cd3b25d5f0ca92603c9422dcd24e6ecb4e6',
      'biv150_konly_q32_numeric_contract.py':'9c75e7a81c0d77fba0690d25db355cb4dfd669b9aed963930697b7951e6c46e8',
      'biv150_konly_numeric_admission.py':'e3bbef4248c0ac1729c60ce0a4a836070d28d894dd51819ffa23f09d9c554622',
      'biv150_prefill_probe_fixtures.py':'289afc781f17e80465626eef045cd4bfb3db5ed7c516ce662d8e75c06a698132'}
RESOURCE_GUARD="""            if not resources['loaded_resource_fields_known'] or not resources['zero_spill_observed']:
                raise ValueError(role+': loaded register/spill/resource fields unknown or spilled')
"""


def transforms():
    for name,digest in PINS.items():
        if frozen.actual.old.sha((frozen.parent.ROOT/'tools'/name).read_bytes())!=digest: raise ValueError('Frozen2d79 host input changed: '+name)
    raw=Path(frozen.__file__).read_text();anchor="            record['loaded_resources']=resources;"
    if raw.count(RESOURCE_GUARD)!=1 or raw.count(anchor)!=1: raise ValueError('Unknown resource observation host grammar')
    changed=raw.replace(RESOURCE_GUARD,'',1)
    if changed.replace(anchor,RESOURCE_GUARD+anchor,1)!=raw: raise ValueError('Resource-only host inverse failed')
    reference=inspect.getsource(numeric.references)
    substitutions={';length=6144':";length=int(cpu['lengths'][0])",
                   "cpu['table'][0,:384]":"cpu['table'][0,:(length+15)//16]",
                   '<4096+torch.tensor(rows)':'<(length-len(cpu[\'q\']))+torch.tensor(rows)'}
    altered=reference
    for before,after in substitutions.items():
        if altered.count(before)!=1: raise ValueError('Unknown CPU reference geometry grammar')
        altered=altered.replace(before,after,1)
    restored=altered
    for before,after in reversed(list(substitutions.items())): restored=restored.replace(after,before,1)
    if restored!=reference: raise ValueError('CPU reference geometry inverse failed')
    return changed,altered,dict(frozen_runtime_commit='2d79d9546a8a240447fee05479c5f6caa00aa296',source_pins=PINS,
        private_host_sha256=frozen.actual.old.sha(changed),private_CPU_reference_sha256=frozen.actual.old.sha(altered),
        host_resource_guard_only_removed=True,resource_counts_observation_only=True,exact_host_byte_inverse=True,
        CPU_reference_geometry_only=True,CPU_reference_exact_inverse=True,kernel_codec_numeric_gate_bytes_unchanged=True)


def changed_metadata(table,length):
    if type(length) is not int or length!=6144 or len(table)!=8192 or len(set(table[:384]))!=384:
        raise ValueError('Only the existing2048-query/4096-context capacity fixture is admitted')
    return table[383:384]+table[:383]+table[384:],6143


class Kernel:
    def __init__(self,original,scope,role): self.original,self.scope,self.role=original,scope,role
    def warmup(self,*args,**kwargs):
        result=self.original.warmup(*args,**kwargs);self.scope.warmups[self.role]=result;return result
    def __getitem__(self,grid):
        launch=self.original[grid]
        def call(**kwargs):
            s=self.scope;torch=fixture.torch;out=kwargs['output_ptr'];role=self.role
            if s.pattern==0:
                s.guard(role,'eager0');result=launch(**kwargs);torch.cuda.synchronize()
                s.observe(role,'eager0',result);s.outputs.setdefault(role,{})['eager0']=out.cpu()
                s.guard(role,'capture0');graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): captured=launch(**kwargs)
                s.graphs[role]=(graph,captured);s.counts['capture']+=1;s.save()
                return result
            if s.pattern!=1 or role not in s.graphs: raise ValueError('Graph must be captured before changing input data')
            s.guard(role,'graph1');graph,captured=s.graphs[role];graph.replay();s.counts['graph_replay']+=1
            torch.cuda.synchronize();s.observe(role,'graph1',captured);s.outputs[role]['graph1']=out.cpu();s.save()
            out.fill_(float('nan'));s.guard(role,'eager1');result=launch(**kwargs);torch.cuda.synchronize()
            s.observe(role,'eager1',result);s.outputs[role]['eager1']=out.cpu();s.save()
            if not torch.equal(s.outputs[role]['graph1'].view(torch.uint8),s.outputs[role]['eager1'].view(torch.uint8)):
                raise ValueError(role+': changed-input Graph output differs bitwise from fresh eager')
            stream=torch.cuda.Stream();current=torch.cuda.current_stream();stream.wait_stream(current)
            s.guard(role,'side1')
            with torch.cuda.stream(stream):
                out.fill_(float('nan'));result=launch(**kwargs)
            current.wait_stream(stream);torch.cuda.synchronize();s.counts['side']+=1
            s.observe(role,'side1',result);s.outputs[role]['side1']=out.cpu();s.save()
            if not torch.equal(s.outputs[role]['side1'].view(torch.uint8),s.outputs[role]['eager1'].view(torch.uint8)):
                raise ValueError(role+': side-stream output differs bitwise from eager')
            s.coherent.append(role);s.save();return result
        return call


class Scope:
    def __init__(self,out):
        self.out=out;self.pattern=None;self.admission=None;self.seal=None;self.warmups={};self.graphs={};self.outputs={}
        self.owned=False
        self.coherent=[];self.resources=[];self.checks=[];self.controls={};self.errors=[]
        self.counts=dict(eager=0,capture=0,graph_replay=0,side=0)
    def guard(self,role,mode):
        a=self.admission
        if a is None or a.state!='active' or not a.token or a.identity()!=a.owner or a.binding()!=a.roots or not a.closed() or a.query()!=[a.native]:
            raise ValueError('Graph/side phase lost the original owned native-PID admission')
        token=frozen.actual.old.sha(a.token);identity=json.dumps((a.owner,a.roots,a.native,token),sort_keys=True)
        if self.seal is None: self.seal=identity
        if self.seal!=identity: raise ValueError('Graph/side identity/token changed')
        self.checks.append(dict(role=role,mode=mode,native_host_pid=a.native,token_sha256=token))
    def observe(self,role,mode,kernel):
        if mode.startswith('eager'): self.counts['eager']+=1
        # Read the original returned/captured CompiledKernel object AFTER real
        # attention execution, without adjusting or requiring any resource count.
        values={key:getattr(kernel,key,None) for key in ('n_regs','n_spills')}
        try: raw=kernel.metadata._asdict() if hasattr(kernel.metadata,'_asdict') else vars(kernel.metadata)
        except TypeError: raw=dict(unrecognized_metadata_repr=repr(kernel.metadata))
        metadata=frozen.parent.freeze(raw)
        self.resources.append(dict(role=role,mode=mode,compiled_hash=str(kernel.hash),**values,metadata=metadata,
            observed_after_real_attention=True,same_object_as_warmup=kernel is self.warmups[role],
            resources_observation_only=True,zero_spill_required=False))
    def inputs(self,original,qlens,klens,seed,pattern):
        cpu=original(qlens,klens,seed,pattern);self.pattern=pattern;torch=fixture.torch
        if pattern==1:
            table,length=changed_metadata(cpu['table'][0].tolist(),int(cpu['lengths'][0]))
            cpu['table'][0]=torch.tensor(table,dtype=torch.int32);cpu['lengths'][0]=length;cpu['cu_k'][-1]=length
        def digest(t): return frozen.actual.old.sha(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes())
        self.controls[str(pattern)]={key:digest(cpu[key]) for key in ('q','table','lengths')}
        self.controls[str(pattern)]['runtime_length']=int(cpu['lengths'][0]);self.save();return cpu
    def save(self):
        if not self.owned or not self.out.is_dir(): return
        changed=set(self.controls)=={'0','1'} and all(self.controls['0'][k]!=self.controls['1'][k] for k in ('q','table','lengths'))
        report=dict(completed=False,coherent_roles=self.coherent,counts=self.counts,changed_controls=self.controls,
            native_phase_checks=self.checks,actual_postlaunch_resources=self.resources,errors=self.errors,
            Graph_verified=len(self.coherent)==3 and changed,side_stream_verified=len(self.coherent)==3 and changed,
            precision_gate=False,timing_attempted=False,performance_verified=False,service_verified=False,
            attention_execution_count_inferred_from_capture=False)
        frozen.parent.save(self.out/'graph-side-summary.json',report)
        if self.outputs:
            with gzip.open(self.out/'graph-side-full-outputs.pt.gz','wb',compresslevel=3) as stream:
                fixture.torch.save(dict(outputs=self.outputs,changed_controls=self.controls,postlaunch_resources=self.resources),stream)
    @contextmanager
    def adapt(self,private,reference_source):
        loader=frozen.parent.module;inputs=fixture.cpu_inputs;reference=numeric.references
        namespace=numeric.__dict__.copy();exec(compile(reference_source,'<private CPU dynamic-length reference>','exec'),namespace)
        def module(path,name):
            result=loader(path,name)
            for role in ('baseline','unhoisted','hoisted'):
                if name=='cjj_konly_numeric_'+role:
                    key='kernel_biv150_prefill_stride' if role=='baseline' else frozen.parent.factory.name('q32')
                    setattr(result,key,Kernel(getattr(result,key),self,role))
            return result
        def admission(*args,**kwargs):
            result=Admission(*args,**kwargs);self.admission=result;self.owned=True
            (self.out/'private-host-runner.py.txt').write_text(self.host_text)
            (self.out/'private-CPU-reference.py.txt').write_text(self.reference_text)
            frozen.parent.save(self.out/'private-host-transform.json',self.transform_proof)
            return result
        frozen.parent.module=module;fixture.cpu_inputs=lambda *args:self.inputs(inputs,*args)
        numeric.references=namespace['references'];private.Admission=admission
        try: yield
        finally: frozen.parent.module=loader;fixture.cpu_inputs=inputs;numeric.references=reference


def run(args):
    if args.out is None or args.out.exists(): raise ValueError('Fresh output required; no existing evidence is modified')
    source,reference,proof=transforms();private=types.ModuleType('cjj_private_q32_graph_side_host')
    private.__file__=str(args.out/'private-host-runner.py.txt')
    exec(compile(source,private.__file__,'exec'),private.__dict__)
    scope=Scope(args.out)
    scope.host_text=source;scope.reference_text=reference;scope.transform_proof=proof
    try:
        with scope.adapt(private,reference): result=private.run(args,private.preflight())
        if set(scope.coherent)!={'baseline','unhoisted','hoisted'}: raise ValueError('Missing actual Graph/side coherence checks')
        if set(scope.controls)!={'0','1'} or any(scope.controls['0'][key]==scope.controls['1'][key] for key in ('q','table','lengths')):
            raise ValueError('Q/table/length must all change before first replay')
        result['legacy_host_reader_entries']=result.pop('reader_launches');result['attention_calls_by_mode']=scope.counts
        result['Graph_verified']=True;result['side_stream_verified']=True;result['Graph_executed']=True;result['side_stream_executed']=True
        result['actual_postlaunch_resources']=scope.resources;result['private_host_transform']=proof
        result['full_validation_passed']=False;result['precision_gate']=False
        frozen.parent.save(args.out/'numeric-summary.json',result)
        scope.save();report=json.loads((args.out/'graph-side-summary.json').read_text());report['completed']=True
        frozen.parent.save(args.out/'graph-side-summary.json',report);return result
    except Exception as error:
        scope.errors.append(repr(error));scope.save()
        if scope.owned and args.out.is_dir() and (args.out/'numeric-summary.json').exists():
            result=json.loads((args.out/'numeric-summary.json').read_text());result['completed']=False
            detail=dict(type=type(error).__name__,message=str(error))
            if detail not in result['errors']: result['errors'].append(detail)
            frozen.parent.save(args.out/'numeric-summary.json',result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-gpu',action='store_true')
    parser.add_argument('--numeric-diagnostic',action='store_true');parser.add_argument('--commit')
    parser.add_argument('--source-root',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    _,_,proof=transforms();pre=frozen.preflight()[0]
    if args.run_gpu:
        if not args.commit or args.source_root is None or args.out is None: parser.error('Exact freeze/source-root/fresh-out required')
        result=run(args)
    else:
        if any((args.commit,args.source_root,args.out,args.numeric_diagnostic)): parser.error('Device flags require explicit --run-gpu')
        result=dict(cpu_only=True,host_transform=proof,capacity_plan=pre['plan'],runtime_lengths=[6144,6143],
                    Q_table_length_change_required=True,Graph_verified=False,side_stream_verified=False,
                    actual_numeric_executed=False,precision_gate=False,timing_attempted=False,
                    performance_verified=False,service_verified=False,capacity_reclaimed=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__': main()
