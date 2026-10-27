#!/usr/bin/env python3
"""Default CPU plan; explicit bounded actual-host attribution, never a score."""
import argparse
import ast
import cProfile
from contextlib import nullcontext
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import threading
import time
import traceback
import tarfile

sys.dont_write_bytecode=True
import bench_biv150_prefill_host_binding as bench
from bench_biv150_prefill_host_binding import MAIN,NATIVE,save,run_cpu
from biv150_prefill_host_binding import (CpuContract,LocalSdkBinding,make_private_reverse_manager,
    NATIVE_ENV,TARGET_PINS,native_runtime_identity)
from biv150_prefill_host_component import PACKING
from biv150_resource_records import resource_evidence,compare_resource_records


def _bootstrap(a,prepared,cpu):
    source=a.source.resolve()
    assert Path(__file__).resolve().parents[1]==source,"Execute tools from the exact frozen source"
    assert subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()==a.commit
    assert not subprocess.check_output(["git","-C",str(source),"status","--porcelain","--untracked-files=no"],text=True).strip()
    subprocess.run(["git","-C",str(source),"diff","--exit-code",MAIN,"--","vllm_fl","csrc"],check=True)
    with socket.socket()as sock:assert sock.connect_ex(("127.0.0.1",9031))!=0,"Existing service; nothing stopped"
    assert "No running processes found"in subprocess.check_output(["/usr/local/corex/bin/ixsmi"],text=True),"GPU busy; nothing stopped"
    assert os.environ.get("VLLM_FL_BIV150_PREFILL_Q32","1")=="1"
    assert os.environ.get("VLLM_FL_BIV150_PREFILL_MULTI2","1")=="1"
    assert hashlib.sha256(a.native_library.read_bytes()).hexdigest()==NATIVE
    # This is a TORCH_LIBRARY SO used for cache identity, not a Python _C module.
    os.environ[NATIVE_ENV]=str(a.native_library.resolve())
    a.out.mkdir(parents=False,exist_ok=False)
    os.environ["TRITON_CACHE_DIR"]=str((a.out/"private-triton-cache").resolve())
    sys.dont_write_bytecode=True
    save(a.out/"cpu-preflight.json",cpu)
    save(a.out/"manifest.json",dict(component_only=True,source_commit=a.commit,
        production_parent=MAIN,expected_native_sha256=NATIVE,model_precision_gate=False,
        actual_GPU_result_pending=True))
    sys.path.insert(0,str(source))
    import torch
    import vllm_fl
    import biv150_prefill_probe_fixtures as fixture
    from bench_biv150_prefill_reverse import Tap,descriptors,resources
    from biv150_resource_records import compare_resource_records
    assert Path(vllm_fl.__file__).resolve()==source/"vllm_fl/__init__.py"
    semantics=importlib.import_module("vllm_fl.compile_semantics")
    assert semantics.native_library_path()==a.native_library.resolve()
    native_before=native_runtime_identity(torch,a.native_library)
    assert native_before["identity_file"]["sha256"]==NATIVE
    for module,pin in [("vllm.model_executor.layers.attention.attention",TARGET_PINS["model_executor/layers/attention/attention.py"]),
                       ("vllm.forward_context",TARGET_PINS["forward_context.py"])]:
        loaded=importlib.import_module(module)
        assert hashlib.sha256(Path(loaded.__file__).read_bytes()).hexdigest()==pin
    fixture.torch=torch;torch.set_num_threads(4)
    props=torch.cuda.get_device_properties(0)
    assert "BI-V150"in props.name and props.multi_processor_count==16
    function,base,states=fixture.install_baseline()
    q32=importlib.import_module("vllm_fl.patches.biv150_prefill_q32")
    m2=importlib.import_module("vllm_fl.patches.biv150_prefill_multi2")
    q32_state=q32.install();m2_state=m2.install()
    assert q32_state["installed"]and m2_state["installed"]
    parent=q32._MANAGER;chain=base.holder[base.kernel_key]
    assert isinstance(chain,m2._Policy)and isinstance(chain.base,q32._Policy)
    assert chain.base.manager is parent and chain is m2._POLICY
    qpolicy=chain.base;original=parent.wrapper
    intact=lambda:(base.holder[base.kernel_key]is chain and chain.base is qpolicy
        and qpolicy.manager is parent and not qpolicy.disabled and not chain.disabled)
    attention=importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    backend=importlib.import_module("vllm.v1.attention.backends.triton_attn")
    fragment=importlib.import_module("vllm_fl.patches.biv150_q_fragment_service")
    reverse=make_private_reverse_manager(attention,backend,parent,q32,prepared,a.out)
    runtime_intact=lambda:(intact()and parent.active and reverse.active
        and parent.async_mode.get()is None and reverse.async_mode.get()is None
        and not parent.raw.pre_run_hooks and not parent.candidate.pre_run_hooks
        and not reverse.raw.pre_run_hooks and not reverse.candidate.pre_run_hooks
        and parent.device in(None,0)and reverse.device in(None,0))
    binding=LocalSdkBinding(function,prepared,parent,chain,reverse.wrapper,
        torch.cuda.is_current_stream_capturing,runtime_intact,source_root=source)
    return locals()


def prefix_contract(source):
    original=next(n for n in ast.parse((source/'tools/bench_biv150_prefill_host_binding.py').read_text()).body
        if isinstance(n,ast.FunctionDef)and n.name=='gpu')
    cutoff=next(i for i,n in enumerate(original.body)if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='result'for t in n.targets))
    own=next(n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_bootstrap')
    assert isinstance(own.body[-1],ast.Return)
    assert ast.dump(ast.Module(body=original.body[:cutoff],type_ignores=[]),include_attributes=False)==ast.dump(
        ast.Module(body=own.body[:-1],type_ignores=[]),include_attributes=False),'Bootstrap drifted from original guards'
    return dict(original_bootstrap_statements_equal=True,original_benchmark_edited=False,
        original_scope_and_qualification_guards_changed=False,profiling_default=False,
        diagnostic_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def usage():
    def get(who):
        r=resource.getrusage(who)
        return {k:getattr(r,k)for k in('ru_utime','ru_stime','ru_minflt','ru_majflt','ru_nvcsw','ru_nivcsw','ru_maxrss')}
    return dict(process=get(resource.RUSAGE_SELF),thread=get(resource.RUSAGE_THREAD)if hasattr(resource,'RUSAGE_THREAD')else None)


def classify(code):
    if not hasattr(code,'co_filename'):
        label=str(code).lower()
        return 'native-sync'if'synchroniz'in label else'native-launch'if'launch'in label else'native-other'
    path,name=code.co_filename,code.co_name
    if path.endswith('biv150_prefill_parent_certificate.py'):return'certificate'
    if name=='_eligible'and('q_fragment'in path or'descriptor-guard'in path):return'descriptor'
    if path.endswith(('biv150_prefill_q32.py','biv150_prefill_multi2.py')):return'opaque-parent-policy'
    if path.endswith('biv150_prefill_host_component.py'):return'route-and-one-shot'
    if path.endswith('biv150_prefill_host_binding.py'):
        return'boundary-and-audit-plumbing'if'BoundaryLaunch'in code.co_qualname else'host-binding-scope'
    if path.endswith('biv150_q_fragment_service.py'):return'opaque-Manager-dispatch'
    if path.endswith('inspect.py')and name in('bind','bind_partial','_bind'):return'SDK-signature-binding'
    if 'prefill_normalized'in path or'triton_unified_attention'in path:return'SDK-and-normalization'
    if 'triton'in path:return'Triton-binder-and-launcher'
    return'other-Python'


class Observer:
    """Observes original frames; never wraps policy/launch functions or disables GC."""
    def __init__(self,audit_code=None,audit_line=None):
        self.audit_code,self.audit_line=audit_code,audit_line
        self.profiler=cProfile.Profile();self.gc_events=[];self.audit_spans=[]
        self.gc_started={};self.audit_started={};self.descriptor_calls={}
        self._trace_callback=self.trace;self._gc_callback=self.gc_callback

    def gc_callback(self,phase,info):
        now=time.perf_counter_ns();generation=info.get('generation');thread=threading.get_ident()
        key=(thread,generation)
        if phase=='start':self.gc_started[key]=now
        start=self.gc_started.pop(key,None)if phase=='stop'else None
        self.gc_events.append(dict(phase=phase,generation=generation,thread=thread,at_ns=now,
            pause_ns=now-start if start is not None else None,
            collected=info.get('collected'),uncollectable=info.get('uncollectable')))

    def trace(self,frame,event,arg):
        code=frame.f_code
        if event=='call'and code.co_name=='_eligible'and('q_fragment'in code.co_filename or'descriptor-guard'in code.co_filename):
            block=frame.f_locals.get('block_m');key=str(block)if type(block)is int else'unknown'
            self.descriptor_calls[key]=self.descriptor_calls.get(key,0)+1
        if code is not self.audit_code:return None
        if event=='line'and frame.f_lineno==self.audit_line:self.audit_started[id(frame)]=time.perf_counter_ns()
        if event=='return':
            start=self.audit_started.pop(id(frame),None)
            if start is not None:self.audit_spans.append(time.perf_counter_ns()-start)
        return self._trace_callback

    def start(self):
        self.previous_trace=sys.gettrace();self.previous_profile=sys.getprofile()
        if self.previous_trace is not None or self.previous_profile is not None:
            raise RuntimeError('Existing debugger/profiler; diagnostic did not replace it')
        self.gc_before=dict(enabled=gc.isenabled(),threshold=gc.get_threshold(),count=gc.get_count())
        gc.callbacks.append(self._gc_callback)
        try:sys.settrace(self._trace_callback);self.profiler.enable()
        except BaseException:
            sys.settrace(self.previous_trace);gc.callbacks.remove(self._gc_callback);raise

    def stop(self):
        self.profiler.disable();sys.settrace(self.previous_trace)
        gc.callbacks.remove(self._gc_callback)
        self._trace_callback=self._gc_callback=None  # avoid retaining observer cycles between arms
        self.gc_after=dict(enabled=gc.isenabled(),threshold=gc.get_threshold(),count=gc.get_count())
        assert self.gc_before['enabled']==self.gc_after['enabled']and self.gc_before['threshold']==self.gc_after['threshold']

    def summary(self):
        rows=[];buckets={}
        for entry in self.profiler.getstats():
            code=entry.code;bucket=classify(code)
            row=dict(bucket=bucket,calls=entry.callcount,recursive=entry.reccallcount,
                self_seconds=entry.inlinetime,inclusive_seconds=entry.totaltime,
                function=code.co_qualname if hasattr(code,'co_qualname')else str(code),
                filename=code.co_filename if hasattr(code,'co_filename')else None,
                first_line=code.co_firstlineno if hasattr(code,'co_firstlineno')else None)
            rows.append(row);buckets[bucket]=buckets.get(bucket,0)+entry.inlinetime
        return dict(exclusive_cProfile_bucket_seconds=buckets,functions=sorted(rows,key=lambda r:r['self_seconds'],reverse=True),
            gc_before=self.gc_before,gc_after=self.gc_after,gc_events=self.gc_events,
            boundary_postlaunch_audit_span_ns=self.audit_spans,live_descriptor_calls_by_block_m=self.descriptor_calls,
            audit_span_overlaps_boundary_self_time=True,profile_and_trace_overhead_not_subtracted=True,
            scheduler_not_directly_measured_by_cProfile=True)


def loaded_resource_record(manager_record):
    snapshot=dict(manager_record['loaded_resources'])
    snapshot['post_launch_resource_evidence']=resource_evidence(snapshot,manager_record)
    return snapshot


def self_test(source):
    callbacks=list(gc.callbacks);trace=sys.gettrace();profile=sys.getprofile()
    enabled,threshold=gc.isenabled(),gc.get_threshold();o=Observer();o.start()
    try:sum(range(7))
    finally:o.stop()
    assert gc.callbacks==callbacks and sys.gettrace()is trace and sys.getprofile()is profile
    assert gc.isenabled()==enabled and gc.get_threshold()==threshold
    assert o.summary()['functions']
    marker=RuntimeError('CPU observer fixture error');failed=Observer()
    try:
        failed.start()
        try:raise marker
        finally:failed.stop()
    except RuntimeError as error:assert error is marker
    assert gc.callbacks==callbacks and sys.gettrace()is trace and sys.getprofile()is profile
    # Use the preserved actual5967 schema, not a guessed synthetic dictionary.
    archive=source/'docs/evidence/prefill-reverse-20261027/host-binding-completed-v4/v4-raw.tar.gz'
    with tarfile.open(archive)as saved:raw=saved.extractfile('out/result.json').read()
    assert hashlib.sha256(raw).hexdigest()=='99f18a8abd926680221a8f32013f3ba65795e56f8843dca875e738d07740e417'
    records=json.loads(raw)['manager_proofs'];left=records['parent']['compile_records'][0];right=records['reverse']['compile_records'][0]
    assert compare_resource_records(loaded_resource_record(left),loaded_resource_record(right))['strict_allowed']
    unknown=dict(left['candidate_resources']);unknown['post_launch_resource_evidence']=resource_evidence(unknown)
    assert not compare_resource_records(unknown,unknown)['strict_allowed']
    mismatched=dict(left,candidate_hash='wrong-binary')
    try:loaded_resource_record(mismatched)
    except ValueError:pass
    else:raise AssertionError('Wrong pre/post manager binary accepted')
    return dict(stdlib_observer_lifecycle_passed=True,error_lifecycle_passed=True,forced_GC=False,Torch_imported=False,
        actual5967_loaded_resource_schema_passed=True,unknown_counts_rejected=True,mismatched_binary_rejected=True)


def device(a,prepared,cpu,contract_proof):
    ctx=_bootstrap(a,prepared,cpu);torch,fixture=ctx['torch'],ctx['fixture']
    binding,function,parent,reverse=ctx['binding'],ctx['function'],ctx['parent'],ctx['reverse']
    from biv150_prefill_host_binding import _BoundaryLaunch
    lines,start=inspect.getsourcelines(_BoundaryLaunch.__call__)
    audit_lines=[start+i for i,line in enumerate(lines)if'b.last=dict('in line]
    assert len(audit_lines)==1
    original_manifest=json.loads((a.out/'manifest.json').read_text());save(a.out/'bootstrap-manifest.json',original_manifest)
    report=dict(diagnostic_only=True,performance_score=False,speedup_not_computed=True,model_precision_gate=False,
        full_component_acceptance=False,source_commit=a.commit,query=a.query,kv=a.kv,calls_per_arm=a.calls,
        source=prepared.proof,native_runtime_before=ctx['native_before'],
        warm_calls_per_arm=3,round_sweep=False,original_contract=contract_proof,arms=[],errors=[],
        caveat='Instrumentation perturbs CPU/GC/queueing. This short warm prehistory differs from the complete numerical harness. Events can include enqueue gaps. Do not compare these timings as benchmark scores or infer unrecorded older-round causes.')
    save(a.out/'manifest.json',dict(diagnostic_only=True,performance_score=False,source_commit=a.commit,
        diagnostic_pending=True,model_precision_gate=False))
    try:
        host=fixture.cpu_inputs([a.query],[a.kv],20261027,0);stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            data={k:v.cuda()for k,v in host.items()};key,value=data['packed'].unbind(1)
            storage=torch.full((a.query+2,16,128),43.,device='cuda',dtype=torch.bfloat16);output=storage[1:-1]
        stream.synchronize()
        kw=dict(q=data['q'],k=key,v=value,out=output,cu_seqlens_q=data['cu_q'],max_seqlen_q=a.query,
            seqused_k=data['lengths'],max_seqlen_k=a.kv,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
            block_table=data['table'],softcap=0,q_descale=None,k_descale=data['k_scale'],v_descale=data['v_scale'])
        contract=lambda:CpuContract(kw['q'],kw['cu_seqlens_q'],kw['seqused_k'],(0,a.query),True,PACKING,True)
        baseline=None
        def proof_record(manager):
            snapshot=manager.snapshot();records=snapshot['compile_records']
            assert snapshot['record_errors']==[]and snapshot['fallback_compiles']==0
            selected=[r for r in records if r.get('mode')=='selected'];assert len(selected)==1
            record=selected[0];proof=record['mechanism']
            assert proof['mechanism_pass']and proof['llvm_q_memory_reads_in_loop']==0 and proof['llvm_q_memory_reads_total']==32
            return record
        def loaded(manager):return loaded_resource_record(proof_record(manager))
        for arm in('baseline','candidate'):
            boundary_before=len(binding.boundary.calls);reverse_before=reverse.snapshot()
            run=(lambda:binding.call(contract=contract(),**kw))if arm=='candidate'else(lambda:function(**kw))
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            with (binding.candidate()if arm=='candidate'else nullcontext()),torch.cuda.stream(stream):
                for _ in range(3):run()
                stream.synchronize()
                expected_reverse=arm=='candidate'and a.kv<=5*a.query
                effective=reverse if expected_reverse else parent
                for field in('abi','options','target'):
                    assert proof_record(parent)[field]==proof_record(effective)[field],'Typed compiler identity changed: '+field
                comparison=ctx['compare_resource_records'](loaded(parent),loaded(effective));assert comparison['strict_allowed']
                observer=Observer(_BoundaryLaunch.__call__.__code__,audit_lines[0]);before=usage();observer.start()
                try:
                    wall=time.perf_counter_ns();phase=time.perf_counter_ns();begin.record();begin_ns=time.perf_counter_ns()-phase
                    calls=[]
                    for _ in range(a.calls):
                        phase=time.perf_counter_ns();run();calls.append(time.perf_counter_ns()-phase)
                    phase=time.perf_counter_ns();end.record();end_record_ns=time.perf_counter_ns()-phase
                    phase=time.perf_counter_ns();end.synchronize();sync_ns=time.perf_counter_ns()-phase
                    wall_ns=time.perf_counter_ns()-wall
                finally:observer.stop()
                after=usage();event_ms=begin.elapsed_time(end)
            observer.profiler.dump_stats(str(a.out/(arm+'.cprofile')))
            current=output.cpu().clone()
            if baseline is None:baseline=current
            comparison=fixture.compare(current,baseline,0,0);assert comparison['passed']and comparison['finite']
            assert bool(torch.all(storage[0]==43))and bool(torch.all(storage[-1]==43))
            new_records=binding.boundary.calls[boundary_before:]
            assert len(new_records)==(3+a.calls if expected_reverse else 0)
            assert all(r['route']=='reverse'and r['payload_identity']for r in new_records)
            if not expected_reverse:
                reverse_after=reverse.snapshot()
                assert all(reverse_before[k]==reverse_after[k]for k in('selected_calls','selected_compiles','fallback_calls','fallback_compiles'))
            assert parent.wrapper is ctx['original']and binding.restored and ctx['intact']()
            deltas={who:({k:after[who][k]-before[who][k]for k in before[who]if k!='ru_maxrss'}if before[who]is not None else None)for who in before}
            record=dict(arm=arm,profiled_wall_ns=wall_ns,profiled_event_ms=event_ms,
                wall_start_ns=wall,wall_end_ns=wall+wall_ns,rusage_bookends_include_observer_setup_cleanup=True,
                begin_record_ns=begin_ns,host_call_ns=calls,end_record_ns=end_record_ns,end_sync_ns=sync_ns,
                rusage_before=before,rusage_after=after,rusage_deltas=deltas,
                maxrss_is_process_highwater_not_phase_delta=True,post_profile_output_exact=comparison,
                expected_route='reverse'if expected_reverse else'parent',route_records=new_records,
                observed_resource_guard=ctx['compare_resource_records'](loaded(parent),loaded(effective)),**observer.summary())
            report['arms'].append(record);save(a.out/'diagnostic.json',report)
        report['native_runtime_after']=native_runtime_identity(torch,a.native_library)
        assert report['native_runtime_after']['identity_file']['sha256']==NATIVE
        report['completed']=True;report['adapter']=binding.snapshot()
        report['manager_snapshots']={k:ctx[k].snapshot()for k in('parent','reverse')}
        report['note']='Manager records are retained; this driver does not claim it saved full compiler IR/cubins.'
    except BaseException as error:
        report['errors'].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));raise
    finally:
        existing_error=sys.exc_info()[0]is not None;cleanup_error=None
        try:binding.close()
        except BaseException as error:
            cleanup_error=error;report['errors'].append(dict(type=type(error).__name__,message='cleanup: '+str(error)))
        finally:
            report['parent_restored']=parent.wrapper is ctx['original']and ctx['intact']()
            report['diagnostic_checks_passed']=bool(report.get('completed',False)and report['parent_restored']and not report['errors'])
            save(a.out/'diagnostic.json',report)
            save(a.out/'manifest.json',dict(diagnostic_only=True,performance_score=False,model_precision_gate=False,
                source_commit=a.commit,diagnostic_completed=report['diagnostic_checks_passed'],parent_restored=report['parent_restored']))
        if cleanup_error is not None and not existing_error:raise cleanup_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('source','sdk-source','helpers-source','target-sources','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-gpu',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--commit');p.add_argument('--native-library',type=Path)
    p.add_argument('--query',type=int,default=2048);p.add_argument('--kv',type=int,default=4096);p.add_argument('--calls',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.calls<=4 or not 1024<=a.query<=4096 or not a.query<=a.kv<=16384:p.error('Bounded query1024..4096, query<=kv<=16384, calls1..4 required')
    if a.run_gpu and(a.self_test or not a.commit or not a.native_library):p.error('Explicit device diagnostic requires commit/native-library and no self-test')
    proof=prefix_contract(a.source.resolve());cpu,prepared=run_cpu(a.source,a.sdk_source,a.helpers_source,a.target_sources)
    if not a.run_gpu:
        a.out.mkdir(parents=True,exist_ok=False);save(a.out/'plan.json',dict(cpu_only=True,actual_host_profiling_executed=False,
            stdlib_self_test_profile_executed=a.self_test,GPU_executed=False,
            production_hook=False,performance_score=False,contract=proof,self_test=self_test(a.source)if a.self_test else None))
        save(a.out/'cpu.json',cpu);assert'torch'not in sys.modules and'triton'not in sys.modules
        print(json.dumps(dict(passed=True,cpu_only=True,actual_host_profile_executed=False)));return
    device(a,prepared,cpu,proof)


if __name__=='__main__':main()
