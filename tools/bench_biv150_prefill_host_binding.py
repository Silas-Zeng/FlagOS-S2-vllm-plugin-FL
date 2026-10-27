#!/usr/bin/env python3
"""Default CPU; explicit, bounded private SDK host-binding GPU diagnostic.

This is not a service installer or an official/model benchmark. No device
dependency is imported unless --run-gpu and all preflights permit it.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from biv150_prefill_host_binding import (CpuContract, LocalSdkBinding,
    make_private_reverse_manager, TARGET_PINS)
from biv150_prefill_host_component import PACKING
from check_biv150_prefill_host_binding_cpu import run_cpu

MAIN = "56f1838192abc940b3fb9accef490022ac125000"
NATIVE = "5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe"
CASES = {"chunk4k":(4096,4096), "chunk16k":(16384,16384), "upper-bound-miss":(4096,16384)}


def save(path, value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)


@contextmanager
def tap_parent(manager,tap):
    original=manager.wrapper;manager.wrapper=tap
    try:yield
    finally:
        if manager.wrapper is not tap:raise RuntimeError("Parent tap ownership changed")
        manager.wrapper=original


def gpu(a,prepared,cpu):
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
    native=importlib.import_module("vllm_fl._C")
    assert Path(native.__file__).resolve()==a.native_library.resolve()
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
        torch.cuda.is_current_stream_capturing,runtime_intact)
    result=dict(completed=False,component_only=True,official_score=False,model_precision_gate=False,
        service_installed=False,source_commit=a.commit,production_parent=MAIN,native_sha256=NATIVE,
        cpu_preflight=cpu,source=prepared.proof,parent_M2_retained=True,
        baseline_installations=dict(states,q32=q32_state,multi2=m2_state),
        private_triton_cache=os.environ["TRITON_CACHE_DIR"],
        baseline_arm_has_no_new_tap=True,timing_includes_component_host_overhead=True,
        timing_is_service_throughput=False,cases=[],errors=[])
    flush=lambda:save(a.out/"result.json",result)
    flush()
    try:
        for name in a.case or ["chunk4k","chunk16k"]:
            kv,upper=CASES[name];qrows=2048;expected="reverse"if upper<=5*qrows else"parent"
            host=fixture.cpu_inputs([qrows],[kv],20261027,0);stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                device={k:v.cuda()for k,v in host.items()};key,value=device["packed"].unbind(1)
                storage=torch.full((qrows+2,16,128),43.,device="cuda",dtype=torch.bfloat16);output=storage[1:-1]
            stream.synchronize()
            kw=dict(q=device["q"],k=key,v=value,out=output,cu_seqlens_q=device["cu_q"],
                max_seqlen_q=qrows,seqused_k=device["lengths"],max_seqlen_k=upper,
                softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=device["table"],
                softcap=0,q_descale=None,k_descale=device["k_scale"],v_descale=device["v_scale"])
            contract=lambda:CpuContract(kw["q"],kw["cu_seqlens_q"],kw["seqused_k"],(0,qrows),True,PACKING,True)
            baseline_view=type("BaselineView",(),{})();baseline_view.wrapper=original;baseline_view.optimized=parent.optimized
            tap=Tap(baseline_view,fragment);candidate_parent_tap=Tap(baseline_view,fragment)
            graphs={};generic_graph=None;generic_compiled=None
            row=dict(name=name,query=qrows,actual_kv=kv,cpu_upper=upper,expected_route=expected,
                completed=False,checks=[],rounds=[],reverse_before=copy.deepcopy(reverse.snapshot()))
            result["cases"].append(row);flush()
            print(json.dumps(dict(case=name,stage="started",expected_route=expected)),flush=True)
            for pattern in (0,1):
                if pattern:
                    host=fixture.cpu_inputs([qrows],[kv],20261027,pattern)
                    with torch.cuda.stream(stream):
                        for field in host:device[field].copy_(host[field])
                with torch.cuda.stream(stream),tap_parent(parent,tap):
                    output.fill_(float("nan"));function(**kw)
                stream.synchronize();baseline=output.cpu().clone()
                before_boundary=len(binding.boundary.calls)
                with binding.candidate()as invoke,torch.cuda.stream(stream):
                    with (tap_parent(parent,candidate_parent_tap)if expected=="parent"else _null()):
                        output.fill_(float("nan"));used=contract();invoke(contract=used,**kw)
                stream.synchronize();candidate=output.cpu().clone();assert intact()and binding.restored
                if expected=="parent":
                    assert len(binding.boundary.calls)==before_boundary
                    selected=SimpleNamespace(grid=candidate_parent_tap.grid,kwargs=candidate_parent_tap.kwargs)
                    compiled=candidate_parent_tap.compiled
                else:
                    observed=binding.boundary.last;assert observed["route"]==expected
                    selected=observed["selected"];compiled=observed["result"]
                effective=reverse if expected=="reverse"else parent
                assert compiled.hash in effective.optimized
                old_record=parent.optimized[tap.compiled.hash];new_record=effective.optimized[compiled.hash]
                for field in ("abi","options","target"):
                    assert old_record[field]==new_record[field],"Compiler identity changed: "+field
                for record in (old_record,new_record):
                    proof=record["mechanism"]
                    assert proof["mechanism_pass"]and proof["llvm_q_memory_reads_in_loop"]==0
                    assert proof["llvm_q_memory_reads_total"]==32
                assert descriptors(selected.kwargs)==descriptors(tap.kwargs)and tuple(selected.grid)==tap.grid
                assert fragment._abi(compiled.src)==fragment._abi(tap.compiled.src)
                if expected=="parent":assert compiled.hash==tap.compiled.hash
                indices,reference=fixture.reference(host,[qrows],[kv],pattern==0,False)
                check=dict(pattern=pattern,direct=fixture.compare(candidate,baseline,0,0),
                    baseline_cpu=fixture.compare(baseline[indices],reference,.005,.005),
                    candidate_cpu=fixture.compare(candidate[indices],reference,.005,.005),
                    reference_guard=fixture.reference_guard(candidate[indices],baseline[indices],reference),
                    actual_route=expected,typed_abi_equal=True,compiler_options_target_equal=True,
                    actual_selected_Q_hoist_verified=True,grid=list(selected.grid),
                    long_parent_bypasses_new_boundary=(expected=="parent"),
                    long_parent_binary_identity=(expected!="parent"or compiled.hash==tap.compiled.hash),graphs=[])
                row["checks"].append(check);flush()
                assert check["direct"]["passed"]and check["reference_guard"]["passed"]
                # Selected-kernel graphs deliberately bypass the host scope.
                # They test actual warmed typed bindings and changed input bytes.
                for label,manager in [("baseline",parent),("candidate",effective)]:
                    if label not in graphs:
                        with torch.cuda.stream(stream):
                            for _ in range(3):manager.candidate[selected.grid](**selected.kwargs)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=stream):
                                manager.candidate[selected.grid](**selected.kwargs)
                        stream.synchronize();graphs[label]=graph
                    with torch.cuda.stream(stream):output.fill_(float("nan"));graphs[label].replay()
                    stream.synchronize();comparison=fixture.compare(output.cpu(),baseline if label=="baseline"else candidate,0,0)
                    check["graphs"].append(dict(kind="warmed-selected-kernel",arm=label,
                        host_scope_bypassed=True,changed_input=bool(pattern),**comparison));assert comparison["passed"]
                # Real SDK-scope capture follows the unchanged parent's capture
                # fallback (BM128), not the Q32 or reverse path. Warm it explicitly.
                raw_kw=dict(tap.kwargs,BLOCK_M=128,BLOCK_Q=16,num_warps=8)
                raw_grid=(qrows//16+1,2)
                with torch.cuda.stream(stream):
                    output.fill_(float("nan"));generic_compiled=base.original_kernel[raw_grid](**raw_kw)
                stream.synchronize();generic_expected=output.cpu().clone()
                generic_guard=fixture.reference_guard(generic_expected[indices],baseline[indices],reference)
                assert generic_guard["passed"]
                if generic_graph is None:
                    before=len(binding.boundary.calls)
                    with binding.candidate()as invoke,torch.cuda.stream(stream):
                        generic_graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(generic_graph,stream=stream):invoke(contract=contract(),**kw)
                    stream.synchronize();assert len(binding.boundary.calls)==before
                with torch.cuda.stream(stream):output.fill_(float("nan"));generic_graph.replay()
                stream.synchronize();comparison=fixture.compare(output.cpu(),generic_expected,0,0)
                check["graphs"].append(dict(kind="actual-SDK-scope-parent-capture-fallback",block_m=128,
                    claimed_Q32_or_reverse=False,binary_identity_independently_traced=False,
                    source_expected_generic_hash=generic_compiled.hash,
                    changed_input=bool(pattern),reference_guard=generic_guard,**comparison))
                assert comparison["passed"]
                # An already consumed CPU contract must not select the reverse.
                before_boundary=len(binding.boundary.calls)
                with binding.candidate()as invoke,torch.cuda.stream(stream),tap_parent(parent,candidate_parent_tap):
                    output.fill_(float("nan"));invoke(contract=used,**kw)
                stream.synchronize();check["stale_contract"]=fixture.compare(output.cpu(),baseline,0,0)
                assert len(binding.boundary.calls)==before_boundary
                assert candidate_parent_tap.compiled.hash==tap.compiled.hash and check["stale_contract"]["passed"]
                side=torch.cuda.Stream();side.wait_stream(stream)
                with binding.candidate()as invoke,torch.cuda.stream(side):
                    output.fill_(float("nan"));invoke(contract=contract(),**kw)
                side.synchronize();check["side_stream"]=fixture.compare(output.cpu(),candidate,0,0)
                assert check["side_stream"]["passed"]
                assert bool(torch.all(storage[0]==43))and bool(torch.all(storage[-1]==43))
                for field in host:assert torch.equal(device[field].cpu().contiguous().view(-1).view(torch.uint8),host[field].contiguous().view(-1).view(torch.uint8))
                check["output_canaries_unchanged"]=check["inputs_unchanged"]=True;flush()
            # Replay the SAME graphs after changing actual KV metadata in place.
            # CPU U is updated conservatively; no captured host decision is rerun.
            changed_kv=kv-1;kw["max_seqlen_k"]=upper if upper>kv else changed_kv
            active=dict(host,lengths=torch.tensor([changed_kv],dtype=torch.int32))
            with torch.cuda.stream(stream),tap_parent(parent,tap):
                device["lengths"].copy_(active["lengths"]);output.fill_(float("nan"));function(**kw)
            stream.synchronize();fresh_parent=output.cpu().clone()
            with binding.candidate()as invoke,torch.cuda.stream(stream):
                with (tap_parent(parent,candidate_parent_tap)if expected=="parent"else _null()):
                    output.fill_(float("nan"));invoke(contract=contract(),**kw)
            stream.synchronize();fresh_candidate=output.cpu().clone()
            positions,reference=fixture.reference(active,[qrows],[changed_kv],False,False)
            probe=dict(actual_kv=changed_kv,cpu_upper=kw["max_seqlen_k"],
                direct=fixture.compare(fresh_candidate,fresh_parent,0,0),
                reference_guard=fixture.reference_guard(fresh_candidate[positions],fresh_parent[positions],reference),
                graph_host_decision_recomputed=False,graphs=[])
            row["changed_metadata"]=probe
            assert probe["direct"]["passed"]and probe["reference_guard"]["passed"]
            before_boundary=len(binding.boundary.calls)
            with binding.candidate()as invoke,torch.cuda.stream(stream),tap_parent(parent,candidate_parent_tap):
                output.fill_(float("nan"));invoke(contract=used,**kw)
            stream.synchronize();probe["stale_contract_after_metadata_change"]=fixture.compare(output.cpu(),fresh_parent,0,0)
            assert probe["stale_contract_after_metadata_change"]["passed"]
            assert len(binding.boundary.calls)==before_boundary and candidate_parent_tap.compiled.hash==tap.compiled.hash
            for label in ("baseline","candidate"):
                with torch.cuda.stream(stream):output.fill_(float("nan"));graphs[label].replay()
                stream.synchronize();comparison=fixture.compare(output.cpu(),fresh_parent if label=="baseline"else fresh_candidate,0,0)
                probe["graphs"].append(dict(kind="warmed-selected-kernel",arm=label,**comparison));assert comparison["passed"]
            with torch.cuda.stream(stream):
                output.fill_(float("nan"));base.original_kernel[raw_grid](**raw_kw)
            stream.synchronize();fresh_generic=output.cpu().clone()
            with torch.cuda.stream(stream):output.fill_(float("nan"));generic_graph.replay()
            stream.synchronize();comparison=fixture.compare(output.cpu(),fresh_generic,0,0)
            probe["graphs"].append(dict(kind="actual-SDK-parent-capture-fallback",**comparison));assert comparison["passed"]
            with torch.cuda.stream(stream):device["lengths"].copy_(host["lengths"])
            stream.synchronize();kw["max_seqlen_k"]=upper;probe["metadata_restored_before_timing"]=True
            assert bool(torch.all(storage[0]==43))and bool(torch.all(storage[-1]==43));flush()
            row["baseline_resources"]=resources(tap.compiled,fragment,a.out,"baseline",parent)
            row["selected_candidate_resources"]=resources(compiled,fragment,a.out,"selected",effective)
            row["generic_capture_parent_resources"]=resources(generic_compiled,fragment,a.out,"generic-capture-parent")
            for record in (row["baseline_resources"],row["selected_candidate_resources"],row["generic_capture_parent_resources"]):
                assert {"source","ttir","ttgir","llir","asm","cubin"}.issubset(record["files"])
            comparison=compare_resource_records(row["baseline_resources"],row["selected_candidate_resources"])
            row["resource_comparison"]=comparison;assert comparison["strict_allowed"]
            # Eager SDK calls: baseline is untapped original. Candidate includes
            # contract allocation, SDK scope, host decision and diagnostic recording.
            for iteration in range(3):
                order=["baseline","candidate"]if iteration%2==0 else["candidate","baseline"]
                timing=dict(iteration=iteration,order=order)
                for arm in order:
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    with (binding.candidate()if arm=="candidate"else _null()),torch.cuda.stream(stream):
                        run=(lambda:binding.call(contract=contract(),**kw))if arm=="candidate"else(lambda:function(**kw))
                        for _ in range(3):run()
                        stream.synchronize();wall=time.perf_counter();begin.record()
                        for _ in range(a.repeats):run()
                        end.record();end.synchronize()
                        timing[arm+"_wall_ms"]=(time.perf_counter()-wall)*1000/a.repeats
                        timing[arm+"_event_ms"]=begin.elapsed_time(end)/a.repeats
                timing["wall_speedup"]=timing["baseline_wall_ms"]/timing["candidate_wall_ms"]
                timing["event_speedup"]=timing["baseline_event_ms"]/timing["candidate_event_ms"]
                row["rounds"].append(timing)
            row["median_wall_speedup"]=statistics.median(x["wall_speedup"]for x in row["rounds"])
            row["median_event_speedup"]=statistics.median(x["event_speedup"]for x in row["rounds"])
            row["reverse_after"]=copy.deepcopy(reverse.snapshot())
            if expected=="parent":
                for field in ("selected_calls","selected_compiles","fallback_calls","fallback_compiles"):
                    assert row["reverse_before"][field]==row["reverse_after"][field],"Long-context touched private reverse"
            row["completed"]=True;flush()
            print(json.dumps(dict(case=name,stage="completed",median_wall_speedup=row["median_wall_speedup"],
                median_event_speedup=row["median_event_speedup"])),flush=True)
        result["completed"]=True
    except Exception as error:
        result["errors"].append(dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        cleanup_error=None
        try:binding.close()
        except Exception as error:
            cleanup_error=error
            result["errors"].append(dict(type=type(error).__name__,message="cleanup:"+str(error)))
        result["parent_restored"]=(parent.wrapper is original and intact())
        result["boundary_records"]=binding.boundary.calls;result["omitted_boundary_records"]=binding.boundary.omitted
        result["direct_parent_sdk_calls"]=binding.direct_parent_calls
        result["scoped_sdk_calls"]=binding.scoped_calls
        result["manager_proofs"]={"parent":parent.snapshot(),"reverse":reverse.snapshot()}
        result["saved_override_ir"]=[]
        artifact_error=None
        try:
            for label,snapshot in result["manager_proofs"].items():
                for record in snapshot["compile_records"]:
                    if "override_filename"not in record:continue
                    name=record["override_filename"];assert Path(name).name==name
                    data=(Path(snapshot["temporary_ir_directory"])/name).read_bytes()
                    assert hashlib.sha256(data).hexdigest()==record["override_sha256"]
                    directory=a.out/"compiler-ir"/label;directory.mkdir(parents=True,exist_ok=True)
                    (directory/name).write_bytes(data)
                    result["saved_override_ir"].append(dict(path=str((directory/name).relative_to(a.out)),
                        bytes=len(data),sha256=record["override_sha256"]))
        except Exception as error:
            artifact_error=error
            result["errors"].append(dict(type=type(error).__name__,message="artifact:"+str(error)))
        result["requested_component_gates_passed"]=bool(result["completed"]and result["parent_restored"]
            and not result["errors"]and binding.boundary.omitted==0)
        result["full_numerical_matrix_verified"]=False
        flush()
        save(a.out/"manifest.json",dict(component_only=True,source_commit=a.commit,
            production_parent=MAIN,expected_native_sha256=NATIVE,model_precision_gate=False,
            actual_GPU_result_pending=False,requested_component_gates_passed=result["requested_component_gates_passed"],
            result_sha256=hashlib.sha256((a.out/"result.json").read_bytes()).hexdigest(),
            excluded_from_portable_evidence=["private-triton-cache","__pycache__"]))
        if cleanup_error is not None and sys.exc_info()[0]is None:raise cleanup_error
        if artifact_error is not None and sys.exc_info()[0]is None:raise artifact_error


@contextmanager
def _null():yield


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","sdk-source","helpers-source","target-sources","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--run-gpu",action="store_true");p.add_argument("--commit")
    p.add_argument("--native-library",type=Path)
    p.add_argument("--case",action="append",choices=CASES)
    p.add_argument("--repeats",type=int,default=16)
    a=p.parse_args()
    if a.run_gpu and(not a.commit or not a.native_library):p.error("GPU requires exact --commit and --native-library")
    if not 1<=a.repeats<=32:p.error("Bounded --repeats must be1..32")
    if a.case and len(a.case)!=len(set(a.case)):p.error("Duplicate cases are not allowed")
    cpu,prepared=run_cpu(a.source,a.sdk_source,a.helpers_source,a.target_sources)
    if not a.run_gpu:
        a.out.mkdir(parents=True,exist_ok=False);save(a.out/"cpu.json",cpu)
        print(json.dumps(dict(passed=True,cpu_only=True,gpu_executed=False)));return
    gpu(a,prepared,cpu)


if __name__=="__main__":main()
