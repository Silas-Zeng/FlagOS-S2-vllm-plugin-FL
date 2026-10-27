#!/usr/bin/env python3
"""Verify ACTUAL target context/custom-op source and local SDK component on CPU."""
import argparse
import ast
import copy
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

from biv150_prefill_host_component import prepare_source, PACKING
import biv150_prefill_host_component as host_protocol
from biv150_prefill_host_binding import CpuContract, LocalSdkBinding, verify_target_sources, target_source_directory
from check_biv150_prefill_host_component_cpu import Entry, values

MAIN = "56f1838192abc940b3fb9accef490022ac125000"


class Mode(Enum):
    NONE=0
    FULL=1
    PIECEWISE=2
    FULL_DECODE_ONLY=3
    def separate_routine(self):return self is Mode.FULL_DECODE_ONLY
    def has_mode(self,mode):return mode is Mode.FULL
    @classmethod
    def valid_runtime_modes(cls):return {cls.NONE,cls.FULL,cls.PIECEWISE}


@dataclass(frozen=True)
class Descriptor:
    num_tokens:int
    num_reqs:object=None
    uniform:bool=False
    has_lora:bool=False
    num_active_loras:int=0


class Unannotated(ast.NodeTransformer):
    def visit_AnnAssign(self, node):
        return ast.copy_location(ast.Assign(targets=[node.target],value=self.visit(node.value)),node) if node.value else None

    def visit_FunctionDef(self, node):
        node.returns=None
        for arg in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:
            arg.annotation=None
        node.decorator_list=[]
        if node.name=="override_forward_context":node.decorator_list=[ast.Name(id="contextmanager",ctx=ast.Load())]
        return self.generic_visit(node)


def source_chain(target,source):
    pins=verify_target_sources(target)
    attention=ast.parse((target/"model_executor/layers/attention/attention.py").read_text())
    context=ast.parse((target/"forward_context.py").read_text())
    funcs={n.name:n for n in attention.body if isinstance(n,ast.FunctionDef)}
    ctx={n.name:n for n in context.body if isinstance(n,ast.FunctionDef)}
    registrations=[n for n in attention.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=="direct_register_custom_op"]
    actual=[n.value for n in registrations if any(k.arg=="op_name" and isinstance(k.value,ast.Constant)
        and k.value.value=="unified_attention_with_output" for k in n.value.keywords)]
    assert len(actual)==1
    keywords={k.arg:k.value for k in actual[0].keywords}
    assert keywords["op_func"].id=="unified_attention_with_output"
    assert keywords["fake_impl"].id=="unified_attention_with_output_fake"
    assert ast.literal_eval(keywords["mutates_args"])==["output","output_block_scale"]
    setctx=ctx["set_forward_context"]
    assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="create_forward_context"
        and isinstance(n.args[0],ast.Name)and n.args[0].id=="attn_metadata" for n in ast.walk(setctx))
    assert any(isinstance(n,ast.With)and any(isinstance(i.context_expr,ast.Call)
        and isinstance(i.context_expr.func,ast.Name)and i.context_expr.func.id=="override_forward_context"
        for i in n.items)and any(isinstance(x,ast.Yield)for x in ast.walk(n))for n in ast.walk(setctx))
    forward=next(n for c in attention.body if isinstance(c,ast.ClassDef)and c.name=="Attention"
        for n in c.body if isinstance(n,ast.FunctionDef)and n.name=="forward")
    assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="unified_attention_with_output"for n in ast.walk(forward))
    assert any(isinstance(n,ast.Call)and ast.unparse(n.func)=="torch.ops.vllm.unified_attention_with_output"for n in ast.walk(forward))
    selected=[ctx[n]for n in ("get_forward_context","create_forward_context","override_forward_context")]
    selected += [funcs[n]for n in ("get_attention_context","unified_attention_with_output")]
    module=Unannotated().visit(ast.Module(body=copy.deepcopy(selected),type_ignores=[]))
    namespace=dict(_forward_context=None,contextmanager=contextmanager,
        CUDAGraphMode=Mode,ForwardContext=lambda **kw:SimpleNamespace(**kw),
        _resolve_layer_name=lambda x:x)
    exec(compile(ast.fix_missing_locations(module),"<actual-target-context-custom-op>","exec"),namespace)
    seen=[]
    class Impl:
        def forward(self,layer,q,k,v,cache,metadata,**kw):seen.append(metadata)
    layer=SimpleNamespace(impl=Impl(),kv_cache=object())
    config=SimpleNamespace(compilation_config=SimpleNamespace(fast_moe_cold_start=False,
        static_forward_context={"layer":layer}))
    first=SimpleNamespace(max_query_len=2048,max_seq_len=4096)
    second=SimpleNamespace(max_query_len=2048,max_seq_len=16384)
    f1=namespace["create_forward_context"]({"layer":first},config,slot_mapping={"layer":object()})
    f2=namespace["create_forward_context"]({"layer":second},config,slot_mapping={"layer":object()})
    # Execute the exact retained target NONE wrapper too, not a hand-written
    # equivalent. Its audited pin is already part of the C3/source evidence.
    wrapper=(source/"tools/reference/breakable_cudagraph_20261024.py").read_bytes()
    assert hashlib.sha256(wrapper).hexdigest()=="722683b44b4461ffd9a6498741da18945cbaac231c3bb01f0b0443dafe9e053e"
    cls=next(n for n in ast.parse(wrapper).body if isinstance(n,ast.ClassDef)and n.name=="BreakableCUDAGraphWrapper")
    call=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=="__call__")
    ws=dict(CUDAGraphMode=Mode,get_forward_context=namespace["get_forward_context"],is_forward_context_available=lambda:True)
    exec(compile(ast.fix_missing_locations(Unannotated().visit(ast.Module(body=[copy.deepcopy(call)],type_ignores=[]))),"<actual-target-NONE-wrapper>","exec"),ws)
    runnable=lambda:namespace["unified_attention_with_output"](object(),object(),object(),object(),"layer")
    invoke=lambda:ws["__call__"](SimpleNamespace(runnable=runnable))
    with namespace["override_forward_context"](f1):
        invoke()
        with namespace["override_forward_context"](f2):invoke()
        invoke()
    assert all(a is b for a,b in zip(seen,[first,second,first]))and len(seen)==3
    assert namespace["_forward_context"]is None
    with namespace["override_forward_context"](f2):invoke()
    assert seen[-1]is second and namespace["_forward_context"]is None
    runner=(source/"vllm_fl/worker/model_runner.py").read_bytes()
    assert hashlib.sha256(runner).hexdigest()=="0161c595ff2f1b115039f03980fbb6de0447f1edf0d55fba1793b6a8055c230c"
    for path,pin in [("biv150_compute_order.py","e60909d8832505566f172ecfe69c63728a5bfdfe3e08f40582fc771ce92cf17b"),
                     ("biv150_mixed_decode.py","a129361f1ef1babe1ce1e6238f6b1a725050612c235201064ddeba676b420c42")]:
        assert hashlib.sha256((source/"vllm_fl/patches"/path).read_bytes()).hexdigest()==pin
    runner_fn=next(n for n in ast.walk(ast.parse(runner))if isinstance(n,ast.FunctionDef)and n.name=="_is_uniform_decode")
    rs={};exec(compile(ast.fix_missing_locations(Unannotated().visit(ast.Module(body=[copy.deepcopy(runner_fn)],type_ignores=[]))),"<actual-runner-uniform>","exec"),rs)
    with tarfile.open(source/"docs/evidence/prefill-reverse-20261027/refined-first-falsifier/host-route-sources-v1.tar.gz")as archive:
        dispatcher=archive.extractfile("sdk/cudagraph_dispatcher.py").read()
    assert hashlib.sha256(dispatcher).hexdigest()=="51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63"
    dc=next(n for n in ast.parse(dispatcher).body if isinstance(n,ast.ClassDef)and n.name=="CudagraphDispatcher")
    df=[n for n in dc.body if isinstance(n,ast.FunctionDef)and n.name in("dispatch","_create_padded_batch_descriptor")]
    ds=dict(CUDAGraphMode=Mode,BatchDescriptor=Descriptor,replace=replace)
    exec(compile(ast.fix_missing_locations(Unannotated().visit(ast.Module(body=copy.deepcopy(df),type_ignores=[]))),"<actual-target-dispatcher>","exec"),ds)
    obj=SimpleNamespace(keys_initialized=True,cudagraph_mode=Mode.FULL_DECODE_ONLY,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=2048),
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=64)),
        uniform_decode_query_len=1,_bs_to_padded_graph_size={1024:1024,2048:2048},
        cudagraph_keys={Mode.FULL:{Descriptor(1024,64,True),Descriptor(2048,64,True)},Mode.PIECEWISE:set()})
    obj._create_padded_batch_descriptor=lambda *a,**k:ds["_create_padded_batch_descriptor"](obj,*a,**k)
    for q in (1024,2048):
        uniform=rs["_is_uniform_decode"](q,1,q,1)
        mode,_=ds["dispatch"](obj,q,uniform_decode=uniform)
        assert uniform is False and mode is Mode.NONE
    return dict(actual_target_sources=pins,source_chain_complete_for_pinned_NONE_path=True,
        real_custom_op_and_fake_distinct=True,direct_and_torch_ops_use_registered_function=True,
        actual_function_ast_reads_current_metadata_each_call=True,nested_context_restored=True,
        current_metadata_sequence=[4096,16384,4096,16384],outer_hook_needed=False,
        actual_NONE_wrapper_AST_executed=True,actual_dispatcher_nonuniform_NONE_AST_executed=True,
        actual_runner_uniform_predicate_AST_executed=True,
        capture_source_preconditions=dict(ordinary_prefill_max_query_gt_one=True,
            fixed_service_FULL_DECODE_ONLY=True,full_keys_uniform_true=True,piecewise_keys_empty=True,
            forced_dummy_capture_not_claimed_ordinary_prefill=True),
        runtime_framework_imported=False,runtime_custom_op_binding_or_device_verified=False,
        source_lines=dict(get_attention_context=[670,710],unified_attention_with_output=[757,784],
            registration=[800,805],get_forward_context=[191,197],create=[204,233],
            override=[235,246],set_forward_context=[307,323]))


def local_binding(root,prepared):
    policies=[]
    for filename,pin in [("biv150_prefill_q32.py","3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b"),
                         ("biv150_prefill_multi2.py","fc3cdd066aacc6dfdc5029bb8aae9589f868b6800534511d2198f1f0d88da342")]:
        raw=subprocess.check_output(["git","-C",str(root),"show",MAIN+":vllm_fl/patches/"+filename])
        assert hashlib.sha256(raw).hexdigest()==pin
        classes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef)]
        ns={"qfragment":SimpleNamespace(Manager=object,_eligible=prepared.descriptor_guard)}
        exec(compile(ast.Module(body=classes,type_ignores=[]),"<actual-parent-policy>","exec"),ns)
        policies.append(ns["_Policy"])
    capture=[False];parent,reverse,multi2,generic=[Entry(x)for x in ("q32","reverse","multi2","generic")]
    def manager(entry):
        return SimpleNamespace(async_mode=SimpleNamespace(get=lambda:None),active=True,
            raw=SimpleNamespace(pre_run_hooks=[]),candidate=SimpleNamespace(pre_run_hooks=[]),
            device=None,wrapper=entry,optimized={entry.label:{}},attention=SimpleNamespace(torch=SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=lambda:capture[0]))))
    qm,mm=manager(parent),manager(multi2);qp=policies[0](generic,qm);chain=policies[1](qp,mm)
    base_values=values()
    sdk=dict(q=base_values["query_ptr"],k=base_values["key_cache_ptr"],v=base_values["value_cache_ptr"],
        out=base_values["output_ptr"],cu_seqlens_q=base_values["query_start_len_ptr"],
        seqused_k=base_values["seq_lens_ptr"],block_table=base_values["block_tables_ptr"],
        max_seqlen_q=2048,max_seqlen_k=4096)
    nested=[False];nested_results=[]
    def function(**kw):
        if kw["max_seqlen_k"]==16384:
            assert qm.wrapper is parent and host_protocol._CURRENT.get()is None
        if nested[0]:
            nested[0]=False
            inner=CpuContract(kw["q"],kw["cu_seqlens_q"],kw["seqused_k"],(0,2048),True,PACKING,True)
            nested_results.append(binding.call(contract=inner,**kw).hash)
        raw=values(kw["q"].shape[0],kw["seqused_k"].shape[0],raw=True)
        raw.update(query_ptr=kw["q"],key_cache_ptr=kw["k"],value_cache_ptr=kw["v"],output_ptr=kw["out"],
            query_start_len_ptr=kw["cu_seqlens_q"],seq_lens_ptr=kw["seqused_k"],block_tables_ptr=kw["block_table"])
        return chain[(kw["q"].shape[0]//16+raw["num_seqs"],2)](**raw)
    binding=LocalSdkBinding(function,prepared,qm,chain,reverse,lambda:capture[0],lambda:chain.base is qp)
    contract=lambda:CpuContract(sdk["q"],sdk["cu_seqlens_q"],sdk["seqused_k"],(0,2048),True,PACKING,True)
    seen=[]
    with redirect_stdout(io.StringIO()):
        assert function(**sdk).hash=="q32"and qm.wrapper is parent
        with binding.candidate()as invoke:
            for upper,label in [(4096,"reverse"),(16384,"q32"),(4096,"reverse")]:
                sdk["max_seqlen_k"]=upper;got=invoke(contract=contract(),**sdk)
                assert got.hash==label;seen.append(label)
            sdk["max_seqlen_k"]=4096
            previous_capture=binding.capturing
            def no_new_capture_query():raise AssertionError("Long bypass queried new capture/tap scope")
            binding.capturing=no_new_capture_query;sdk["max_seqlen_k"]=16384
            assert invoke(contract=contract(),**sdk).hash=="q32"
            binding.capturing=previous_capture;sdk["max_seqlen_k"]=4096
            once=contract();assert invoke(contract=once,**sdk).hash=="reverse"
            assert invoke(contract=once,**sdk).hash=="q32"
            assert invoke(**sdk).hash=="q32"
            wrong=CpuContract(object(),sdk["cu_seqlens_q"],sdk["seqused_k"],(0,2048),True,PACKING,True)
            assert invoke(contract=wrong,**sdk).hash=="q32"
            nested[0]=True
            assert invoke(contract=contract(),**sdk).hash=="reverse"and nested_results==["q32"]
            before=len(binding.boundary.calls);capture[0]=True
            assert invoke(contract=contract(),**sdk).hash=="generic"
            assert len(binding.boundary.calls)==before;capture[0]=False
            two=values(1026,2)
            kw2=dict(sdk,q=two["query_ptr"],k=two["key_cache_ptr"],v=two["value_cache_ptr"],
                out=two["output_ptr"],cu_seqlens_q=two["query_start_len_ptr"],
                seqused_k=two["seq_lens_ptr"],block_table=two["block_tables_ptr"],
                max_seqlen_q=513,max_seqlen_k=2048)
            contract2=CpuContract(kw2["q"],kw2["cu_seqlens_q"],kw2["seqused_k"],(0,513,1026),True,PACKING,True)
            assert invoke(contract=contract2,**kw2).hash=="multi2"
            assert multi2.gets[-1]==(34,2)and len(binding.boundary.calls)==before
        assert qm.wrapper is parent and binding.restored
        error=RuntimeError("private launch error");reverse.error=error
        try:
            with binding.candidate()as invoke:invoke(contract=contract(),**sdk)
        except RuntimeError as got:assert got is error
        else:raise AssertionError("Launch error swallowed")
        assert qm.wrapper is parent and binding.restored;reverse.error=None
    binding.close()
    return dict(reference_parent=MAIN,explicit_sdk_maxima_local_scope=True,
        sequence=seen,baseline_no_new_tap=True,current_M2_Q32_identity_preserved=True,
        sdk_capture_uses_original_generic_parent_not_Q32=True,capture_claimed_Q32=False,
        stale_missing_wrong_contract_keep_parent=True,exception_restores_original_wrapper=True,
        nested_sdk_call_parent_then_outer_reverse=True,
        long_miss_has_no_new_frame_tap_or_capture_query=True,
        unchanged_M2_bypasses_single_sequence_boundary_grid=[34,2],
        boundary_records=binding.boundary.calls,outer_forward_context_hook_installed=False,
        real_framework_imported=False,device_verified=False)


def run_cpu(source,sdk,helpers,target):
    for name,pin in [("biv150_prefill_host_component.py","3cf4023500b32bf743f4c546fb60008d4d91c007f07dcc49ac9da7b79932a8bc"),
                     ("check_biv150_prefill_host_component_cpu.py","1310110d16eef94e0ab76b2a1822f50edea1759d6f9c75414b2c4f92769d1a35")]:
        assert hashlib.sha256((source/"tools"/name).read_bytes()).hexdigest()==pin
    prepared=prepare_source(sdk.read_text(),helpers.read_text(),source)
    with target_source_directory(target)as actual_target:
        chain=source_chain(actual_target,source)
    report=dict(cpu_only=True,gpu_executed=False,jit_executed=False,service_installed=False,
        host_binding_device_verified=False,typed_abi_device_verified=False,
        host_binding_Q_hoist_device_verified=False,performance_verified=False,
        model_precision_gate=False,official_score=False,source=prepared.proof,
        target_source_chain=chain,local_binding=local_binding(source,prepared))
    report["tool_sha256"]={name:hashlib.sha256((source/"tools"/name).read_bytes()).hexdigest()
        for name in ("biv150_prefill_host_binding.py","check_biv150_prefill_host_binding_cpu.py",
                     "bench_biv150_prefill_host_binding.py")}
    assert "torch"not in sys.modules and "triton"not in sys.modules
    return report,prepared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--sdk-source",type=Path,required=True)
    p.add_argument("--helpers-source",type=Path,required=True);p.add_argument("--target-sources",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    report,_=run_cpu(a.source,a.sdk_source,a.helpers_source,a.target_sources)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/"cpu.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(passed=True,cpu_only=True,target_source_chain_complete=True,
        local_sdk_route_sequence=report["local_binding"]["sequence"])))


if __name__=="__main__":main()
