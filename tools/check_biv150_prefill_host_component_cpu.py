#!/usr/bin/env python3
"""Execute host component/source/real-parent-policy CPU checks; no device entry."""
import argparse
import ast
from contextlib import redirect_stdout
from dataclasses import replace
from enum import IntEnum
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from biv150_prefill_host_component import (
    BOUNDARY, PACKING, PROVENANCE, Call, ComponentScope, CpuMetadata, Router,
    cpu_metadata, prepare_source,
)

MAIN = "56f1838192abc940b3fb9accef490022ac125000"
DEVICE = SimpleNamespace(type="cuda", index=0)
FORBIDDEN_TOUCHES = []


class Tensor:
    def __init__(self, shape, strides, address, dtype="torch.bfloat16"):
        self.shape, self.strides, self.address, self.dtype = shape, strides, address, dtype
        self.ndim, self.device = len(shape), DEVICE

    def stride(self, axis=None):
        return self.strides if axis is None else self.strides[axis]

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.address)

    def __eq__(self, other):
        FORBIDDEN_TOUCHES.append("tensor_equality")
        raise AssertionError("Tensor equality must not bind metadata")

    def item(self):
        FORBIDDEN_TOUCHES.append("scalar_read")
        raise AssertionError("GPU scalar read forbidden")

    cpu = numpy = item


class Entry:
    def __init__(self, label):
        self.label, self.gets, self.calls, self.error = label, [], [], None

    def __getitem__(self, grid):
        self.gets.append(grid)
        def invoke(*args, **kwargs):
            self.calls.append((args, kwargs))
            if self.error is not None:
                raise self.error
            return SimpleNamespace(hash=self.label)
        return invoke


def values(rows=2048, sequences=1, raw=False):
    return dict(IS_3D=False, HEAD_SIZE=128, HEAD_SIZE_PADDED=128,
        num_query_heads=16, num_queries_per_kv=8, BLOCK_M=128 if raw else 256,
        BLOCK_Q=16 if raw else 32, BLOCK_SIZE=16, TILE_SIZE=64,
        NUM_SEGMENTS_PER_SEQ=1, KV_QUANT_MODE=0, USE_CAUSAL=True,
        SLIDING_WINDOW=0, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1,
        num_warps=8 if raw else 16, num_stages=1, num_seqs=sequences,
        query_ptr=Tensor((rows,16,128),(2048,128,1),1),
        output_ptr=Tensor((rows,16,128),(2048,128,1),2),
        key_cache_ptr=Tensor((1024,16,2,128),(8192,256,128,1),3),
        value_cache_ptr=Tensor((1024,16,2,128),(8192,256,128,1),3),
        query_start_len_ptr=Tensor((sequences+1,),(1,),4,"torch.int32"),
        seq_lens_ptr=Tensor((sequences,),(1,),5,"torch.int32"),
        block_tables_ptr=Tensor((sequences,8192),(8192,1),6,"torch.int32"))


def scope(parent, reverse, chain):
    return ComponentScope(BOUNDARY, PACKING, "NONE", True, True, True, True,
                          chain, parent, reverse)


def metadata(call, q=2048, upper=4096, **changes):
    data = CpuMetadata(call.stamp, call.kwargs["query_ptr"], q, upper, (0,q),
                       True, PROVENANCE, False)
    return replace(data, **changes) if changes else data


def call(parent, chain, rows=2048):
    return Call(parent, (), [rows//32+1,2], values(rows), chain)


def assert_payload(original, selected):
    assert selected.args is original.args and selected.grid is original.grid
    assert selected.kwargs is original.kwargs and selected.parent_chain is original.parent_chain
    assert selected.stamp is original.stamp


def check_router(prepared):
    parent, reverse, chain = Entry("parent"), Entry("reverse"), object()
    router = Router(prepared, scope(parent, reverse, chain))
    assert router.enabled
    cases = []
    for q, upper, hit in [(1023,4096,False),(1024,1023,False),(1024,1024,True),
                          (1024,5119,True),(1024,5120,True),(1024,5121,False),
                          (1025,4096,True),(2048,4096,True),(2048,10240,True),
                          (2048,10241,False),(2048,16384,False),
                          (4096,16384,True),(16384,16384,True)]:
        original = call(parent, chain, q)
        before = tuple((k,id(v)) for k,v in original.kwargs.items())
        with cpu_metadata(metadata(original,q,upper)):
            selected = router.select(original)
            result=selected.run(); assert result.hash == ("reverse" if hit else "parent")
        assert (selected is not original) is hit
        assert selected.entry is (reverse if hit else parent)
        assert_payload(original,selected)
        assert tuple((k,id(v)) for k,v in original.kwargs.items()) == before
        assert selected.entry.gets[-1] is original.grid
        assert all(selected.entry.calls[-1][1][k] is v for k,v in original.kwargs.items())
        cases.append(dict(query=q,upper=upper,reverse=hit))
    rejected=[]
    original=call(parent,chain)
    bad = {
        "q_bool":dict(qmax=True),"q_float":dict(qmax=2048.0),
        "upper_float":dict(kv_upper=4096.0),"upper_tensor":dict(kv_upper=original.kwargs["query_ptr"]),
        "upper_huge":dict(kv_upper=2**100),"unknown_bound":dict(upper_bound_proven=None),
        "missing_provenance":dict(provenance=None),"unknown_provenance":dict(provenance="service-guess"),
        "provenance_tensor":dict(provenance=original.kwargs["query_ptr"]),
        "capture":dict(capturing=True),"unknown_capture":dict(capturing=None),
        "stale_stamp":dict(stamp=object()),"wrong_query":dict(query=values()["query_ptr"]),
        "nonzero_start":dict(qmax=2048,cu_q=(1,2049)),"missing_cu":dict(cu_q=None),
        "cu_gpu_tensor":dict(cu_q=original.kwargs["query_start_len_ptr"]),
        "cu_tensor_element":dict(cu_q=(0,original.kwargs["query_ptr"])),
        "padded_query":dict(qmax=1024,cu_q=(0,1024)),
    }
    for name, change in bad.items():
        with cpu_metadata(metadata(original,**change)):
            assert router.select(original) is original
        rejected.append(name)
    for name, bad_meta in [("missing",None),("dict",{}),("opaque",object())]:
        with cpu_metadata(bad_meta):assert router.select(original) is original
        rejected.append(name)
    for name, changes in [("multi2",dict(num_seqs=2)),("zero_seq",dict(num_seqs=0)),
        ("wrong_tile",dict(TILE_SIZE=128)),("ir_override",dict(ir_override="no")),
        ("old_sdk_geometry",dict(BLOCK_M=128,BLOCK_Q=16,num_warps=8)),
        ("tensor_constexpr",dict(BLOCK_M=original.kwargs["query_ptr"])),
        ("alias_output",dict(output_ptr=original.kwargs["query_ptr"]))]:
        wrong=replace(original,kwargs=dict(original.kwargs,**changes))
        with cpu_metadata(metadata(wrong)):assert router.select(wrong) is wrong
        rejected.append(name)
    for name, wrong in [("positional",replace(original,args=(original.kwargs["query_ptr"],))),
                        ("grid",replace(original,grid=[64,2])),
                        ("wrong_parent",replace(original,parent_chain=object())),
                        ("wrong_entry",replace(original,entry=Entry("other"))),
                        ("bad_stamp_type",replace(original,stamp="constant-label"))]:
        with cpu_metadata(metadata(wrong)):assert router.select(wrong) is wrong
        rejected.append(name)
    base_scope=scope(parent,reverse,chain)
    for field,value in [("boundary","SDK-raw"),("packing",None),("packing","V8-bf16-view"),
                        ("runtime_mode","FULL"),("component_only",False),
                        ("hooks_empty",False),("async_mode_none",None),
                        ("snapshot_coherent_for_launch",False),("parent_chain",None),
                        ("reverse_entry",None),("reverse_entry",object()),
                        ("reverse_entry",parent)]:
        disabled=Router(prepared,replace(base_scope,**{field:value}))
        with cpu_metadata(metadata(original)):assert disabled.select(original) is original
        label=str(value) if type(value) in (str,bool,type(None)) else type(value).__name__
        rejected.append("scope_"+field+"_"+label)
    with cpu_metadata(metadata(original)):assert Router(prepared).select(original) is original
    class Mode(IntEnum):NONE=0
    enum_call=replace(original,kwargs=dict(original.kwargs,KV_QUANT_MODE=Mode.NONE))
    with cpu_metadata(metadata(enum_call)):assert router.select(enum_call).entry is reverse
    assert router.select(original) is original
    once=metadata(original)
    with cpu_metadata(once):
        assert router.select(original).entry is reverse
        assert router.select(original) is original
    with cpu_metadata(once):assert router.select(original) is original
    revoked=Router(prepared,base_scope)
    with cpu_metadata(metadata(original)):
        pending=revoked.select(original);revoked.invalidate();before=len(reverse.gets)
        assert pending.run().hash=="parent" and len(reverse.gets)==before
    with cpu_metadata(metadata(original)):assert revoked.select(original) is original
    with cpu_metadata(metadata(original)):stored=router.select(original)
    before=len(reverse.gets)
    assert stored.run().hash=="parent" and len(reverse.gets)==before
    try:stored.run()
    except RuntimeError:pass
    else:raise AssertionError("Expired decision was reused")
    outer=metadata(original)
    with cpu_metadata(outer):
        with cpu_metadata(metadata(original)):assert router.select(original) is original
        assert router.select(original).entry is reverse
    try:
        with cpu_metadata(metadata(original)):raise RuntimeError("context body")
    except RuntimeError:pass
    assert router.select(original) is original
    # A long-context miss cannot touch the reverse entry or even its Q32
    # descriptor guard; it returns the caller's old invocation directly.
    def forbidden_guard(*a,**kw):raise AssertionError("Long context entered candidate guard")
    long_router=Router(replace(prepared,descriptor_guard=forbidden_guard),base_scope)
    before=len(reverse.gets)
    with cpu_metadata(metadata(original,upper=16384)):
        assert long_router.select(original) is original
    assert len(reverse.gets)==before
    # No per-call source reads/hashes/Git/signature binding; setup is over.
    with patch.object(Path,"read_bytes",side_effect=AssertionError("hot read")), \
         patch.object(Path,"read_text",side_effect=AssertionError("hot read")), \
         patch.object(hashlib,"sha256",side_effect=AssertionError("hot hash")), \
         patch.object(subprocess,"check_output",side_effect=AssertionError("hot Git")), \
         patch.object(inspect,"signature",side_effect=AssertionError("hot bind")):
        with cpu_metadata(metadata(original)):assert router.select(original).entry is reverse
        with cpu_metadata(metadata(original,upper=16384)):assert router.select(original) is original
    for target, upper in [(parent,16384),(reverse,4096)]:
        error=RuntimeError(target.label+" launch failure");target.error=error
        other=reverse if target is parent else parent; before=len(other.gets)
        try:
            with cpu_metadata(metadata(original,upper=upper)):router.run(original)
        except RuntimeError as got:assert got is error
        else:raise AssertionError("Launch error swallowed")
        assert len(other.gets)==before;target.error=None
    return dict(cases=cases,rejected=rejected,parent_identity_and_payload_preserved=True,
        tensor_equality_or_scalar_read=False,one_shot_stale_missing_nested_checked=True,
        explicit_scope_revocation_before_selection_or_launch=True,
        long_context_new_device_entry_touches=0,whole_host_cost_claimed_zero=False,
        full_batch_upper_bound_conservative_miss=dict(query=2048,actual_kv=4096,upper=16384,reverse=False),
        launch_errors_propagate_without_retry=True,hot_path_io_hash_git_bind=False)


def parent_policy_check(root,prepared):
    specs=[("biv150_prefill_q32.py","3906f34e89b814cafa94ef04a0f268ffc3cf237b99ecb011a62c9abfef56878b"),
           ("biv150_prefill_multi2.py","fc3cdd066aacc6dfdc5029bb8aae9589f868b6800534511d2198f1f0d88da342")]
    policies=[]
    for filename,pin in specs:
        raw=subprocess.check_output(["git","-C",str(root),"show",MAIN+":vllm_fl/patches/"+filename])
        assert hashlib.sha256(raw).hexdigest()==pin
        classes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef)]
        ns={"qfragment":SimpleNamespace(Manager=object,_eligible=prepared.descriptor_guard)}
        exec(compile(ast.Module(body=classes,type_ignores=[]),"<actual-main-"+filename+">","exec"),ns)
        policies.append(ns["_Policy"])
    baseline,reverse,multi2,generic=Entry("q32"),Entry("reverse"),Entry("multi2"),Entry("generic")
    def manager(entry):
        return SimpleNamespace(async_mode=SimpleNamespace(get=lambda:None),active=True,
            raw=SimpleNamespace(pre_run_hooks=[]),candidate=SimpleNamespace(pre_run_hooks=[]),
            device=None,wrapper=entry,optimized={entry.label:{}},
            attention=SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda:False))))
    qmanager,mmanager=manager(baseline),manager(multi2)
    qpolicy=policies[0](generic,qmanager);chain=policies[1](qpolicy,mmanager)
    router=Router(prepared,scope(baseline,reverse,chain))
    for upper,expected in [(4096,"reverse"),(16384,"q32")]:
        stamp=object();raw=values(raw=True)
        class ComponentBoundary:
            def __getitem__(self,grid):
                def invoke(*args,**kwargs):
                    original=Call(baseline,args,grid,kwargs,chain,stamp)
                    return router.run(original)
                return invoke
        qmanager.wrapper=ComponentBoundary()
        meta=CpuMetadata(stamp,raw["query_ptr"],2048,upper,(0,2048),True,PROVENANCE,False)
        with redirect_stdout(io.StringIO()),cpu_metadata(meta):
            assert chain[(129,2)](**raw).hash==expected
        chosen=reverse if expected=="reverse"else baseline
        assert chosen.gets[-1]==(65,2) and chosen.calls[-1][1]["BLOCK_M"]==256
        assert raw["BLOCK_M"]==128 and chain.base is qpolicy and qpolicy.manager is qmanager
    # The main M2 parent remains outermost. Its two-sequence case bypasses the
    # single-Q32 boundary and retains SDK rows//32+num_sequences, including tails.
    raw=values(1026,2,raw=True);before=len(reverse.gets)
    meta=CpuMetadata(object(),raw["query_ptr"],513,16384,(0,513,1026),True,PROVENANCE,False)
    with redirect_stdout(io.StringIO()),cpu_metadata(meta):assert chain[(66,2)](**raw).hash=="multi2"
    assert multi2.gets[-1]==(34,2) and len(reverse.gets)==before
    return dict(reference_parent=MAIN,actual_q32_and_multi2_class_ast=True,
        original_outer_m2_identity_preserved=True,raw_sdk_BM128_parent_selected_BM256=True,
        short_reverse_and_long_original_q32=True,multi2_513_513_grid=[34,2],
        no_install_or_real_framework_import=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--sdk-source",type=Path,required=True)
    p.add_argument("--helpers-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();prepared=prepare_source(a.sdk_source.read_text(),a.helpers_source.read_text(),a.source)
    report=dict(cpu_only=True,gpu_executed=False,jit_executed=False,service_installed=False,
        host_component_numerical_verified=False,host_component_performance_verified=False,
        model_precision_gate=False,official_score=False,source_proof=prepared.proof,
        router=check_router(prepared),parent_policy=parent_policy_check(a.source,prepared))
    report["tool_sha256"]={name:hashlib.sha256((a.source/"tools"/name).read_bytes()).hexdigest()
        for name in ("biv150_prefill_host_component.py","check_biv150_prefill_host_component_cpu.py")}
    assert "torch"not in sys.modules and "triton"not in sys.modules and FORBIDDEN_TOUCHES==[]
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"baseline.py").write_text(prepared.baseline);(a.out/"reverse.py").write_text(prepared.reverse)
    (a.out/"cpu.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(passed=True,cpu_only=True,boundary_cases=len(report["router"]["cases"]),
        rejected_cases=len(report["router"]["rejected"]),source_sha256=prepared.proof["reversed_source_sha256"])))


if __name__=="__main__":main()
