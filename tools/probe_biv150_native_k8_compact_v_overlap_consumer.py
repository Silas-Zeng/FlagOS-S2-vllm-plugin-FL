#!/usr/bin/env python3
"""Identity-only PR100 consumer adapter; default CPU, no new numeric implementation."""
import argparse
import ast
import importlib
import inspect
import json
from pathlib import Path
import sys
import types

import biv150_native_k8_compact_v_overlap_factory as candidate

PARENT='51da6fd73e7fff1513081fb330323ce72c00856c'
EXTRA_REVIEW=('actual_K_then_V_issue_order_preserved','actual_V_wait_after_QK_and_P_softmax_packing')
SELF='probe_biv150_native_k8_compact_v_overlap_consumer.py'


def frozen_module(name):
    path=candidate.ROOT/'tools'/(name+'.py')
    assert path.read_bytes()==candidate.blob(PARENT,'tools/'+path.name)
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve()==path.resolve()
    return module


def replace_exact(original,edits):
    changed=original
    for old,new in edits:
        assert changed.count(old)==1,old
        changed=changed.replace(old,new)
    restored=changed
    for old,new in reversed(edits):
        assert restored.count(new)==1,new
        restored=restored.replace(new,old)
    assert restored==original
    ast.parse(changed)
    return changed,dict(parent_sha256=candidate.sha(original.encode()),
        adapted_sha256=candidate.sha(changed.encode()),inverse_byte_exact=True,
        edits=[dict(old=a,new=b) for a,b in edits])


def source_pins():
    return {name:candidate.sha((candidate.ROOT/'tools'/name).read_bytes()) for name in
        (SELF,'biv150_native_k8_compact_v_overlap_factory.py','compile_biv150_native_k8_compact_v_overlap.py')}


def modules():
    old=frozen_module('probe_biv150_native_k8_compact_consumer')
    original=frozen_module('biv150_native_k8_compact_consumer_contract')
    # Keep the compact layout/LiveEpoch and frozen mathematical fixture API.
    factory=types.SimpleNamespace(**dict(vars(old.factory)))
    factory.verify=candidate.verify;factory.path=candidate.path
    factory.NAMESPACE=candidate.NAMESPACE;factory.FILES=candidate.FILES
    binding_source,binding_proof=replace_exact(inspect.getsource(original.compile_binding),[
        ('import probe_biv150_native_k8_compact as frontend',
         'import compile_biv150_native_k8_compact_v_overlap as frontend'),
        ('frontend.build_module().CUDA_FLAGS','frontend.CUDA_FLAGS')])
    ns=dict(vars(original),f=factory,REVIEW_FLAGS=original.REVIEW_FLAGS+EXTRA_REVIEW)
    exec(compile(binding_source,'<compact-v-overlap-existing-compile-binding>','exec'),ns)
    for name in ('review_template','admission'):
        fn=getattr(original,name);ns[name]=types.FunctionType(fn.__code__,ns,name)
    contract=types.SimpleNamespace(**ns)
    return old,factory,contract,binding_proof


def replay_frontend():
    old,factory,_,_=modules()
    return types.SimpleNamespace(factory=factory,compact_bytes=old.compact_bytes)


def adapted_run():
    old,_,_,_=modules()
    return replace_exact(inspect.getsource(old.run),[
        ('torch.ops.cjj_native_k8_compact_probe.produce',
         'torch.ops.cjj_native_k8_compact_v_overlap_probe.produce'),
        ("('probe_biv150_native_k8_compact_consumer.py','biv150_native_k8_compact_consumer_contract.py')",
         "('probe_biv150_native_k8_compact_consumer.py','biv150_native_k8_compact_consumer_contract.py','"+SELF+"','biv150_native_k8_compact_v_overlap_factory.py','compile_biv150_native_k8_compact_v_overlap.py')")])


def plan():
    old,factory,contract,binding_proof=modules()
    result=types.FunctionType(old.plan.__code__,dict(vars(old),factory=factory),'plan')()
    result['component']='compact12544 deferred-V-wait vs original22e full O/M/L/final; not service'
    result['variant_adapter']=dict(parent=PARENT,run=adapted_run()[1],compile_binding=binding_proof,
        frontend_sources=source_pins(),required_actual_review_flags=list(contract.REVIEW_FLAGS),
        qualification_inherited_from40f9_or8b=False)
    return result


def review_template(summary):
    contract=modules()[2]
    result=contract.review_template(summary)
    result['notes']+='; actual QK and P softmax/packing precede wait; rho exp/running-L FMA may follow it'
    return result


def execute(a):
    assert a.root_owned_exclusive
    old,factory,contract,_=modules()
    source,_=adapted_run()
    ns=dict(vars(old),factory=factory,contract=contract,plan=plan)
    exec(compile(source,'<compact-v-overlap-unchanged-four-case-run>','exec'),ns)
    return ns['run'](a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--gpu',action='store_true');mode.add_argument('--review-template',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true')
    for name in ('out','compile-summary','assembly-review','compact-library','original-library','codec-summary'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--assembly-review-sha256');p.add_argument('--codec-proof-sha256')
    a=p.parse_args()
    if a.gpu:
        if not all((a.root_owned_exclusive,a.out,a.compile_summary,a.assembly_review,a.assembly_review_sha256,a.compact_library,a.original_library,a.codec_summary,a.codec_proof_sha256)):
            p.error('Root ownership and all actual compile/review/library/codec receipts required')
        if not all(v.is_absolute() for v in vars(a).values() if isinstance(v,Path)):
            p.error('Absolute paths required')
        return execute(a)
    if a.review_template:
        if not a.compile_summary or any((a.root_owned_exclusive,a.out,a.assembly_review,a.assembly_review_sha256,a.compact_library,a.original_library,a.codec_summary,a.codec_proof_sha256)):
            p.error('Review template accepts only actual compile summary plus final SO inspection')
        result=review_template(a.compile_summary)
    else:
        if any(v for k,v in vars(a).items() if k not in ('gpu','review_template')):p.error('Device arguments require --gpu or --review-template')
        result=plan()
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
