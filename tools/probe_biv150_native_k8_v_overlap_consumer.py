#!/usr/bin/env python3
"""Reuse the unchanged four-case consumer after actual V-overlap SO review.

The actual SO wait must follow QK and P softmax/packing; the compiler may
schedule rho exp and running-L FMA after it. This describes the machine-code
review boundary, not an additional change to CUDA or the numeric consumer.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import biv150_native_k8_v_overlap_factory as factory
from compile_biv150_native_k8_v_overlap import CUDA_FLAGS

PARENT='3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'
PINS={
 'biv150_konly_decode_quant_reference':'38807ab17bf28e64afee6fb581af92d4e715856d954dc42db4fdd1dd6d91bc18',
 'biv150_konly_i8_qk_reference':'63d8f901b3648dad52f9dceab251e1d451def2aaa96f2bc30bd0aaff102e64e3',
 'biv150_native_k8_codec_fixture':'490034bfe77b69a22d6109a03f422b60883e7791bc0c48eceecb44c31019044d',
 'biv150_native_k8_codec_receipt':'e02e6668d24258b81c0d339a04015673983c0633c7e2b1b7b7a1ba8d79914788',
 'biv150_native_k8_consumer_fixture':'1448f8e5bc6efa3941965a2bae9268f4eba48cc7468104871abb65a5d029b11c',
 'biv150_native_k8_consumer_v2_fixture':'de650b4e1568a8a5288b6a19e0e53870bba257270ffbe365f5298ac75feec037',
 'biv150_native_k8_pagewise_reference':'ac23913bdfa3f59025d2e66d09eee48da176f14fa8c41226e159bce1b9f5898a',
 'probe_biv150_native_k8_consumer_v2':'4d83297d4a1848faf1e880895c485ecac144df48f8800940e66986f6f70a14ac',
}
REDUCER='vllm_fl/patches/biv150_gqa_decode.py'
REDUCER_SHA='f1877ad06e8be4cef5960ef2d66d59045b4ac1ac0d7f2fe448618bbdec0d7f40'
REVIEW_FLAGS=('actual_K_then_V_issue_order_preserved','actual_V_wait_after_QK_and_P_softmax_packing',
              'actual_wait_tail_clear_barrier_before_PV','original_QK_softmax_BF16_PV_preserved',
              'actual_SO_codec_equivalent_to_original22e','resources_reviewed')
CASES=((64,2,2048),(64,2,2560),(32,4,2048),(32,4,2560))


def sha(path):
    return factory.sha(Path(path).read_bytes())


def parent_files(root):
    assert root.is_absolute() and root.is_dir(),'explicit frozen parent checkout required'
    root=root.resolve()
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==PARENT
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    for name,digest in PINS.items():
        path=root/'tools'/(name+'.py');assert sha(path)==digest,name
        if name in sys.modules:
            assert Path(sys.modules[name].__file__).resolve()==path.resolve(),'foreign loaded module: '+name
    assert sha(root/REDUCER)==REDUCER_SHA
    return root


def adapted_run(root):
    text=(parent_files(root)/'tools/probe_biv150_native_k8_consumer_v2.py').read_text()
    fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    original=ast.get_source_segment(text,fn);changed=original
    edits=(('f.sha(args.library.read_bytes())==f.LIBRARY_SHA',
            'f.sha(args.library.read_bytes())==args.variant_sha256',2),
           ('torch.ops.cjj_native_k8_qk_probe.produce',
            'torch.ops.cjj_native_k8_v_overlap_probe.produce',1),
           ('result=dict(f.plan(),codec_precondition=receipt,',
            'result=dict(variant_plan(),codec_precondition=receipt,',1))
    for old,new,count in edits:
        assert changed.count(old)==count;changed=changed.replace(old,new)
    inverse=changed
    for old,new,count in reversed(edits):
        assert inverse.count(new)==count;inverse=inverse.replace(new,old)
    assert inverse==original
    return changed,dict(parent_commit=PARENT,parent_run_sha256=factory.sha(original.encode()),
                       adapter_run_sha256=factory.sha(changed.encode()),inverse_byte_exact=True,
                       edits=[dict(old=a,new=b,count=n) for a,b,n in edits])


def compile_binding(path):
    assert path.is_absolute() and path.is_file()
    path=path.resolve();j=json.loads(path.read_text());proof=factory.verify()
    assert j['completed'] is True and j['compile_passed'] is True and not j['errors']
    assert j['source_transform']==proof and j['explicit_cuda_flags']==CUDA_FLAGS
    for key in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'):
        assert j[key]==0
    files={}
    for row in j['artifact_manifest']:
        name=Path(row['relative']);assert not name.is_absolute() and '..' not in name.parts
        assert str(name) not in files
        p=path.parent/name;assert p.resolve().is_relative_to(path.parent)
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];files[str(name)]=row
    for role in factory.FILES:
        p=factory.path(role);assert (path.parent/p.name).read_bytes()==p.read_bytes()
    for name in ('biv150_native_k8_v_overlap_qk.ll','biv150_native_k8_v_overlap_qk.s'):
        assert files[name]['bytes']>0
    library=Path(j['library']['path']);assert library.is_absolute()
    assert library.resolve().is_relative_to(path.parent)
    assert library.name==factory.NAMESPACE+'.so'
    rel=str(library.resolve().relative_to(path.parent))
    assert files[rel]['sha256']==j['library']['sha256']==sha(library)
    return dict(compile_summary_sha256=sha(path),library_sha256=sha(library),
                namespace=factory.NAMESPACE,source_proof=proof)


def review_template(binding):
    return dict(version=1,reviewer='CJJ1008',**binding,
                **{key:False for key in REVIEW_FLAGS},
                actual_SO_disassembly={kind:dict(path=None,sha256=None) for kind in ('IR','SASS')},
                resources={str(s):dict(vgpr=None,sgpr=None,shared_bytes=None,spill_bytes=None) for s in (2,4)},
                notes='UNREVIEWED: actual SO full/tail ordering, codec and resource review required; '
                      'wait follows QK and P softmax/packing; rho exp and running-L FMA may follow wait')


def review_fields(review):
    assert review['version']==1 and review['reviewer']=='CJJ1008'
    for key in REVIEW_FLAGS:assert review[key] is True,'unreviewed actual SO: '+key
    assert set(review['resources'])=={'2','4'}
    for row in review['resources'].values():
        assert type(row['vgpr']) is int and row['vgpr']>0
        assert type(row['sgpr']) is int and row['sgpr']>0
        assert type(row['shared_bytes']) is int and row['shared_bytes']==8192
        assert type(row['spill_bytes']) is int and row['spill_bytes']==0
    assert type(review['notes']) is str and len(review['notes'])>=20 and not review['notes'].startswith('UNREVIEWED')


def admission(a):
    assert a.root_owned_exclusive,'Root must confirm prior owner terminal and exclusive device'
    binding=compile_binding(a.compile_summary)
    assert re.fullmatch('[0-9a-f]{64}',a.assembly_review_sha256 or '')
    assert a.assembly_review.is_absolute() and sha(a.assembly_review)==a.assembly_review_sha256
    review=json.loads(a.assembly_review.read_text())
    for key,value in binding.items():assert review[key]==value,'review binding: '+key
    review_fields(review)
    assert set(review['actual_SO_disassembly'])=={'IR','SASS'}
    for item in review['actual_SO_disassembly'].values():
        path=Path(item['path']);assert path.is_absolute() and path.is_file() and path.stat().st_size>0
        assert re.fullmatch('[0-9a-f]{64}',item['sha256']) and sha(path)==item['sha256']
    assert a.library.is_absolute() and sha(a.library)==binding['library_sha256']
    assert a.codec_proof_sha256=='3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263'
    assert a.codec_summary.is_absolute() and a.out.is_absolute() and not a.out.exists()
    parent=parent_files(a.parent_source)
    assert not a.out.resolve().is_relative_to(parent) and not a.out.resolve().is_relative_to(factory.ROOT)
    return parent,dict(**binding,assembly_review_sha256=a.assembly_review_sha256,review=review,
                       external_exclusion_authority='Root serial scheduling; no namespace-based idle inference')


def execute(a):
    parent,binding=admission(a)  # All source/receipt gates precede framework imports.
    run_source,adapter=adapted_run(parent)
    sys.path.insert(0,str(parent/'tools'))
    name='cjj_native_k8_v_overlap_parent_consumer';assert name not in sys.modules
    spec=importlib.util.spec_from_file_location(name,parent/'tools/probe_biv150_native_k8_consumer_v2.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    parent_files(parent);assert module.f.CASES==CASES
    a.variant_sha256=binding['library_sha256']
    def variant_plan():
        plan=module.f.plan();plan['library_sha256']=a.variant_sha256
        plan.update(kernel_or_dtype_changed=True,numeric_expression_or_dtype_changed=False,
                    variant=dict(compile_admission=binding,host_adapter=adapter,
                                 source_proof=factory.verify(),qualification_inherited_from_old_reader=False))
        return plan
    namespace=dict(vars(module),variant_plan=variant_plan)
    exec(compile(run_source,'<V-overlap-exact-parent-consumer-adapter>','exec'),namespace)
    return namespace['run'](a)


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--gpu',action='store_true');mode.add_argument('--review-template',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true')
    for key in ('parent-source','compile-summary','assembly-review','library','codec-summary','out'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--assembly-review-sha256');p.add_argument('--codec-proof-sha256')
    a=p.parse_args()
    if a.gpu:
        if not all((a.root_owned_exclusive,a.parent_source,a.compile_summary,a.assembly_review,
                    a.assembly_review_sha256,a.library,a.codec_summary,a.codec_proof_sha256,a.out)):
            p.error('GPU needs explicit Root ownership, frozen parent, actual compile/library/review and codec receipts')
        return execute(a)
    if a.review_template:
        if not a.compile_summary or any((a.root_owned_exclusive,a.parent_source,a.assembly_review,a.library,
                                         a.codec_summary,a.out,a.assembly_review_sha256,a.codec_proof_sha256)):
            p.error('Review template accepts only an actual compile summary')
        result=review_template(compile_binding(a.compile_summary))
    else:
        if any(v for k,v in vars(a).items() if k not in ('gpu','review_template')):
            p.error('Device arguments require --gpu or --review-template')
        result=dict(source_proof=factory.verify(),parent_commit=PARENT,consumer_module_pins=PINS,
                    cases=CASES,review_required=REVIEW_FLAGS,readonly_CPU_default=True,
                    new_numerical_framework=False,precision_gate=False,GPU_executed=False)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(result,sort_keys=True,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
