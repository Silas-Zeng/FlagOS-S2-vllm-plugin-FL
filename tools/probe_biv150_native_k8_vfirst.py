#!/usr/bin/env python3
"""CPU plan default. Root opts into new compile, reviewed numeric, then fullstep."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys
import biv150_native_k8_vfirst_factory as factory


def load_module(name):
    p=factory.ROOT/'tools'/name;spec=importlib.util.spec_from_file_location('cjj_vfirst_'+p.stem,p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def plan():
    return dict(component='private V-first K8 issue-order experiment',source_proof=factory.verify(),
        phases=['compile-only no launches','Root CPU actual LLVM/ASM review','fresh variant consumer','fresh variant vs b3 BF16 fullstep'],
        root_owned_exclusive_required=True,actual_scheduling_verified=False,numeric_verified=False,
        timing_executed=False,model_verified=False,precision_gate=False,service_installed=False,capacity_reclaimed=False,
        default_imports_framework=False,parent_library_is_not_variant_qualification=True)

def compile_record(path):
    path=Path(path).resolve();raw=path.read_bytes();j=json.loads(raw)
    assert j['completed'] and j['compile_passed'] and not j['errors']
    assert j['vfirst_source_proof']==factory.verify()
    assert all(j[k]==0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    artifacts={}
    for row in j['artifact_manifest']:
        name=Path(row['relative']);assert not name.is_absolute() and '..' not in name.parts
        p=path.parent/name;assert p.resolve().is_relative_to(path.parent)
        b=p.read_bytes();assert len(b)==row['bytes'] and factory.sha(b)==row['sha256'];artifacts[str(name)]=row
    for role in factory.FILES:
        local=factory.path(role);assert (path.parent/local.name).read_bytes()==local.read_bytes()
    header=factory.ROOT/'tools/kernels/biv150_native_k8_codec.cuh'
    assert (path.parent/header.name).read_bytes()==header.read_bytes()
    assert j['explicit_cuda_flags']==load_module('compile_biv150_native_k8_qk.py').CUDA_FLAGS
    ll='biv150_native_k8_vfirst_qk.ll';asm='biv150_native_k8_vfirst_qk.s'
    assert ll in artifacts and asm in artifacts and artifacts[ll]['bytes'] and artifacts[asm]['bytes']
    assert re.fullmatch('[0-9a-f]{64}',j['library']['sha256'])
    return dict(compile_summary_sha256=factory.sha(raw),variant_library_sha256=j['library']['sha256'],
        qk_LLVM_sha256=artifacts[ll]['sha256'],qk_ASM_sha256=artifacts[asm]['sha256'],
        variant_CUDA_sha256=factory.verify()['files']['qk.cu']['variant_sha256'])

def review_template(path):
    return dict(version=1,**compile_record(path),reviewer='CJJ1008',
        actual_V_issue_before_K_packing=False,actual_wait_barrier_tail_preserved=False,
        original_I8_QK_BF16_PV_and_codec_preserved=False,
        notes='UNREVIEWED: inspect the actual full/tail LLVM and ASM; source order alone is insufficient')

def admitted(a):
    assert a.root_owned_exclusive,'Root must coordinate the existing job terminal state and exclusive device'
    assert a.compile_summary and a.assembly_review and a.assembly_review_sha256 and a.variant_library
    binding=compile_record(a.compile_summary);raw=a.assembly_review.read_bytes()
    assert factory.sha(raw)==a.assembly_review_sha256;review=json.loads(raw)
    for k,v in binding.items():assert review[k]==v
    assert review['version']==1 and review['reviewer']=='CJJ1008'
    for k in ('actual_V_issue_before_K_packing','actual_wait_barrier_tail_preserved','original_I8_QK_BF16_PV_and_codec_preserved'):
        assert review[k] is True,'actual compile mechanism remains unreviewed: '+k
    assert type(review['notes']) is str and len(review['notes'])>=20 and not review['notes'].startswith('UNREVIEWED')
    assert factory.sha(a.variant_library.read_bytes())==binding['variant_library_sha256']
    a.variant_sha256=binding['variant_library_sha256']
    return dict(**binding,review_sha256=a.assembly_review_sha256,review=review,
                exclusion_authority='Root external serial scheduling; no process-namespace inference')

def variant_receipt(path,expected_sha,variant_sha):
    # Reuse every saved-array, exact-input, guard, metric and Graph test; only
    # replace the immutable parent run/library identities by this new run.
    m=load_module('biv150_native_k8_consumer_receipt.py')
    original=factory.blob('tools/biv150_native_k8_consumer_receipt.py').decode()
    fn=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='require_consumer')
    source=ast.get_source_segment(original,fn);text=source
    old="expected_sha=='c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128' and "
    assert text.count(old)==1;text=text.replace(old,'')
    assert text.count('f.LIBRARY_SHA')==2;text=text.replace('f.LIBRARY_SHA','variant_sha256')
    assert text.replace('variant_sha256','f.LIBRARY_SHA').replace('assert f.sha(raw)==expected_sha','assert '+old+'f.sha(raw)==expected_sha')==source
    ns=dict(vars(m),variant_sha256=variant_sha);exec(compile(text,'<vfirst-consumer-receipt-identity-adapter>','exec'),ns)
    raw=Path(path).read_bytes();j=json.loads(raw)
    assert j['variant']['source_proof']==factory.verify() and j['library_sha256']==variant_sha
    return ns['require_consumer'](path,expected_sha)

def execute(a):
    proof=admitted(a)
    kind='numeric' if a.numeric else 'fullstep';file='probe_biv150_native_k8_consumer_v2.py' if a.numeric else 'probe_biv150_native_k8_fullstep.py'
    parent=load_module(file);source,adapter=factory.host_source(kind)
    extra={}
    if a.fullstep:
        assert a.variant_consumer_summary and a.variant_consumer_sha256
        extra['actual_variant_consumer']=variant_receipt(a.variant_consumer_summary,a.variant_consumer_sha256,a.variant_sha256)
        assert a.consumer_summary and a.consumer_proof_sha256 and a.bf16_library
        a.k8_library=a.variant_library
    else:
        assert a.codec_summary and a.codec_proof_sha256;a.library=a.variant_library
    def variant_plan(args):
        j=parent.f.plan();j['K8_library_sha256' if a.fullstep else 'library_sha256']=args.variant_sha256
        if not a.fullstep:j.update(kernel_or_dtype_changed=True,numeric_expression_or_dtype_changed=False)
        j['variant']=dict(source_proof=factory.verify(),compile_admission=proof,host_adapter=adapter,
                          qualification_inherited_from_old_library=False)
        j.update(extra);return j
    ns=dict(vars(parent),variant_plan=variant_plan);exec(compile(source,'<vfirst-exact-parent-'+kind+'-adapter>','exec'),ns)
    return ns['run'](a)

def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group()
    m.add_argument('--compile-only',action='store_true');m.add_argument('--review-template',action='store_true')
    m.add_argument('--numeric',action='store_true');m.add_argument('--fullstep',action='store_true')
    p.add_argument('--out',type=Path);p.add_argument('--compiler',default='/usr/local/corex/bin/clang++')
    p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--compile-summary',type=Path)
    p.add_argument('--assembly-review',type=Path);p.add_argument('--assembly-review-sha256');p.add_argument('--variant-library',type=Path)
    p.add_argument('--codec-summary',type=Path);p.add_argument('--codec-proof-sha256')
    p.add_argument('--consumer-summary',type=Path);p.add_argument('--consumer-proof-sha256');p.add_argument('--bf16-library',type=Path)
    p.add_argument('--variant-consumer-summary',type=Path);p.add_argument('--variant-consumer-sha256');p.add_argument('--case',choices=('4k','16k'),action='append')
    a=p.parse_args();result=plan()
    if a.compile_only:
        assert a.out and a.root_owned_exclusive
        parent=load_module('compile_biv150_native_k8_qk.py');j=parent.cpu_plan()
        sources=[factory.path(k) for k in ('qk.cpp','qk.cu','codec.cu')]
        j['parent_source_transform']=j.pop('source_transform')
        j.update(CPU_only=False,source_transform=factory.verify(),vfirst_source_proof=factory.verify(),private_source={str(x.relative_to(factory.ROOT)):factory.sha(x.read_bytes()) for x in sources+[parent.HEADER]},
                 requested='isolated V-first compile-only; actual ordering unknown until Root review')
        text,_=factory.host_source('compile');ns=dict(vars(parent),SOURCES=sources)
        exec(compile(text,'<vfirst-exact-parent-compile-adapter>','exec'),ns);return ns['compile_only'](a,j)
    if a.review_template:
        assert a.compile_summary;result=review_template(a.compile_summary)
    elif a.numeric or a.fullstep:
        assert a.out
        if a.numeric:assert a.case is None,'small consumer cases stay exact'
        elif a.case:assert len(set(a.case))==len(a.case)
        return execute(a)
    else:
        assert not any((a.out,a.compile_summary,a.assembly_review,a.variant_library,a.codec_summary,a.consumer_summary,a.variant_consumer_summary,a.case,a.root_owned_exclusive))
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(result,sort_keys=True,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
