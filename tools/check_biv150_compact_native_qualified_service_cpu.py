#!/usr/bin/env python3
"""Real compact admission/dispatch with existing CPU metadata recorders; no GPU."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import check_biv150_native_k8_service_cpu as old

ROOT=Path(__file__).resolve().parents[1]
SO=Path('/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out/build/cjj_native_k8_compact_probe.so')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def compact_load(name):
    return load('cjj_compact_qualified_'+name,ROOT/'vllm_fl/patches'/('biv150_konly_compact_'+name+'.py'))


def adapted_fixture():
    text=(ROOT/'tools/check_biv150_native_k8_service_cpu.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='fixture')
    original=ast.get_source_segment(text,node);changed=original
    edits=[('VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','VLLM_FL_BIV150_KONLY_COMPACT_SERVICE',1),
           ('4*16384+512','4*12544+512',1),('(8192,4096,256,128,1)','(6272,2176,256,128,1)',1),
           ('(8192,256,128,1)','(6272,256,128,1)',2),
           ('(6272,256,128,1),4224)', '(6272,256,128,1),2304)',1),
           ('k_stride_block=8192','k_stride_block=6272',1),('v_stride_block=8192','v_stride_block=6272',1)]
    for a,b,n in edits:
        assert changed.count(a)==n,a;changed=changed.replace(a,b)
    reverse=changed
    for a,b,n in reversed(edits):
        assert reverse.count(b)==n,b;reverse=reverse.replace(b,a)
    assert reverse==original
    def ops(schema):
        value=old.Ops(schema);value.cjj_native_k8_compact_probe=value.cjj_native_k8_v_overlap_probe
        return value
    ns=dict(vars(old),load=compact_load,Ops=ops)
    exec(compile(changed,'<existing-service-fixture-compact-geometry-only>','exec'),ns)
    return ns['fixture'],dict(inverse_byte_exact=True,original_fixture_sha256=hashlib.sha256(original.encode()).hexdigest())


def check(path):
    # A fake package permits stdlib native source loading without running real
    # package bootstrap. Actual pre-Torch guard and C3 are checked separately.
    package=ModuleType('vllm_fl');package.__path__=[str(ROOT/'vllm_fl')];sys.modules['vllm_fl']=package
    q=load('vllm_fl.compact_native_qualification',ROOT/'vllm_fl/compact_native_qualification.py')
    native=compact_load('native');flags={q.ENV:'1','VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1',
        q.PATH_ENV:str(path),q.SHA_ENV:q.QUALIFICATION['library_sha256']}
    assert q.preflight({}) is None
    binding=q.preflight(flags);assert binding['qualification']==q.pin()
    refused=[]
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError):refused.append(name)
        else:raise AssertionError('Accepted '+name)
    for name,change in [('no_format',{'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'0'}),
        ('two_formats',{'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1'}),
        ('16KiB_native',{'VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1'}),
        ('103_not_included',{'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST':'1'}),
        ('missing_SHA',{q.SHA_ENV:None}),('8b_library',{q.PATH_ENV:str(old.SO)})]:
        reject(name,lambda change=change:q.preflight({**flags,**change}))
    original=q.QUALIFICATION
    try:
        for name,change in [('wrong_page',{'page_bytes':16384}),('wrong_writer',{'writer_source_sha256':'0'*64}),
            ('wrong_views',{'views_source_sha256':'0'*64}),('wrong_CUDA',{'source_CUDA_sha256':'0'*64}),
            ('40_native',{'scopes':[[40,4,2048]]})]:
            q.QUALIFICATION={**original,**change};reject(name,q.pin)
        q.QUALIFICATION={k:v for k,v in original.items() if k!='fullstep_summary_sha256'}
        reject('missing_fullstep',q.pin)
    finally:q.QUALIFICATION=original
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==native.LIBRARY_SHA
    assert raw[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',raw,18)[0]==62
    assert native.SCHEMA.split('::',1)[1].encode() in raw
    assert all(n.encode() in raw for n in ('launch_cjj_native_k8_compact_s2','launch_cjj_native_k8_compact_s4'))
    fixture,fixture_proof=adapted_fixture();cases=[]
    for n,s,pitch in ((64,2,2048),(64,2,2560),(32,4,2048),(32,4,2560)):
        runtime,entry,kw,producer,kernels,ops,capture,service,mod=fixture(native,n,s,pitch,path)
        with contextlib.redirect_stdout(io.StringIO()):
            producer((n,2,s),**kw);capture[0]=True;producer((n,2,s),**kw);capture[0]=False
        assert len(ops.op.calls)==2 and not kernels['decode'].calls
        assert ops.op.calls[-1][1].stride(0)==12544
        assert ops.op.calls[-1][3].data_ptr()-ops.op.calls[-1][1].data_ptr()==4352
        reject('scalar_mismatch',lambda:producer((n,2,s),**dict(kw,block_table_stride=1)))
        reject('output_alias',lambda:producer((n,2,s),**dict(kw,segm_output_ptr=kw['query_ptr'])))
        keep=entry.generation;entry.generation=runtime.registry.generation+1
        reject('stale_entry',lambda:producer((n,2,s),**kw));entry.generation=keep
        assert len(ops.op.calls)==2
        with contextlib.redirect_stdout(io.StringIO()):producer((n,2,s),**dict(kw,num_stages=2))
        assert len(kernels['decode'].calls)==1
        ops.op.throw=True;reject('postlaunch_no_retry',lambda:producer((n,2,s),**kw))
        assert len(ops.op.calls)==3 and len(kernels['decode'].calls)==1
        assert runtime.native_decode.verify(runtime)['library']['sha256']==q.pin()['library_sha256']
        cases.append(dict(batch=n,segments=s,query_pitch=pitch,native_recorded_calls=3,
            cache_page_bytes=12544,V_offset4352=True,postlaunch_error_not_retried=True))
    fallbacks=[]
    for n in (40,48):
        runtime,entry,kw,producer,kernels,ops,capture,service,mod=fixture(native,n,16,2560,path)
        with contextlib.redirect_stdout(io.StringIO()):producer((n,2,16),**kw)
        assert not ops.op.calls and len(kernels['decode'].calls)==1
        record=runtime.native_decode.stats()['private_Triton_fallback_records']
        assert len(record)==1 and record[0]['shape']==n and record[0]['segments']==16
        fallbacks.extend(record)
    # The actual early guard accepts this exact SO, with no framework import.
    tree=ast.parse((ROOT/'vllm_fl/__init__.py').read_text())
    guard=next(n for n in tree.body if isinstance(n,ast.If) and q.ENV in ast.unparse(n.test))
    from unittest.mock import patch
    with patch.dict(os.environ,flags):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'<actual-package-guard>','exec'),
             {'os':os,'__package__':'vllm_fl'})
    sem=load('_compact_qualified_standalone_semantics',ROOT/'vllm_fl/compile_semantics.py')
    config=NS(model_config=NS(hf_config=NS(**sem._GEOMETRY)))
    base={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0'}
    off=sem.build_fingerprint(config,'iluvatar',environ=base,abi={'CPU_only':True})
    tl=sem.build_fingerprint(config,'iluvatar',environ={**base,'VLLM_FL_BIV150_KONLY_COMPACT_SERVICE':'1'},abi={'CPU_only':True})
    on=sem.build_fingerprint(config,'iluvatar',environ={**base,**flags},abi={'CPU_only':True})
    assert len({x['sha256'] for x in (off,tl,on)})==3
    assert on['konly_compact']['native_decode']['qualification']==q.pin()
    assert on['konly_compact']['page_bytes']==12544 and not on['konly_compact']['q32_hoist_enabled']
    unchanged={}
    for prefix,commit in [('compact','dc5fd2e5b86a3d57aceaad26e8556d169ec7462a'),
                           ('encoded','a455a3de364051fbea80226870c239de3a48e716')]:
        for role in ('writer','decode','q32','multi2','raw3d','normalized2d128'):
            name=f'vllm_fl/patches/biv150_konly_{prefix}_{role}.py'
            data=(ROOT/name).read_bytes();assert data==subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name])
            unchanged[name]=hashlib.sha256(data).hexdigest()
    for role in ('qk.cu','qk.cpp','codec.cu'):
        name='tools/kernels/biv150_native_k8_compact_'+role;data=(ROOT/name).read_bytes()
        assert data==subprocess.check_output(['git','-C',str(ROOT),'show','51da6fd73e7fff1513081fb330323ce72c00856c:'+name])
        unchanged[name]=hashlib.sha256(data).hexdigest()
    identity=compact_load('identity');manifest=identity.verify_package()
    assert manifest['native_decode']['qualification_binding']==q.pin()
    assert not {'torch','triton','numpy','vllm'}.intersection(sys.modules)
    return dict(passed=True,CPU_only=True,actual_SO_read_not_loaded=True,library_sha256=native.LIBRARY_SHA,
        qualification=q.pin(),fixture_adapter=fixture_proof,cases=cases,private_fallbacks=fallbacks,
        negative_cases=refused,actual_package_guard_before_Torch=True,C3_fingerprints_distinct=True,
        GPU_sources_byte_exact=unchanged,service_GPU_executed=False,model_verified=False,
        capacity_reclaimed=False,performance_verified=False,precision_gate=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,default=SO);a=p.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):result=check(a.library)
    print(json.dumps(result,sort_keys=True,indent=2))
