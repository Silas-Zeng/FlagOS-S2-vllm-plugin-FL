#!/usr/bin/env python3
"""Finite source/descriptor checks; optional tiny NumPy codec/reference checks."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType,SimpleNamespace
import biv150_native127_v8_dense_timing as d


def refuse(fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,TypeError):return
    raise AssertionError('pending/bad input admitted')


def numpy_checks(parent):
    import numpy as np
    f=d.c.load(parent,'biv150_native_k8_fullstep_fixture','cjj127_dense_cpu_fullstep')
    old=d.c.load(parent,'biv150_native_k8_consumer_fixture','cjj127_dense_cpu_consumer')
    source,*aux=f.chunk_sources(0,2);packed=f.encode_chunk(0,source,*aux,f.codebook())
    controls=[f.control(64,4096,2048,p) for p in (0,1)]
    for ctrl in controls:ctrl['lengths'][:]=4096
    assert np.array_equal(controls[0]['lengths'],controls[1]['lengths'])
    assert not np.array_equal(controls[0]['query'],controls[1]['query'])
    assert not np.array_equal(controls[0]['table'],controls[1]['table'])
    book=d.fixture.vbook(f.quantize_v2);control,private=d.fixture.pack_v(source,packed,book)
    count=0
    for peak in range(64):
        for basis in range(-peak,peak+1):
            code=int(book[0][peak,basis+63]);scale=float(book[1][peak])
            product=np.float32(code)*np.float32(scale);bits=product.view(np.uint32)
            actual=int((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16)
            assert actual==d.fixture.small.dequant_bf16(code,scale);count+=1
    for p in range(2):
        for slot in range(16):
            for head in range(2):
                assert source[p,0,slot,head].tolist()==old.vector(1,0,p*16+slot,head)
                assert source[p,1,slot,head].tolist()==old.vector(2,0,p*16+slot,head)
                codes,scale=f.quantize_v2([old.bf16_value(int(b)) for b in source[p,1,slot,head]])
                assert bytes(c&255 for c in codes)==private[p,8192+slot*256+head*128:8192+slot*256+(head+1)*128].tobytes()
                import struct
                assert struct.pack('<f',scale)==private[p,12288+slot*8+head*4:12292+slot*8+head*4].tobytes()
                expect=[d.fixture.small.dequant_bf16(c,scale) for c in codes]
                assert np.frombuffer(control[p,8192+slot*512+head*256:8192+slot*512+(head+1)*256],'<u2').tolist()==expect
    assert np.array_equal(control[:,:8192],private[:,:8192]) and np.array_equal(control[:,:8192],packed[:,:8192])
    # Prove neutral last-KV substitution against the independent original ideal
    # consumer on a tiny extent; no device, no new numerical oracle or tolerance.
    query=np.array([[old.vector(0,1,r,h) for h in range(16)] for r in range(2)],dtype='<u2').reshape(2,2048)
    ctrl=dict(query=query,table=np.array([[0,-1],[1,-1]],'<i4'),lengths=np.array([16,16],'<i4'),
      new_k=source[:,0,15].copy(),new_v=source[:,1,15].copy())
    proxy=f.golden(source,control,ctrl,2,'bf16')['output']
    cold=source.copy();cold[:,1]=control[:,8192:].reshape(2,8192).view('<u2').reshape(2,16,2,128)
    ctrl['new_v']=cold[:,1,15].copy();gold=f.golden(cold,control,ctrl,2,'k8')
    pairs=[[f.quantize_v2([old.bf16_value(int(x)) for x in head]) for head in query[r].reshape(16,128)] for r in range(2)]
    case=dict(rows=2,segments=2,pitch=2048,query_bits=[old.BF16_GUARD]*64+query.reshape(-1).tolist()+[old.BF16_GUARD]*64,
      lengths=[-1234567]*64+[16,16]+[-1234567]*64,table=[-1]*64+[0,-1,1,-1]+[-1]*64,
      qcodes=[[x[0] for x in row] for row in pairs],qscales=[[x[1] for x in row] for row in pairs],
      kcodes=control[:,:4096].view(np.int8).reshape(2,16,2,128),
      kscales=control[:,4096:4224].copy().view('<f4').reshape(2,16,2),key_bits=cold[:,0].tolist(),value_bits=cold[:,1].tolist())
    independent=FunctionType(old.golden.__code__,dict(vars(old),PAGES=2,COLS=2))(case)
    for field in ('output','maximum','summation'):
        active=np.isfinite(independent[field]);assert old.metrics(gold[field][active],independent[field][active])['passed']
    assert np.isfinite(proxy).all() and not np.array_equal(source[:,1],cold[:,1])
    return dict(v_RNE_lookup_pairs=count,all64_token_head_vectors_exact_original_codec=True,
      fullD128_counter_equals_original_scalar=True,neutral_last_KV_matches_original_ideal_consumer=True,
      original_V_and_dequantized_V_differ_proxy_kept=True,torch_imported=False)


def check(a):
    parent=d.c.previous.parent(a.parent_source)
    source,proof=d.adapted_run(parent);assert proof['inverse_byte_exact'];compile(source,'<dense-worker>','exec')
    def funcs(text):return {n.name:ast.get_source_segment(text,n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef)}
    inherited=d.short.adapted_run(parent)[0];old,new=funcs(inherited),funcs(source)
    names=('npblob','write_json','host','bf16_float','poison','change','snapshot','check','producer','reducer','eager')
    assert all(old[n]==new[n] for n in names)
    assert d.prepare_case is d.short.prepare_case and d.run_pair is d.short.run_pair
    assert d.clock_source(parent)==d.short.clock_source(parent)
    assert d.fixture.CASES==((64,2,2048),) and (d.fixture.PAGES,d.fixture.COLS,d.fixture.CACHE_BYTES)==(34276,8192,561578240)
    full=d.c.load(parent,'biv150_native_k8_fullstep_fixture','cjj127_dense_scope_fullstep')
    mapping=full.fixed_table(64,4096)
    flat=[p for row in mapping for p in row];assert len(flat)==len(set(flat))==16384 and min(flat)>=0 and max(flat)<34276
    b=[row[1:]+row[:1] for row in mapping];assert b!=mapping and set(sum(b,[]))==set(flat)
    refuse(lambda:d.evidence(SimpleNamespace(dense_admission_sha256=None,dense_admission=None)))
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        saved=d.fixture.CACHE_BYTES
        try:
            d.fixture.CACHE_BYTES=512;raw=b'\x5a'*128+b'\xa5'*256+b'\x5a'*128
            descriptor=d.fixture.cache_chunks(raw,raw,Path(folder),dict(CPU_test_only=True))
            assert descriptor['bytes']==512 and descriptor['kind']=='regenerable_cache_all_bytes_compared_v1'
            refuse(lambda:d.fixture.cache_chunks(raw[:200]+b'\0'+raw[201:],raw,Path(folder),dict(CPU_test_only=True)))
            assert (Path(folder)/'failed-cache/0-actual.bin').is_file()
            bad=b'\0'+raw[1:];refuse(lambda:d.fixture.cache_chunks(bad,bad,Path(folder),{}))
        finally:d.fixture.CACHE_BYTES=saved
    script=Path(__file__).with_name('probe_biv150_native127_v8_dense_timing.py')
    p=subprocess.run([sys.executable,'-B','-S',str(script)],text=True,capture_output=True,check=True)
    assert json.loads(p.stdout)==json.loads(json.dumps(d.plan()))
    for args in (['--gpu'],['--timing'],['--gpu','--timing','--root-owned-exclusive']):
        assert subprocess.run([sys.executable,'-B','-S',str(script),*args],capture_output=True).returncode==2
    result=dict(passed=True,parent=d.PARENT,dense_layout=d.fixture.layout(),source_adapter=proof,
      original_consumer_nested_functions_byte_exact=list(names),prepare_capture_and_pair_clock_byte_exact=True,
      all16384_active_physical_pages_unique=True,A_B_same_cache_lengths_different_Q_table=True,
      full_cache_mutation_and_guard_refused_bad_chunk_saved=True,
      missing_dense_receipt_refused_before_framework=True,default_stdlib_only=True,
      GPU_executed=False,numeric_verified=False,performance_verified=False,service_verified=False)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    evidence_paths=(a.dense_admission,a.numeric_summary,a.numeric_review,a.timing_admission)
    assert all(evidence_paths) or not any(evidence_paths)
    if all(evidence_paths):
        args=SimpleNamespace(root_timing_approved=True,dense_admission=a.dense_admission,dense_admission_sha256=d.DENSE_ADMISSION_SHA,
          numeric_summary=a.numeric_summary,numeric_review=a.numeric_review,numeric_review_sha256=d.short.NUMERIC['Root_review_sha256'],
          timing_admission=a.timing_admission,timing_admission_sha256=d.short.TIMING_ADMISSION_SHA)
        result['actual5824_preflight']=d.evidence(args)
        with tempfile.TemporaryDirectory() as folder:
            pin=d.DENSE_ADMISSION_SHA
            try:
                for key,value in [('cases',[[64,2,2560]]),('pages',389),('writer',True)]:
                    receipt=json.loads(a.dense_admission.read_text());receipt['scope'][key]=value
                    path=Path(folder)/(key+'.json');path.write_text(json.dumps(receipt))
                    bad=copy.copy(args);bad.dense_admission=path;bad.dense_admission_sha256=d.c.sha(path)
                    d.DENSE_ADMISSION_SHA=bad.dense_admission_sha256 # CPU fault pin only; test strict schema beyond SHA.
                    refuse(lambda:d.evidence(bad))
            finally:d.DENSE_ADMISSION_SHA=pin
        result['actual_receipt_scope_mutants_refused']=['second_pitch','old389page_arena','writer']
        result['Root_flag_in_positive_preflight_was_CPU_stub_only']=True
    if a.numpy_oracles:result['tiny_CPU_oracles']=numpy_checks(parent)
    assert not {'torch','triton'}.intersection(sys.modules)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True);p.add_argument('--numpy-oracles',action='store_true')
    for name in ('dense-admission','numeric-summary','numeric-review','timing-admission'):p.add_argument('--'+name,type=Path)
    print(json.dumps(check(p.parse_args()),indent=2,sort_keys=True))
