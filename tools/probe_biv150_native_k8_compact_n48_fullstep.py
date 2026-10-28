#!/usr/bin/env python3
"""Default CPU plan; actual N48 consumer and Root review precede fullstep."""
import argparse
import ast
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import biv150_native_k8_compact_n48_fullstep_fixture as f
import biv150_native_k8_compact_n48_fullstep_source as source
import biv150_native_k8_compact_n48_consumer_contract as contract
import probe_biv150_native_k8_fullstep as parent
import probe_biv150_native_k8_compact_fullstep as compact

CONSUMER_SOURCE='1c0de57574f66f5401c0d1443ccbc57bb1d44a67'
CONSUMER_SHA='493e210f8eeaac6006b25a94ed55e08fbe932337d6481fe3d8ec6e86d0033272'
APPROVAL_SHA='14c8f26fa0d7afb7d00d480aad9ec5cf84f490d11cca8da8cb5f6a959a114a50'
CONSUMER_REVIEW_SHA='c07937c3d6c3ffc78a5c7b213271e4f864a9463775fe932f9f2aadbff08ef4d0'


def plan():
    text,proof=source.adapted_source();source.production_bytes()
    return dict(f.plan(),source_adapter=proof,production_reader=dict(commit=source.PRODUCTION,path=source.READER,sha256=source.READER_SHA,
      quant_helper=source.HELPER,quant_helper_sha256=source.HELPER_SHA),
      writer_source=compact.FROZEN,writer_commit=compact.WRITER_COMMIT,
      actual_consumer_summary_sha256=CONSUMER_SHA,consumer_source_bytes=CONSUMER_SOURCE,
      independent_consumer_review_required=True,consumer_review_sha256=CONSUMER_REVIEW_SHA,
      actual_source_sha256=f.sha(text.encode()))


def actual_consumer_archive(a):
    """Mechanical archive verification; this alone is not numerical approval."""
    root=a.consumer_raw.resolve();summary=root/'out/consumer-summary.json'
    inventory=root/'download-manifest.json'
    assert f.sha(summary.read_bytes())==CONSUMER_SHA
    assert re.fullmatch('[0-9a-f]{64}',a.consumer_archive_sha256 or '')
    assert f.sha(inventory.read_bytes())==a.consumer_archive_sha256
    download=json.loads(inventory.read_text());assert download['read_only'] is True and download['model_rerun'] is False
    owner=download['original_owner'];assert owner['same_start'] is True and owner['state'] in ('Z','X')
    assert (owner['pid'],owner['start_ticks'])==(62050,355994499)
    assert len(download['files'])==download['raw_files'] and sum(x['bytes'] for x in download['files'])==download['raw_bytes']
    for row in download['files']:contract.original.artifact(root,row)
    j=json.loads(summary.read_text())
    for flag in ('completed','requested_consumer_gates_passed','numeric_verified','full_first48_bitwise_verified','Graph_verified','side_stream_verified'):
        assert j[flag] is True,flag
    assert not j['errors'] and j['thresholds_changed'] is False and j['timing'] is None
    approved=contract.contracts()[0].admission(a.compile_summary,a.assembly_review,APPROVAL_SHA,a.k8_library)
    assert j['candidate_admission']==approved and approved['library_sha256']==f.LIBRARY_SHA
    rows={}
    for row in j['artifact_manifest']:
        assert row['relative'] not in rows;rows[row['relative']]=row;contract.original.artifact(summary.parent,row)
    for name,digest in j['frontend_sources'].items():
        original=compact.blob(CONSUMER_SOURCE,'tools/'+name)
        assert f.sha(original)==digest and (summary.parent/'frontends'/name).read_bytes()==original
    expected_cases={(48,4,pitch,live) for pitch in (2048,2560) for live in (41,42)}
    assert len(j['records'])==4 and {tuple(r['case']) for r in j['records']}==expected_cases
    checked=set()
    def array(meta):
        row=rows[meta['path']];assert row['bytes']==meta['stored_bytes'] and row['sha256']==meta['stored_sha256']
        if meta['path'] not in checked:
            raw=gzip.decompress((summary.parent/meta['path']).read_bytes())
            assert len(raw)==meta['bytes'] and f.sha(raw)==meta['sha256'];checked.add(meta['path'])
    for r in j['records']:
        assert r['completed'] is True
        assert set(r['first48_complete_field_bitwise'])==set(importlib.import_module('probe_biv150_native_k8_compact_n48_consumer').PHASES)
        assert all(set(v)=={'partial','maximum','summation','output'} and all(x is True for x in v.values()) for v in r['first48_complete_field_bitwise'].values())
        for arm in ('candidate','reference'):
            item=r['arms'][arm];assert item['completed'] is True and item['closed_epoch_rejected'] is True
            assert len(item['checks'])==6 and all(all(v is True for v in c['checks'].values()) for c in item['checks'])
            for snapshot in item['snapshots'].values():
                assert set(snapshot)=={'query','cache','lengths','table','partial','maximum','summation','output'}
                for meta in snapshot.values():array(meta)
    return dict(summary_sha256=CONSUMER_SHA,archive_sha256=a.consumer_archive_sha256,
      actual_compile_and_SO=approved,full_artifacts_rehashed=True,full_snapshot_encodings_rehashed=True,
      candidate_actual_schema=j['actual_schemas']['candidate'],
      original_BF16_proxy_passed=j['original_BF16_proxy_passed'],consumer_only=True,
      large_context_or_performance_qualification_inherited=False)


def actual_consumer_gate(a):
    """Actual archive plus the separately issued independent full-array review."""
    proof=actual_consumer_archive(a)
    assert a.consumer_review_sha256==CONSUMER_REVIEW_SHA
    review_raw=a.consumer_review.read_bytes();assert f.sha(review_raw)==a.consumer_review_sha256
    review=json.loads(review_raw)
    assert review['completed'] is True and review['errors']==[] and review['admissions_replayed'] is True
    assert review['summary_sha256']==CONSUMER_SHA and review['download_manifest_sha256']==a.consumer_archive_sha256
    assert review['candidate_library_sha256']==f.LIBRARY_SHA and review['reference_library_sha256']==contract.REFERENCE_LIBRARY_SHA
    expected=dict(FP64_hard=176,BF16_proxy=32,first48_field_pairs=96,golden_arrays=168,epoch_receipts=24,host_refusals=2,
      full_snapshots=48,readonly_whole_buffers=336,ideal_diagnostic=48,ideal_diagnostic_failed=48)
    assert review['counts']==expected
    assert review['device_thresholds_unchanged']==dict(RRMSE=.005,pointwise_atol=.005,pointwise_rtol=.005)
    assert review['raw_files_rehashed']==170 and review['raw_bytes_rehashed']==59569068
    assert review['scope']['rows']==48 and review['scope']['segments']==4 and review['scope']['max_context']==193
    assert all(review['scope'][name] is False for name in ('fullstep','model','performance','writer','encoder','capacity42layers'))
    return dict(proof,independent_array_review_sha256=a.consumer_review_sha256)


def save_frontend_sources(out,result):
    folder=out/'frontends';folder.mkdir()
    result['frontend_sources']={}
    for name in ('probe_biv150_native_k8_compact_n48_fullstep.py','biv150_native_k8_compact_n48_fullstep_source.py','biv150_native_k8_compact_n48_fullstep_fixture.py'):
        raw=(f.ROOT/'tools'/name).read_bytes();(folder/name).write_bytes(raw);result['frontend_sources'][name]=f.sha(raw)


def verify_loaded_reader(reader):
    helper=importlib.import_module('vllm_fl.patches.biv150_konly_encoded_writer')
    assert f.sha(Path(helper.__file__).read_bytes())==source.HELPER_SHA
    assert reader.quantize_bf16_rne is helper.quantize_bf16_rne
    assert f.sha(Path(reader.__file__).read_bytes())==source.READER_SHA
    return dict(reader_file=str(Path(reader.__file__).resolve()),reader_sha256=source.READER_SHA,
      quant_helper_file=str(Path(helper.__file__).resolve()),quant_helper_sha256=source.HELPER_SHA,
      quantizer_object_identity_verified=True)


def tl_arguments(q,z,lens,table,s):
    o,m,l=z['partial'],z['maximum'],z['summation'];k,v=z['key'],z['value']
    return dict(segm_output_ptr=o,segm_max_ptr=m,segm_expsum_ptr=l,query_ptr=q,
      query_code_ptr=k,query_scale_ptr=z['scale'],key_scale_ptr=z['scale'],key_cache_ptr=k,value_cache_ptr=v,
      seq_lens_ptr=lens,block_table_ptr=table,softmax_scale_log2e=128**-.5*1.4426950408889634,
      queries_per_kv=8,block_table_stride=table.stride(0),query_stride_token=q.stride(0),query_stride_head=q.stride(1),
      k_stride_block=k.stride(0),k_stride_slot=k.stride(1),k_stride_head=k.stride(2),
      v_stride_block=v.stride(0),v_stride_slot=v.stride(1),v_stride_head=v.stride(2),
      segment_stride_token=o.stride(0),segment_stride_head=o.stride(1),segment_stride_segment=o.stride(2),
      max_stride_token=m.stride(0),max_stride_head=m.stride(1),HEAD_SIZE=128,PAGE_SIZE=16,ROWS=8,SEGMENTS=s,num_warps=1,num_stages=1)


def typed_signature(handle):
    result={};names=handle.src.fn.arg_names
    for key,value in handle.src.signature.items():
        if isinstance(key,tuple):assert len(key)==1;key=key[0]
        name=names[key] if type(key) is int else key
        assert name in names and name not in result;result[name]=str(value)
    return result


def compiled_receipt(handle,blob):
    def values(obj):
        if hasattr(obj,'_asdict'):return {str(k):repr(v) for k,v in obj._asdict().items()}
        if hasattr(obj,'__dict__'):return {str(k):repr(v) for k,v in vars(obj).items()}
        return repr(obj)
    return dict(hash=str(handle.hash),n_regs=getattr(handle,'n_regs',None),n_spills=getattr(handle,'n_spills',None),
      source_sha256=f.sha(handle.src.fn.src.encode()),signature=typed_signature(handle),
      constants={repr(k):repr(v) for k,v in handle.src.constants.items()},attrs=values(handle.src.attrs),metadata=values(handle.metadata),
      artifacts={ext:blob(data.encode() if isinstance(data,str) else data) for ext,data in handle.asm.items() if isinstance(data,(str,bytes))})


def verify_actual_handles(handles,compiled,rec,writer,reader,base):
    expected=dict(initial_encoder=writer.encode_append,tl_append=writer.encode_append,native_append=writer.encode_append,
      tl_producer=reader._gqa_decode_konly_i8,reducer=base._reduce_partials)
    assert set(handles)==set(expected)
    for role,fn in expected.items():
        h=handles[role];assert h.src.fn is fn
        assert type(h.n_regs) is int and h.n_regs>0 and h.n_spills==0,'unknown/failed postlaunch resources: '+role
    names=['key_source','value_source','slots','code_full','scale_full','value_full','key_row_stride','value_row_stride','key_rows','value_rows','slot_count','cache_pages']
    for role in ('tl_append','native_append'):
        signature=typed_signature(handles[role])
        assert [signature[n] for n in names]==['*bf16','*bf16','*i64','*i8','*fp32','*bf16']+['i64']*6
    reader_signature=typed_signature(handles['tl_producer'])
    assert [reader_signature[n] for n in ('segm_output_ptr','segm_max_ptr','segm_expsum_ptr','query_ptr','query_code_ptr','query_scale_ptr','key_scale_ptr','key_cache_ptr','value_cache_ptr','seq_lens_ptr','block_table_ptr')]==['*fp32']*3+['*bf16','*i8','*fp32','*fp32','*i8','*bf16','*i32','*i32']
    assert set(rec['actual_compiled'])==set(expected)
    rec['stride2560_actual_JIT_and_resources_verified']=True


def validate_roles(views,arenas,cases,pages,torch):
    return {arm:compact.validate_roles(views,{'k8':arena},{'k8':cases[arm]},pages,torch) for arm,arena in arenas.items()}


def execute(a):
    assert a.root_owned_exclusive and not a.out.exists()
    text,_=source.adapted_source()
    ns=dict(vars(parent),f=f,actual_consumer_gate=actual_consumer_gate,compact_plan=plan,
      compact_contract=compact.compact_contract,compact_chunk=compact.compact_chunk,
      materialize=compact.materialize,materialize_reader=source.materialize_reader,
      initial_sources=compact.initial_sources,check_initial_sources=compact.check_initial_sources,
      validate_roles=validate_roles,verify_loaded_reader=verify_loaded_reader,tl_arguments=tl_arguments,
      verify_actual_handles=verify_actual_handles,compiled_receipt=compiled_receipt,save_frontend_sources=save_frontend_sources)
    exec(compile(text,'<existing-fullstep-N48-TL-vs-native>','exec'),ns)
    return ns['run'](a)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--preflight-cpu',action='store_true')
    for name in ('consumer-raw','consumer-review','compile-summary','assembly-review','out'):p.add_argument('--'+name,type=Path)
    p.add_argument('--native-library',dest='k8_library',type=Path)
    for name in ('consumer-archive-sha256','consumer-review-sha256'):p.add_argument('--'+name)
    p.add_argument('--case',choices=f.CASES,action='append');a=p.parse_args()
    if a.preflight_cpu:
        assert not any((a.gpu,a.root_owned_exclusive,a.out,a.case))
        assert all(getattr(a,k) for k in ('consumer_raw','consumer_review','compile_summary','assembly_review','k8_library','consumer_archive_sha256','consumer_review_sha256'))
        plan()
        proof=actual_consumer_gate(a);assert not {'torch','triton','numpy'}&set(sys.modules)
        print(json.dumps(proof,indent=2,sort_keys=True));return 0
    if not a.gpu:
        assert not any(v for k,v in vars(a).items() if k!='gpu')
        result=plan();assert not {'torch','triton','numpy'}&set(sys.modules);print(json.dumps(result,indent=2,sort_keys=True));return 0
    assert a.root_owned_exclusive and all(v for k,v in vars(a).items() if k not in ('preflight_cpu','case'))
    assert all(p.is_absolute() for p in vars(a).values() if isinstance(p,Path))
    if a.case:assert len(a.case)==len(set(a.case))
    return execute(a)


if __name__=='__main__':raise SystemExit(main())
