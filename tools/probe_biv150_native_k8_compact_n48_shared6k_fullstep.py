#!/usr/bin/env python3
"""CPU-default direct old8k/new6k ABBA. New consumer qualification remains pending."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types

import biv150_native_k8_compact_n48_fullstep_fixture as old_fixture
import biv150_native_k8_compact_n48_shared6k_fullstep_source as source
import biv150_native_k8_compact_n48_shared6k_admission as contract
import probe_biv150_native_k8_compact_n48_fullstep as old
import probe_biv150_native_k8_compact_n48_shared6k_consumer as new_consumer
from probe_biv150_native_k8_compact_v_overlap_fullstep import replace_exact

OLD_SO='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
OLD_FULLSTEP='5a6aefeb7befb777e890402fa9742774ea23ae5e696be67c74f04418df37b131'
OLD_FULLSTEP_REVIEW='2d8d5b873eb30084da7dd0e25332088462dd5cfef62c45fe45c75bccd07ede5d'
CONSUMER_SOURCE='c91f1cac3b7e69e03cee8dab9f7aa7b313674d8b'
# Only replace these with actual completed consumer evidence and a separately
# issued full-array review. Unknown values intentionally block preflight/GPU.
NEW_CONSUMER=dict(summary_sha256=None,review_sha256=None,manifest_sha256=None,
                  owner=None,raw_files=None,raw_bytes=None)
ORDER=('old8k','new6k','new6k','old8k')


def fixture():
    f=types.SimpleNamespace(**vars(old_fixture));f.ORDER=ORDER;f.LIBRARY_SHA=contract.SO
    def check_arm(arm):assert arm in ('old8k','new6k');return 'native'
    def patched_chunk(arm,start,initial,control_):return old_fixture.patched_chunk(check_arm(arm),start,initial,control_)
    def golden(cold,packed,control_,segments,arm):return old_fixture.golden(cold,packed,control_,segments,check_arm(arm))
    f.patched_chunk=patched_chunk;f.golden=golden
    return f


f=fixture()


def candidate_gate_source():
    source.verify_pins()
    archive=inspect.getsource(old.actual_consumer_archive)
    archive,ap=replace_exact(archive,[("assert (owner['pid'],owner['start_ticks'])==(62050,355994499)",
      "assert (owner['pid'],owner['start_ticks'])==tuple(NEW_CONSUMER['owner'])")])
    gate=inspect.getsource(old.actual_consumer_gate)
    gate,gp=replace_exact(gate,[("assert review['raw_files_rehashed']==170 and review['raw_bytes_rehashed']==59569068",
      "assert review['raw_files_rehashed']==NEW_CONSUMER['raw_files'] and review['raw_bytes_rehashed']==NEW_CONSUMER['raw_bytes']")])
    return archive+gate,dict(archive=ap,review=gp,all_original_array_Graph_side_artifact_epoch_counts_and_threshold_gates_preserved=True)


def pending():return [k for k,v in NEW_CONSUMER.items() if v is None]


def actual_consumer_gate(a):
    assert not pending(),'Actual113 consumer evidence pending: '+','.join(pending())
    assert a.consumer_archive_sha256==NEW_CONSUMER['manifest_sha256']
    assert a.consumer_review_sha256==NEW_CONSUMER['review_sha256']
    text,_=candidate_gate_source()
    ns=dict(vars(old),f=f,contract=contract,CONSUMER_SOURCE=CONSUMER_SOURCE,
      CONSUMER_SHA=NEW_CONSUMER['summary_sha256'],APPROVAL_SHA=contract.ROOT_REVIEW,
      CONSUMER_REVIEW_SHA=NEW_CONSUMER['review_sha256'],NEW_CONSUMER=NEW_CONSUMER)
    exec(compile(text,'<original112-actual-consumer-gates-for-113>','exec'),ns)
    new=ns['actual_consumer_gate'](a)
    summary=json.loads((a.consumer_raw/'out/consumer-summary.json').read_text())
    executed=summary['executed_consumer_source'];raw=(a.consumer_raw/'out'/executed['relative']).read_bytes()
    expected,proof=new_consumer.adapted_source()
    assert raw==expected.encode() and f.sha(raw)==executed['sha256'] and len(raw)==executed['bytes']
    assert executed['source_adapter']==proof
    # Existing2ad keeps its exact already-approved archive/report gates. No
    # additional array math is run here, and its fullstep evidence is immutable.
    old_a=types.SimpleNamespace(consumer_raw=a.old_consumer_raw,consumer_review=a.old_consumer_review,
      compile_summary=a.old_compile_summary,assembly_review=a.old_assembly_review,k8_library=a.old_library,
      consumer_archive_sha256=f.sha((a.old_consumer_raw/'download-manifest.json').read_bytes()),consumer_review_sha256=old.CONSUMER_REVIEW_SHA)
    previous=old.actual_consumer_gate(old_a)
    raw=a.old_fullstep_summary.read_bytes();assert f.sha(raw)==OLD_FULLSTEP
    step=json.loads(raw);assert step['completed'] is True and step['errors']==[]
    for flag in ('requested_component_gates_passed','numerical_verified','Graph_verified','side_stream_verified','timing_executed'):
        assert step[flag] is True
    assert step['actual_native_library']['sha256']==OLD_SO
    raw=a.old_fullstep_review.read_bytes();assert f.sha(raw)==OLD_FULLSTEP_REVIEW
    review=json.loads(raw);assert review['passed'] is True and review['native_SO']==OLD_SO and review['summary_sha256']==OLD_FULLSTEP
    assert review['consumer_sha256']==old.CONSUMER_SHA and review['consumer_original_report_sha256']==old.CONSUMER_REVIEW_SHA
    assert review['runtime_n_spills_zero_gate_preserved'] is True and review['GPU_numeric_thresholds_changed'] is False
    assert f.sha(a.old_library.read_bytes())==OLD_SO and f.sha(a.k8_library.read_bytes())==contract.SO
    return dict(old8k=previous,new6k=new,arm_schemas=dict(old8k=previous['candidate_actual_schema'],new6k=new['candidate_actual_schema']),
      old_fullstep_summary_sha256=OLD_FULLSTEP,old_fullstep_review_sha256=OLD_FULLSTEP_REVIEW,
      previous_record_errata=dict(GPU_execution=review['GPU_execution_record_erratum'],TL_resources=review['resource_review_erratum']),
      qualification='two separately admitted actual N48/S4 libraries; no new model or performance qualification')


def plan():
    text,proof=source.adapted_source();_,handles=source.handles_source();_,gate=candidate_gate_source()
    result=old_fixture.plan()
    result.update(component='Direct actual2ad old8k vs actual5cf6 new6k full step; not service',
      arms=['actual2ad old8k N48/S4','actual5cf6 new6k N48/S4'],ABBA_order=list(ORDER),library_sha256=contract.SO,
      libraries=dict(old8k=OLD_SO,new6k=contract.SO),source_adapter=proof,remaining_JIT_checks=handles,
      candidate_consumer_gate_adapter=gate,actual_source_sha256=f.sha(text.encode()),
      writer_source=old.compact.FROZEN,writer_commit=old.compact.WRITER_COMMIT,
      new_consumer_source=CONSUMER_SOURCE,new_consumer_evidence=dict(NEW_CONSUMER),pending_fields=pending(),
      fullstep_execution_allowed=not pending(),old_fullstep_summary_sha256=OLD_FULLSTEP,old_fullstep_review_sha256=OLD_FULLSTEP_REVIEW,
      JIT_roles=['initial_encoder','old8k_append','new6k_append','reducer'],TL_not_executed=True,
      Native_producer_resource_binding='Each exact SO/compile/review/consumer; no invented Triton handles',
      previous_TL_ratio_used_as_performance_baseline=False,source_shared_QKV_allocation_qualification=False)
    return result


def save_frontend_sources(out,result):
    folder=out/'frontends';folder.mkdir();result['frontend_sources']={}
    for name in ('probe_biv150_native_k8_compact_n48_shared6k_fullstep.py','biv150_native_k8_compact_n48_shared6k_fullstep_source.py',
      'probe_biv150_native_k8_compact_n48_fullstep.py','biv150_native_k8_compact_n48_fullstep_source.py','biv150_native_k8_compact_n48_fullstep_fixture.py',
      'biv150_native_k8_compact_n48_shared6k_admission.py','probe_biv150_native_k8_compact_n48_shared6k_consumer.py'):
        raw=(f.ROOT/'tools'/name).read_bytes();(folder/name).write_bytes(raw);result['frontend_sources'][name]=f.sha(raw)
    text,proof=source.adapted_source();(out/'executed-fullstep-source.py').write_text(text)
    result['executed_source']=dict(relative='executed-fullstep-source.py',sha256=f.sha(text.encode()),source_adapter=proof)
    # Preserve admissions used by this actual run, not only their path strings.
    for arm,paths in (('old8k',('old_compile_summary','old_assembly_review','old_consumer_review','old_fullstep_summary','old_fullstep_review')),
      ('new6k',('compile_summary','assembly_review','consumer_review'))):
        for name in paths:
            path=getattr(ACTUAL_ARGS,name);(out/(arm+'-'+name+'-input.json')).write_bytes(path.read_bytes())


def execute(a):
    assert a.root_owned_exclusive and not a.out.exists() and not pending()
    global ACTUAL_ARGS
    ACTUAL_ARGS=a
    text,_=source.adapted_source();handles,_=source.handles_source()
    helper_ns=dict(vars(old),f=f)
    exec(compile(handles,'<existing112-JIT-gates-without-unexecuted-TL>','exec'),helper_ns)
    ns=dict(vars(old.parent),f=f,actual_consumer_gate=actual_consumer_gate,compact_plan=lambda:dict(plan(),GPU_executed=None),
      compact_contract=old.compact.compact_contract,compact_chunk=old.compact.compact_chunk,
      materialize=old.compact.materialize,initial_sources=old.compact.initial_sources,check_initial_sources=old.compact.check_initial_sources,
      validate_roles=old.validate_roles,verify_actual_handles=helper_ns['verify_actual_handles'],compiled_receipt=old.compiled_receipt,
      save_frontend_sources=save_frontend_sources)
    exec(compile(text,'<direct2ad-vs5cf6-existing-fullstep>','exec'),ns)
    return ns['run'](a)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--preflight-cpu',action='store_true')
    for name in ('consumer-raw','consumer-review','compile-summary','assembly-review','out','old-library','old-consumer-raw','old-consumer-review',
      'old-compile-summary','old-assembly-review','old-fullstep-summary','old-fullstep-review'):p.add_argument('--'+name,type=Path)
    p.add_argument('--native-library',dest='k8_library',type=Path)
    for name in ('consumer-archive-sha256','consumer-review-sha256'):p.add_argument('--'+name)
    p.add_argument('--case',choices=f.CASES,action='append');a=p.parse_args()
    if a.preflight_cpu:
        assert not any((a.gpu,a.root_owned_exclusive,a.out,a.case))
        assert all(v for k,v in vars(a).items() if k not in ('gpu','root_owned_exclusive','out','case'))
        plan();result=actual_consumer_gate(a);assert not {'torch','triton','numpy'}&set(sys.modules)
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    if not a.gpu:
        assert not any(v for k,v in vars(a).items() if k!='gpu')
        result=plan();assert not {'torch','triton','numpy'}&set(sys.modules);print(json.dumps(result,indent=2,sort_keys=True));return 0
    assert a.root_owned_exclusive and all(v for k,v in vars(a).items() if k not in ('preflight_cpu','case'))
    assert all(v.is_absolute() for v in vars(a).values() if isinstance(v,Path))
    if a.case:assert len(a.case)==len(set(a.case))
    return execute(a)
if __name__=='__main__':raise SystemExit(main())
