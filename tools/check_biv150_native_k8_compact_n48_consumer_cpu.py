#!/usr/bin/env python3
"""CPU fixture, pointer-plan and pre-device evidence-refusal checks only."""
import argparse
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile

import probe_biv150_native_k8_compact_n48_consumer as frontend
import biv150_native_k8_compact_n48_consumer_contract as contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path);p.add_argument('--golden-cpu',action='store_true')
    for name in ('actual-candidate-compile-summary','pending-candidate-review','actual-reference-consumer-summary'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    plan=frontend.plan();records=[];integer_dot_checks=0
    for live,pitch in frontend.CASES:
        cases=frontend.fixtures(live,pitch)
        for phase,c in cases.items():
            x,y=c['case48'],c['case64'];e=c['expected'];g=64
            assert x['rows']==48 and y['rows']==64 and x['segments']==y['segments']==4
            assert sum(v>0 for v in x['lengths'][g:-g])==sum(v>0 for v in y['lengths'][g:-g])==live
            assert y['lengths'][g+48:-g]==[0]*16
            assert x['lengths'][g:-g]==y['lengths'][g:g+48]
            assert x['table'][g:-g]==y['table'][g:g+48*16] and y['table'][g+48*16:-g]==[-1]*(16*16)
            assert x['query_bits'][g:-g]==y['query_bits'][g:g+48*pitch] and y['query_bits'][g+48*pitch:-g]==[0]*(16*pitch)
            assert x['qcodes']==y['qcodes'][:48] and x['qscales']==y['qscales'][:48]
            # Assemble actual row slices of the same64-row backing, never a
            # regenerated local-index0..15 query for the second reference call.
            sliced=[y['query_bits'][g+row*pitch:g+row*pitch+2048] for start in (0,32) for row in range(start,start+32) if row<48]
            assert sliced==[x['query_bits'][g+row*pitch:g+row*pitch+2048] for row in range(48)]
            assert all(len(set(scales))==16 for scales in x['qscales'])
            # Full128D scalar INT64-control values at distinct rows/heads.
            for row,head in ((0,0),(31,7),(32,8),(47,15)):
                page=(row*3+phase*5)%29;slot=(row+phase)%16;kh=head//8
                d0=sum(int(k)*int(q) for k,q in zip(x['kcodes'][page][slot][kh],x['qcodes'][row][head]))
                d1=sum(int(k)*int(q) for k,q in zip(y['kcodes'][page][slot][kh],y['qcodes'][row][head]))
                assert d0==d1 and abs(d0)<=128*127**2;integer_dot_checks+=1
            assert len(e['cache'])==128+29*12544+128
            assert e['cache'][:128]==x['cache'][:128] and e['cache'][-128:]==x['cache'][-128:]
            assert all(cases[0]['expected'][key]!=cases[1]['expected'][key] for key in ('query','cache','lengths','table'))
            record=dict(live_rows=live,pitch=pitch,phase=phase,query_sha256=frontend.source.sha(e['query']),
              cache_sha256=frontend.source.sha(e['cache']),lengths_sha256=frontend.source.sha(e['lengths']),table_sha256=frontend.source.sha(e['table']),
              split32_row_order_and16_zero_padding_exact=True)
            if a.golden_cpu:
                import numpy as np
                gx=frontend.f.golden(x);gy=frontend.f.golden(y)
                for field in ('partial','maximum','summation','output'):
                    assert np.array_equal(gx[field],gy[field][:48],equal_nan=True),field
                record['CPU_complete_INT64_FP64_first48_match']=True
                record['full_D128_integer_products']=gx['full_D128_integer_products']
            records.append(record)
    # Real source/compile admission stays unavailable until actual records.
    rejections=[]
    with tempfile.TemporaryDirectory(prefix='cjj-n48-consumer-contract-') as directory:
        p=Path(directory)/'incomplete-reference.json'
        p.write_text(json.dumps(dict(completed=False)))
        try:contract.reference_consumer(p,frontend.source.sha(p.read_bytes()),{},contract.REFERENCE_LIBRARY_SHA)
        except (AssertionError,KeyError):rejections.append('unfinished actual reference rejected')
        else:raise AssertionError('unfinished reference accepted')
        try:contract.reference_consumer(p,'0'*64,{},contract.REFERENCE_LIBRARY_SHA)
        except AssertionError:rejections.append('wrong reference summary SHA rejected')
        else:raise AssertionError('wrong SHA accepted')
        candidate_contract,_,_=contract.contracts()
        scope=dict(rows=48,segments=4,namespace=frontend.source.NAMESPACE,consumer_only=True,service_or_model_verified=False)
        for label,scope_,approved in (('Root approval absent',scope,False),('S2 scope',dict(scope,segments=2),True),('N32 scope',dict(scope,rows=32),True)):
            review=dict(scope=scope_,manual_Root_approval=approved)
            p.write_text(json.dumps(review));digest=frontend.source.sha(p.read_bytes())
            try:candidate_contract.admission(Path(directory)/'not-yet-compiled.json',p,digest,Path(directory)/'unknown.so')
            except AssertionError:rejections.append(label+' rejected before absent compile/library access')
            else:raise AssertionError('scope or approval accepted')
    cpp=frontend.source.path('qk.cpp').read_text()
    assert cpp.index('TORCH_CHECK(segments==4')<cpp.index('"Q BF16 shape"')<cpp.index('c10::cuda::CUDAGuard guard(q.device())')<cpp.index(' launch(q.data_ptr()')
    assert cpp.count('TORCH_CHECK(segments==4')==1
    actual=None
    paths=(a.actual_candidate_compile_summary,a.pending_candidate_review,a.actual_reference_consumer_summary)
    if any(paths):
        assert all(paths)
        candidate_contract,_,_=contract.contracts()
        binding=candidate_contract.compile_binding(a.actual_candidate_compile_summary)
        pending=json.loads(a.pending_candidate_review.read_text())
        assert pending['manual_Root_approval'] is False and all(pending[k] is False for k in candidate_contract.REVIEW_FLAGS)
        assert set(pending['static_resources'])=={'S4'}
        try:candidate_contract.admission(a.actual_candidate_compile_summary,a.pending_candidate_review,frontend.source.sha(a.pending_candidate_review.read_bytes()),Path('/unopened-unknown-library.so'))
        except AssertionError:pass
        else:raise AssertionError('pending actual-SO approval admitted')
        ref=json.loads(a.actual_reference_consumer_summary.read_text())
        checked=contract.reference_consumer(a.actual_reference_consumer_summary,contract.REFERENCE_CONSUMER_SHA,ref['compact_admission'],contract.REFERENCE_LIBRARY_SHA)
        actual=dict(candidate_compile_artifacts_checked=binding,pending_actual_SO_review_rejected=True,reference_consumer_summary_and_all_artifacts_checked=checked,
          reference_compile_SO_admission_not_reexecuted_by_this_CPU_checker=True,device_prerequisites_still_pending_Root_approval=True)
    assert not {'torch','triton'}&set(sys.modules)
    if not a.golden_cpu:assert 'numpy' not in sys.modules
    report=dict(schema=1,passed=True,plan=plan,fixture_cases=records,scalar_full128D_control_checks=integer_dot_checks,
      prerequisite_refusals=rejections,golden_CPU_requested=a.golden_cpu,actual_candidate_SO=None,actual_reference_consumer_qualified=False,
      actual_host_rejection_source_order_verified=True,actual_host_rejections_executed=False,
      actual_prerequisite_CPU_checks=actual,
      device_executed=False,numeric_GPU_verified=False,Graph_verified=False,side_stream_verified=False,
      timing=None,service_activated=False,kernels_changed=False)
    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if a.out:assert not a.out.exists();a.out.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
