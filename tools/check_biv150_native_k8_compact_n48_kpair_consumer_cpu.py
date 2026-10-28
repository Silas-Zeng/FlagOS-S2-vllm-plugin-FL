#!/usr/bin/env python3
"""Narrow local CPU checks of the 114 frontend identity adapter; no numerical GPU test."""
import argparse,hashlib,inspect,json,sys,tempfile
from pathlib import Path

import probe_biv150_native_k8_compact_n48_kpair_consumer as adapter
import probe_biv150_native_k8_compact_n48_consumer as original
import biv150_native_k8_compact_n48_kpair_admission as admission


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--root-review',type=Path,required=True);p.add_argument('--wrong-library',type=Path,required=True);p.add_argument('--foreign-review',type=Path,required=True);a=p.parse_args()
    ns,proof=adapter.frontend()
    assert ns['CASES']==original.CASES==((41,2048),(41,2560),(42,2048),(42,2560))
    assert ns['FIELDS']==original.FIELDS and ns['SHAPES']==original.SHAPES and ns['PHASES']==original.PHASES
    assert ns['f'] is original.f and ns['require_codec'] is original.require_codec
    assert ns['admission'].reference_consumer is original.admission.reference_consumer
    assert ns['fixtures'].__code__.co_code==original.fixtures.__code__.co_code
    c=admission.contracts()[0];summary=a.raw/'out/compile-summary.json'
    library=a.raw/'out/build/cjj_native_k8_compact_n48_kpair_probe.so'
    accepted=c.admission(summary,a.root_review,admission.ROOT_REVIEW,library)
    rejected=[]
    def reject(name,fn):
        try:fn()
        except (AssertionError,FileNotFoundError,KeyError) as exc:rejected.append(dict(case=name,rejection=type(exc).__name__))
        else:raise AssertionError('accepted invalid input: '+name)
    reject('wrong_actual_library',lambda:c.admission(summary,a.root_review,admission.ROOT_REVIEW,a.wrong_library))
    reject('wrong_receipt_sha',lambda:c.admission(summary,a.root_review,'0'*64,library))
    with tempfile.TemporaryDirectory(prefix='native114-CPU-only-') as td:
        d=Path(td);receipt=json.loads(a.root_review.read_text())
        for field,value in (('manual_Root_approval',False),('actual_static_review_passed',False),('producer_reviews',[])):
            changed=dict(receipt);changed[field]=value;path=d/(field+'.json');path.write_text(json.dumps(changed))
            reject('changed_pinned_Root_receipt_'+field,lambda path=path:c.admission(summary,path,admission.ROOT_REVIEW,library))
        reject('missing_actual_receipt',lambda:c.admission(summary,d/'missing.json',admission.ROOT_REVIEW,library))
        oldroot=adapter.ROOT
        try:
            target=d/adapter.ORIGINAL;target.parent.mkdir(parents=True);raw=(oldroot/adapter.ORIGINAL).read_bytes();target.write_bytes(raw.replace(b'pointwise_atol=.005',b'pointwise_atol=.006'))
            assert target.read_bytes()!=raw;adapter.ROOT=d
            reject('changed_original_numeric_threshold',adapter.adapted_source)
        finally:adapter.ROOT=oldroot
    foreign=json.loads(a.foreign_review.read_text());assert foreign['SO_sha256']=='5cf698e8a60272d86b2545364f580faf4c40114e0044e91557c1a137f2521bed'
    reject('actual_PR113_receipt_not_PR114',lambda:c.admission(summary,a.foreign_review,admission.ROOT_REVIEW,library))
    observe=lambda pointer:dict(pointer=pointer)
    aligned=admission.record_default_alignment(4096,4096,4096+4352,observe)
    assert aligned['pointer_mod8']==dict(key=0,scale=0,value=0) and aligned['shifted_4B_non8B_exercised'] is False
    reject('whole_cache_4B_not8B_out_of_current_scope',lambda:admission.record_default_alignment(4100,4100,4100+4352,observe))
    reject('invalid_V_offset',lambda:admission.record_default_alignment(4096,4096,4096+4360,observe))
    assert len(rejected)==10 and not {'torch','numpy','triton'}&set(sys.modules)
    print(json.dumps(dict(completed=True,errors=[],CPU_only=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      adapter=proof,actual_SO_sha256=accepted['library_sha256'],actual_Root_review_sha256=accepted['review_sha256'],
      scope=accepted['scope'],original_fixtures_and_reference_functions_identical=True,refusals=rejected,
      CPU_fake_pointer_alignment_scope_check_only=aligned,actual_device_alignment_queried=False,
      negative_checks_refuse_modified_receipt_by_exact_SHA=True,GPU_numeric_verified=False,Graph_verified=False,
      side_stream_verified=False,fullstep_enabled=False,fullstep_consumer_summary_sha256=None,performance_verified=False),indent=2,sort_keys=True))
if __name__=='__main__':main()
