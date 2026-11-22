#!/usr/bin/env python3
"""CPU-only invocation of frozen 4d fullstep's existing complete saved-array replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'tools'))
import probe_biv150_native_k8_compact_v_overlap_fullstep as adapter

SOURCE='4d1166792a5247b774d6b1977e60466574e6a34c'
SUMMARY='6f719fe84e83e547d0e42f7c2be5bdb408e64cdb107cc455994a681705e34c7c'
MANIFEST='4435696c50214a8ba60f6a4e4fbf7ff1970ce3f8fb202cb13f7b82b4c4c5613a'
REVIEW='90f6e18e142813e732694f453779b876048daaa8c122fd3e2d985dc2969cdb17'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','compile-raw','parent-source','codec-summary','original-library','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=REPO/'tools/probe_biv150_native_k8_compact_v_overlap_fullstep.py'
    assert source.read_bytes()==subprocess.check_output(['git','-C',str(REPO),'show',SOURCE+':tools/'+source.name])
    assert sha(a.raw/'download-manifest.json')==MANIFEST and sha(a.raw/'out/consumer-summary.json')==SUMMARY
    manifest=json.loads((a.raw/'download-manifest.json').read_text())
    assert (manifest['raw_files'],manifest['raw_bytes'])==(138,32796687)
    assert manifest['original_owner']==dict(pid=61196,start_ticks=355697119,same_start=True,state='Z')
    assert {str(p.relative_to(a.raw)) for p in a.raw.rglob('*') if p.is_file()}=={'download-manifest.json',*(r['relative'] for r in manifest['files'])}
    review=a.raw/'out/actual-so-review-input.json';assert sha(review)==REVIEW
    assert json.loads(review.read_text())['manual_Root_approval'] is True
    old=adapter.base_module();old.prepare(a.parent_source)
    args=SimpleNamespace(compact_consumer_raw=a.raw.resolve(),compile_summary=(a.compile_raw/'out/compile-summary.json').resolve(),
        assembly_review=review.resolve(),assembly_review_sha256=REVIEW,codec_summary=a.codec_summary.resolve(),
        original_library=a.original_library.resolve(),k8_library=(a.compile_raw/'out/build/cjj_native_k8_compact_v_overlap_probe.so').resolve(),
        variant_consumer_sha256=SUMMARY,variant_archive_manifest_sha256=MANIFEST)
    result=adapter.replay_consumer(args)
    result['review_wrapper_sha256']=sha(__file__)
    result['scope']['full_array_replay_without_new_reference_or_threshold']=True
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    assert not {'torch','triton'}.intersection(sys.modules)
    print(json.dumps(dict(report=str(a.out),sha256=sha(a.out),hard_checks=result['full_array_numeric_hard_checks'],
        cross_arm_byte_checks=result['full_field_cross_arm_byte_comparisons'],ideal_partial_failures=result['ideal_partial_diagnostic_failures_preserved'],
        GPU_or_model_acceptance=False,CPU_seconds=result['CPU_seconds']),sort_keys=True))

if __name__=='__main__':main()
