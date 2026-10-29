#!/usr/bin/env python3
"""Default CPU plan; Root-only same-descriptor S1/S2 numeric plus one ABBA."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import biv150_konly_q32_s2_timing_source as t


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true');p.add_argument('--cpu-preflight',action='store_true')
    p.add_argument('--timing',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true')
    p.add_argument('--numeric-diagnostic',action='store_true');p.add_argument('--commit');p.add_argument('--out',type=Path)
    p.add_argument('--preflight',type=Path);p.add_argument('--preflight-sha256')
    p.add_argument('--numeric-review',type=Path);p.add_argument('--numeric-review-sha256')
    p.add_argument('--admission',type=Path);p.add_argument('--admission-sha256');a=p.parse_args()
    t.need(not(a.gpu and a.cpu_preflight),'explicit CPU preflight or GPU')
    if a.gpu or a.cpu_preflight:
        t.need(all((a.preflight,a.preflight_sha256,a.numeric_review,a.numeric_review_sha256)),'actual prepared proof and final numerical review')
        body=a.preflight.read_bytes();t.need(a.preflight_sha256==t.PREFLIGHT_SHA and t.sha(body)==t.PREFLIGHT_SHA,'fixed dual-root prepared proof')
        pre=json.loads(body);body=a.numeric_review.read_bytes()
        t.need(a.numeric_review_sha256==t.NUMERIC_REVIEW_SHA and t.sha(body)==t.NUMERIC_REVIEW_SHA,'fixed final numerical review')
        t.evidence(pre,json.loads(body))
        if a.gpu:
            t.need(all((a.timing,a.root_owned_exclusive,a.commit,a.out,a.admission,a.admission_sha256)),'Root timing/exclusive/exact new source/new output/separate decision required')
            value=t.run(a,pre)
        else:
            t.need(not any((a.timing,a.root_owned_exclusive,a.numeric_diagnostic,a.commit,a.out,a.admission,a.admission_sha256)),'CPU preflight rejects device flags')
            value=dict(plan=t.plan(),evidence_sha256=t.NUMERIC_REVIEW_SHA,preflight_sha256=t.PREFLIGHT_SHA,Root_timing_admission=False,GPU_executed=False)
    else:
        t.need(not any(v for k,v in vars(a).items() if k not in ('gpu','cpu_preflight')),'options require an explicit mode')
        value=t.plan()
    t.need(a.gpu or not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'CPU must stay stdlib')
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
