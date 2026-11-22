#!/usr/bin/env python3
"""Default pending plan; Root-only short paired native consumer, no writer."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
import biv150_native127_v8_timing as t


def execute(a):
    evidence=t.evidence(a) # Pending actual result pins refuse before a framework.
    parent,binding=t.c.admission(a)
    source,adapter=t.adapted_run(parent)
    clock=t.clock_source(parent)
    a.out.mkdir();result=dict(plan=t.plan(),evidence=evidence,binding=binding,source_adapter=adapter,
      clock_source_sha256=t.c.factory.sha(clock.encode()),completed=False,errors=[],records=[],performance_verified=False,
      numeric_verified=False,precision_gate=False,model_verified=False,service_verified=False,capacity_recovered=False)
    def save():(a.out/'v8-consumer-timing-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    save();shared={}
    try:
        for case in t.fixture.CASES:
            # One call frame per role/case is essential: no loop-cell retargeting.
            states={};arms={}
            for role,(digest,namespace) in t.c.ROLES.items():
                module=t.c.load(parent,'probe_biv150_native_k8_consumer_v2','cjj127_timing_'+role)
                f=t.fixture.namespace(parent,role,shared,t.c.load,lambda r,case=case:dict(t.plan(),role=r,cases=[case]))
                f.CASES=(case,)
                ns=dict(vars(module),f=f,prepare_case=t.prepare_case,PREPARED=states)
                exec(compile(source,'<127-original-consumer-new-timing-capture-only>','exec'),ns)
                out=a.out/('q'+str(case[2])+'-'+role)
                args=SimpleNamespace(library=a.control_library if role=='control' else a.library,
                  role_sha256=digest,namespace=namespace,role=role,out=out,codec_summary=a.codec_summary,codec_proof_sha256=a.codec_proof_sha256)
                status=ns['run'](args);assert status==0,'necessary new capture numeric gate refused '+role
                arms[role]=dict(summary_relative=str((out/'consumer-summary.json').relative_to(a.out)))
            import torch
            ns=dict(torch=torch,time=time);exec(compile(clock,'<original3eb-clock-byte-exact>','exec'),ns)
            record=t.run_pair(a,torch,states,ns['timed'],case);record['arms']=arms
            for role,x in arms.items():x['summary_sha256']=t.c.sha(a.out/x['summary_relative'])
            result['records'].append(record);save()
            del states
        result.update(completed=True,timing_executed=True,new_capture_B_verified=True,original_numeric_evidence_inherited=True,side_newly_executed=False)
    except Exception as exc:
        result['errors'].append(repr(exc));(a.out/'failure.txt').write_text(traceback.format_exc())
    save();print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['completed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--timing',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--root-timing-approved',action='store_true')
    for name in ('parent-source','compile-summary','static-review','control-library','library','codec-summary','numeric-summary','numeric-review','timing-admission','out'):p.add_argument('--'+name,type=Path)
    for name in ('static-review-sha256','codec-proof-sha256','numeric-review-sha256','timing-admission-sha256'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.gpu:
        if not all(v for k,v in vars(a).items() if k!='gpu'):p.error('explicit Root timing/serial flags and exact previous numeric/static/source/library/codec inputs required')
        return execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device/timing arguments require --gpu')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(t.plan(),indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
