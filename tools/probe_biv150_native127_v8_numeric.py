#!/usr/bin/env python3
"""Default CPU plan. Root-only private V8 vs explicit BF16 dequant-V consumer."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import biv150_native127_v8_numeric as c
import biv150_native127_v8_fixture as fixture


def execute(a):
    parent,binding=c.admission(a) # Before framework import/input construction.
    source,adapter=c.adapted_run(parent)
    a.out.mkdir();result=dict(plan=c.plan(),binding=binding,consumer_source_adapter=adapter,
      completed=False,errors=[],arms={},numeric_verified=False,performance_verified=False,
      precision_gate=False,model_verified=False,service_verified=False,capacity_recovered=False)
    def save():(a.out/'v8-consumer-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    save();shared={}
    try:
        for role,(digest,namespace) in c.ROLES.items():
            module=c.load(parent,'probe_biv150_native_k8_consumer_v2','cjj127_run_'+role)
            f=fixture.namespace(parent,role,shared,c.load,c.plan)
            ns=dict(vars(module),f=f)
            exec(compile(source,'<127-frozen-consumer-finite-ABI-adapter>','exec'),ns)
            args=SimpleNamespace(library=a.control_library if role=='control' else a.library,
              role_sha256=digest,namespace=namespace,role=role,out=a.out/role,
              codec_summary=a.codec_summary,codec_proof_sha256=a.codec_proof_sha256)
            status=ns['run'](args);p=args.out/'consumer-summary.json'
            result['arms'][role]=dict(summary_relative=str(p.relative_to(a.out)),summary_sha256=c.sha(p));save()
            assert status==0,'Original consumer hard gate refused '+role
        arms={role:json.loads((a.out/x['summary_relative']).read_text()) for role,x in result['arms'].items()}
        result['cross_arm']=c.compare(arms)
        proxy=[item['metrics']['original_BF16_proxy']['passed'] for report in arms.values() for row in report['records'] for item in row['checks'] if 'original_BF16_proxy' in item['metrics']]
        result.update(completed=True,numeric_verified=True,full_OML_final_bitwise=True,
          original_BF16_proxy_passed=all(proxy),Graph_verified=True,side_stream_verified=True)
    except Exception as exc:
        result['errors'].append(repr(exc));(a.out/'failure.txt').write_text(traceback.format_exc())
    save();print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['completed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true')
    for name in ('parent-source','compile-summary','static-review','control-library','library','codec-summary','out'):p.add_argument('--'+name,type=Path)
    for name in ('static-review-sha256','codec-proof-sha256'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.gpu:
        if not all(v for k,v in vars(a).items() if k!='gpu'):p.error('Root serial slot, exact static/compile/codec proofs, both separate libraries, frozen3eb parent and new output required')
        return execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device arguments require --gpu')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(c.plan(),indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
