#!/usr/bin/env python3
"""Default CPU; Root-only exact 8b vs5898 N64/S2 frozen consumer pair."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import traceback
import biv150_native117_pair_contract as c

def execute(a):
    root,binding=c.admission(a)
    source,adapter=c.consumer_source(root)
    a.out.mkdir();result=dict(plan=c.plan(),binding=binding,completed=False,errors=[],arms={},
        full_OML_final_bitwise=False,numeric_verified=False,performance_verified=False,
        precision_gate=False,service_verified=False)
    def checkpoint():
        (a.out/'consumer-pair-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    checkpoint()
    try:
        for role,(digest,namespace) in c.ROLES.items():
            module=c.load(root,'probe_biv150_native_k8_consumer_v2','cjj117_consumer_'+role)
            def role_plan():
                p=module.f.plan();p.update(cases=c.CASES,library_sha256=digest,pair_binding=binding,
                    pair_role=role,pair_plan=c.plan(),consumer_adapter=adapter,
                    kernel_or_dtype_changed=True,numeric_expression_or_dtype_changed=False,
                    qualification_inherited_from_old_reader=False)
                return p
            args=SimpleNamespace(library=a.main8b_library if role=='main8b' else a.library,
                role_sha256=digest,namespace=namespace,out=a.out/role,
                codec_summary=a.codec_summary,codec_proof_sha256=a.codec_proof_sha256)
            ns=dict(vars(module),role_plan=role_plan,PAIR_CASES=c.CASES)
            exec(compile(source,'<117-exact-v2-consumer-pair-adapter>','exec'),ns)
            status=ns['run'](args)
            p=args.out/'consumer-summary.json'
            result['arms'][role]=dict(summary_relative=str(p.relative_to(a.out)),summary_sha256=c.sha(p))
            checkpoint();assert status==0,'Original consumer refused '+role
        summaries={role:json.loads((a.out/x['summary_relative']).read_text()) for role,x in result['arms'].items()}
        result['bitwise']=c.compare_pair(summaries)
        result.update(completed=True,full_OML_final_bitwise=True,numeric_verified=True)
    except Exception as exc:
        result['errors'].append(repr(exc));(a.out/'failure.txt').write_text(traceback.format_exc())
    checkpoint();print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['completed'] else 1

def arguments():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true')
    for key in ('parent-source','compile-summary','assembly-review','main8b-library','library','codec-summary','out'):p.add_argument('--'+key,type=Path)
    for key in ('assembly-review-sha256','codec-proof-sha256'):p.add_argument('--'+key)
    return p

def main():
    p=arguments();a=p.parse_args()
    if a.gpu:
        if not all(v for k,v in vars(a).items() if k!='gpu'):p.error('Root ownership, frozen3eb parent, actual5898 compile/static receipt, both exact libraries, original codec and new out required')
        return execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device arguments require --gpu')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(c.plan(),indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
