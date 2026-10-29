#!/usr/bin/env python3
"""Bounded CPU source/scope/binding faults; does not run reference arrays or a framework."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import biv150_native117_pair_contract as c

def reject(call):
    try:call()
    except (AssertionError,KeyError,TypeError,ValueError):return
    raise AssertionError('Fault was accepted')

def check(a):
    source,adapter=c.consumer_source(a.parent_source)
    receipt,receipt_adapter=c.receipt_source(a.parent_source)
    assert len(ast.parse(source).body)==len(ast.parse(receipt).body)==1
    assert adapter['inverse_byte_exact'] and receipt_adapter['inverse_byte_exact']
    assert c.CASES==((64,2,2048),(64,2,2560))
    script=Path(__file__).with_name('probe_biv150_native117_n64_consumer.py')
    result=subprocess.run([sys.executable,'-B','-S',str(script)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==c.plan()
    cli=[]
    for args in (['--gpu'],['--gpu','--root-owned-exclusive'],['--parent-source',str(a.parent_source)]):
        r=subprocess.run([sys.executable,'-B','-S',str(script),*args],capture_output=True,text=True)
        assert r.returncode==2;cli.append(args)
    # Small synthetic blob metadata is solely a CPU counterexample to bitwise
    # comparison; it is never a consumer result/receipt and admits no device.
    summaries={role:dict(records=[dict(case=list(case),snapshots={stage:{
        field:dict(bytes=4,sha256='0'*64,path='synthetic-not-a-device-result')
        for field in ('partial','maximum','summation','output')} for stage in c.STAGES}) for case in c.CASES]) for role in c.ROLES}
    c.compare_pair(summaries);faults=[]
    for field in ('partial','maximum','summation','output'):
        bad=copy.deepcopy(summaries);bad['lookahead']['records'][0]['snapshots']['A-after'][field]['sha256']='1'*64
        reject(lambda:c.compare_pair(bad));faults.append('cross_arm_'+field+'_bit_difference')
    bad=copy.deepcopy(summaries);bad['lookahead']['records'][0]['case'][0]=32
    reject(lambda:c.compare_pair(bad));faults.append('N32_out_of_scope')
    bad=copy.deepcopy(summaries)
    for arm in bad.values():del arm['records'][0]['snapshots']['graphB-after']
    reject(lambda:c.compare_pair(bad));faults.append('both_arms_missing_Graph_stage')
    if a.actual_raw:
        root=a.actual_raw/'out/compile'
        args=SimpleNamespace(root_owned_exclusive=True,parent_source=a.parent_source,
            compile_summary=root/'compile-summary.json',assembly_review=a.actual_review,
            assembly_review_sha256=c.REVIEW_SHA,main8b_library=a.actual_main8b,
            library=root/'build/cjj_native_k8_klookahead_probe.so',codec_summary=a.actual_codec,
            codec_proof_sha256=c.CODEC_SHA,out=Path('/private/tmp/cjj117-cpu-admission-no-device-created-20261029'))
        c.admission(args)
        for field,value in (('root_owned_exclusive',False),('assembly_review_sha256',None),
            ('main8b_library',args.library),('library',args.main8b_library),('codec_proof_sha256',None)):
            bad=copy.copy(args);setattr(bad,field,value);reject(lambda:c.admission(bad));faults.append('admission_'+field)
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(completed=True,default_stdlib_only=True,consumer_adapter=adapter,
        receipt_adapter=receipt_adapter,negative_checks=faults,CLI_rejections=cli,
        actual_CPU_binding_replayed=bool(a.actual_raw),new_reference_framework=False,
        GPU_executed=False,numeric_verified=False,performance_verified=False,precision_gate=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    for name in ('actual-raw','actual-review','actual-main8b','actual-codec'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    assert all((a.actual_raw,a.actual_review,a.actual_main8b,a.actual_codec)) or not any((a.actual_raw,a.actual_review,a.actual_main8b,a.actual_codec))
    print(json.dumps(check(a),indent=2,sort_keys=True))
