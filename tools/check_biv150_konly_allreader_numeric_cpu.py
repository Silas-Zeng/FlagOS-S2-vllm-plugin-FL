#!/usr/bin/env python3
"""Bounded CPU replay/refusals for the two actual allreader compile feeds."""
from bisect import bisect_right
from collections import Counter
import copy
import json
import sys

import biv150_konly_allreader_numeric_contract as c
from biv150_konly_prefill_reference import address,coverage


def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Unknown allreader contract admitted')


def partition(p):
    prefix=[0]
    for q in p['qlens']: prefix.append(prefix[-1]+q)
    bq=p['config']['BLOCK_Q'];starts=[n//bq+i for i,n in enumerate(prefix)];seen=[]
    for program in range(p['grid'][0]):
        seq=bisect_right(starts,program)-1;local=(program-starts[seq])*bq
        seen.extend(prefix[seq]+q for q in range(local,min(local+bq,p['qlens'][seq])))
    assert Counter(seen)==Counter(range(p['total']))
    return dict(tokens=p['total'],programs=p['grid'][0],exact_once=True)


def run():
    records={};geometry={}
    for profile in c.PINS:
        files,summary,root=c.original(profile);rows=[]
        for record in summary['records']:
            raw={e:files['out/'+record['files'][e]['path']] for e in ('source','ttir','ttgir','llir','asm','cubin')}
            proof=c.bind_actual(profile,record,record,raw,raw);rows.append(dict(role=record['label'],hash=record['hash'],matrix=proof['actual_native_matrix_counts']))
            bad=copy.deepcopy(record);bad['actual_options']['shared']=None;reject(lambda:c.bind_actual(profile,bad,record,raw,raw))
            bad=copy.deepcopy(record);mutated=dict(raw,asm=raw['asm']+b'\nsl_jump X;\n');bad['files']['asm']['sha256']=c.parent.factory.sha(mutated['asm'])
            reject(lambda:c.bind_actual(profile,bad,record,mutated,raw))
        records[profile]=dict(source_root=root,actual_roles=rows)
        geometry[profile]=[dict(plan=c.plan(profile,i),partition=partition(c.plan(profile,i))) for i in (0,1)]
    assert geometry['multi2'][0]['plan']['qlens']==[2047,1] and geometry['multi2'][1]['plan']['qlens']==[2046,2]
    assert all(g['plan']['total']==2048 for g in geometry['multi2'])
    assert all(max(g['plan']['qlens'])==1 and g['plan']['total']==64 for g in geometry['raw3d'])
    coverage('multi2',[33,2],[63,5]);coverage('raw3d',[1,1],[17,33],16)
    address(2**31-1,15,1,127)
    reject(lambda:c.plan('unknown'));reject(lambda:c.plan('raw3d',True))
    checks=dict(baseline_hard_passed=True,quantized_operand=dict(finite=True,pointwise_passed=True),candidate_numeric_passed=False)
    hard=dict.fromkeys(c.HARD,True);assert not c.policy(hard,checks,True)['precision_gate']
    reject(lambda:c.policy(hard,checks,False))
    for name in hard:
        reject(lambda name=name:c.policy(dict(hard,**{name:False}),checks,True))
        reject(lambda name=name:c.policy({k:v for k,v in hard.items() if k!=name},checks,True))
    reject(lambda:c.policy(hard,dict(checks,baseline_hard_passed=False),True))
    reject(lambda:c.policy(hard,dict(checks,quantized_operand=dict(finite=False,pointwise_passed=True)),True))
    reject(lambda:c.policy(hard,dict(checks,quantized_operand=dict(finite=True,pointwise_passed=False)),True))
    assert not any(name in sys.modules for name in ('torch','triton','numpy','vllm_fl'))
    return dict(cpu_only=True,actual_compile_originals=records,cases=geometry,default_framework_imports=False,
        source_ABI_options_IR_and_ASM_mutants_refused=True,hard_gate_missing_or_failed_refused=True,
        strict_default=True,original_proxy_failure_explicit_diagnostic_only=True,
        actual_binding_verified=False,actual_I8_operand_hoist_verified=False,raw3D_qlen2_verified=False,
        Graph_verified=False,timing_attempted=False,precision_gate=False,service_verified=False)


if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
