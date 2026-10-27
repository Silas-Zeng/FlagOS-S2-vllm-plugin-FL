#!/usr/bin/env python3
"""CPU refusals for actual-main selection and per-length diagnostic admission."""
import json
import sys
from types import SimpleNamespace

import bench_biv150_konly_q32_attention_perf as perf


def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Unqualified attention performance admitted')


def run():
    source=perf.main_control.source_contract()
    assert source['accepted_main_commit']=='dac1b8fb48e7983e48283032efeee9b592c5fbe5'
    assert source['accepted_attention_bytes_equal_to56f'] and not source['actual_parent_selected']
    geometries={str(k):perf.plan(k) for k in (4096,6144,16384)}
    assert [(v['pages'],v['context']) for v in geometries.values()]==[(517,2048),(773,4096),(2053,14336)]
    reject(lambda:perf.plan(6143));reject(lambda:perf.plan(True))
    manager=SimpleNamespace(optimized={'good':dict(mode='selected',mechanism=dict(mechanism_pass=True))})
    kernel=SimpleNamespace(hash='good');perf.main_control.require_selected(manager,kernel)
    reject(lambda:perf.main_control.require_selected(manager,SimpleNamespace(hash='raw96KB')))
    reject(lambda:perf.main_control.require_selected(manager,None))
    for value in (dict(mode='fallback',mechanism=dict(mechanism_pass=True)),dict(mode='selected',mechanism=dict(mechanism_pass=False))):
        reject(lambda value=value:perf.main_control.require_selected(SimpleNamespace(optimized={'good':value}),kernel))
    # Current candidate checkout/evidence head is not the separate DAC1 main.
    reject(lambda:perf.main_control.root_contract(perf.frozen.parent.ROOT))
    checks=dict(baseline_hard_passed=True,quantized_operand=dict(finite=True,pointwise_passed=True),candidate_numeric_passed=False)
    hard=dict.fromkeys(perf.HARD,True)
    admitted=perf.timing_gate(hard,checks,True)
    assert admitted['implementation_gate'] and not admitted['precision_gate'] and admitted['fullcost_ms'] is None and admitted['writer_ms'] is None
    reject(lambda:perf.timing_gate(hard,checks,False))
    for field in hard:
        reject(lambda field=field:perf.timing_gate(dict(hard,**{field:False}),checks,True))
        reject(lambda field=field:perf.timing_gate({k:v for k,v in hard.items() if k!=field},checks,True))
    reject(lambda:perf.timing_gate(hard,dict(checks,baseline_hard_passed=False),True))
    reject(lambda:perf.timing_gate(hard,dict(checks,quantized_operand=dict(finite=True,pointwise_passed=False)),True))
    owned=perf.Owned(lambda:None,lambda:{})
    owned.token='fresh';owned.a=SimpleNamespace(state='active',token='fresh',owner={'pid':7},roots={'main':'dac1'},native=701,
        identity=lambda:{'pid':7},binding=lambda:{'main':'dac1'},closed=lambda:True,query=lambda:[701])
    owned.check('numeric');owned.a.query=lambda:[7];reject(lambda:owned.check('timing'))
    owned.a.query=lambda:[701,702];reject(lambda:owned.check('timing'))
    owned.a.query=lambda:[701];owned.token='stale';reject(lambda:owned.check('timing'))
    assert not any(name in sys.modules for name in ('torch','triton','numpy','vllm_fl'))
    return dict(cpu_only=True,accepted_main=source,one_case_geometries=geometries,
        selected_Manager_and_lowered_proof_required=True,old_raw96KB_refused_as_timing_baseline=True,
        per_actual_length_fresh_numerics_required=True,implementation_failure_not_diagnostic_bypassable=True,
        numeric_proxy_failure_requires_explicit_diagnostic=True,writer_ms=None,fullcost_ms=None,
        actual_materializer_executed=False,actual_timing_executed=False,precision_gate=False,
        performance_verified=False,service_verified=False,cache_flush_or_parameter_sweep=False)


if __name__=='__main__': print(json.dumps(run(),indent=2,sort_keys=True))
