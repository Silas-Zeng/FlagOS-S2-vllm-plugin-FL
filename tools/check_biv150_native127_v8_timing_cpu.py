#!/usr/bin/env python3
"""Finite-source/pending admission checks; no numeric mirror or device mocks."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import biv150_native127_v8_timing as t


def reject(fn):
    try:fn()
    except (AssertionError,KeyError,TypeError,ValueError):return
    raise AssertionError('Pending/bad evidence admitted')


def check(a):
    parent=t.c.previous.parent(a.parent_source)
    source,adapter=t.adapted_run(parent);assert adapter['inverse_byte_exact'];compile(source,'<timing-worker>','exec')
    original=t.c.adapted_run(parent)[0]
    def functions(text):return {n.name:ast.get_source_segment(text,n) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef)}
    new,old=functions(source),functions(original)
    names=('blob','npblob','write_json','host','bytes_gpu','bf16_float','poison','change','snapshot','check','producer','reducer','eager')
    assert all(new[name]==old[name] for name in names)
    clock=t.clock_source(parent);compile(clock,'<original-clock>','exec')
    raw=t.c.previous.function(parent,'probe_biv150_native_k8_fullstep','run')
    assert clock==functions(raw)['timed']
    assert (t.UNROLL,t.WARM,t.REPLAYS)==(8,128,64) and t.ORDER==('control','packed_v8','packed_v8','control')
    plan=t.plan();assert plan['complete_steps_per_sample']==512 and plan['event_blocks_total']==8
    t.ready()
    pending=copy.deepcopy(t.NUMERIC)
    try:
        t.NUMERIC.update({key:None for key in t.NUMERIC});reject(t.ready)
        reject(lambda:t.evidence(SimpleNamespace(root_timing_approved=True)))
        t.NUMERIC.clear();t.NUMERIC.update(pending);t.NUMERIC['raw_files']=True;reject(t.ready)
        t.NUMERIC['raw_files']=pending['raw_files'];t.NUMERIC['raw_bytes']=None;reject(t.ready)
    finally:t.NUMERIC.clear();t.NUMERIC.update(pending)
    actual=(a.numeric_summary,a.numeric_review,a.timing_admission)
    assert all(actual) or not any(actual)
    evidence=None
    if all(actual):
        args=SimpleNamespace(root_timing_approved=True,numeric_summary=a.numeric_summary,numeric_review=a.numeric_review,
          numeric_review_sha256=t.NUMERIC['Root_review_sha256'],timing_admission=a.timing_admission,timing_admission_sha256=t.TIMING_ADMISSION_SHA)
        evidence=t.evidence(args)
        for field,value in (('root_timing_approved',False),('root_timing_approved',1),('numeric_review_sha256',None),('timing_admission_sha256',None),('numeric_summary',a.numeric_review)):
            bad=copy.copy(args);setattr(bad,field,value);reject(lambda:t.evidence(bad))
    script=Path(__file__).with_name('probe_biv150_native127_v8_timing.py')
    p=subprocess.run([sys.executable,'-B','-S',str(script)],capture_output=True,text=True,check=True)
    assert json.loads(p.stdout)==json.loads(json.dumps(t.plan()))
    for args in (['--gpu'],['--gpu','--timing','--root-owned-exclusive'],['--timing']):
        assert subprocess.run([sys.executable,'-B','-S',str(script),*args],capture_output=True,text=True).returncode==2
    tree=ast.parse(script.read_text());outer=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='t.fixture.CASES')
    assert any(isinstance(n,ast.Assign) and any(ast.unparse(x)=='f.CASES' for x in n.targets) and ast.unparse(n.value)=='(case,)' for n in ast.walk(outer))
    assert 'gold={1:f.golden(cases[1])}' in source and "for p in (0,1)}" in source
    assert 'timing_capture_B_verified=True' in source and 'side_stream_verified=False' in source
    assert "f.cpu_oracles(cases,blob)" not in source
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    return dict(passed=True,adapter=adapter,original_numeric_functions_byte_exact=list(names),
      original3eb_clock_source_byte_exact=True,clock_source_sha256=t.c.factory.sha(clock.encode()),
      original_warm128_unroll8_replays64_AB_then_BA=True,complete_steps_per_event_block=512,
      no_same_full_numeric_matrix_rerun=True,new_capture_only_B_validation=True,
      per_role_case_separate_run_frame=True,actual_numeric_and_scope_receipt_replayed=bool(evidence),actual_evidence=evidence,
      missing_pin_and_unapproved_device_refused_before_framework=True,Root_flag_was_CPU_stub_only=bool(evidence),
      default_stdlib_only=True,GPU_executed=False,numeric_verified=False,performance_verified=False,
      precision_gate=False,model_verified=False,service_verified=False,capacity_recovered=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-source',type=Path,required=True)
    for name in ('numeric-summary','numeric-review','timing-admission'):p.add_argument('--'+name,type=Path)
    print(json.dumps(check(p.parse_args()),indent=2,sort_keys=True))
