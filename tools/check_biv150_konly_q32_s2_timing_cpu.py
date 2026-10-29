#!/usr/bin/env python3
"""Stdlib checks for the exact post-numeric ABBA seam and closed Root gates."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
sys.dont_write_bytecode=True
import biv150_konly_q32_s2_timing_source as t


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',type=Path);p.add_argument('--numeric-review',type=Path);a=p.parse_args()
    checks=[]
    def check(v,label):t.need(v,label);checks.append(label)
    def reject(fn,label):
        try:fn()
        except (ValueError,AssertionError,KeyError):checks.append(label);return
        raise ValueError('mutation accepted '+label)
    plan=t.plan();source=t.run_source();block=t.timing_block();tree=ast.parse(source)
    check(source.count(block)==1 and source.index(block)>source.index("result.update(numeric_verified=True,Graph_verified=True,side_stream_verified=True)"),'original timing follows all numeric/Graph/side stages')
    check("('s1','s2','s2','s1')" in block and 'for _ in range(16):replay(role,mode)' in block and 'for _ in range(8):replay(role,mode)' in block,'one original ABBA with original warm/replay counts')
    check('begin.elapsed_time(end)/64' in block and 'wall*1000/64' in block,'event and host per64 complete steps')
    check(block.index('wall=time.perf_counter()-start')<block.index('post=snapshot(guarded[role])'),'host timing excludes post-validation/readback')
    check("cache_after==byte_cpu(uploaded[1])" in block and "row['readonly']==immutable" in block and "byte_cpu(gold_outputs[1,mode][role])" in block,'every timed block retains complete output/cache/readonly checks')
    check('role in f.ROLES' in block and 'for mode in f.MODES' in block,'both same actual arms and reader/append-reader modes')
    check((plan['timing_pattern'],plan['timing_total_KV'],plan['aligned6144_timing_measured'])==(1,6143,False),'only final6143 pattern timed;6144 numeric-only')
    check(8+8*2*2+16*4*2==168 and plan['event_blocks_total']==8 and plan['complete_steps_per_event_block']==64,'graph replay arithmetic; no inferred hardware count')
    check(all(plan[k] is False for k in ('GPU_executed','Root_timing_admission','Root_final_numeric_review_confirmed','Root_A4K_B4K_pair_closed_confirmed','timing_requested','performance_measured','precision_gate','service_verified')),'default plan grants no timing/device/model permission')
    check("graph_b={mode:stage('Graph-B-first'" in source and "original_BF16_quality_passed=all(r['original_BF16_quality_passed']" in source,'numeric Graph ordering and original BF16 failure preserved')
    check("'precision_gate'" not in block and 'threshold' not in block,'ABBA does not alter original numerical policy')
    check(any(isinstance(n,ast.Attribute) and n.attr=='Event' for n in ast.walk(tree)),'only the explicit timing worker contains device events')
    actual=None
    if a.preflight or a.numeric_review:
        t.need(a.preflight and a.numeric_review,'both actual small evidence files')
        prebody=a.preflight.read_bytes();body=a.numeric_review.read_bytes()
        check(t.sha(prebody)==t.PREFLIGHT_SHA and t.sha(body)==t.NUMERIC_REVIEW_SHA,'exact1fec prepared proof and Root-replayed fbf review')
        pre=json.loads(prebody);review=json.loads(body);t.evidence(pre,review);check(True,'same actual S1/S2 full implementation/Graph/side proof')
        for key,value in (('source_commit','0'*40),('model_verified',True),('original_BF16_quality_passed',True),('Graph_before_fresh_and_changed_counterexample',False)):
            bad=copy.deepcopy(review);bad[key]=value;reject(lambda:t.evidence(pre,bad),'reject numerical evidence '+key)
        with tempfile.TemporaryDirectory(prefix='cjj-124-timing-cpu-') as tmp:
            path=Path(tmp)/'pending.json';args=types.SimpleNamespace(timing=True,preflight_sha256=t.PREFLIGHT_SHA,numeric_review=a.numeric_review,numeric_review_sha256=t.NUMERIC_REVIEW_SHA,admission=path,commit='a'*40)
            base=t.admission_template(args.commit);flags=('Root_timing_approved','Root_final_numeric_review_confirmed','Root_current_A4K_B4K_pair_closed_no_continuation')
            path.write_text(json.dumps(base));args.admission_sha256=t.sha(path.read_bytes());reject(lambda:t.admit(args,pre),'pending Root receipt rejected')
            for key in flags:
                bad=dict(base,**{f:True for f in flags});bad[key]=False
                path.write_text(json.dumps(bad));args.admission_sha256=t.sha(path.read_bytes());reject(lambda:t.admit(args,pre),'independent Root gate '+key)
            for key,value in (('Root_timing_approved',1),('timing_total_KV',6144),('precision_qualified',True),('ABBA_rounds',2)):
                bad=dict(base,**{f:True for f in flags});bad[key]=value
                path.write_text(json.dumps(bad));args.admission_sha256=t.sha(path.read_bytes());reject(lambda:t.admit(args,pre),'reject timing receipt '+key+'/'+str(value))
            args.timing=False;reject(lambda:t.admit(args,pre),'explicit timing flag required')
        actual=dict(preflight_sha256=t.PREFLIGHT_SHA,Root_replayed_numeric_review_sha256=t.NUMERIC_REVIEW_SHA,
            original_BF16_quality_passed=False,Root_timing_approved=False)
    check(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'stdlib only/no device')
    print(json.dumps(dict(schema='cjj-124-timing-cpu-v1',checks=checks,count=len(checks),actual=actual,
        derived_worker_sha256=t.sha(source),plan_sha256=t.sha(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n'),
        GPU_executed=False,performance_measured=False,precision_gate=False,service_verified=False),sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
