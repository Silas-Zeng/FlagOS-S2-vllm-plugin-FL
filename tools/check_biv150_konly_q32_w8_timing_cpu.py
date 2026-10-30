#!/usr/bin/env python3
"""Finite ABBA source inverse and pending-evidence refusal; no framework imports."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import types
sys.dont_write_bytecode=True
import biv150_konly_q32_w8_timing_source as t

def main():
    checks=[]
    def check(ok,label):t.need(ok,label);checks.append(label)
    def reject(fn,label):
        try:fn()
        except (ValueError,AssertionError,KeyError):checks.append(label);return
        raise ValueError('accepted '+label)
    plan=t.plan();source=t.run_source();block=t.timing_block();tree=ast.parse(source)
    original=t.n.original('probe_biv150_konly_q64_screen.py')
    begin=original.index('        for mode in f.MODES:\n            with torch.cuda.stream(stream):\n                for role in f.ROLES:\n                    for _ in range(8):replay(role,mode)')
    end=original.index('        result.update(completed=True,performance_measured=True,',begin)
    inverse=block.replace("('w16','w8','w8','w16')","('q32','q64','q64','q32')").replace("W16_over_W8=average['w16']/average['w8']","Q32_over_Q64=average['q32']/average['q64']")
    check(inverse==original[begin:end],'exact original timing-body inverse: only role/ratio labels')
    check(source.count(block)==1 and source.index(block)>source.index("result.update(numeric_verified=True,Graph_verified=True,side_stream_verified=True)"),'ABBA after all numeric/Graph/side gates')
    check('full_output_reference=full[\'finite\'] and full[\'pointwise_passed\']' in source and t.full_output_metrics is t.numeric.full_output_metrics,'complete4M gate and identical metric helper retained')
    check("graph_b={mode:stage('Graph-B-first'" in source and source.index("graph_b={mode:stage('Graph-B-first'")<source.index("stage('eager-B-after-Graph'"),'changed Graph B before fresh eager unchanged')
    check('for _ in range(8):replay(role,mode)' in block and 'for _ in range(16):replay(role,mode)' in block and "('w16','w8','w8','w16')" in block,'original warm8/timed16 ABBA per mode')
    check('begin.elapsed_time(end)/64' in block and 'wall*1000/64' in block and block.index('wall=time.perf_counter()-start')<block.index('post=snapshot(guarded[role])'),'event and host per64; validation excluded')
    check("cache_after==byte_cpu(uploaded[1])" in block and "row['readonly']==immutable" in block and "byte_cpu(gold_outputs[1,mode][role])" in block,'each timed block full same-arm bits/cache/readonly check')
    check((plan['timing_total_KV'],plan['aligned6144_timing_measured'],plan['event_blocks_total'])==(6143,False,8),'6143only/8blocks; aligned6144 numeric-only')
    check(plan['warps']=={'w16':16,'w8':8} and t.WARPS==t.numeric.WARPS and plan['stages']=={'w16':1,'w8':1},'actual w16/S1 and w8/S1 forwarded')
    names={n.attr for n in ast.walk(tree)if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='f'}
    check(all(hasattr(t,name)for name in names),'all reused worker contract attributes forwarded')
    check('original_BF16_quality_passed=all(r[\'original_BF16_quality_passed\']' in source and t.numeric.REVIEW_SHA=='5409b37ec09961b1459fbab689fd7407ee610341250eeb48857b3761e8a8c18d','original BF16 diagnostic and actual static cache binding retained')
    check(all(getattr(t,k) is None for k in ('NUMERIC_REVIEW_SHA','NUMERIC_MANIFEST_SHA','NUMERIC_SUMMARY_SHA','NUMERIC_RECEIPT_SHA','NUMERIC_RAW_FILES','NUMERIC_RAW_BYTES')),'actual numeric evidence fields explicitly unknown')
    reject(t.qualification_ready,'None cannot qualify None')
    reject(lambda:t.evidence({},{}),'no unverified evidence can reach timing')
    reject(lambda:t.admit(types.SimpleNamespace(timing=True),{}),'explicit timing flag cannot replace pending numeric evidence')
    for mode in ('--cpu-preflight','--gpu'):
        run=subprocess.run([sys.executable,'-B','-S',str(t.ROOT/'tools/probe_biv150_konly_q32_w8_timing.py'),mode],capture_output=True,text=True)
        check(run.returncode!=0 and 'actual numeric originals/Root array review still pending' in run.stderr,'actual '+mode+' refuses before file/framework/device work')
    check(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'stdlib CPU only')
    print(json.dumps(dict(schema='cjj-w16-w8-timing-pending-cpu-v1',count=len(checks),checks=checks,
        worker_sha256=t.sha(source),plan_sha256=t.sha(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n'),
        numeric_evidence_pending=True,GPU_executed=False,performance_measured=False,Root_timing_admission=False,service_verified=False),sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
