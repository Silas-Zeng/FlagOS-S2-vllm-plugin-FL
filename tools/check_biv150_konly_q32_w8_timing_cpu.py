#!/usr/bin/env python3
"""Finite ABBA source inverse and pending-evidence refusal; no framework imports."""
import ast
import argparse
import copy
import tempfile
import json
from pathlib import Path
import subprocess
import sys
import types
sys.dont_write_bytecode=True
import biv150_konly_q32_w8_timing_source as t

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',type=Path);parser.add_argument('--numeric-review',type=Path);parser.add_argument('--numeric-raw',type=Path);a=parser.parse_args()
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
    check(t.qualification_fields_known(),'six actual numeric evidence pins known')
    t.qualification_ready()
    reject(lambda:t.evidence({},{}),'missing proof structure cannot qualify')
    reject(lambda:t.admit(types.SimpleNamespace(timing=False),{}),'explicit timing opt-in remains required')
    for mode in ('--cpu-preflight','--gpu'):
        run=subprocess.run([sys.executable,'-B','-S',str(t.ROOT/'tools/probe_biv150_konly_q32_w8_timing.py'),mode],capture_output=True,text=True)
        check(run.returncode!=0 and 'actual prepared proof and final numerical review' in run.stderr,'actual '+mode+' refuses absent evidence paths before framework/device work')
    actual=None
    if any((a.preflight,a.numeric_review,a.numeric_raw)):
        t.need(all((a.preflight,a.numeric_review,a.numeric_raw)),'all three actual input paths')
        prebody=a.preflight.read_bytes();body=a.numeric_review.read_bytes()
        check(t.sha(prebody)==t.PREFLIGHT_SHA and t.sha(body)==t.NUMERIC_REVIEW_SHA,'exact actual9a78 prepared and b9f3 Root review')
        pre=json.loads(prebody);review=json.loads(body);t.evidence(pre,review);check(True,'actual complete8-stage implementation/Graph/side/full-output review admitted as evidence')
        raw=a.numeric_raw.resolve();body=(raw/'download-manifest.json').read_bytes()
        check(t.sha(body)==t.NUMERIC_MANIFEST_SHA,'actual numeric manifest identity')
        manifest=json.loads(body)
        check(manifest['raw_files']==len(manifest['files'])==t.NUMERIC_RAW_FILES and manifest['raw_bytes']==t.NUMERIC_RAW_BYTES,'actual92-file numeric archive geometry')
        for item in manifest['files']:
            path=raw/item['relative'];t.need(path.resolve().is_relative_to(raw),'archive path escape')
            data=path.read_bytes();t.need(len(data)==item['bytes'] and t.sha(data)==item['sha256'],'actual raw SHA '+item['relative'])
        check(True,'all92 actual raw file SHA verified; no array/model rerun')
        body=(raw/'out/summary.json').read_bytes();summary=json.loads(body)
        check(t.sha(body)==t.NUMERIC_SUMMARY_SHA and summary['commit']==t.PARENT and summary['completed'] is True and summary['errors']==[] and summary['Root_numeric_admission_receipt_sha256']==t.NUMERIC_RECEIPT_SHA,'actual summary/source/numeric receipt bound')
        for key,value in (('elements',None),('elements',4194303),('pointwise_violations',1),('finite',False)):
            bad=copy.deepcopy(review);metric=bad['stage_checks'][0]['full_output_reference']
            if value is None:metric.pop(key)
            else:metric[key]=value
            reject(lambda:t.evidence(pre,bad),'reject full-output metric '+key+'/'+str(value))
        with tempfile.TemporaryDirectory(prefix='cjj-w8-timing-pins-cpu-') as tmp:
            path=Path(tmp)/'pending.json';args=types.SimpleNamespace(timing=True,preflight_sha256=t.PREFLIGHT_SHA,numeric_review=a.numeric_review,numeric_review_sha256=t.NUMERIC_REVIEW_SHA,admission=path,commit='f'*40)
            receipt=t.admission_template(args.commit);path.write_text(json.dumps(receipt));args.admission_sha256=t.sha(path.read_bytes())
            reject(lambda:t.admit(args,pre),'actual numeric proof cannot replace pending Root timing decision')
            flags=('Root_timing_approved','Root_final_numeric_review_confirmed','Root_current_queue_closed_no_continuation')
            for key in flags:
                value=dict(receipt,**{k:True for k in flags});value[key]=False;path.write_text(json.dumps(value));args.admission_sha256=t.sha(path.read_bytes())
                reject(lambda:t.admit(args,pre),'independent Root timing gate '+key)
        actual=dict(preflight_sha256=t.PREFLIGHT_SHA,Root_numeric_review_sha256=t.NUMERIC_REVIEW_SHA,manifest_sha256=t.NUMERIC_MANIFEST_SHA,summary_sha256=t.NUMERIC_SUMMARY_SHA,raw_files=t.NUMERIC_RAW_FILES,raw_bytes=t.NUMERIC_RAW_BYTES,original_BF16_quality_passed=review['original_BF16_quality_passed'],Root_timing_admission=False)
    check(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'stdlib CPU only')
    print(json.dumps(dict(schema='cjj-w16-w8-timing-pending-cpu-v1',count=len(checks),checks=checks,
        worker_sha256=t.sha(source),plan_sha256=t.sha(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n'),
        actual=actual,numeric_evidence_pending=False,GPU_executed=False,performance_measured=False,Root_timing_admission=False,service_verified=False),sort_keys=True,indent=2,allow_nan=False))

if __name__=='__main__':main()
