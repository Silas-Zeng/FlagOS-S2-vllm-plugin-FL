#!/usr/bin/env python3
"""Narrow CPU regression for frozen-source reuse, sequencing, fail-closed entry."""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import biv150_native_i8_gate2048_numeric_contract as c
import check_biv150_native_i8_gate2048_cpu as original


def main():
 p=argparse.ArgumentParser();p.add_argument('--static-review',type=Path);a=p.parse_args()
 checks=[]
 def check(value,label):
  assert value,label
  checks.append(label)
 quant=c.source_proof();plan=c.plan()
 check(plan['static_review_pending'] and plan['static_review_sha256'] is None and plan['Root_admission_sha256'] is None,'unknown actual review and Root receipt remain pending')
 check(plan['stages']==['eagerA','graphA','graphB','eagerB','sideC'],'five original numerical stages')
 check(plan['fixed_wq_cols_marker_across_all_epochs'] is True,'fixed vendor cached-weight identity')
 check('torch.round' in quant and 'clamp_(-127, 127)' in quant,'original frozen weight quantizer reused')
 old=c.git('show',c.COMPILE_SOURCE+':tools/check_biv150_native_i8_gate2048_cpu.py')
 check(old==(c.ROOT/'tools/check_biv150_native_i8_gate2048_cpu.py').read_bytes(),'original panel checker byte exact')
 # The only host baseline override recognizes the ordinary merge of main122.
 # The original panel/fragment enumeration and candidate GPU inputs stay intact.
 base=original.BASE
 try:original.BASE=c.MAIN;mapping=original.check()
 finally:original.BASE=base
 check(mapping['passed'] and mapping['unique_shared_word_writers']==3072 and mapping['unique_CTA_outputs']==4096,'original complete panel/fragment CPU check')
 src=(c.ROOT/'tools/probe_biv150_native_i8_gate2048_numeric.py').read_text();tree=ast.parse(src)
 run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
 statement=ast.get_source_segment(src,run)
 check(statement.index('binding=c.preflight(a)')<statement.index('import torch'),'no Torch import before actual admission')
 check(statement.count('fixed_weight(torch).to')==1 and statement.index('fixed_weight(torch).to')<statement.index('def upload_epoch'),'weight prepared once outside changing epochs')
 upload=next(n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name=='upload_epoch')
 destinations=[ast.unparse(n.func.value) for n in ast.walk(upload) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='copy_']
 check(destinations==['x','aq','rows'],'epoch updates activation buffers only')
 ref=next(n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name=='reference')
 text=ast.get_source_segment(src,ref)
 check('torch.ops.biv150_native.gate_silu_cached_probe(x,wq,cols,marker,256)' in text,'actual accepted native GEMM and epilogue direct reference')
 check('torch.ops.biv150_native.quantize_rows_probe(x,256)' in statement,'actual accepted activation quantizer')
 check("checkpoint('graphB');poison();reference();candidate();checkpoint('eagerB')" in statement,'changed graph B checked before fresh eager B')
 check("with torch.cuda.stream(side):reference();candidate();done.record()" in statement and 'wait_event(done)' in statement,'side-stream execution and completion handoff')
 check('torch.testing.assert_close(output[\'candidate\'],output[\'reference\'],atol=0,rtol=0)' in statement,'whole output original zero tolerance')
 check('0x7fc1' in statement and "all(record['guards'].values())" in statement,'NaN output poison; guard checks remain explicit')
 check(not any(x in src for x in ('elapsed_time','do_bench','perf_counter','torch.compile','cpp_extension')),'no timing, code compilation or service path')
 for approval in (b'{}',b'{"approved_numeric_only":true}',b'null'):
  try:c.require_admissions(approval,approval)
  except ValueError as e:check(str(e)=='actual static review and Root approval are pending','pending receipt rejects '+approval.decode())
  else:raise AssertionError('pending approval accepted')
 with tempfile.TemporaryDirectory(prefix='cjj128-numeric-cpu-') as d:
  out=Path(d)/'not-created'
  args=types.SimpleNamespace(root_owned_exclusive=True,expected_source_commit=c.git('rev-parse','HEAD').decode().strip(),out=out)
  try:c.preflight(args)
  except ValueError as e:check('pending' in str(e) and not out.exists(),'CPU preflight refuses before files/device access')
  else:raise AssertionError('unknown static proof accepted')
  entry=c.ROOT/'tools/probe_biv150_native_i8_gate2048_numeric.py'
  default=subprocess.run([sys.executable,'-B','-S','-W','error',str(entry)],capture_output=True,text=True)
  check(default.returncode==0 and json.loads(default.stdout)==plan,'real default subprocess is CPU plan')
  argv=[sys.executable,'-B','-S','-W','error',str(entry),'--run-numeric','--root-owned-exclusive','--expected-source-commit',args.expected_source_commit]
  for name in ('compile-summary','library','reference-library','static-review','root-admission'):argv+=['--'+name,str(Path(d)/name)]
  argv+=['--out',str(out)]
  rejected=subprocess.run(argv,capture_output=True,text=True)
  check(rejected.returncode!=0 and 'actual static review and Root approval are pending' in rejected.stderr and not out.exists(),'actual numeric CLI stops at pending tripwire without importing Torch')
 actual_review=None
 if a.static_review:
  raw=a.static_review.read_bytes();review=json.loads(raw)
  check(c.review_fields(review),'actual final-SO static schema fields (not Root admission)')
  actual_review=c.sha(raw)
  for key,value in [('passed',False),('static_ready_for_numeric',False),('library_sha256','0'*64),('source_commit','0'*40),('summary_sha256','0'*64),('numeric_verified',True),('object_embedded_code_identical_to_final_SO',False)]:
   bad=copy.deepcopy(review);bad[key]=value
   try:c.review_fields(bad)
   except ValueError:check(True,'reject altered actual static field '+key)
   else:raise AssertionError('mutated review accepted '+key)
 check(not {'torch','triton','numpy'}.intersection(sys.modules),'CPU checker imports no device/math frameworks')
 result=dict(CPU_only=True,actual_static_fields_checked_sha256=actual_review,passed=True,checks=checks,checks_count=len(checks),original_mapping=mapping,
  original_mapping_source_sha256=c.sha(old),original_mapping_host_base=c.MAIN,
  plan=plan,numeric_verified=False,GPU_executed=False,service_verified=False,
  sources={p.name:c.sha(p.read_bytes()) for p in (Path(__file__),c.ROOT/'tools/biv150_native_i8_gate2048_numeric_contract.py',c.ROOT/'tools/probe_biv150_native_i8_gate2048_numeric.py')})
 print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__':main()
