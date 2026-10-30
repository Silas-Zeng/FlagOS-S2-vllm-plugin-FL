#!/usr/bin/env python3
"""Small stdlib source/producer/admission checks; no fake cold/device proof."""
import ast,json,subprocess,sys
from pathlib import Path
import run_biv150_dynamic_hidden_gather_helper as worker
c=worker.c;root=worker.ROOT

def run():
 plan=worker.spec.plan();cases=plan['cases'];c.need(len(cases)==4,'two geometries and two real producer formulas')
 for x in cases:
  ends=[None]*x['N'];total=0
  for request in x['permutation']:total+=x['scheduled'][request];ends[request]=total-1
  c.need(total==x['M'] and ends==x['logits_indices'] and len(set(ends))==x['N'] and min(ends)>=0 and max(ends)==x['M']-1,'independent producer/index boundaries')
  c.need(x['phases'][0]['byte_sha256']['hidden']!=x['phases'][1]['byte_sha256']['hidden'] and x['phases'][0]['byte_sha256']['expected']!=x['phases'][1]['byte_sha256']['expected'],'real payload changes with valid producer indices')
 transform=json.loads((root/'docs/evidence/dynamic-hidden-gather-entry-20261029/runner-transform.json').read_text())
 runner=(root/'vllm_fl/worker/model_runner.py').read_bytes();restore=runner
 for item in reversed(transform['edits']):
  before,after=item['before'].encode(),item['after'].encode();c.need(restore.count(after)==1,'one reversible original seam');restore=restore.replace(after,before)
 original=lambda name:subprocess.check_output(['git','-C',str(root),'show',worker.MAIN+':'+name])
 c.need(restore==original('vllm_fl/worker/model_runner.py'),'runner inverse exact main899; original protected function bodies preserved')
 order=(root/'vllm_fl/patches/biv150_compute_order.py').read_bytes()
 c.need(order.replace(transform['runner_sha256'].encode(),transform['original_sha256'].encode())==original('vllm_fl/patches/biv150_compute_order.py'),'main899 Plan/source identity preserved except runner pin')
 semantic=(root/'vllm_fl/compile_semantics.py').read_bytes()
 c.need(semantic.count(transform['semantics_addition'].encode())==1 and semantic.replace(transform['semantics_addition'].encode(),b'')==original('vllm_fl/compile_semantics.py'),'main899 mixed/5898 C3 preserved plus122 feature')
 for name in ('biv150_dynamic_hidden_gather.py','biv150_dynamic_hidden_gather_kernel.py'):
  name='vllm_fl/patches/'+name;c.need((root/name).read_bytes()==subprocess.check_output(['git','-C',str(root),'show',bits_parent+':'+name]),'production helper/kernel unchanged22ba '+name)
 tree=ast.parse(Path(worker.__file__).read_text());assignments=[t for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)) for t in (n.targets if isinstance(n,ast.Assign) else [n.target])]
 protected={'execute_model','_prepare_inputs','_get_cumsum_and_arange','_cold','_codes','_Receipt','_SEAL','prepare','try_gather','_launch'}
 c.need(not any(isinstance(t,ast.Attribute) and t.attr in protected for t in assignments),'no helper/caller guard or protected method replaced')
 patches=[ast.literal_eval(n.args[1]) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='object' and isinstance(n.func.value,ast.Name) and n.func.value.id=='patch']
 c.need(sorted(patches)==['_biv150_compute_order_plan','get_pp_group','set_forward_context'],'only explicit outer infrastructure fixture globals')
 source=Path(worker.__file__).read_text();c.need(source.count('fn.warmup(')==1 and 'for m,n in bits.CASES' in source,'bounded actual warmup sites, not84-shape sweep')
 c.need("sys.setprofile(previous)" in source and "if a.timing:" in source and "timings_not_profiled=True" in source,'profiling bounded to correctness, cost scope explicit')
 c.need("postblock full output/hidden guard and readonly" in source and "torch.index_select" in source,'same-input index_select and full post-cost output checks')
 cli=[]
 for args,rc in (([],0),(['--root-owned-exclusive'],2),(['--run-helper'],2),(['--timing'],2)):
  p=subprocess.run([sys.executable,'-B','-S',worker.__file__,*args],capture_output=True,text=True);c.need(p.returncode==rc,'default/admission CLI')
  if not args:c.need(json.loads(p.stdout)==plan,'exact stdlib default plan')
  cli.append(dict(args=args,exit_code=p.returncode))
 c.need(not {'torch','triton','numpy'}.intersection(sys.modules),'no framework import/device pretense')
 return dict(passed=True,stdlib_only=True,main=worker.MAIN,parent=bits_parent,source_inverse_main899=True,production_helper_unchanged=True,
  four_valid_producer_fixture_cases=True,public_service_gates_not_weakened=True,only_enumerated_outer_fixture_globals=patches,
  worker_sha256=c.sha(Path(worker.__file__).read_bytes()),plan_sha256=c.sha(Path(worker.spec.__file__).read_bytes()),CLI=cli,
  actual_cold_or_SDK_constructor_tested=False,device_numeric_verified=False,actual_timing_verified=False,service_verified=False)
bits_parent='22ba8b7cdc3be4e9619f4fa3f4e91c9cca6cfdfc'
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
