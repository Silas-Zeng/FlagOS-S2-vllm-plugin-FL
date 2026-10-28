#!/usr/bin/env python3
"""Finite immutable-Git diagnosis; no framework imports or device execution."""
import argparse,ast,difflib,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('/Users/chenjunjie/Documents/flagos/repos/FlagOS-S2-vllm-plugin-FL'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists()
refs={'main_a455':'a455a3de364051fbea80226870c239de3a48e716','PR109_fc25':'fc25c86a13fc273bb7368b83af4ca264524b4e12','PR111_a6d':'a6d654a7444bba72cbbd48d3d9bfd33172cfd129'}
runner='vllm_fl/worker/model_runner.py';order='vllm_fl/patches/biv150_compute_order.py';worker='vllm_fl/worker/worker.py'
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(ref,path):return subprocess.check_output(['git','-C',str(a.repo),'show',ref+':'+path])
def defs(raw):
 lines=raw.splitlines(keepends=True)
 return {n.name:b''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def constant(raw,name):return next(ast.literal_eval(n.value) for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
b={k:{p:blob(v,p) for p in (runner,order,worker)} for k,v in refs.items()};rows={}
for k,v in b.items():
 pin=constant(v[order],'EXPECTED_RUNNER');manifest=None
 if k!='main_a455':manifest=json.loads(blob(refs[k],'vllm_fl/patches/biv150_konly_compact_sources.json'))['binding_files']
 rows[k]={'commit':refs[k],'files':{p:{'bytes':len(x),'sha256':sha(x),'CRLF':x.count(b'\r\n'),'LF':x.count(b'\n'),'normalized_LF_sha256':sha(x.replace(b'\r\n',b'\n'))} for p,x in v.items()},'EXPECTED_RUNNER':pin,'runtime_read_bytes_pin_matches':pin==sha(v[runner]),'read_text_newline_normalization_would_match':pin==sha(v[runner].replace(b'\r\n',b'\n')),'compact_binding_files':manifest}
assert [v['runtime_read_bytes_pin_matches'] for v in rows.values()]==[True,True,False]
old,new=b['PR109_fc25'][runner],b['PR111_a6d'][runner]
start_old=old.index(b'# Reject two cache formats');start_new=new.index(b'# Reject simultaneous cache formats');marker=b'# Private encoded-cache diagnostic:'
old_region=old[start_old:old.index(marker,start_old)];new_region=new[start_new:new.index(marker,start_new)]
assert new.count(new_region)==1 and new.replace(new_region,old_region,1)==old
assert defs(old)==defs(new)==defs(b['main_a455'][runner])
class_bytes=defs(new)['ModelRunnerFL'];cls=ast.parse(class_bytes).body[0]
methods=[n.name for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
assert b['PR109_fc25'][order]==b['PR111_a6d'][order]
assert len({sha(v[worker]) for v in b.values()})==1
for name in ('Plan','make_plan'):
 assert ast.dump(ast.parse(defs(b['main_a455'][order])[name]),include_attributes=False)==ast.dump(ast.parse(defs(b['PR111_a6d'][order])[name]),include_attributes=False)
assert rows['PR111_a6d']['compact_binding_files']['worker/model_runner.py']==sha(new)
report={'schema':'native111-compute-order-source-diagnosis-v1','static_audit_passed':True,'service_acceptance_passed':False,'GPU_executed':False,'raw_service_results_read':False,'refs':rows,'all_top_level_runner_definitions_byte_equal':sorted(defs(new)),'ModelRunnerFL_entire_class_byte_equal':True,'ModelRunnerFL_class_sha256':sha(class_bytes),'ModelRunnerFL_method_count':len(methods),'ModelRunnerFL_methods_byte_unchanged':methods,'whole_runner_exact_inverse_to_PR109':True,'inverse_old_region_bytes':old_region.decode(),'inverse_new_region_bytes':new_region.decode(),'worker_loader_byte_equal':True,'worker_loader_rule':'worker.py imports ModelRunnerFL from vllm_fl.worker.model_runner before constructing it','compute_order_109_to_111_byte_equal':True,'Plan_and_make_plan_main_to_111_AST_equal':True,'diagnosis':'111 module-tail Q32 admission changed the worker bytes; compact manifest records current d444 but EXPECTED_RUNNER still requires109 9ee. The guard label conflates absent module and raw hash mismatch. Module absence is not established by that text.','fallback_boundary':'_cold verifies source bytes before _main_stack and before staging allocation. make_plan catches at this pre-write point, sets its private state False and returns None; future make_plan calls on that runner return None. This is not runtime method-contract or accuracy proof.','limitations':['No worker/torch/triton/GPU imported; no live module globals/closures claimed from this static audit.','Only original a6d failure can be diagnosed; changing a future pin cannot retroactively qualify a6d.','No throughput loss attribution or new performance/model qualification.'],'patch_origin_commit':'c0bbdf0a9dcb44573eba8840b7f5804d467cc69d'}
a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'report':str(a.out),'sha256':sha(a.out.read_bytes()),'methods':len(methods),'whole_class_sha256':sha(class_bytes)}))
