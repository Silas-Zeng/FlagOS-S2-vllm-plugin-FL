#!/usr/bin/env python3
"""Equivalent pre-cold gate: real pure functions and CPU control-flow sentinels."""
import ast,json,os,subprocess,sys,types
from pathlib import Path
from unittest.mock import patch
import check_biv150_dynamic_hidden_gather_tensor_cost_cpu as cost_cpu
x=cost_cpu.x;c=x.c;ROOT=x.ROOT;HELPER='vllm_fl/patches/biv150_dynamic_hidden_gather.py';checks=[]
def ok(name,v=True):c.need(v,name);checks.append(name)
def load(name,source):
 m=types.ModuleType(name);m.__file__=str(ROOT/HELPER);sys.modules[name]=m;exec(compile(source,m.__file__,'exec'),m.__dict__);return m
def outcome(f,scheduled,total):
 try:f(scheduled,total)
 except Exception as e:return(type(e).__name__,str(e))
 return('accepted',)
def run():
 old=subprocess.check_output(['git','-C',str(ROOT),'show',x.PARENT+':'+HELPER]).decode();new=(ROOT/HELPER).read_text();ot=ast.parse(old);nt=ast.parse(new)
 funcs=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
 of,nf=funcs(ot),funcs(nt);validator=nf['_validate_sizes'];chunk=ast.get_source_segment(new,validator)+'\n\n\n'
 restored=new.replace(chunk,'',1).replace('        _validate_sizes(sizes, total)\n','        cpu_indices(sizes, total)\n',1)
 ok('exact whole-helper inverse; only validator and discarded call changed',new.count(chunk)==1 and restored==old)
 ok('strict three constraints and error order AST exact',ast.dump(ast.Module(body=validator.body[1:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=of['cpu_indices'].body[1:4],type_ignores=[]),include_attributes=False))
 for name in of:ok('unchanged original function '+name,ast.get_source_segment(new,nf[name]).replace('_validate_sizes(sizes, total)','cpu_indices(sizes, total)')==ast.get_source_segment(old,of[name]))
 a=load('_sizes_gate_old_cpu',old);b=load('_sizes_gate_new_cpu',new)
 class IntSubclass(int):pass
 cases=[]
 for n in range(66):
  v=[1]*n;cases.append((v,n));cases.append((tuple(v),n))
  if n:
   cases.append(([2048-n+1]+[1]*(n-1),2048))
   for bad in (0,-1,True,1.0,IntSubclass(1)):cases.append(([bad]+[1]*(n-1),n))
 for total in (None,True,False,1.0,IntSubclass(1),-1,0,2048,2049,10**30):cases.append(([1],total))
 for v in (None,iter([1]),[None],[1,'2'],[],[2048],[2049]):cases.append((v,2048))
 for fixture in x.base.spec.plan()['cases']:cases.append((tuple(fixture['scheduled']),fixture['M']))
 for scheduled,total in cases:ok_pair=outcome(a.cpu_indices,scheduled,total)==outcome(b._validate_sizes,scheduled,total);c.need(ok_pair,'acceptance/rejection or error order differs '+repr((scheduled,total)))
 ok('strict pure-function bounds/type acceptance and exception order',len(cases)>500)
 for n in (1,2,9,61,64):
  sizes=tuple([2048-n+1]+[1]*(n-1))
  for permutation in (None,tuple(range(n)),tuple(reversed(range(n)))):c.need(a.cpu_indices(sizes,2048,permutation)==b.cpu_indices(sizes,2048,permutation),'full genuine permutation/prefix oracle changed')
 ok('all full prefix/index/permutation results remain exact')
 # These sentinels NEVER return a cold state. They check rejection timing only,
 # not SDK types, caller admission, a successful cold path or device behavior.
 int32=object()
 class Array:
  dtype=int32
  def __init__(self,values):self.values=values;self.shape=(len(values),);self.flags=types.SimpleNamespace(c_contiguous=True)
  def __iter__(self):return iter(self.values)
 np=types.SimpleNamespace(ndarray=Array,int32=int32);torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))
 timing=[]
 for values,total,expect in (([1],1,1),([2040]+[1]*8,2048,1),([1988]+[1]*60,2048,1),([1]*64,64,1),([],0,0),([1]*65,65,0),([0],1,0),([-1],1,0),([1],True,0),([1],1.0,0),([1],2049,0),([1],2,0)):
  observed=[]
  for module in (a,b):
   calls=[]
   def cold(*args):calls.append('cold attempted');raise ValueError('CPU cold sentinel: no qualification')
   r=types.SimpleNamespace(speculative_config=None,input_batch=types.SimpleNamespace(num_reqs=len(values)))
   with patch.dict(sys.modules,torch=torch,numpy=np),patch.dict(os.environ,{module.ENV:'1'}),patch.object(module,'_cold',cold):c.need(module.prepare(r,None,Array(values),total,None,None) is None,'sentinel must never produce receipt')
   observed.append(len(calls))
  c.need(observed==[expect,expect],'pre-cold rejection timing differs');timing.append(dict(scheduled=values,total=total,old_cold_attempts=observed[0],new_cold_attempts=observed[1]))
 ok('invalid sizes rejected before cold; valid inputs reach same sentinel once')
 inherited=cost_cpu.run();ok('all inherited true-Tensor four-case CPU/cost guards pass',inherited['passed'] and inherited['check_count']==28)
 ok('no actual frameworks imported',not {'torch','triton','numpy'}.intersection(sys.modules))
 return dict(passed=True,CPU_only=True,parent=x.PARENT,helper_sha256=c.sha(new.encode()),exact_helper_inverse=True,strict_gate_comparisons=len(cases),checks=checks,check_count=len(checks),pre_cold_control_flow_sentinels=timing,sentinels_not_real_cold_or_SDK_qualification=True,inherited_tensor_cost=inherited,
  GPU_source_diff=False,original_runner_and_post_model_guards_unchanged=True,cross_call_cache_added=False,Plan_or_MN_routing_changed=False,device_numeric_verified=False,actual_new_host_cost_measured=False,service_verified=False)
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
