#!/usr/bin/env python3
"""Small CPU contract/cost-boundary checks; fake calls do not model GPU results."""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import probe_biv150_native_i8_gate2048_timing as t


def main():
 checks=[]
 def check(ok,label):assert ok,label;checks.append(label)
 plan=t.plan();t.c.source_proof()
 check(t.c.git('show',t.NUMERIC_SOURCE+':tools/probe_biv150_native_i8_gate2048_numeric.py')==(t.c.ROOT/'tools/probe_biv150_native_i8_gate2048_numeric.py').read_bytes(),'frozen original five-stage numeric driver unchanged')
 check(plan['warm_batches_per_role_and_mode']==128 and plan['unroll']==4 and plan['timed_batches_per_arm']==16 and plan['steps_per_timed_arm']==64,'fixed512 warm and64 timed steps')
 check(plan['modes']==['graph','eager'] and plan['order']==['reference','candidate','candidate','reference'],'two distinct ABBA measurement scopes')
 calls=[];x=NS(device='cpu-fake');wq=object();cols=object();marker=object();aq=object();rows=object();output=object()
 def reference(*args):calls.append(('reference',args));return output
 def quant(*args):calls.append(('quantizer',args));return aq,rows
 def empty(shape,**kwargs):calls.append(('empty',shape,kwargs));return output
 def gate(*args):calls.append(('gate',args))
 torch=NS(bfloat16='BF16',empty=empty,ops=NS(biv150_native=NS(gate_silu_cached_probe=reference,quantize_rows_probe=quant),cjj_native_i8_gate2048_probe=NS(gate_out=gate)))
 check(t.reference_step(torch,x,wq,cols,marker) is output and calls==[('reference',(x,wq,cols,marker,256))],'actual A wrapper one original call, returned output, no validation copy')
 calls.clear()
 for _ in range(2):check(t.candidate_step(torch,x,wq,cols) is output,'actual B wrapper returns allocated output')
 expected=[('quantizer',(x,256)),('empty',(2048,6144),dict(device=x.device,dtype='BF16')),('gate',(aq,wq,rows,cols,output))]
 check(calls==expected*2,'actual B repeats quantization, allocation and fused gate every step')
 # Execute the real measured loops with no-device protocol doubles, to verify
 # step counts and that host stop is after event synchronization.
 for mode in ('graph','eager'):
  events=[];clock=iter((0.,1.));n=iter(('begin','end'))
  class Event:
   def __init__(self,**kwargs):self.label=next(n);check(kwargs==dict(enable_timing=True),'events enable timing')
   def record(self):events.append(self.label)
   def synchronize(self):events.append('sync')
   def elapsed_time(self,other):return 64.
  def now():events.append('clock');return next(clock)
  def replay():events.append('replay')
  def step(role):events.append('step-'+role)
  fake=NS(cuda=NS(Event=Event,stream=lambda s:nullcontext()))
  measured=t.measure(fake,NS(perf_counter=now),object(),mode,'reference',step,{'reference':NS(replay=replay)})
  check(events.count('replay')==(16 if mode=='graph' else 0) and events.count('step-reference')==(64 if mode=='eager' else 0),'actual '+mode+' timed loop count')
  check(events[:2]==['clock','begin'] and events[-3:]==['end','sync','clock'],'actual '+mode+' event/host completion ordering')
  check(measured==dict(role='reference',event_ms_per_step=1.,wall_ms_per_step=15.625),'actual '+mode+' divisor64; synthetic numbers only')
 source=Path(t.__file__).read_text();tree=ast.parse(source)
 check(source.index("'composed-Graph-B-before-fresh'")<source.index("'fresh-eager-B-after-Graph'"),'new composed Graph B before fresh B check')
 check(source.count('range(128)')==1 and source.count('range(4)')==3,'warm128 and fixed four-step capture/eager loops')
 check('biv150_native_i8_gate2048_numeric_contract' in source and 'preflight(a)' in source,'original source/library admission reused')
 check(not any(x in source for x in ('cpp_extension','torch.compile','pip install','site-packages')),'no compiler, package installation or production edits')
 actual=subprocess.run([sys.executable,'-B','-S','-W','error',str(Path(t.__file__))],capture_output=True,text=True)
 check(actual.returncode==0 and json.loads(actual.stdout)==plan,'real default CLI remains CPU plan')
 rejected=subprocess.run([sys.executable,'-B','-S','-W','error',str(Path(t.__file__)),'--run-timing'],capture_output=True,text=True)
 check(rejected.returncode!=0 and 'all fixed evidence and Root exclusive fields required' in rejected.stderr,'actual missing-evidence CLI refuses before device import')
 check(not {'torch','triton','numpy'}.intersection(sys.modules),'CPU checks import no device/math framework')
 print(json.dumps(dict(passed=True,count=len(checks),checks=checks,plan=plan,source_sha256=t.c.sha(Path(t.__file__).read_bytes()),checker_sha256=t.c.sha(Path(__file__).read_bytes()),GPU_executed=False,performance_verified=False,CPU_doubles_are_only_call_order_checks=True),indent=2,sort_keys=True))
if __name__=='__main__':main()
