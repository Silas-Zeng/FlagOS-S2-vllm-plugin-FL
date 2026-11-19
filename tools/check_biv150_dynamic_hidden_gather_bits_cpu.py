#!/usr/bin/env python3
"""Small complete-fixture and refusal checks; no Torch, SDK, fake GPU or timing."""
import ast,json,struct,subprocess,sys
from pathlib import Path
import run_biv150_dynamic_hidden_gather_bits as w

def main():
 need=w.need;records=[];negative=0
 for m,n in w.CASES:
  prior=None
  for phase in (0,1):
   f=w.fixture(m,n,phase);idx=list(struct.unpack('<'+'i'*n,f['indices'][w.IGUARD*4:-w.IGUARD*4]));need(idx==f['logical_indices'],'saved CPU indices')
   # Derive oracle again from complete guarded input bytes and independent byte offsets.
   oracle=b''.join(f['hidden'][(w.GUARD+x*w.D)*2:(w.GUARD+(x+1)*w.D)*2] for x in idx)
   need(oracle==f['expected'] and len(oracle)==n*w.D*2,'complete byte-offset oracle')
   good=f['output_poison'][:2*w.GUARD]+oracle+f['output_poison'][-2*w.GUARD:]
   need(all(w.checks(f,f['hidden'],f['indices'],good).values()),'full valid fixture')
   need(tuple(struct.unpack('<8H',oracle[:16]))==w.SPECIAL,'exact special BF16 bit payloads')
   need(idx[0:3]==([m-1,0,m-1] if phase==0 else [0,m-1,0]),'extreme repeated indices')
   if prior:
    need(all(prior[k]!=f[k] for k in ('hidden','indices','expected','output_poison')),'changed-input B')
   def flip(b,at):return b[:at]+bytes([b[at]^1])+b[at+1:]
   mutants=[(f['hidden'],f['indices'],flip(good,2*w.GUARD)),(flip(f['hidden'],0),f['indices'],good),
    (f['hidden'],flip(f['indices'],0),good),(f['hidden'],f['indices'],flip(good,0)),
    (f['hidden'],f['indices'],flip(good,len(good)-1)),(f['hidden'],f['indices'],f['output_poison'])]
   for hidden,index,output in mutants:need(not all(w.checks(f,hidden,index,output).values()),'byte/readonly/guard/poison mutant rejected');negative+=1
   records.append(dict(M=m,N=n,phase='AB'[phase],hidden_bytes=len(f['hidden']),output_words=len(oracle)//2,
    input_sha256={k:w.sha(v) for k,v in f.items() if isinstance(v,bytes)},all_uint16_oracle_and_guards=True))
   prior=f
 need(w.packed((4,1,0))['expected_launch_block']==[256,1,1],'derived dimensions')
 for bad in (None,[4,1,0],(4,True,0),(4,1,False),(4,1,1),(2,1,0),(4,1,0,0)):
  try:w.packed(bad)
  except ValueError:negative+=1
  else:raise ValueError('unknown/changed packed metadata accepted')
 for bad in ({'type':'unknown'},):
  try:w.typed(bad)
  except ValueError:negative+=1
  else:raise ValueError('unknown saved ABI accepted')
 tree=ast.parse(Path(w.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 need(not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('warmup','compile') for n in calls),'no warmup or IR compile call')
 need(sum(isinstance(n.func,ast.Name) and n.func.id=='CompiledKernel' for n in calls)==1,'one explicit existing artifact load')
 for args in ([],['--root-owned-exclusive'],['--run-bits']):
  p=subprocess.run([sys.executable,'-B',w.__file__,*args],capture_output=True,text=True)
  need(p.returncode==(0 if not args else 2),'default/admission CLI')
  if not args:need(json.loads(p.stdout)==w.c.freeze(w.plan()),'exact stdlib default plan')
 need(not {'torch','triton','numpy'}.intersection(sys.modules),'stdlib only')
 return dict(passed=True,stdlib_only=True,worker_sha256=w.sha(Path(w.__file__).read_bytes()),production_kernel_sha256=w.c.KERNEL_SHA,
  complete_fixture_records=records,negative_checks=negative,default_and_missing_admission_checked=True,
  existing_binary_load_not_new_warmup=True,actual_SDK_loader_or_device_tested=False,
  device_numeric_verified=False,helper_hit_verified=False,performance_verified=False,service_verified=False)

if __name__=='__main__':print(json.dumps(main(),sort_keys=True,indent=2))
