"""Local readonly artifact admission and exact-parent CPU checks; no Torch import."""
import argparse,ast,builtins,inspect,json,os,shutil,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
import run_biv150_hidden_gather_service_object_numeric as n
c=n.c

def run(a):
 checks=[]
 def ok(value,label):c.need(value,label);checks.append(label)
 def refuses(fn,label):
  try:fn()
  except (ValueError,FileNotFoundError):checks.append(label)
  else:raise AssertionError(label)
 text=n.expanded();compile(text,'exact-parent-execute','exec');ok(True,'seven edits reverse exactly to frozen original execute')
 ok(not c.git('diff',n.cache.SOURCE,'--','vllm_fl','csrc'),'all production and GPU bytes remain8b7')
 for name,h in n.REUSED.items():ok(c.sha((Path(__file__).parent/name).read_bytes())==h,'unchanged '+name)
 ok(n.plan()['device_executed'] is False and n.plan()['service_verified'] is False,'CPU plan grants no device or service result')
 ok(n.base.c.OPTIONS==dict(BLOCK_R=8,BLOCK_D=128,num_warps=4,num_stages=1),'original launch options without use_sme')
 st=ast.parse(inspect.getsource(n.seed));ok(not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Subscript) for x in ast.walk(st)),'seed has no indexed kernel launch')
 ok(any(isinstance(x,ast.Attribute) and x.attr=='warmup' for x in ast.walk(st)),'seed uses normal warmup')
 for word in ('prohibit_candidate_compile','compiled is seeded','cold_SME_prediction_if_uncached','output_guards(seeded'):
  ok(word in text,'expanded guard '+word)
 summary,record=n.cache.proof(a.service_out,a.review)
 ok(summary['original_failed_state']['completed'] is False,'actual service failed state preserved')
 ok(record['hash']==n.cache.HASH and record['options']['use_sme']==1,'actual SME1 identity')
 original={str(p.relative_to(a.service_out)):c.sha(p.read_bytes()) for p in [a.service_out/k for k in n.cache.PINS]}
 original.update({str(p.relative_to(a.service_out)):c.sha(p.read_bytes()) for p in (a.service_out/'triton-cache'/n.cache.DIRECTORY).iterdir()})
 with tempfile.TemporaryDirectory(prefix='cjj122-service-object-CPU-') as tmp:
  root=Path(tmp);group=n.cache.prepare(a.service_out,root/'cache',a.review)
  class Manager:
   def get_group(self,name):return dict(group['paths'])
  ok(n.cache.validate_group(Manager(),group)==group['paths'],'original standard-cache group validation')
  for name,digest in group['sha256'].items():ok(c.sha(Path(group['paths'][name]).read_bytes())==digest,'copied exact '+name)
  ok(c.sha(Path(group['host'][0]['path']).read_bytes())==n.cache.LAUNCHER,'copied exact4105 host launcher')
  changed=Path(group['paths']['gather_hidden_bits.cubin']);saved=changed.read_bytes();changed.write_bytes(saved+b'bad')
  refuses(lambda:n.cache.validate_group(Manager(),group),'changed private cubin refused');changed.write_bytes(saved)
  copy=root/'original-copy';copy.mkdir()
  relatives=set(n.cache.PINS)|{'triton-cache/'+n.cache.DIRECTORY+'/'+n.cache.NAME+'.'+e for e in record['files']}
  for rel in relatives:p=copy/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((a.service_out/rel).read_bytes())
  for rel in ('gather-smoke-snapshots/smoke-after.json','triton-cache/'+n.cache.HANDLER,'triton-cache/'+n.cache.DIRECTORY+'/gather_hidden_bits.json'):
   p=copy/rel;b=p.read_bytes();p.write_bytes(b+b'bad');refuses(lambda:n.cache.proof(copy,a.review),'changed original refused '+rel);p.write_bytes(b)
  wrong=root/'review.json';wrong.write_bytes(a.review.read_bytes()+b'bad');refuses(lambda:n.cache.proof(a.service_out,wrong),'wrong independent review refused')
  for m,k in n.base.bits.CASES:
   for phase in (0,1):
    f=n.base.bits.fixture(m,k,phase);g=n.base.bits.GUARD*2;out=f['output_poison'][:g]+f['expected']+f['output_poison'][-g:]
    ok(all(n.base.bits.checks(f,f['hidden'],f['indices'],out).values()),'unchanged full CPU bit oracle '+str((m,k,phase)))
    bad=bytes([out[0]^1])+out[1:];ok(not all(n.base.bits.checks(f,f['hidden'],f['indices'],bad).values()),'full guard catches corruption '+str((m,k,phase)))
  refuses(lambda:n.execute(types.SimpleNamespace(timing=True)),'timing refused before device')
  refuses(lambda:n.execute(types.SimpleNamespace(timing=False,root_owned_exclusive=False)),'pending Root admission refused')
  if a.ready_prefix:
   ok(not c.git('status','--porcelain','--untracked-files=all'),'clean frozen actual-ready CPU prefix')
   args=types.SimpleNamespace(timing=False,root_owned_exclusive=True,expected_source_commit=c.git('rev-parse','HEAD'),compile_out=a.compile_out,actual_review=a.actual_review,bits_review=a.bits_review,production_out=a.service_out,production_review=a.review,out=root/'never-device-out')
   imported=builtins.__import__;calls=[]
   def tripwire(name,*rest,**kwargs):
    if name in ('torch','triton','numpy'):calls.append(name);raise RuntimeError('CPU_TRIPWIRE before any framework import')
    return imported(name,*rest,**kwargs)
   with patch.dict(os.environ),patch('builtins.__import__',tripwire):result=n.execute(args)
   ok(calls==['torch'] and result['completed'] is False and result['errors']==["RuntimeError('CPU_TRIPWIRE before any framework import')"],'real execute admission and private copy reach first Torch import only')
   ok(len(result['reviewed_production_cache']['paths'])==7 and result['production_warmups_attempted']==0,'actual ready prefix copied reviewed group with0warmup')
 for rel,digest in original.items():ok(c.sha((a.service_out/rel).read_bytes())==digest,'readonly original '+rel)
 ok(not {'torch','triton','numpy'}.intersection(sys.modules),'no framework import or GPU activity')
 return dict(passed=True,CPU_only=True,device_numeric_verified=False,service_verified=False,checks=checks,check_count=len(checks),source_commit=c.git('rev-parse','HEAD'),expanded_execute_sha256=c.sha(text.encode()),independent_review_sha256=n.cache.REVIEW,actual_hash=n.cache.HASH,actual_cubin=n.cache.CUBIN,actual_launcher=n.cache.LAUNCHER,actual_ready_prefix_checked=a.ready_prefix)

def main():
 p=argparse.ArgumentParser();p.add_argument('--service-out',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--ready-prefix',action='store_true')
 for name in ('compile-out','actual-review','bits-review'):p.add_argument('--'+name,type=Path)
 a=p.parse_args()
 if a.ready_prefix and not all((a.compile_out,a.actual_review,a.bits_review)):p.error('original real proofs required for ready prefix')
 result=run(a);c.save(a.out,result);print(json.dumps(dict(passed=True,check_count=result['check_count'],sha256=c.sha(a.out.read_bytes()))))
if __name__=='__main__':main()
