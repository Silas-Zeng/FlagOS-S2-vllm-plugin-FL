#!/usr/bin/env python3
"""CPU real-cache byte tests and compile-refusal regressions, no SDK imports."""
import ast,hashlib,json,shutil,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import biv150_dynamic_hidden_gather_reviewed_cache as r
c=r.c
ORIGINAL=Path('/private/tmp/cjj-dynamic-hidden-gather-helper-numeric-v1-raw-20261029/out')
REVIEW=Path('/private/tmp/cjj-agent-dynamic-hidden-gather-helper-specialization-review-20261029.json')
checks=[]
def ok(name,v=True):c.need(v,name);checks.append(name)
def bad(name,call):
 try:call()
 except (ValueError,KeyError,FileNotFoundError):checks.append(name);return
 raise ValueError('negative accepted '+name)
def inventory(root):return {str(p.relative_to(root)):c.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
def run():
 before=inventory(ORIGINAL);s,a=r.proof(ORIGINAL,REVIEW);ok('bind actualv1 preserved refusal, natural5 and new static review',a['options']['use_sme']==5 and a['hash']==r.HASH)
 with tempfile.TemporaryDirectory(prefix='cjj-gather-cache-cpu-') as tmp:
  root=Path(tmp);group=r.prepare(ORIGINAL,root/'cache',REVIEW)
  ok('only actual afc seven children, two exact hostlibs',len(group['paths'])==7 and len(group['host'])==2 and all('/V7F3TADD' in p for p in group['paths'].values()))
  for name,p in group['paths'].items():
   old=ORIGINAL/'triton-cache'/Path(p).parent.name/name;ok('copied original child exact '+name,Path(p).read_bytes()==old.read_bytes())
  class Manager:
   def get_group(self,name):c.need(name=='gather_hidden_bits.json','group lookup role');return dict(group['paths'])
  manager=Manager();r.validate_group(manager,group);ok('exact relocated group loads')
  file=Path(group['paths']['gather_hidden_bits.ttir']);raw=file.read_bytes();file.write_bytes(raw+b'\n');bad('changed cached IR child',lambda:r.validate_group(manager,group));file.write_bytes(raw)
  file.unlink();bad('missing child',lambda:r.validate_group(manager,group));file.write_bytes(raw)
  redirect=NS(get_group=lambda name:{**group['paths'],'gather_hidden_bits.cubin':'/tmp/other.cubin'});bad('redirected group',lambda:r.validate_group(redirect,group))
  incomplete=NS(get_group=lambda name:{k:v for k,v in group['paths'].items() if not k.endswith('.source')});bad('incomplete group',lambda:r.validate_group(incomplete,group))
  wrong=root/'wrong-review.json';wrong.write_bytes(REVIEW.read_bytes()+b'\n');bad('mutated static review',lambda:r.proof(ORIGINAL,wrong))
  stub=root/'stub';stub.mkdir();(stub/'helper-summary.json').write_bytes((ORIGINAL/'helper-summary.json').read_bytes()+b'\n');bad('mutated original summary',lambda:r.proof(stub,REVIEW))
  # Compile barriers are source-level CPU function tests, never fake SDK success.
  calls=[]
  class Backend:
   def add_stages(self,*a):calls.append('backend')
  original=Backend.__dict__['add_stages']
  with r.prohibit_backend_compile(Backend):bad('unknown backend compile miss rejected',lambda:Backend().add_stages())
  ok('backend barrier restored, not executed',Backend.__dict__['add_stages'] is original and not calls)
  fn=NS(compile=lambda *a,**kw:calls.append('candidate'))
  original=fn.compile
  with r.prohibit_candidate_compile(fn):bad('candidate memory cache miss rejected',lambda:fn.compile('unknown signature'))
  ok('candidate original compiler restored, not executed',fn.compile is original and not calls)
  try:
   with r.prohibit_candidate_compile(fn):raise RuntimeError('simulated launch failure')
  except RuntimeError:pass
  ok('candidate compiler restored after failure',fn.compile is original)
 ok('original source/cache remains byte-exact',inventory(ORIGINAL)==before)
 source=(c.ROOT/'tools/run_biv150_dynamic_hidden_gather_helper.py').read_text();helper=(c.ROOT/'vllm_fl/patches/biv150_dynamic_hidden_gather.py').read_bytes()
 import subprocess
 ok('production helper source exact04fa',helper==subprocess.check_output(['git','-C',str(c.ROOT),'show',r.SOURCE+':vllm_fl/patches/biv150_dynamic_hidden_gather.py']))
 ok('no forced SME in normal options', 'use_sme' not in c.OPTIONS and 'use_sme=' not in source and 'TRITON_DISABLE_SME\']' not in source)
 ok('natural SME recorded and guarded',"natural_SME_prediction" in source and "never force5 or0" in source)
 ok('first licensed warmup then candidate-only barrier','cache.warmup(fn' in source and 'cache.prohibit_candidate_compile(fn)' in source and 'candidate_guard.__exit__' in source)
 ok('actual options ABI resources handler gates still exact',all(v in source for v in ("c.runtime_ABI(compiled)==old['runtime_ABI']","c.metadata(compiled)==old['actual_options']","c.resources(compiled)==old['resources']","bits.LAUNCHER_SHA")))
 ok('all device paths default unqualified',not {'torch','triton','numpy'}.intersection(sys.modules))
 return dict(passed=True,CPU_only=True,checks=checks,check_count=len(checks),actual_source_summary_sha256=r.SUMMARY,actual_static_review_sha256=r.REVIEW,
  expected_actual_hash=r.HASH,expected_actual_cubin=r.CUBIN,natural_SME_expected=5,old0_not_forced=True,reviewed7children_exact=True,original_cache_untouched=True,
  helper_entry_or_numeric_verified=False,actual_SDK_cache_hit_verified=False,performance_verified=False,service_verified=False,
  cache_module_sha256=c.sha(Path(r.__file__).read_bytes()),worker_sha256=c.sha((c.ROOT/'tools/run_biv150_dynamic_hidden_gather_helper.py').read_bytes()))
if __name__=='__main__':print(json.dumps(run(),sort_keys=True,indent=2))
