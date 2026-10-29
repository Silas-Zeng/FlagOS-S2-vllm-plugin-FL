#!/usr/bin/env python3
"""Default CPU plan; explicit Root-only full controlled-frame Tensor OFF/ON cost."""
import contextlib,inspect,json,os,sys
from pathlib import Path
from types import SimpleNamespace as NS
import run_biv150_dynamic_hidden_gather_helper as base
c=base.c;ROOT=base.ROOT;PARENT='8e65fcc27ac26a6ab08403b9e2208d3acf9eeb39'
PINS={'vllm_fl/worker/worker.py':'a8de1780d1b41bf4889c26f3d593e6ec54e91a5f61996ce64b4d79f2a97efa24',
 'vllm_fl/utils.py':'1c841db50d0c4370d3d04fa0383a87eaa3ceba667df0ffe66aca8f482830c429',
 'vllm_fl/envs.py':'ecae13cf9bb3cbc83c0c411bd15420031a543743a94b0fe1d41dd959c803222f',
 'vllm_fl/dispatch/config/iluvatar.yaml':'e37660b6cd5b8dbe0b927e03c9f3bc4e130e42ed643d099eb20981b9afdf95a7',
 'vllm_fl/dispatch/config/__init__.py':'2493096111a4a51548ae9d1c7f3fae7e89ecdf16c226f0dcaef37759d067bc90'}
ENV_KEYS=('USE_FLAGGEMS','VLLM_FL_PREFER_ENABLED','VLLM_FL_PREFER','VLLM_FL_FLAGOS_WHITELIST','VLLM_FL_FLAGOS_BLACKLIST','VLLM_FL_FLAGOS_BLACKLIST_APPEND','FLAGTUNE','USE_C_EXTENSION','VLLM_BATCH_INVARIANT')
SERVICE=dict(source='69827253756a0ef9e618ae0322e9ff361e93504f',manifest='bff4dcb835c5c0de0bf9eeef29829da6f777059558f402aaa1b7e33bce2da16e',controller='637d29e30c86a56e0536548b5678067954c8900046dc2d1a9dd78a96f4fa7199',server_log='e13480803a70ce788665ff2820cadae2b4fdc6b0ad04d974d5755a33afce27c2')

class TensorModel:
 def __init__(self,controls,parent):self.controls=controls;self.parent=parent
 def __call__(self,**kwargs):return self.controls.hidden
 def compute_logits(self,sample):return self.parent.compute_logits(sample)

def transformed_execute():
 """Only three private timing-loop edits; original numeric/ABI/receipt gates exact."""
 c.need((ROOT/'tools/run_biv150_dynamic_hidden_gather_helper.py').read_bytes()==__import__('subprocess').check_output(['git','-C',str(ROOT),'show',PARENT+':tools/run_biv150_dynamic_hidden_gather_helper.py']),'exact frozen parent worker')
 text=inspect.getsource(base.execute)
 edits=(("for _ in range(2):invoke(x,helper,arm=='candidate')","for _ in range(8):invoke(x,helper,arm=='candidate')"),
  ('torch.cuda.synchronize();start=torch.Event(enable_timing=True);end=torch.Event(enable_timing=True);start.record();t=time.perf_counter_ns()',
   "torch.cuda.synchronize();_tensor_cost_block('before',x,arm);start=torch.Event(enable_timing=True);end=torch.Event(enable_timing=True);start.record();t=time.perf_counter_ns()"),
  ('event_ms=start.elapsed_time(end)/32;',"_tensor_cost_block('after',x,arm);event_ms=start.elapsed_time(end)/32;"))
 for before,after in edits:c.need(text.count(before)==1,'exact private cost edit');text=text.replace(before,after)
 return text,edits

def plan():
 for path,digest in PINS.items():c.need(c.sha((ROOT/path).read_bytes())==digest,'service dispatch source '+path)
 text,edits=transformed_execute()
 return dict(default_CPU=True,parent=PARENT,production_feature_default='OFF',production_diff=False,cases=[[m,n,k] for m,n in base.bits.CASES for k in ('ordinary','Plan')],
  service_policy_evidence=SERVICE,dispatch_source_pins=PINS,control='same actual Tensor; feature OFF executes original hidden_states[logits_indices] seam',candidate='same actual Tensor and original execute_model/_prepare_inputs/receipt/new output/_launch',
  FlagGems='same Worker rank0 default enable(record=True,once=True); output recording path isolated; vendor exclusions not overridden; actual index function/code/cache must be observed before timing',
  numeric_candidate_calls_planned=8,additional_real_Tensor_OFF_numeric_calls_planned=8,production_warmups_planned=4,timing_opt_in=True,order='ABBA',warmup_calls_per_block=8,measured_calls_per_block=32,private_loop_edits=len(edits),
  first_baseline_compile_outside_timing=True,timed_backend_compile_prohibited=True,timed_memory_cache_and_file_inventory_must_be_unchanged=True,
  full_call_scope='controlled original execute_model, original input preparation/H2D/event wait, receipt, allocation and host/launch; model/head arithmetic excluded; not service',
  real_make_plan_or_C3=False,Graph_numeric=False,side_stream=False,performance_verified=False,service_verified=False,actual_device_work=0)

class Adapter:
 def __init__(self,a,generated):
  self.a=a;self.generated=generated;self.original_fixture=base.fixture;self.original_observe=base.observe
  self.current=None;self.torch=None;self.index_codes=set();self.functions={};self.active=None;self.numeric=[];self.dispatch=[];self.blocks=[];self.loaded={};self.policy=None
 def save(self):c.save(self.a.out/'tensor-cost-records.json',dict(service_policy=self.policy,numeric=self.numeric,dispatch=self.dispatch,timing_blocks=self.blocks,loaded=self.loaded))
 def blob(self,b):
  import gzip
  p=self.a.out/'arrays'/('sha256-'+c.sha(b)+'.bin.gz');p.parent.mkdir(exist_ok=True)
  if not p.exists():p.write_bytes(gzip.compress(b,compresslevel=1,mtime=0))
  return dict(path=str(p.relative_to(self.a.out)),bytes=len(b),sha256=c.sha(b))
 def read(self,t):return t.detach().view(self.torch.int16).cpu().numpy().tobytes()
 def install(self,torch):
  for name in ENV_KEYS:c.need(name not in os.environ,'service default environment differs '+name)
  for name,value in os.environ.items():
   if name.startswith(('VLLM_FL_','BIV150_')):c.need(name in (c.ENV,'VLLM_FL_BIV150_HEAD_TEMPERATURE') and value=='0','unexpected component feature/config '+name)
  for path,digest in PINS.items():c.need(c.sha((ROOT/path).read_bytes())==digest,'service dispatch source '+path)
  # Set by execute before ANY framework/runner import, not just before enable.
  c.need(os.environ.get('FLAGGEMS_CACHE_DIR')==str(self.a.out/'flaggems-cache') and os.environ.get('FLAGGEMS_ENABLE_OPLIST_PATH')==str(self.a.out/'flaggems-oplist.txt'),'private pre-import FlagGems paths')
  import flag_gems
  from vllm_fl import envs
  from vllm_fl.utils import get_flag_gems_whitelist_blacklist
  c.need(envs.USE_FLAGGEMS is True and get_flag_gems_whitelist_blacklist()==(None,None),'same actual Worker default branch')
  module=Path(flag_gems.__file__).resolve();c.need(module==Path('/usr/local/lib/python3.12/site-packages/flag_gems/__init__.py'),'actual installed FlagGems')
  self.policy=dict(service=SERVICE,environment={k:os.environ.get(k) for k in ENV_KEYS},USE_FLAGGEMS=True,whitelist=None,blacklist=None,enable_kwargs=dict(record=True,once=True,path=envs.FLAGGEMS_ENABLE_OPLIST_PATH),installed_version=str(getattr(flag_gems,'__version__',None)),installed_sources={})
  # Exact Worker all-default API call; its vendor-default exclusions remain active.
  flag_gems.enable(record=True,once=True,path=envs.FLAGGEMS_ENABLE_OPLIST_PATH)
  fn=flag_gems.index
  while True:
   code=getattr(fn,'__code__',None)
   if code is not None:self.index_codes.add(code)
   if not hasattr(fn,'__wrapped__'):break
   fn=fn.__wrapped__
  c.need(self.index_codes,'unknown actual FlagGems index implementation')
  for role,file in (('FlagGems_init',module),('FlagGems_index',Path(inspect.getsourcefile(fn)).resolve())):
   c.need(file.is_relative_to(module.parent),'installed index source scope');b=file.read_bytes();p=self.a.out/'actual-dispatch-sources';p.mkdir(exist_ok=True);(p/(role+'.py')).write_bytes(b);self.policy['installed_sources'][role]=dict(path=str(file),sha256=c.sha(b))
  (self.a.out/'derived-private-execute.py').write_text(self.generated);self.torch=torch;self.save()
 def fixture(self,module,helper,torch,np,m,n,kind,hidden):
  if self.torch is None:self.install(torch)
  x=self.original_fixture(module,helper,torch,np,m,n,kind,hidden);x.runner.model=TensorModel(x.controls,x.runner.model);self.current=x;return x
 def export(self,kernel):
  name=c.metadata(kernel).get('name');key=str(kernel.hash);root=self.a.out/'actual-dispatch-kernels'/key
  if key in self.loaded:return
  root.mkdir(parents=True);files={}
  for ext,value in kernel.asm.items():
   if not isinstance(value,(str,bytes)):continue
   c.need(ext.replace('_','').isalnum(),'actual artifact extension');b=value.encode() if isinstance(value,str) else value;p=root/('actual.'+ext);p.write_bytes(b);files[ext]=dict(path=str(p.relative_to(self.a.out)),bytes=len(b),sha256=c.sha(b))
  self.loaded[key]=dict(name=name,hash=key,actual_options=c.metadata(kernel),typed_ABI={k:c.typed(getattr(kernel.src,k,None)) for k in ('signature','constants','attrs')},resources=c.resources(kernel),packed_metadata=c.typed(kernel.packed_metadata),files=files)
  launch=getattr(getattr(kernel,'_run',None),'launch',None);owner=getattr(launch,'__self__',None);path=getattr(owner,'__file__',None)
  if path:
   b=Path(path).read_bytes();(root/'actual-launcher.so').write_bytes(b);self.loaded[key]['actual_handler']=dict(type=type(kernel._run).__module__+'.'+type(kernel._run).__qualname__,path=str(path),sha256=c.sha(b))
 def profile(self,frame,event,value):
  if event=='call' and frame.f_code in self.index_codes and frame.f_locals.get('inp') is self.current.controls.hidden:
   ii=frame.f_locals.get('indices');c.need(isinstance(ii,(list,tuple)) and len(ii)==1,'actual Tensor advanced-index arguments')
   self.dispatch.append(dict(M=self.current.case['M'],N=self.current.case['N'],kind=self.current.case['source'],function=frame.f_code.co_name,source=frame.f_code.co_filename,source_sha256=c.sha(Path(frame.f_code.co_filename).read_bytes()),hidden=c.descriptor(frame.f_locals['inp']),indices=c.descriptor(ii[0]),actual_GPU_indices_before=self.blob(ii[0].detach().cpu().numpy().tobytes())))
  if frame.f_code.co_name=='run' and '/triton/' in frame.f_code.co_filename:
   fn=frame.f_locals.get('self')
   if hasattr(fn,'device_caches') and hasattr(fn,'fn'):
    self.functions[id(fn)]=fn
    if event=='return' and hasattr(value,'asm') and hasattr(value,'src'):self.export(value)
 @contextlib.contextmanager
 def observed(self,helper,records):
  with self.original_observe(helper,records):
   previous=sys.getprofile()
   def profile(frame,event,value):previous(frame,event,value);self.profile(frame,event,value)
   sys.setprofile(profile)
   try:yield
   finally:sys.setprofile(previous)
  if not any(x['kind']=='_launch' for x in records):return
  # Additional numeric-only OFF call uses the same real Tensor and CPU producer.
  x=self.current;candidate=next(x['output'] for x in records if x['kind']=='try_gather' and x['output'] is not None);start=len(self.dispatch);events=[]
  arm=x.controls.arm;enabled=os.environ.get(helper.ENV);x.controls.arm='control'
  try:
   with self.original_observe(helper,events):
    previous=sys.getprofile()
    def profile(frame,event,value):previous(frame,event,value);self.profile(frame,event,value)
    sys.setprofile(profile)
    try:control=base.invoke(x,helper,False)
    finally:sys.setprofile(previous)
   self.torch.cuda.synchronize();c.need(not any(v['kind']=='_launch' for v in events),'OFF must not launch helper')
   c.need(self.read(control)==self.read(candidate) and len(self.dispatch)>start,'real OFF full bits and actual FlagGems index ingress')
   h=x.controls.hidden;entire=h.as_strided((h.untyped_storage().nbytes()//2,),(1,),storage_offset=0)
   self.numeric.append(dict(M=x.case['M'],N=x.case['N'],kind=x.case['source'],phase='AB'[sum(r['M']==x.case['M'] and r['kind']==x.case['source'] for r in self.numeric)],actual_control=self.blob(self.read(control)),actual_hidden_after=self.blob(self.read(entire)),dispatch_records=[start,len(self.dispatch)],feature_OFF_helper_launches_observed=0))
   self.save()
  finally:x.controls.arm=arm;os.environ[helper.ENV]=enabled
 def memory(self):
  result={}
  for fn in self.functions.values():
   memory=fn.device_caches[self.torch.cuda.current_device()][0]
   result[str(id(fn))]=dict(name=fn.fn.__name__,keys={str(k):dict(object=id(v),hash=str(v.hash)) for k,v in memory.items()})
   for kernel in memory.values():self.export(kernel)
  return result
 def files(self):
  result={}
  for root in (self.a.out/'triton-cache',self.a.out/'flaggems-cache'):
   if root.exists():
    for p in root.rglob('*'):
     if p.is_file() and p.suffix in ('.py','.json','.source','.ttir','.ttgir','.llir','.asm','.cubin','.so','.db','.sqlite','.sqlite3','.db-wal','.db-shm','.sqlite-wal','.sqlite-shm','.sqlite3-wal','.sqlite3-shm'):result[str(p.relative_to(self.a.out))]=c.sha(p.read_bytes())
  return result
 def block(self,stage,x,arm):
  if stage=='before':
   c.need(self.active is None and len(self.numeric)==8 and self.dispatch,'all actual OFF numeric dispatch before timing')
   baseline=dict(memory=self.memory(),files=self.files());fn=next(iter(self.functions.values()));backend=fn.device_caches[self.torch.cuda.current_device()][3]
   guard=base.cache.prohibit_backend_compile(type(backend));guard.__enter__();self.active=(guard,baseline,x.case,arm)
  else:
   c.need(self.active is not None,'matching timing block');guard,before,case,saved_arm=self.active;guard.__exit__(None,None,None);self.active=None
   after=dict(memory=self.memory(),files=self.files());self.blocks.append(dict(case=case['source'],M=case['M'],N=case['N'],arm=arm,before=before,after=after,unchanged=before==after));self.save();c.need(arm==saved_arm and before==after,'timed cache changed: stop review, no cost qualification')

def main():
 if '--run-helper-cost' not in sys.argv:
  c.need(len(sys.argv)==1,'explicit --run-helper-cost required for all device/cost arguments');print(json.dumps(plan(),sort_keys=True,indent=2));return 0
 # Reuse the exact existing CLI/admission, fixture loops, numeric and ABBA worker.
 sys.argv[sys.argv.index('--run-helper-cost')]='--run-helper';text,_=transformed_execute();old_fixture=base.fixture;old_observe=base.observe
 original_execute=base.execute;adapter=None
 def execute(a):
  nonlocal adapter
  c.need(not {'torch','triton','numpy'}.intersection(sys.modules),'pre-framework private FlagGems paths')
  os.environ['FLAGGEMS_CACHE_DIR']=str(a.out/'flaggems-cache');os.environ['FLAGGEMS_ENABLE_OPLIST_PATH']=str(a.out/'flaggems-oplist.txt')
  adapter=Adapter(a,text);base.fixture=adapter.fixture;base.observe=adapter.observed;base._tensor_cost_block=adapter.block
  namespace=dict(base.__dict__);exec(compile(text,str(a.out/'derived-private-execute.py'),'exec'),namespace);run=namespace['execute'];namespace['_tensor_cost_block']=adapter.block
  try:
   result=run(a)
   result.update(tensor_cost_actual_control='real Tensor advanced index at original fallback seam',tensor_cost_numeric=adapter.numeric,tensor_cost_actual_dispatch=adapter.dispatch,tensor_cost_policy=adapter.policy,tensor_cost_warmups_per_block=8,tensor_cost_timed_cache_checks=adapter.blocks,
    component_full_call_timing_observed=result['timing_executed'],performance_verified=False,service_verified=False)
   for record in result['timings']:record['control']='real Tensor[indices] with production feature OFF and same installed Worker FlagGems policy';record['warmups_per_block']=8;record['candidate_calls_inferred_from_loop_not_profile_observed']=80;record['control_calls_inferred_from_loop']=80
   c.save(a.out/'helper-summary.json',result);return result
  finally:
   if adapter.active is not None:adapter.active[0].__exit__(None,None,None)
   base.fixture=old_fixture;base.observe=old_observe
 base.execute=execute
 try:return base.main()
 finally:base.execute=original_execute
if __name__=='__main__':raise SystemExit(main())
