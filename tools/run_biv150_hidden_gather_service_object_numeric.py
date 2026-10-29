"""Numeric-only test of the exact failed-smoke SME1 group; default CPU plan."""
import argparse,inspect,json,os,types
from pathlib import Path
import run_biv150_dynamic_hidden_gather_helper as base
import biv150_hidden_gather_service_object_cache as cache
c=base.c
PARENT_SHA='bc3ff1848c9c11968549d1d63eadfc03da520114ca951088e2c39fdd0be9000f'
REUSED={
 'run_biv150_dynamic_hidden_gather_helper.py':PARENT_SHA,
 'biv150_dynamic_hidden_gather_reviewed_cache.py':'030d8a6b4559ffcf949bf979a42fe1ba7405f2eaa3d2744aba2ba9df5c769992',
 'run_biv150_dynamic_hidden_gather_bits.py':'38da26d0cc01bdecc798f6290c78155d62ca720f50067dc872cd2ccd026df55f',
 'plan_biv150_dynamic_hidden_gather_helper.py':'1e289d17ad51dbc90047ceba92678b891098644e125aa784d921e6d42b5adac2'}

def plan():
 return dict(device_executed=False,service_verified=False,performance_verified=False,timing_executed=False,
  original_failed_service_source=cache.SOURCE,actual_group=cache.HASH,actual_cubin=cache.CUBIN,actual_launcher=cache.LAUNCHER,
  original_service_Plan_host_submits=0,natural_seed=[18,1],seed_warmup_only_no_kernel=True,
  helper_cases=[dict(M=m,N=n,kind=k) for m,n in base.bits.CASES for k in ('ordinary','Plan')],
  helper_phases='AB',constructed_Plan_not_service_coverage=True,additional_direct_guard_launches=4,
  expected_total_warmups=5,expected_helper_numeric_launches=8,private_cache_only=True,backend_compile_prohibited=True,
  original_helper_new_output_allocation_has_no_injected_output_canary=True,additional_guards_use_same_loaded_object=True,
  no_math_105_or_service_qualification=True)

def seed(fn,torch,result,out,save):
 """Normal binder and natural18/1 SME selection, with the actual cache immutable."""
 h=torch.empty((18,2048),dtype=torch.bfloat16,device='cuda');i=torch.empty((1,),dtype=torch.int32,device='cuda');o=torch.empty((1,2048),dtype=torch.bfloat16,device='cuda')
 jit=base.importlib.import_module('triton.runtime.jit');c.need(c.sha(Path(jit.__file__).read_bytes())==c.JIT_SHA,'actual reviewed SME selector')
 device=torch.cuda.current_device();prop=torch.cuda.get_device_properties(device)
 c.need('BI-V150' in prop.name and prop.multi_processor_count==16,'actual target device')
 _,_,_,_,binder=fn.device_caches[device];bound,specialization,_=binder(h,i,o,18,1,**c.OPTIONS)
 record=dict(M=18,N=1,descriptors={k:c.descriptor(t) for k,t in dict(hidden=h,indices=i,output=o).items()},natural_SME=jit.get_corex_sme(bound.values(),specialization),kernel_launches=0,warmup_attempted=False,warmup_completed=False)
 result['natural_seed']=record;save();c.need(record['natural_SME']==1,'natural N1 must select1; never force use_sme')
 record['warmup_attempted']=True;result['production_warmups_attempted']+=1;save()
 compiled=cache.warmup(fn,device,result['reviewed_production_cache'],lambda:fn.warmup(h,i,o,18,1,grid=(1,16),**c.OPTIONS))
 record['warmup_completed']=True;result['production_warmups_completed']+=1;compiled._init_handles()
 folder=out/'actual-seed-18-1';folder.mkdir();files={}
 for ext in ('source','ttir','ttgir','llir','asm','cubin'):
  b=compiled.asm[ext];b=b.encode() if isinstance(b,str) else b;(folder/('actual.'+ext)).write_bytes(b);files[ext]=dict(bytes=len(b),sha256=c.sha(b))
 owner=getattr(getattr(compiled._run,'launch',None),'__self__',None)
 c.need(isinstance(owner,types.ModuleType) and getattr(owner,'__file__',None),'seed actual loaded host extension')
 c.need(Path(owner.__file__).resolve()==Path(result['reviewed_production_cache']['host'][0]['path']).resolve(),'actual private original launcher path')
 lb=Path(owner.__file__).read_bytes();(folder/'actual-launcher.so').write_bytes(lb)
 record.update(python_object_id=id(compiled),hash=str(compiled.hash),ABI=c.runtime_ABI(compiled),options=c.metadata(compiled),resources=c.resources(compiled),files=files,launcher_sha256=c.sha(lb));save()
 expected=result['service_actual_object']
 c.need(record['hash']==cache.HASH and record['ABI']==expected['ABI'] and record['options']==expected['options'] and record['resources']==expected['resources'] and files==expected['files'] and c.sha(lb)==cache.LAUNCHER,'exact original service object and launcher before numeric launch')
 return compiled

def output_guards(compiled,torch,result,blob,save):
 """Original bits fixture/checks, separate from unmodified helper allocation."""
 bits=base.bits;result['additional_direct_guard_records']=[]
 def read(t):return t.detach().cpu().numpy().tobytes()
 for m,n in bits.CASES:
  buffers=None;descriptors=None;previous=None
  for phase in (0,1):
   f=bits.fixture(m,n,phase);r=dict(M=m,N=n,phase='AB'[phase],compiled_python_object_id=id(compiled),inputs={k:blob(v) for k,v in f.items() if isinstance(v,bytes)},logical_indices=f['logical_indices'],launch_attempted=False,launch_completed=False)
   result['additional_direct_guard_records'].append(r);save()
   host=[torch.frombuffer(bytearray(f[k]),dtype=d) for k,d in [('hidden',torch.int16),('indices',torch.int32),('output_poison',torch.int16)]]
   if buffers is None:buffers=tuple(t.clone().to('cuda') for t in host)
   else:
    for dst,src in zip(buffers,host):dst.copy_(src)
   h,i,o=buffers;hv=h.narrow(0,bits.GUARD,m*bits.D).view(torch.bfloat16).reshape(m,bits.D);iv=i.narrow(0,bits.IGUARD,n);ov=o.narrow(0,bits.GUARD,n*bits.D).view(torch.bfloat16).reshape(n,bits.D)
   desc={k:c.descriptor(t) for k,t in dict(hidden=hv,indices=iv,output=ov).items()};r['descriptors']=desc
   c.need(all(x['pointer_mod64']==0 for x in desc.values()),'original guard alignment')
   if descriptors is None:descriptors=desc
   else:c.need(desc==descriptors,'B reuses all exact guarded storages/descriptors')
   if previous is not None:c.need(all(previous[k]!=f[k] for k in ('hidden','indices','expected')),'B changes consumed values and indices')
   r['launch_attempted']=True;save();compiled[((n+7)//8,16,1)](hv,iv,ov,m,n,8,128);torch.cuda.synchronize();r['launch_completed']=True
   actual={k:read(t) for k,t in zip(('hidden','indices','output'),buffers)};r['actual']={k:blob(v) for k,v in actual.items()};r['hard_checks']=bits.checks(f,actual['hidden'],actual['indices'],actual['output']);save()
   c.need(all(r['hard_checks'].values()),'full guarded output/poison/readonly failure');previous=f
 c.need(len(result['additional_direct_guard_records'])==4,'four additional direct launches')

# Each edit is an exact unique seam; inverse restoration is checked on CPU.
EDITS=[
 (' candidate_guard=None'," result['service_actual_object']=actual;result['service_object_numeric_plan']=plan();result['service_actual_object_numeric_verified']=False\n candidate_guard=None"),
 ('  with torch.inference_mode():\n   for m,n in bits.CASES:', '  with torch.inference_mode():\n   seeded=seed(fn,torch,result,out,save);candidate_guard=cache.prohibit_candidate_compile(fn);candidate_guard.__enter__()\n   for m,n in bits.CASES:'),
 ('     # Two actual production warmups only per M/N; no84-shape sweep. Export before gate.', '     # Existing case warmups must reuse the naturally seeded exact SME1 object.'),
 ("record['natural_SME_prediction']", "record['cold_SME_prediction_if_uncached']"),
 ("     result['production_warmups_completed']+=1;compiled._init_handles()", "     result['production_warmups_completed']+=1;compiled._init_handles();c.need(compiled is seeded,'same actual service object for every original helper case')"),
 ("'actual loaded launcher differs from5124 review'", "'actual loaded launcher differs from service4105'"),
 ("   # No profiling inside timed original frames. All numeric modes passed first.", "   output_guards(seeded,torch,result,blob,save)\n   c.need(result['production_warmups_completed']==5,'one seed plus four original warmups');result['service_actual_object_numeric_verified']=True\n   # Timing code retained byte-for-byte but this entry forbids timing.")]

def expanded():
 for name,h in REUSED.items():c.need(c.sha((Path(__file__).parent/name).read_bytes())==h,'unchanged reused source '+name)
 old=inspect.getsource(base.execute);new=old
 for before,after in EDITS:
  count=2 if before=="record['natural_SME_prediction']" else 1
  c.need(new.count(before)==count,'unique reviewed execute seam');new=new.replace(before,after)
 inverse=new
 for before,after in reversed(EDITS):inverse=inverse.replace(after,before)
 c.need(inverse==old,'exact inverse to unchanged original execute')
 return new

def execute(a):
 c.need(a.timing is False,'this entry is numeric only, no cost repeats')
 c.need(not c.git('diff',cache.SOURCE,'--','vllm_fl','csrc'),'all production/GPU bytes remain frozen8b7')
 c.need(os.environ.get('CJJ_HIDDEN_GATHER_DIAGNOSTICS','0')=='0','component diagnostics stay OFF')
 namespace=dict(vars(base),cache=cache,bits=types.SimpleNamespace(**dict(vars(base.bits),LAUNCHER_SHA=cache.LAUNCHER)),seed=seed,output_guards=output_guards,plan=plan)
 exec(compile(expanded(),str(Path(__file__).resolve())+':exact-parent-execute','exec'),namespace)
 return namespace['execute'](a)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-helper',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--expected-source-commit')
 for name in ('compile-out','actual-review','bits-review','production-out','production-review','out'):p.add_argument('--'+name,type=Path)
 a=p.parse_args();a.timing=False
 if not a.run_helper:
  if any(v for k,v in vars(a).items() if k not in ('run_helper','timing')):p.error('device arguments require explicit --run-helper')
  result=plan()
 else:
  if not all(getattr(a,k) for k in ('root_owned_exclusive','expected_source_commit','compile_out','actual_review','bits_review','production_out','production_review','out')):p.error('Root/source/original proofs/service proof/new out required')
  result=execute(a)
 print(json.dumps(result,sort_keys=True,indent=2));return int(a.run_helper and not result['completed'])
if __name__=='__main__':raise SystemExit(main())
