#!/usr/bin/env python3
"""Default-off PR120 full host validate timing on real CUDA Tensors; no producer launch."""
import argparse,hashlib,importlib.util,json,math,socket,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAIN='899ce77470b0b539aebc108fd122f668c2d5894d'
CANDIDATE='855475037fcbb46fc7288d7d87b7140b58864c27'
LIBRARY_SHA='5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'
NATIVE='vllm_fl/patches/biv150_konly_encoded_native.py'
PINS={NATIVE:'ce5feb765a635ded45924bfafce109a6b6ccfe04cc00495599b1c5170fe039b7',
 'vllm_fl/patches/biv150_konly_encoded_runtime.py':'0199c2664665798ff68e161803e65c5ce19f13893a1a1af80adba85210d8bb70',
 'vllm_fl/patches/biv150_konly_encoded_views.py':'018c993858803df250ec1fa089778dbed5627e2c1720c3ccd12aad4e40344ede',
 'vllm_fl/patches/biv150_konly_encoded_sources.json':'749c7ba477cb1445458ee8c6806c8432dc3804026cb511845ff3ae1e0baac56e'}
REFERENCE_SHA='4a2f387d835dc9addcf38a88714bbb01d8c0e526a9b196c946ae616c4749df89'
CASES=((64,2,2048,0),(32,4,2560,41),(61,2,2560,20),(30,4,2560,41))
WARMUP=128;CALLS=512;ROUNDS=3;PAGES=29;LAYERS=42
sha=lambda b:hashlib.sha256(b).hexdigest()
def need(ok,msg):
 if not ok:raise RuntimeError('PR120 actual validate: '+msg)
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def plan():
 return dict(schema='pr120-actual-tensor-full-validate-v1',baseline=MAIN,candidate=CANDIDATE,source_pins=PINS,reference_sha256=REFERENCE_SHA,library_sha256=LIBRARY_SHA,cases=[dict(N=n,S=s,Q_pitch=p,registry_entry=i) for n,s,p,i in CASES],registry_entries=LAYERS,cache_pages_per_entry=PAGES,warmup_per_arm=WARMUP,calls_per_sample=CALLS,ABBA_rounds=ROUNDS,streams=['default','side'],timing_scope='complete Adapter.validate, including verify/owner/recipe/roles/bounds/alias/scalar checks; host perf_counter_ns',stream_scope='actual default and side stream contexts; validate itself has no stream query, no added per-call query',actual_CUDA_required=True,default_enabled=False,GPU_executed=False,attention_writer_encoder_launches=0,performance_verified=False,C3_service_verified=False,model_precision_verified=False)
def preflight(commit,library):
 need(commit==git('rev-parse','HEAD').decode().strip(),'exact frozen HEAD')
 need(not git('status','--porcelain','--untracked-files=all'),'clean worktree')
 need(not git('diff',CANDIDATE,commit,'--','vllm_fl','csrc','tools/kernels'),'unchanged candidate production/GPU bytes')
 git('merge-base','--is-ancestor',CANDIDATE,commit)
 for p,h in PINS.items():need(sha((ROOT/p).read_bytes())==h and (ROOT/p).read_bytes()==git('show',commit+':'+p),'source '+p)
 reference=git('show',MAIN+':'+NATIVE);need(sha(reference)==REFERENCE_SHA,'reference adapter')
 for p in PINS:
  if p.endswith(('encoded_runtime.py','encoded_views.py')):need(git('show',MAIN+':'+p)==(ROOT/p).read_bytes(),'unchanged common runtime/broker')
 me=Path(__file__).relative_to(ROOT).as_posix();need(Path(__file__).read_bytes()==git('show',commit+':'+me),'tracked actual entry')
 need(library.is_absolute() and library.is_file() and sha(library.read_bytes())==LIBRARY_SHA,'exact existing5898 library')
 return reference

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def timed_validate(adapter,runtime,entry,kw,count,clock=time.perf_counter_ns):
 start=clock()
 for _ in range(count):adapter.validate(runtime,entry,kw)
 return clock()-start

def abba():return [arm for _ in range(ROUNDS) for arm in ('reference','candidate','candidate','reference')]

def compare(adapters,runtime,entry,kw,accept,expected_error=None):
 expected=[kw[k] for k in ('query_ptr','key_cache_ptr','key_scale_ptr','value_cache_ptr','seq_lens_ptr','block_table_ptr','segm_output_ptr','segm_max_ptr','segm_expsum_ptr')];rows=[]
 for arm in ('reference','candidate'):
  try:out=adapters[arm].validate(runtime,entry,kw)
  except RuntimeError as e:rows.append(dict(accepted=False,error_type=type(e).__name__,error=str(e)))
  else:
   need(type(out) is list and len(out)==9 and all(a is b for a,b in zip(out,expected)),'exact returned operands')
   rows.append(dict(accepted=True,exact_returned_objects=True))
 need(rows[0]==rows[1] and rows[0]['accepted'] is accept,'paired accept/reject result')
 if expected_error is not None:need(rows[0].get('error')=='NativeK8 service: '+expected_error,'original error order/message')
 return rows[0]

def make_inputs(torch,entry,n,s,pitch):
 # All backing allocations are real. No content is read, initialized or used by a kernel.
 if pitch==2560:
  backing=torch.empty((n,2560),dtype=torch.bfloat16,device='cuda:0')
  q=backing.as_strided((n,16,128),(2560,128,1))
  kv=backing.as_strided((n,2,128),(2560,128,1),2048)
  vv=backing.as_strided((n,2,128),(2560,128,1),2304)
 else:
  q=torch.empty((n,16,128),dtype=torch.bfloat16,device='cuda:0');kv=vv=None
 lens=torch.empty((n,),dtype=torch.int32,device='cuda:0');table=torch.empty((n,8192),dtype=torch.int32,device='cuda:0')
 partial=torch.empty((n,16,16,128),dtype=torch.float32,device='cuda:0').as_strided((n,16,s,128),(32768,2048,128,1))
 maximum=torch.empty((n,16,16),dtype=torch.float32,device='cuda:0').as_strided((n,16,s),(256,16,1))
 summation=torch.empty((n,16,16),dtype=torch.float32,device='cuda:0').as_strided((n,16,s),(256,16,1))
 role=entry.roles['decode79'];kw=dict(role,query_ptr=q,query_code_ptr=role['key_cache_ptr'],query_scale_ptr=role['key_scale_ptr'],seq_lens_ptr=lens,block_table_ptr=table,segm_output_ptr=partial,segm_max_ptr=maximum,segm_expsum_ptr=summation,query_stride_token=pitch,query_stride_head=128,block_table_stride=8192,segment_stride_token=32768,segment_stride_head=2048,segment_stride_segment=128,max_stride_token=256,max_stride_head=16,SEGMENTS=s)
 return kw,(kv,vv)

def device(args,reference):
 result=plan();result.update(completed=False,errors=[],source_commit=args.commit,actual_shapes=[],behavior=[],timings=[],native_produce_called=False,C3_service_verified=False)
 args.out.mkdir();(args.out/'sources').mkdir();refpath=args.out/'sources/main899_native.py';refpath.write_bytes(reference)
 def save(): (args.out/'summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 save()
 try:
  before=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True);need('No running processes found' in before,'GPU owner process table must be idle')
  with socket.socket() as sock:need(sock.connect_ex(('127.0.0.1',9031))==111,'service9031 must be closed')
  result['prelaunch_ixsmi']=before
  import torch
  result['CUDA_initialized_after_import']=torch.cuda.is_initialized();result['GPU_executed']=result['CUDA_initialized_after_import']
  need(str(torch.__version__)=='2.10.0' and torch._C._GLIBCXX_USE_CXX11_ABI is False,'actual Torch ABI')
  result['CUDA_API_entered']=True;result['GPU_executed']=True
  need(torch.cuda.is_available() and torch.cuda.device_count()==1,'single CUDA device');torch.cuda.set_device(0)
  prop=torch.cuda.get_device_properties(0);need('BI-V150' in prop.name and (prop.major,prop.minor)==(7,1),'actual BI-V150/71')
  result['GPU_executed']=True;result['device']=dict(name=prop.name,major=prop.major,minor=prop.minor,total_memory=prop.total_memory);result['torch']=dict(version=str(torch.__version__),file=str(Path(torch.__file__).resolve()),cxx11_abi=False)
  broker=load('pr120_actual_broker',ROOT/'vllm_fl/patches/biv150_konly_encoded_views.py');runtime_mod=load('pr120_actual_runtime',ROOT/'vllm_fl/patches/biv150_konly_encoded_runtime.py')
  modules=dict(reference=load('pr120_main899_native',refpath),candidate=load('pr120_actual_candidate',ROOT/NATIVE))
  runtime=runtime_mod.Runtime(torch,broker,{},None);runtime.registry.armed=True
  source=torch.empty((64,2,128),dtype=torch.bfloat16,device='cuda:0');entries=[]
  for _ in range(LAYERS):
   owner=torch.empty((PAGES,2,16,2,128),dtype=torch.bfloat16,device='cuda:0')
   entries.append(runtime.registry.claim(object(),owner,source,source))
  need(len(runtime.registry.entries)==LAYERS,'42 real Registry entries')
  config=dict(path=str(args.library.resolve()),sha256=LIBRARY_SHA)
  adapters={key:mod.Adapter(runtime,config) for key,mod in modules.items()}
  result['adapters']={key:adapter.imports for key,adapter in adapters.items()}
  need(adapters['reference'].op is adapters['candidate'].op,'same actual registered operator')
  inputs=[]
  for n,s,pitch,index in CASES:
   entry=entries[index];kw,shared_kv=make_inputs(torch,entry,n,s,pitch);inputs.append((entry,kw,shared_kv))
   result['behavior'].append(dict(case=[n,s,pitch,index],mutation='original',outcome=compare(adapters,runtime,entry,kw,True)))
   result['actual_shapes'].append(dict(N=n,S=s,Q_pitch=pitch,registry_entry=index,operands={k:modules['candidate'].geometry(v) for k,v in kw.items() if isinstance(v,torch.Tensor)},owner=modules['candidate'].geometry(entry.owner),QKV_shared_backing=shared_kv[0] is not None))
  entry,kw,_=inputs[1]
  changes=[('scalar_mismatch',dict(query_stride_token=1),'launch scalar/actual Tensor metadata mismatch'),('scratch_same_object',dict(segm_expsum_ptr=kw['segm_max_ptr']),'scratch backing alias'),('unused_Q_foreign',dict(query_code_ptr=kw['query_ptr']),'foreign unused private-Q operands')]
  alias=entry.owner.as_strided(tuple(kw['query_ptr'].shape),tuple(kw['query_ptr'].stride()))
  changes.append(('whole_storage_cache_alias',dict(query_ptr=alias),'source/live-cache backing alias'))
  changes.append(('wrong_lengths_dtype',dict(seq_lens_ptr=torch.empty((32,),dtype=torch.int64,device='cuda:0')),'I32 metadata'))
  changes.append(('CPU_lengths',dict(seq_lens_ptr=torch.empty((32,),dtype=torch.int32,device='cpu')),'single actual CUDA0 roles'))
  for label,update,message in changes:result['behavior'].append(dict(mutation=label,outcome=compare(adapters,runtime,entry,{**kw,**update},False,message)))
  for label,attribute,value,message in [('epoch','generation',runtime.registry.generation+1,'unowned/stale epoch'),('owner','owner',torch.empty_like(entry.owner),'registered allocation changed')]:
   old=getattr(entry,attribute)
   try:
    setattr(entry,attribute,value);result['behavior'].append(dict(mutation=label,outcome=compare(adapters,runtime,entry,kw,False,message)))
   finally:setattr(entry,attribute,old)
  prior=entry.roles['decode79']['key_cache_ptr'];replacement=torch.empty_like(prior)
  try:
   entry.roles['decode79']['key_cache_ptr']=replacement;result['behavior'].append(dict(mutation='role',outcome=compare(adapters,runtime,entry,kw,False,'typed role changed key_cache_ptr')))
  finally:entry.roles['decode79']['key_cache_ptr']=prior
  fresh_backing=torch.empty((32*2560+64,),dtype=torch.bfloat16,device='cuda:0');fresh=fresh_backing.as_strided((32,16,128),(2560,128,1),64)
  for label,q in [('new_storage_offset',fresh),('original_after_changes',kw['query_ptr'])]:result['behavior'].append(dict(mutation=label,outcome=compare(adapters,runtime,entry,{**kw,'query_ptr':q},True)))
  result['behavior'].append(dict(mutation='two_errors_original_order',outcome=compare(adapters,runtime,entry,{**kw,'query_ptr':alias,'segm_expsum_ptr':kw['segm_max_ptr']},False,'source/live-cache backing alias')))
  streams=dict(default=torch.cuda.default_stream(0),side=torch.cuda.Stream(device=0));torch.cuda.synchronize()
  for stream_name,stream in streams.items():
   with torch.cuda.stream(stream):
    need(torch.cuda.current_stream(0)==stream and not torch.cuda.is_current_stream_capturing(),'actual noncapturing selected stream')
    for case,(entry,kw,shared_kv) in zip(CASES,inputs):
     for arm in ('reference','candidate'):
      for _ in range(WARMUP):adapters[arm].validate(runtime,entry,kw)
     samples=[]
     for arm in abba():
      elapsed=timed_validate(adapters[arm],runtime,entry,kw,CALLS)
      need(type(elapsed) is int and elapsed>0,'positive complete host interval');samples.append(dict(arm=arm,calls=CALLS,elapsed_ns=elapsed,ns_per_call=elapsed/CALLS))
     medians={arm:statistics.median(x['ns_per_call'] for x in samples if x['arm']==arm) for arm in ('reference','candidate')}
     result['timings'].append(dict(case=list(case),stream=stream_name,actual_stream=int(stream.cuda_stream),samples=samples,median_ns=medians,reference_over_candidate=medians['reference']/medians['candidate'],saved_ns_per_call=medians['reference']-medians['candidate']))
     compare(adapters,runtime,entry,kw,True)
    need(torch.cuda.current_stream(0)==stream and not torch.cuda.is_current_stream_capturing(),'stream unchanged')
  result['actual_behavior_verified']=True;result['host_timing_completed']=True;result['performance_verified']=False
  result['scope']='Full Python validate on real CUDA tensors, not a CUDA kernel/Graph replay/service throughput measurement; 42 metadata entries of29pages are not a model capacity test.'
  result['completed']=True
 except Exception as exc:
  result['errors'].append(dict(type=type(exc).__name__,message=str(exc)));raise
 finally:save()
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actual-cuda',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--commit');p.add_argument('--library',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
 if not a.actual_cuda:
  need(not a.root_owned_exclusive and a.commit is None and a.library is None and a.out is None,'default plan takes no device arguments');print(json.dumps(plan(),indent=2,sort_keys=True));return
 need(a.root_owned_exclusive and a.commit and a.library and a.out,'explicit exclusive, frozen commit, actual library and new output required')
 need(not a.out.exists() and a.out.parent.is_dir(),'new output with existing parent')
 reference=preflight(a.commit,a.library);device(a,reference)
if __name__=='__main__':main()
