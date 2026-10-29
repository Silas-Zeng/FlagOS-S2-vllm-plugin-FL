#!/usr/bin/env python3
"""CPU plan by default. Root-only actual helper/caller-frame component, never service."""
import argparse,contextlib,gzip,importlib,json,os,socket,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import run_biv150_dynamic_hidden_gather_bits as bits
import biv150_dynamic_hidden_gather_reviewed_cache as cache
import plan_biv150_dynamic_hidden_gather_helper as spec
c=bits.c;ROOT=Path(__file__).resolve().parents[1]
HELPER='vllm_fl.patches.biv150_dynamic_hidden_gather'
RUNNER='vllm_fl.worker.model_runner'
MAIN='899ce77470b0b539aebc108fd122f668c2d5894d'

def fixture(module,helper,torch,np,m,n,kind,hidden):
 """Real class/code/Buffer; synthetic batch/config/model outside the protected seam."""
 r=module.ModelRunnerFL.__new__(module.ModelRunnerFL);r.device=hidden.device
 cfg=NS(model_config=NS(model='/component/MiniCPM5-2B',dtype=torch.bfloat16,quantization=None,is_encoder_decoder=False,
     hf_text_config=NS(hidden_size=2048,num_hidden_layers=42,num_attention_heads=16,num_key_value_heads=2)),
  parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,decode_context_parallel_size=1,data_parallel_size=1,use_ubatching=False,num_ubatches=1),
  speculative_config=None,lora_config=None,cache_config=NS(kv_sharing_fast_prefill=False,mamba_cache_mode='none'))
 r.vllm_config=cfg;r.model_config=cfg.model_config;r.cache_config=cfg.cache_config;r.parallel_config=cfg.parallel_config;r.speculative_config=None;r.lora_config=None
 for name,value in dict(is_pooling_model=False,supports_mm_inputs=False,enable_prompt_embeds=False,use_aux_hidden_state_outputs=False,uses_mrope=False,uses_xdrope_dim=0,
  use_async_spec_decode=False,use_async_scheduling=True,num_spec_tokens=0,prev_num_spec_tokens=0,_draft_token_ids=None,mamba_prev_last_scheduled_idx=None,pin_memory=True,
  execute_model_state=None,routed_experts_initialized=False,cascade_attn_enabled=False,calculate_kv_scales=False,broadcast_pp_output=False,num_accepted_tokens_event=None,
  valid_sampled_token_count_gpu=None,max_num_tokens=2048,max_num_reqs=64).items():setattr(r,name,value)
 r.prepare_inputs_event=torch.Event();r.num_prompt_logprobs={};r.kv_cache_config=NS(kv_cache_groups=[]);r.attn_groups=[]
 # Original _make_buffer produces the exact installed CpuGpuBuffer, with pinnedCPU alias.
 for name,sz,dtype in (('query_start_loc',65,torch.int32),('query_pos',2048,torch.int64),('input_ids',2048,torch.int32),
  ('req_indices',2048,torch.int64),('num_scheduled_tokens',64,torch.int32),('prev_positions',64,torch.int64),('num_accepted_tokens',64,torch.int32),('discard_request_mask',64,torch.bool)):
  setattr(r,name,r._make_buffer(sz,dtype=dtype))
 r.arange_np=np.arange(2048,dtype=np.int64)
 r.positions=torch.zeros(2048,dtype=torch.int64,device=r.device);r.seq_lens=torch.zeros(64,dtype=torch.int32,device=r.device)
 r.num_computed_tokens=torch.zeros(64,dtype=torch.int32,device=r.device);r.optimistic_seq_lens_cpu=torch.zeros(64,dtype=torch.int32,pin_memory=True)
 computed=torch.zeros(64,dtype=torch.int32,pin_memory=True);token_ids=torch.zeros((64,2048),dtype=torch.int32,pin_memory=True)
 case=spec.producer_case(m,n,kind);scheduled=np.asarray(case['scheduled'],dtype=np.int32);ids=['component-'+str(i) for i in range(n)]
 tables=NS(commit_block_table=lambda *a:None,compute_slot_mapping=lambda *a:None)
 r.input_batch=NS(num_reqs=n,req_ids=ids,num_computed_tokens_cpu_tensor=computed,num_computed_tokens_cpu=computed.numpy(),
  token_ids_cpu_tensor=token_ids,token_ids_cpu=token_ids.numpy(),req_prompt_embeds={},block_table=tables,prev_req_id_to_index={})
 r.requests={key:NS(num_tokens=int(scheduled[i])) for i,key in enumerate(ids)}
 output=NS(total_num_scheduled_tokens=m,num_scheduled_tokens=dict(zip(ids,map(int,scheduled))),scheduled_encoder_inputs={},scheduled_spec_decode_tokens={},num_common_prefix_blocks=[],
  scheduled_cached_reqs=NS(resumed_req_ids=[]),has_structured_output_requests=False,pending_structured_output_tokens=False)
 controls=NS(arm='candidate',graph_mode=module.CUDAGraphMode.NONE,hidden=hidden)
 class IndexSelectProxy:
  # Control only: exact original fallback callsite dispatches to requested index_select.
  def __getitem__(self,index):return torch.index_select(controls.hidden,0,index)
 proxy=IndexSelectProxy()
 class Model:
  def __call__(self,**kwargs):return controls.hidden if controls.arm=='candidate' else proxy
  def compute_logits(self,sample):return sample # head/model arithmetic excluded, no sampling call.
 r.model=Model()
 # All three protected methods remain actual class functions with exact original code.
 r._update_states=lambda *a:None
 r._determine_batch_execution_and_padding=lambda **kw:(controls.graph_mode,NS(num_tokens=m,num_reqs=n),False,None,None)
 r._get_slot_mappings=lambda **kw:({},None)
 r._build_attention_metadata=lambda **kw:({},None)
 r._preprocess=lambda *a:(None,None,None,None,{},None)
 r.maybe_get_kv_connector_output=lambda *a,**kw:contextlib.nullcontext(None)
 r._prepare_input_ids=lambda out,n,total,ends:r.input_ids.copy_to_gpu(total)
 plan=None
 if kind=='Plan':
  order=importlib.import_module('vllm_fl.patches.biv150_compute_order');state=dict(mode=module.CUDAGraphMode,selected=0)
  table=r._make_buffer(64,1,dtype=torch.int32);table.np.fill(0);state['table_buffer']=table
  for key,shape,dtype in (('table',(64,1),torch.int32),('positions',(2048,),torch.int64),('seq',(64,),torch.int32),('computed',(64,),torch.int32),('prompt',(64,),torch.int32),('logits',(64,),torch.int32)):
   cpu=torch.zeros(shape,dtype=dtype,pin_memory=True);state[key+'_cpu']=cpu;state[key+'_np']=cpu.numpy()
  perm=np.asarray(case['permutation'],dtype=np.int64);ends=np.asarray(case['logits_indices'],dtype=np.int32)+1
  state['logits_np'][:n]=ends-1;state['seq_np'][:n]=scheduled[perm]
  plan=order.Plan(state,perm,np.argsort(perm),scheduled[perm],ends,m,1,n-2)
 return NS(runner=r,scheduler=output,controls=controls,plan=plan,case=case,
  stubs=['_update_states','_determine_batch_execution_and_padding','_get_slot_mappings','_build_attention_metadata','_preprocess','maybe_get_kv_connector_output','_prepare_input_ids','controlled no-op model/head'],
  Plan_constructed_fixture_not_make_plan_or_C3=(kind=='Plan'))

@contextlib.contextmanager
def frame_environment(module,x):
 # Isolated process, enumerated surrounding infrastructure only. Never patch helper,
 # _cold/_codes/_Receipt, execute_model, _prepare_inputs or _get_cumsum_and_arange.
 with contextlib.ExitStack() as stack:
  stack.enter_context(patch.object(module,'get_pp_group',lambda:NS(is_last_rank=True)))
  stack.enter_context(patch.object(module,'set_forward_context',lambda *a,**kw:contextlib.nullcontext()))
  if x.plan is not None:stack.enter_context(patch.object(module,'_biv150_compute_order_plan',lambda *a:x.plan))
  yield

def invoke(x,helper,enabled):
 os.environ[helper.ENV]='1' if enabled else '0';x.runner.execute_model_state=None
 x.runner.execute_model(x.scheduler)
 state=x.runner.execute_model_state;c.need(state is not None,'original frame did not reach common gather/return')
 return state.sample_hidden_states

@contextlib.contextmanager
def observe(helper,records):
 """Correctness only; profiling is fully disabled during cost measurement."""
 previous=sys.getprofile();c.need(previous is None,'unknown existing profiler')
 def profile(frame,event,value):
  if event=='return' and frame.f_code is helper.prepare.__code__:
   records.append(dict(kind='prepare',receipt=value,source=None if value is None else value.source))
  elif event=='return' and frame.f_code is helper._cold.__code__:records.append(dict(kind='_cold',state=value))
  elif event=='call' and frame.f_code is helper._launch.__code__:records.append(dict(kind='_launch',M=frame.f_locals['input_rows'],N=frame.f_locals['selected_rows']))
  elif event=='return' and frame.f_code is helper.try_gather.__code__:records.append(dict(kind='try_gather',output=value))
 sys.setprofile(profile)
 try:yield
 finally:sys.setprofile(previous)

def execute(a):
 c.need(a.root_owned_exclusive and a.expected_source_commit==c.git('rev-parse','HEAD') and not c.git('status','--porcelain','--untracked-files=all'),'Root explicit admission and exact clean source')
 c.git('merge-base','--is-ancestor',MAIN,'HEAD')
 c.need(not {'torch','triton','numpy'}.intersection(sys.modules),'pre-framework admission')
 c.need(os.environ.get(c.ENV,'0')=='0' and not any(os.environ.get(k) for k in ('TRITON_KERNEL_OVERRIDE','TRITON_OVERRIDE_DIR')),'production defaults OFF, no foreign override')
 with socket.socket() as s:c.need(s.connect_ex(('127.0.0.1',9031))!=0,'service9031 active')
 c.need(a.out.is_absolute() and not a.out.exists() and not a.out.resolve().is_relative_to(ROOT) and not a.out.resolve().is_relative_to(a.compile_out) and not a.out.resolve().is_relative_to(a.production_out),'fresh isolated out')
 c.need(c.sha(a.bits_review.read_bytes())=='9fe896614a0ace0320be69dc0e91d8a933bda6279c9629d6cb3bf055a93ee58f','accepted direct-bit review identity')
 br=json.loads(a.bits_review.read_text());c.need(br['passed'] is True and br['source_commit']=='22ba8b7cdc3be4e9619f4fa3f4e91c9cca6cfdfc' and br['boundaries']['direct_kernel_device_bitcopy_verified'] is True,'prior direct bits scope')
 c.need(all(p.is_absolute() for p in (a.compile_out,a.actual_review,a.bits_review,a.production_out,a.production_review)),'absolute readonly proof paths')
 s,reference=bits.proof(a.compile_out,a.actual_review)
 production_summary,actual=cache.proof(a.production_out,a.production_review)
 old=dict(runtime_ABI=actual['ABI'],actual_options=actual['options'],resources=actual['resources'])
 a.out.mkdir(parents=True);out=a.out;result=dict(source_commit=a.expected_source_commit,main_dependency=MAIN,completed=False,errors=[],records=[],timings=[],
  actual_helper_entry_verified=False,actual_helper_cold_verified=False,Plan_make_plan_or_C3_verified=False,Graph_numeric_verified=False,performance_verified=False,service_verified=False,timing_executed=False,production_warmups_attempted=0,production_warmups_completed=0,numeric_candidate_launches_profile_observed=0,helper_output_canary_allocation_not_injected=True)
 candidate_guard=None
 def save():c.save(out/'helper-summary.json',result)
 def blob(b):
  p=out/'arrays'/('sha256-'+c.sha(b)+'.bin.gz');p.parent.mkdir(exist_ok=True)
  if not p.exists():p.write_bytes(gzip.compress(b,compresslevel=1,mtime=0))
  return dict(path=str(p.relative_to(out)),bytes=len(b),sha256=c.sha(b))
 save()
 try:
  os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache');os.environ['VLLM_FL_BIV150_HEAD_TEMPERATURE']='0'
  # Copy the reviewed cache, never mutate the original compile directory.
  group=cache.prepare(a.production_out,out/'triton-cache',a.production_review);result['reviewed_production_cache']=group;result['binary_source_root']=group['original_source_root'];save()
  import torch,numpy as np,triton
  c.need(torch.__version__=='2.10.0' and not torch.cuda.is_initialized(),'actual fresh Torch2.10 context')
  c.need(not torch.are_deterministic_algorithms_enabled(),'unknown deterministic empty auto-fill policy; do not change global state')
  result['CUDA_initialized_before']=False
  result['SME_before_runner_import']=dict(torch_corex=c.typed(getattr(torch,'corex',None)),TRITON_DISABLE_SME=os.environ.get('TRITON_DISABLE_SME'))
  module=importlib.import_module(RUNNER);helper=importlib.import_module(HELPER);order=importlib.import_module('vllm_fl.patches.biv150_compute_order')
  importlib.import_module('vllm.v1.worker.gpu_input_batch');importlib.import_module('vllm.v1.utils')
  c.need(Path(module.__file__).resolve()==ROOT/'vllm_fl/worker/model_runner.py' and c.sha(Path(module.__file__).read_bytes())==order.EXPECTED_RUNNER,'actual runner source/pin')
  c.need(c.sha(Path(helper.__file__).read_bytes())==spec.sha(spec.HELPER.read_bytes()),'actual production helper bytes')
  result['SME_after_runner_import']=dict(torch_corex=c.typed(getattr(torch,'corex',None)),TRITON_DISABLE_SME=os.environ.get('TRITON_DISABLE_SME'))
  kernel=importlib.import_module('vllm_fl.patches.biv150_dynamic_hidden_gather_kernel');fn=kernel.gather_hidden_bits
  result['actual_JIT_function_cache_key']=str(fn.cache_key);c.need(str(fn.cache_key)==s['actual_JIT_function_cache_key'],'production function key differs: stop for actual review')
  result['actual_helper_sha256']=c.sha(Path(helper.__file__).read_bytes());result['actual_runner_sha256']=order.EXPECTED_RUNNER;save()
  def read(t):return t.detach().view(torch.int16).cpu().numpy().tobytes()
  timing_jobs=[]
  with torch.inference_mode():
   for m,n in bits.CASES:
    raw=bits.fixture(m,n,0)['hidden'];backing=torch.frombuffer(bytearray(raw),dtype=torch.int16).clone().to('cuda');hidden=backing[128:-128].view(torch.bfloat16).reshape(m,2048)
    for kind in ('ordinary','Plan'):
     x=fixture(module,helper,torch,np,m,n,kind,hidden);record=dict(M=m,N=n,kind=kind,stubs=x.stubs,Plan_constructed_fixture_not_make_plan_or_C3=x.Plan_constructed_fixture_not_make_plan_or_C3,phases=[]);result['records'].append(record);save()
     # Two actual production warmups only per M/N; no84-shape sweep. Export before gate.
     ii=torch.tensor(x.case['logits_indices'],dtype=torch.int32,device='cuda');oo=torch.empty((n,2048),dtype=torch.bfloat16,device='cuda')
     jit=importlib.import_module('triton.runtime.jit');c.need(c.sha(Path(jit.__file__).read_bytes())==c.JIT_SHA,'actual SME-selection JIT source')
     memory,_,_,backend,binder=fn.device_caches[torch.cuda.current_device()];bound,specialization,_=binder(hidden,ii,oo,m,n,**c.OPTIONS)
     record['natural_SME_prediction']=jit.get_corex_sme(bound.values(),specialization);save();c.need(record['natural_SME_prediction']==5,'natural production SME differs; never force5 or0')
     result['production_warmups_attempted']+=1;save()
     call=lambda:fn.warmup(hidden,ii,oo,m,n,grid=((n+7)//8,16),**c.OPTIONS)
     if candidate_guard is None:
      compiled=cache.warmup(fn,torch.cuda.current_device(),group,call);guard=cache.prohibit_candidate_compile(fn);guard.__enter__();candidate_guard=guard
     else:compiled=call()
     result['production_warmups_completed']+=1;compiled._init_handles()
     folder=out/('actual-'+str(m)+'-'+kind);folder.mkdir();files={}
     for ext in ('source','ttir','ttgir','llir','asm','cubin'):
      b=compiled.asm[ext];b=b.encode() if isinstance(b,str) else b;(folder/('actual.'+ext)).write_bytes(b);files[ext]=dict(bytes=len(b),sha256=c.sha(b))
     record['actual_compiled']=dict(python_object_id=id(compiled),hash=str(compiled.hash),options=c.metadata(compiled),ABI=c.runtime_ABI(compiled),resources=c.resources(compiled),files=files);save()
     c.need(str(compiled.hash)==cache.HASH and files['cubin']['sha256']==cache.CUBIN and c.runtime_ABI(compiled)==old['runtime_ABI'] and c.metadata(compiled)==old['actual_options'] and c.resources(compiled)==old['resources'],'actual production specialization differs; saved evidence, stop review before launch')
     record.update(bits.packed(compiled.packed_metadata));launch=getattr(compiled._run,'launch',None);owner=getattr(launch,'__self__',None)
     record['actual_handler']=dict(type=type(compiled._run).__module__+'.'+type(compiled._run).__qualname__,owner_type=type(owner).__name__)
     c.need(getattr(owner,'__file__',None),'actual handler extension missing')
     launcher=Path(owner.__file__).resolve();lb=launcher.read_bytes();(folder/'actual-launcher.so').write_bytes(lb);record['actual_handler'].update(path=str(launcher),sha256=c.sha(lb));save()
     c.need(c.sha(lb)==bits.LAUNCHER_SHA,'actual loaded launcher differs from5124 review')
     with frame_environment(module,x):
      for phase in (0,1):
       fresh=bits.fixture(m,n,phase);backing.copy_(torch.frombuffer(bytearray(fresh['hidden']),dtype=torch.int16))
       expected=b''.join(fresh['hidden'][(128+i*2048)*2:(128+(i+1)*2048)*2] for i in x.case['logits_indices'])
       events=[];x.controls.arm='candidate'
       with observe(helper,events):output=invoke(x,helper,True)
       torch.cuda.synchronize();receipts=[v['receipt'] for v in events if v['kind']=='prepare'];launches=[v for v in events if v['kind']=='_launch']
       c.need(len(receipts)==1 and type(receipts[0]) is helper._Receipt and receipts[0].consumed and len(launches)==1,'actual original frame/helper receipt/one launch not reached')
       c.need(read(output)==expected and read(backing)==fresh['hidden'],'full bits and hidden guards/readonly')
       receipt=receipts[0];c.need(receipt.source==('original_query_start_loc_minus_one' if kind=='ordinary' else 'original_compute_order_logits_copy'),'actual CPU index source')
       result['numeric_candidate_launches_profile_observed']+=len(launches)
       c.need(tuple(map(int,receipt.indices.cpu().numpy()))==tuple(x.case['logits_indices']),'entire GPU index vector equals original CPU producer after gather')
       c.need(receipt.runtime is getattr(x.runner,helper._STATE) and tuple(receipt.values)==tuple(x.case['logits_indices']),'real cold state and producer values')
       # Same actual GPU hidden and indices for direct index_select numeric control.
       control=torch.index_select(hidden,0,receipt.indices);torch.cuda.synchronize();c.need(read(control)==read(output)==expected,'torch.index_select exact same inputs')
       record['phases'].append(dict(phase='AB'[phase],inputs=blob(fresh['hidden']),actual=blob(read(output)),expected=blob(expected),control=blob(read(control)),indices=blob(receipt.indices.cpu().numpy().tobytes()),
        receipt_source=receipt.source,consumed=True,cold_return_observed=any(v['kind']=='_cold' for v in events),hidden_descriptor=c.descriptor(hidden),index_descriptor=c.descriptor(receipt.indices),output_descriptor=c.descriptor(output),bitwise_verified=True));save()
      # Plan's original check_dispatch must throw before gather when mode is FULL;
      # ordinary execution skips receipt and reaches unchanged fallback callsite.
      x.controls.arm='control';x.controls.graph_mode=module.CUDAGraphMode.FULL;events=[]
      with observe(helper,events):
       if kind=='Plan':
        try:invoke(x,helper,True)
        except RuntimeError as error:
         c.need('compute-order' in str(error),'unexpected Plan failure');record['Plan_FULL_dispatch_rejected_without_retry']=str(error)
        else:raise ValueError('Plan FULL dispatch improperly accepted')
       else:
        fallback=invoke(x,helper,True);torch.cuda.synchronize();c.need(read(fallback)==expected,'ordinary FULL fallback bits')
      c.need(not any(e['kind'] in ('prepare','_launch') for e in events),'FULL original seam must not reach receipt/candidate')
      record['Graph_mode_FULL_candidate_refusal_verified']=True;x.controls.graph_mode=module.CUDAGraphMode.NONE
      events=[]
      with observe(helper,events):fallback=invoke(x,helper,False)
      torch.cuda.synchronize();c.need(not any(e['kind']=='_launch' for e in events) and read(fallback)==expected,'OFF fallback no candidate')
      record['feature_OFF_fallback_verified']=True
      # Actual capture guard is checked before caller inspection. This external
      # early-refusal test does NOT claim a captured original execute_model.
      os.environ[helper.ENV]='1';graph=torch.cuda.CUDAGraph();scheduled=np.asarray(x.case['scheduled'],dtype=np.int32)
      with torch.cuda.graph(graph):rejected=helper.prepare(x.runner,None,scheduled,m,None,None)
      c.need(rejected is None,'actual capture must reject before candidate/caller inspection')
      record['actual_capture_early_refusal_external_call_only']=True
      c.need(helper.prepare(x.runner,None,scheduled,m,None,object()) is None,'spec metadata early refusal')
      record['spec_early_refusal_external_call_only']=True
      x.expected=expected;x.hidden_backing=backing;x.hidden_bytes=fresh['hidden'];timing_jobs.append(x);save()
   c.need(result['numeric_candidate_launches_profile_observed']==8 and all(r['phases'][0]['cold_return_observed'] for r in result['records']),'all four source/geometry cases numeric plus genuine helper cold first')
   # No profiling inside timed original frames. All numeric modes passed first.
   if a.timing:
    for x in timing_jobs:
     with frame_environment(module,x):
      blocks=[]
      for arm in ('control','candidate','candidate','control'):
       x.controls.arm=arm
       for _ in range(2):invoke(x,helper,arm=='candidate')
       torch.cuda.synchronize();start=torch.Event(enable_timing=True);end=torch.Event(enable_timing=True);start.record();t=time.perf_counter_ns()
       for _ in range(32):invoke(x,helper,arm=='candidate')
       end.record();end.synchronize();host=(time.perf_counter_ns()-t)/1e6
       actual=read(x.runner.execute_model_state.sample_hidden_states);c.need(actual==x.expected and read(x.hidden_backing)==x.hidden_bytes,'postblock full output/hidden guard and readonly')
       event_ms=start.elapsed_time(end)/32;c.need(event_ms>0 and host>0,'positive complete-frame cost')
       blocks.append(dict(arm=arm,iterations=32,event_ms_per_call=event_ms,host_ms_per_call=host/32,postblock_actual=blob(actual),postblock_bits_and_hidden_readonly=True))
      result['timings'].append(dict(M=x.case['M'],N=x.case['N'],kind=x.case['source'],order='ABBA',blocks=blocks,cost_scope='whole controlled original execute_model frame INCLUDING common original input preparation/H2D/event wait, receipt, new output allocation, host and launch; not isolated gather or service',
       control='production feature OFF; original fallback dispatches a prebuilt control-only proxy to torch.index_select; same real hidden and original index values, fresh indices per call as production',
       timings_not_profiled=True,candidate_calls_inferred_from_loop_not_profile_observed=68,control_calls_inferred_from_loop=68))
      result['timing_executed']=True;save()
  result.update(completed=True,actual_helper_entry_verified=True,actual_helper_cold_verified=all(r['phases'][0]['cold_return_observed'] for r in result['records']))
 except Exception as error:
  result['errors'].append(repr(error));(out/'failure.txt').write_text(traceback.format_exc())
 finally:
  if candidate_guard is not None:candidate_guard.__exit__(None,None,None)
  os.environ[c.ENV]='0'
  if 'torch' in sys.modules:result['CUDA_initialized_after']=sys.modules['torch'].cuda.is_initialized()
  result['artifacts']=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=c.sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='helper-summary.json'];save()
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-helper',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--expected-source-commit');p.add_argument('--compile-out',type=Path);p.add_argument('--actual-review',type=Path);p.add_argument('--bits-review',type=Path);p.add_argument('--production-out',type=Path);p.add_argument('--production-review',type=Path);p.add_argument('--out',type=Path);p.add_argument('--timing',action='store_true');a=p.parse_args()
 if not a.run_helper:
  if any((a.root_owned_exclusive,a.expected_source_commit,a.compile_out,a.actual_review,a.bits_review,a.production_out,a.production_review,a.out,a.timing)):p.error('device arguments require explicit --run-helper')
  result=spec.plan()
 else:
  if not all((a.root_owned_exclusive,a.expected_source_commit,a.compile_out,a.actual_review,a.bits_review,a.production_out,a.production_review,a.out)):p.error('Root/source/original-proof/fresh-out admission required')
  result=execute(a)
 print(json.dumps(result,sort_keys=True,indent=2));return int(a.run_helper and not result['completed'])
if __name__=='__main__':raise SystemExit(main())
