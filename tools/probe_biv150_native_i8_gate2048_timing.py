#!/usr/bin/env python3
"""Fixed PR128 complete gate cost: warm-weight Graph component and eager path."""
import argparse
import json
from pathlib import Path
import sys
import biv150_native_i8_gate2048_numeric_contract as c

NUMERIC_SOURCE='5de17dea9195f6643f651b1d1c6196af7f4e3503'
NUMERIC_MANIFEST='e4c47a4f18872c2cacad7a4665ea93aa45502abab7a93122569af7ea9260503d'
NUMERIC_SUMMARY='eb90ad81af65aeb086ec1beb7015bdca30a80566a9ab7c9d1d599a0b95522b7f'
NUMERIC_REVIEW='57bc147def00db9c3d7b1c0421835f9e9d23e3ae2b91129b172cf71040775958'
TIMING_ADMISSION='0bff2b0abe2a43b45832cfce74724c4b2b6052da32e49d6e245ff5b57703d4c7'
ROLES=('reference','candidate');ORDER=('reference','candidate','candidate','reference')


def plan():
 return dict(c.plan(),numeric_source=NUMERIC_SOURCE,numeric_manifest_sha256=NUMERIC_MANIFEST,
  numeric_summary_sha256=NUMERIC_SUMMARY,numeric_review_sha256=NUMERIC_REVIEW,
  timing_admission_sha256=TIMING_ADMISSION,modes=['graph','eager'],order=list(ORDER),
  unroll=4,warm_batches_per_role_and_mode=128,timed_batches_per_arm=16,
  steps_per_timed_arm=64,activation_quantization_included=True,output_allocation_in_step=True,
  reference_validation_copy_included=False,weights_and_pointers_fixed=True,
  Graph_wall_is_not_eager_host_cost=True,timings=False,performance_measured=False,
  scheduling_requires_Root=True,Root_timing_scope_not_service_qualification=True)


def preflight(a):
 binding=c.preflight(a)
 for path,h in ((a.numeric_raw/'download-manifest.json',NUMERIC_MANIFEST),(a.numeric_raw/'out/numeric-summary.json',NUMERIC_SUMMARY),(a.numeric_review,NUMERIC_REVIEW),(a.timing_admission,TIMING_ADMISSION)):
  c.need(c.sha(path.read_bytes())==h,'actual numeric/timing evidence '+path.name)
 m=json.loads((a.numeric_raw/'download-manifest.json').read_text())
 c.need(m['raw_files']==len(m['files'])==19 and m['raw_bytes']==sum(x['bytes'] for x in m['files'])==188976810,'complete numeric archive')
 for row in m['files']:
  p=a.numeric_raw/row['relative'];c.need(p.resolve().is_relative_to(a.numeric_raw.resolve()),'safe prior numeric file');b=p.read_bytes();c.need((len(b),c.sha(b))==(row['bytes'],row['sha256']),'full original numeric evidence')
 s=json.loads((a.numeric_raw/'out/numeric-summary.json').read_text());r=json.loads(a.numeric_review.read_text());approval=json.loads(a.timing_admission.read_text())
 c.need(s['binding']['frontend_source']==r['source_commit']==NUMERIC_SOURCE and r['actual']['summary_sha256']==NUMERIC_SUMMARY and r['actual']['manifest_sha256']==NUMERIC_MANIFEST,'numeric source/summary/manifest')
 c.need(r['errors']==[] and all(r[k] is True for k in ('review_completed','numeric_verified','all_file_hashes_checked','all_full_outputs_bitwise','Graph_B_before_fresh_B_verified','side_C_verified')),'Root-replayed full original gates')
 c.need(r['candidate_library_sha256']==c.LIBRARY and r['reference_library_sha256']==c.REFERENCE,'same actual numerical pair')
 c.need(approval['approved_component_timing_only'] is True and approval['actual_numeric_review_sha256']==NUMERIC_REVIEW and approval['actual_numeric_summary_sha256']==NUMERIC_SUMMARY and approval['source_commit']==NUMERIC_SOURCE,'Root exact timing scope')
 c.need(all(approval[k] is False and r[k] is False for k in ('performance_verified','model_precision_verified','service_verified')),'no borrowed service/performance qualification')
 return binding,s


def reference_step(torch,x,wq,cols,marker):
 # No validation copy: this is the actual main M2048 gate path, including AQ/SA.
 return torch.ops.biv150_native.gate_silu_cached_probe(x,wq,cols,marker,256)


def candidate_step(torch,x,wq,cols):
 # Both activation quantization and output allocation are inside every step.
 aq,rows=torch.ops.biv150_native.quantize_rows_probe(x,256)
 out=torch.empty((2048,6144),device=x.device,dtype=torch.bfloat16)
 torch.ops.cjj_native_i8_gate2048_probe.gate_out(aq,wq,rows,cols,out)
 return out


def measure(torch,time,stream,mode,role,step,graphs):
 begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 start=time.perf_counter()
 with torch.cuda.stream(stream):
  begin.record()
  for _ in range(16):
   if mode=='graph':graphs[role].replay()
   else:
    for _ in range(4):step(role)
  end.record()
 end.synchronize();wall=time.perf_counter()-start
 return dict(role=role,event_ms_per_step=begin.elapsed_time(end)/64,wall_ms_per_step=wall*1000/64)


def execute(a):
 binding,prior=preflight(a)
 import time
 import torch
 c.need(torch.__version__.split('+')[0]=='2.10.0' and not torch.cuda.is_initialized(),'fresh accepted Torch ABI/process')
 c.need(not hasattr(torch.ops.biv150_native,'gate_silu_cached_probe') and not hasattr(torch.ops.cjj_native_i8_gate2048_probe,'gate_out'),'no preloaded operators')
 a.out.mkdir(parents=True);(a.out/'arrays').mkdir()
 result=dict(plan(),binding=binding,frontend_source=a.expected_source_commit,completed=False,errors=[],checks=[],timing_records=[],performance_measured=False,default_CPU_only=False,GPU_executed=False)
 def save():(a.out/'timing-summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 def old_bytes(meta):return (a.numeric_raw/'out'/meta['path']).read_bytes()
 def blob(t):
  host=t.detach().contiguous().cpu();b=host.reshape(-1).view(torch.uint8).numpy().tobytes();h=c.sha(b);p=a.out/'arrays'/(h+'.bin')
  if not p.exists():p.write_bytes(b)
  return dict(path=str(p.relative_to(a.out)),sha256=h,bytes=len(b),shape=list(t.shape),dtype=str(t.dtype))
 def upload(meta):
  dtype={'torch.bfloat16':torch.bfloat16,'torch.int8':torch.int8,'torch.float32':torch.float32}[meta['dtype']]
  return torch.frombuffer(bytearray(old_bytes(meta)),dtype=dtype).clone().reshape(meta['shape']).to('cuda')
 save()
 try:
  torch.ops.load_library(str(a.reference_library));torch.ops.load_library(str(a.library))
  result['loaded_libraries']={str(Path(p).resolve()):c.sha(Path(p).read_bytes()) for p in torch.ops.loaded_libraries if Path(p).resolve() in {a.library.resolve(),a.reference_library.resolve()}}
  c.need(set(result['loaded_libraries'].values())=={c.LIBRARY,c.REFERENCE},'actual two loaded libraries')
  with torch.inference_mode():
   inputs=prior['input_epochs'][0]['arrays'];x=upload(inputs['x']);result['GPU_executed']=True
   wq=upload(inputs['wq']);cols=upload(inputs['column_scales']);marker_base=upload(inputs['marker_base']);marker=marker_base.expand(12288,2048)
   tensors=dict(x=x,wq=wq,cols=cols,marker_base=marker_base)
   def fingerprints():return {k:blob(t) for k,t in tensors.items()}
   def descriptors():return {k:dict(pointer=t.data_ptr(),storage=t.untyped_storage().data_ptr(),shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset()) for k,t in tensors.items()}
   fixed_descriptors=descriptors();expected=fingerprints();result['input_A']=expected;result['fixed_descriptors']=fixed_descriptors
   for key,original in (('x','x'),('wq','wq'),('cols','column_scales'),('marker_base','marker_base')):c.need(expected[key]['sha256']==inputs[original]['sha256'],'exact original fixed input '+key)
   last={};graphs={};graph_outputs={}
   def step(role):
    value=reference_step(torch,x,wq,cols,marker) if role=='reference' else candidate_step(torch,x,wq,cols)
    last[role]=value
    return value
   def checked(value,role,label,epoch):
    torch.cuda.synchronize();item=blob(value);data=(a.out/item['path']).read_bytes()
    old=next(r for r in prior['records'] if r['stage']==('eagerA' if epoch==0 else 'eagerB'))
    gold=old_bytes(old['output_backings'][role])[128:-128]
    current=fingerprints();equal=data==gold
    result['checks'].append(dict(label=label,role=role,epoch=epoch,full_output=item,bitwise_equal_original_numeric=equal,readonly=current==expected,descriptors_stable=descriptors()==fixed_descriptors));save()
    c.need(value.dtype==torch.bfloat16 and tuple(value.shape)==(2048,6144) and equal,'complete output equals original zero-tolerance golden')
    c.need(current==expected and descriptors()==fixed_descriptors,'all inputs/weights and descriptors unchanged')
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());result['stream']=stream.cuda_stream
   for role in ROLES:
    with torch.cuda.stream(stream):
     for _ in range(3):step(role)
    stream.synchronize();checked(last[role],role,'eager-A-before-capture',0)
    with torch.cuda.graph(torch_graph:=torch.cuda.CUDAGraph(),stream=stream):
     for _ in range(4):step(role)
    graphs[role]=torch_graph;graph_outputs[role]=last[role]
   with torch.cuda.stream(stream):
    for role in ROLES:graphs[role].replay()
   stream.synchronize()
   for role in ROLES:checked(graph_outputs[role],role,'Graph-A',0)
   # Original B bytes, same x pointer. The newly composed quant+gate graphs see
   # changed x before any fresh eager B; all actual AQ/SA work remains in graph.
   b=prior['input_epochs'][1]['arrays']['x'];x.copy_(upload(b));torch.cuda.synchronize();expected=fingerprints();result['input_B']=expected
   c.need(expected['x']['sha256']==b['sha256'] and expected['x']['sha256']!=result['input_A']['x']['sha256'],'actual changed B activation')
   for key in ('wq','cols','marker_base'):c.need(expected[key]==result['input_A'][key],'fixed warm weight/scale/marker')
   with torch.cuda.stream(stream):
    for role in ROLES:graphs[role].replay()
   stream.synchronize()
   for role in ROLES:checked(graph_outputs[role],role,'composed-Graph-B-before-fresh',1)
   with torch.cuda.stream(stream):
    for role in ROLES:step(role)
   stream.synchronize()
   for role in ROLES:checked(last[role],role,'fresh-eager-B-after-Graph',1)
   result['composed_quant_gate_Graph_verified']=True;save()
   for mode in ('graph','eager'):
    with torch.cuda.stream(stream):
     for role in ROLES:
      for _ in range(128):
       if mode=='graph':graphs[role].replay()
       else:
        for _ in range(4):step(role)
    stream.synchronize()
    for index,role in enumerate(ORDER):
     row=measure(torch,time,stream,mode,role,step,graphs);row.update(mode=mode,index=index,warm_steps_per_role=512,timed_steps=64)
     result['timing_records'].append(row);save()
     checked(graph_outputs[role] if mode=='graph' else last[role],role,'timed-'+mode+'-'+str(index),1)
    rows=[r for r in result['timing_records'] if r['mode']==mode]
    means={metric:{role:sum(r[metric] for r in rows if r['role']==role)/2 for role in ROLES} for metric in ('event_ms_per_step','wall_ms_per_step')}
    result.setdefault('means',{})[mode]=means
    result.setdefault('reference_over_candidate',{})[mode]={metric:values['reference']/values['candidate'] for metric,values in means.items()};save()
   result.update(completed=True,performance_measured=True,timings=True)
 except Exception as error:result['errors'].append(type(error).__name__+': '+str(error))
 finally:save()
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-timing',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--expected-source-commit')
 names=('compile-summary','library','reference-library','static-review','root-admission','numeric-raw','numeric-review','timing-admission','out')
 for name in names:p.add_argument('--'+name,type=Path)
 a=p.parse_args()
 if not a.run_timing and not a.preflight_only:
  if any(v for k,v in vars(a).items() if k not in ('run_timing','preflight_only')):p.error('default plan takes no device arguments')
  result=plan()
 else:
  if a.run_timing and a.preflight_only:p.error('choose preflight or timing')
  if not all(getattr(a,k.replace('-','_')) for k in ('root-owned-exclusive','expected-source-commit',*names)):p.error('all fixed evidence and Root exclusive fields required')
  result=dict(binding=preflight(a)[0],plan=plan()) if a.preflight_only else execute(a)
 print(json.dumps(result,sort_keys=True,indent=2));return int(a.run_timing and not result['completed'])
if __name__=='__main__':raise SystemExit(main())
