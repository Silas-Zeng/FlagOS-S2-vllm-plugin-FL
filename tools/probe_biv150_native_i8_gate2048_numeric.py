#!/usr/bin/env python3
"""One fixed full-output gate component; real original reference, no timing."""
import argparse
import json
from pathlib import Path
import sys
import biv150_native_i8_gate2048_numeric_contract as c


def fixture(torch,epoch):
 # Fixed CPU integer formulas, then BF16. Only activation changes across epochs;
 # cached-weight packing cannot be invalidated or hidden by this experiment.
 r=torch.arange(2048,dtype=torch.int64)[:,None]
 k=torch.arange(2048,dtype=torch.int64)[None,:]
 x=(((17*r+13*k+(r*k)%37+23*epoch)%257)-128).float()/64
 x*= (1.0,0.75,1.5)[epoch]
 x[0]=0; x[1]*=2**-18; x[2]*=(4 if epoch==2 else 0.5)
 return x.to(torch.bfloat16)


def fixed_weight(torch):
 n=torch.arange(12288,dtype=torch.int64)[:,None]
 k=torch.arange(2048,dtype=torch.int64)[None,:]
 return ((((7*n+11*k+(n*k)%31)%255)-127).float()/8192).to(torch.bfloat16)


def metrics(torch,left,right):
 # Original gate_auto comparisons require atol=rtol=0. Also report actual error
 # if exact comparison fails, without silently turning it into a tolerance pass.
 d=left.double()-right.double()
 finite=bool(torch.isfinite(left).all() and torch.isfinite(right).all())
 return dict(full_elements=left.numel(),all_finite=finite,
  bitwise_equal=bool(torch.equal(left.view(torch.int16),right.view(torch.int16))),
  unequal_bits=int((left.view(torch.int16)!=right.view(torch.int16)).sum()),
  max_abs=float(d.abs().max()) if finite else None,relative_RMSE=float(d.square().mean().sqrt()/right.double().square().mean().sqrt().clamp_min(1e-30)) if finite else None)


def execute(a):
 binding=c.preflight(a)  # All receipts/source/library gates precede Torch import.
 import torch
 c.need(torch.__version__.split('+')[0]=='2.10.0','actual accepted Torch ABI')
 c.need(not torch.cuda.is_initialized(),'fresh component process')
 c.need(not hasattr(torch.ops.biv150_native,'gate_silu_cached_probe') and
        not hasattr(torch.ops.cjj_native_i8_gate2048_probe,'gate_out'),'no prior operator registration')
 result=dict(c.plan(),binding=binding,completed=False,errors=[],records=[],input_epochs=[],
  stage_order=[],candidate_python_calls=0,reference_python_calls=0,graph_replays=0,
  quantize_probe_calls=0,full_output_bitwise_verified=False,Graph_verified=False,
  side_stream_verified=False,frozen_candidate_and_reference_sources=True)
 result.update(default_CPU_only=False,GPU_executed=False)
 a.out.mkdir(parents=True);(a.out/'arrays').mkdir()
 def save():(a.out/'numeric-summary.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
 def blob(t):
  host=t.detach().contiguous().cpu();raw=host.reshape(-1).view(torch.uint8).numpy().tobytes();h=c.sha(raw);p=a.out/'arrays'/(h+'.bin')
  if not p.exists():p.write_bytes(raw)
  return dict(path=str(p.relative_to(a.out)),sha256=h,bytes=len(raw),dtype=str(host.dtype),shape=list(host.shape))
 def desc(t):return dict(pointer=t.data_ptr(),storage_pointer=t.untyped_storage().data_ptr(),storage_bytes=t.untyped_storage().nbytes(),offset=t.storage_offset(),shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype))
 save()
 try:
  torch.ops.load_library(str(a.reference_library));torch.ops.load_library(str(a.library))
  result['loaded_libraries']={str(Path(p).resolve()):c.sha(Path(p).read_bytes()) for p in torch.ops.loaded_libraries if Path(p).resolve() in {a.library.resolve(),a.reference_library.resolve()}}
  c.need(set(result['loaded_libraries'].values())=={c.REFERENCE,c.LIBRARY},'actual two loaded libraries')
  quant_ns={'torch':torch};exec(compile(c.source_proof(),str(c.ROOT/'vllm_fl/patches/biv150_native_w8a8.py')+':quantize_weight','exec'),quant_ns)
  with torch.inference_mode():
   weight=fixed_weight(torch).to('cuda');result['GPU_executed']=True;save()
   wq,cols=quant_ns['quantize_weight'](weight)
   marker_base=torch.zeros((),device='cuda',dtype=torch.bfloat16);marker=marker_base.expand(12288,2048)
   x=fixture(torch,0).to('cuda');aq,rows=torch.ops.biv150_native.quantize_rows_probe(x,256);result['quantize_probe_calls']+=1
   q0,s0=torch.ops.biv150_native.quantize_rows_probe(x,0);result['quantize_probe_calls']+=1
   c.need(torch.equal(aq,q0) and torch.equal(rows,s0),'original and cached quantizer exact for actual initial input')
   result['original_quantizer_crosscheck']=dict(cached=blob(aq),original=blob(q0),cached_scales=blob(rows),original_scales=blob(s0),bitwise=True)
   inputs=dict(x=x,aq=aq,row_scales=rows,wq=wq,column_scales=cols,weight_BF16=weight,marker_base=marker_base)
   initial_desc={k:desc(t) for k,t in inputs.items()}
   storage={role:torch.full((2048*6144+2*c.GUARD,),0x7fc1,dtype=torch.int16,device='cuda') for role in ('reference','candidate')}
   output={role:buf[c.GUARD:-c.GUARD].view(torch.bfloat16).reshape(2048,6144) for role,buf in storage.items()}
   out_desc={role:desc(t) for role,t in output.items()}
   def poison():
    for buf in storage.values():buf.fill_(0x7fc1)
   def candidate():
    result['candidate_python_calls']+=1
    torch.ops.cjj_native_i8_gate2048_probe.gate_out(aq,wq,rows,cols,output['candidate'])
   def reference():
    result['reference_python_calls']+=1
    # This is the original native _int_mm plus native silu_epilogue, including
    # its two BF16 projection rounds; no Python substitute activation formula.
    output['reference'].copy_(torch.ops.biv150_native.gate_silu_cached_probe(x,wq,cols,marker,256))
   def upload_epoch(epoch):
    x.copy_(fixture(torch,epoch));q,s=torch.ops.biv150_native.quantize_rows_probe(x,256);result['quantize_probe_calls']+=1
    aq.copy_(q);rows.copy_(s);torch.cuda.synchronize()
    current={k:desc(t) for k,t in inputs.items()};c.need(current==initial_desc,'all input descriptors fixed across epochs')
    record={k:blob(t) for k,t in inputs.items()};result['input_epochs'].append(dict(epoch=epoch,arrays=record,descriptors=current));save();return record
   expected={k:blob(t) for k,t in inputs.items()};result['input_epochs'].append(dict(epoch=0,arrays=expected,descriptors=initial_desc));save()
   originals={k:expected[k] for k in ('wq','column_scales','weight_BF16','marker_base')}
   snapshots={}
   def checkpoint(stage):
    torch.cuda.synchronize()
    current={k:blob(t) for k,t in inputs.items()};all_out={role:blob(buf) for role,buf in storage.items()}
    record=dict(stage=stage,epoch=0 if stage.endswith('A') else 1 if stage.endswith('B') else 2,
      readonly=current==expected,fixed_weight_unchanged=all(current[k]==v for k,v in originals.items()),
      input_arrays=current,output_backings=all_out,output_descriptors={role:desc(t) for role,t in output.items()},
      guards={role:bool(torch.all(buf[:c.GUARD]==0x7fc1) and torch.all(buf[-c.GUARD:]==0x7fc1)) for role,buf in storage.items()},
      comparison=metrics(torch,output['candidate'].cpu(),output['reference'].cpu()))
    snapshots[stage]={role:output[role].detach().clone() for role in output}
    result['records'].append(record);result['stage_order'].append(stage);save()
    c.need(record['readonly'] and record['fixed_weight_unchanged'] and all(record['guards'].values()),'readonly input/full backing guards')
    c.need(record['output_descriptors']==out_desc,'stable provided outputs')
    c.need(record['comparison']['all_finite'] and record['comparison']['bitwise_equal'],'complete output bitwise mismatch; original zero tolerance retained')
    torch.testing.assert_close(output['candidate'],output['reference'],atol=0,rtol=0)
   # Warm both actual operators on A before capture; no new JIT or recompilation.
   reference();candidate();torch.cuda.synchronize();poison();reference();candidate();checkpoint('eagerA')
   graphs={role:torch.cuda.CUDAGraph() for role in output}
   for role,call in (('reference',reference),('candidate',candidate)):
    with torch.cuda.graph(graphs[role]):call()
   poison()
   for graph in graphs.values():graph.replay();result['graph_replays']+=1
   checkpoint('graphA')
   for role in output:c.need(torch.equal(snapshots['eagerA'][role].view(torch.int16),snapshots['graphA'][role].view(torch.int16)),'A replay agrees with eager')
   expected=upload_epoch(1)
   c.need(all(result['input_epochs'][0]['arrays'][k]['sha256']!=expected[k]['sha256'] for k in ('x','aq','row_scales')),'B changes all activation operands')
   before=(result['candidate_python_calls'],result['reference_python_calls']);poison()
   for graph in graphs.values():graph.replay();result['graph_replays']+=1
   c.need(before==(result['candidate_python_calls'],result['reference_python_calls']),'Graph B precedes any fresh B gate')
   checkpoint('graphB');poison();reference();candidate();checkpoint('eagerB')
   for role in output:c.need(torch.equal(snapshots['graphB'][role].view(torch.int16),snapshots['eagerB'][role].view(torch.int16)),'changed B graph agrees with later fresh B')
   result['Graph_verified']=True;save()
   expected=upload_epoch(2);c.need(all(result['input_epochs'][1]['arrays'][k]['sha256']!=expected[k]['sha256'] for k in ('x','aq','row_scales')),'C changes activation operands')
   poison();side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());done=torch.cuda.Event()
   result['streams']=dict(default=torch.cuda.current_stream().cuda_stream,side=side.cuda_stream)
   c.need(result['streams']['default']!=result['streams']['side'],'actual distinct side stream')
   with torch.cuda.stream(side):reference();candidate();done.record()
   torch.cuda.current_stream().wait_event(done);checkpoint('sideC')
   result['side_stream_verified']=True
   c.need(tuple(result['stage_order'])==c.STAGES,'all five real stages')
   result.update(completed=True,numeric_verified=True,full_output_bitwise_verified=True)
 except Exception as error:
  result['errors'].append(type(error).__name__+': '+str(error))
 finally:save()
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-numeric',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--expected-source-commit')
 for name in ('compile-summary','library','reference-library','static-review','root-admission','out'):p.add_argument('--'+name,type=Path)
 a=p.parse_args()
 if not a.run_numeric and not a.preflight_only:
  if any(v for k,v in vars(a).items() if k not in ('run_numeric','preflight_only')):p.error('default plan accepts no device arguments')
  result=c.plan()
 else:
  if a.run_numeric and a.preflight_only:p.error('choose preflight or device numeric')
  if not all(getattr(a,k) for k in ('root_owned_exclusive','expected_source_commit','compile_summary','library','reference_library','static_review','root_admission','out')):p.error('all frozen source/compile/library/review/Root fields required')
  result=c.preflight(a) if a.preflight_only else execute(a)
 print(json.dumps(result,sort_keys=True,indent=2));return int(a.run_numeric and not result['completed'])
if __name__=='__main__':raise SystemExit(main())
