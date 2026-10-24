"""C03: change partial-output strides, reuse shipping producer/reducer unchanged.

Same scratch capacity and scalar max/sum layout; no service hook or extra copy.
Measure complete producer+reducer, not only a favorable half of the pipeline.
"""

import argparse
import hashlib
import json
import socket
import statistics
import subprocess
import traceback
from functools import partial
from pathlib import Path

import torch

from biv150_benchmark_utils import full_call,graph_time,inputs,load_module

PIN='2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1'
BATCHES=(1,8,16,29,30,31,32,40,60,61,62,63,64)


def reference(case,row):
    length=int(case['lengths'][row].cpu())
    if length==0:return torch.zeros(16,128,dtype=torch.bfloat16)
    pages=case['table'][row,:(length+15)//16].long()
    k=case['k'][pages].reshape(-1,2,128)[:length].cpu().double()
    v=case['v'][pages].reshape(-1,2,128)[:length].cpu().double()
    q=case['q'][row].reshape(2,8,128).cpu().double()
    scores=torch.einsum('hgd,thd->hgt',q,k)*(128**-.5)
    return torch.einsum('hgt,thd->hgd',scores.softmax(-1),v).reshape(16,128).bfloat16()


def candidate_case(original):
    candidate=dict(original)
    b=original['q'].shape[0]
    # Expose exactly the same [B,head,segment,dim] logical shape; physical
    # partial storage is [B,segment,head,dim]. No scalar layout change because
    # the shipping producer deliberately assumes unit scalar segment stride.
    candidate['scratch']=torch.empty((b,16,16,128),device='cuda',dtype=torch.float32).transpose(1,2)
    candidate['maxima']=torch.empty_like(original['maxima'])
    candidate['sums']=torch.empty_like(original['sums'])
    candidate['out']=torch.empty_like(original['out'])
    return candidate


def poison(case):
    for key in ('scratch','maxima','sums'):case[key].fill_(float('nan'))


def main(args):
    with socket.socket() as port:assert port.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    root=Path(__file__).resolve().parents[1];source=root/'vllm_fl/patches/biv150_gqa_decode.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==PIN
    module=load_module(source,'partial_layout_shipping')
    assert module.TARGET_PROGRAMS==256
    torch.set_num_threads(2);torch.manual_seed(20261025)
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(component_only=True,service_installed=False,same_producer_reducer_source_sha256=PIN,
        same_scratch_capacity=True,scalar_layout_unchanged=True,extra_transpose_copy=False,
        inference_mode=torch.is_inference_mode_enabled(),pure_shipping_triton_producer_reducer=True,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),records=[])
    save=lambda:(args.out/'result.json').write_text(json.dumps(result,indent=2))
    for batch in BATCHES:
        for length in (4096,5119,16384,17407):
            for ragged in (False,True):
                case=inputs(batch,length,mixed=ragged)
                packed=torch.empty((case['k'].shape[0],2,16,2,128),device='cuda',dtype=torch.bfloat16)
                k,v=packed.unbind(1);k.copy_(case['k']);v.copy_(case['v']);case.update(k=k,v=v)
                other=candidate_case(case);segments=module._pick_segments(batch,2,16)
                row=dict(batch=batch,kv=length,ragged=ragged,segments=segments,
                    native_strides=list(case['scratch'].stride()),candidate_strides=list(other['scratch'].stride()),
                    scratch_bytes=case['scratch'].numel()*4,kv_strides=list(k.stride()))
                try:
                    assert case['scratch'].numel()==other['scratch'].numel()
                    poison(case);poison(other)
                    full_call(module,case,segments);full_call(module,other,segments)
                    torch.testing.assert_close(other['out'],case['out'],atol=0,rtol=0)
                    for index in sorted({0,min(1,batch-1),batch-1}):
                        gold=reference(case,index)
                        torch.testing.assert_close(other['out'][index].cpu(),gold,atol=.005,rtol=.005)
                    original_q=case['q'].clone();original_lengths=case['lengths'].clone()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):full_call(module,other,segments)
                    for factor in (.5,1.25):
                        case['q'].copy_(original_q*factor);case['lengths'].copy_(original_lengths//2)
                        case['lengths'][-1]=0
                        poison(case);poison(other);full_call(module,case,segments)
                        graph.replay();torch.cuda.synchronize()
                        torch.testing.assert_close(other['out'],case['out'],atol=0,rtol=0)
                        assert torch.equal(other['out'][-1],torch.zeros_like(other['out'][-1]))
                    case['q'].copy_(original_q);case['lengths'].copy_(original_lengths)
                    del graph,original_q,original_lengths
                    rounds=[]
                    for iteration in range(3):
                        calls=[('baseline_ms',case),('candidate_ms',other)]
                        if iteration%2:calls.reverse()
                        times={name:graph_time(partial(full_call,module,c,segments),repeats=9,unroll=4)[0] for name,c in calls}
                        times['speedup']=times['baseline_ms']/times['candidate_ms'];rounds.append(times)
                    row.update(passed=True,bit_exact_layout_change=True,cpu_fp64_sampled_rows=True,changed_input_graph=True,
                        rounds=rounds,median_speedup=statistics.median(t['speedup'] for t in rounds),
                        min_round_speedup=min(t['speedup'] for t in rounds))
                except Exception as error:
                    row.update(passed=False,error=repr(error),traceback=traceback.format_exc())
                result['records'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ('rounds','traceback')}),flush=True)
                del case,other,packed,k,v
    result['summary']=dict(records=len(result['records']),expected=104,
        passed=sum(r['passed'] for r in result['records']),
        promising=sum(r['passed'] and r.get('median_speedup',0)>=1.03 and r.get('min_round_speedup',0)>=1 for r in result['records']))
    result['diagnostic_completed']=len(result['records'])==104;save()
    if not all(r['passed'] for r in result['records']):raise RuntimeError('Layout correctness failed; no promotion')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    with torch.inference_mode():main(p.parse_args())
