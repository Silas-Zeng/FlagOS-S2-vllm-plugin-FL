"""Test conditional/lazy online-softmax reference updates without changing math.

A reference need not equal the running maximum. Margin bounds exponent growth;
segment recombination still accounts for the actual reference used in each part.
No serving registration is installed by this probe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import torch
from biv150_benchmark_utils import load_module,inputs,full_call,graph_time,main_call
from bench_biv150_int8_kv_decode import dense_reference


def source_for(text,margin):
    before='''        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(V.dtype), V
        )'''
    after='''        if tl.max((new_max > running_max).to(tl.int32), 0) != 0:
            rescale = tl.math.exp2(running_max - new_max)
            running_sum *= rescale
            accumulator *= rescale[:, None]
        running_sum += tl.sum(probability, 1)
        accumulator = tl.dot(probability.to(V.dtype), V, accumulator)'''
    assert text.count(before)==2
    text=text.replace(before,after)
    old='        new_max = tl.maximum(running_max, tl.max(scores, 1))'
    assert text.count(old)==2
    if margin:
        new=f'''        tile_max = tl.max(scores, 1)
        new_max = tl.where(tile_max > running_max + {float(margin)}, tile_max, running_max)'''
        text=text.replace(old,new)
    return text.replace('def _gqa_decode_attention(',f'def _gqa_decode_lazy_rescale_{margin}(',1)


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    torch.backends.cuda.matmul.allow_tf32=False
    args.out.parent.mkdir(parents=True,exist_ok=True)
    original=args.source.read_text();base=load_module(args.source,'lazy_rescale_baseline')
    variants={}
    for margin in [0,2,4,8]:
        p=args.out.parent/f'lazy-rescale-{margin}.py';p.write_text(source_for(original,margin))
        mod=load_module(p,'lazy_rescale_'+str(margin));mod._gqa_decode_attention=getattr(mod,f'_gqa_decode_lazy_rescale_{margin}')
        variants[margin]=mod
    records=[]
    for b,kv,mixed,amplitude in [(64,4096,False,1),(32,16384,False,1),(31,16896,False,1),
                               (8,1024,True,1),(1,100,False,1),(8,4096,False,4)]:
        c=inputs(b,kv,mixed=mixed)
        if amplitude!=1:c['q'].mul_(amplitude)
        segments=base._pick_segments(b,2,16)
        full_call(base,c,segments);reference=c['out'].clone()
        before=graph_time(lambda:full_call(base,c,segments),repeats=15,unroll=4)[0]
        for margin,module in variants.items():
            row=dict(batch=b,kv=kv,mixed=mixed,query_amplitude=amplitude,margin_log2=margin,segments=segments,baseline_ms=before)
            try:
                kernel=main_call(module,c,segments)
                full_call(module,c,segments);result=c['out'].clone()
                torch.testing.assert_close(result,reference,atol=.005,rtol=.005)
                row['candidate_ms']=graph_time(lambda:full_call(module,c,segments),repeats=15,unroll=4)[0]
                row['speedup']=before/row['candidate_ms']
                row.update(n_regs=kernel.n_regs,n_spills=kernel.n_spills,shared=kernel.metadata.shared,
                    bit_exact=torch.equal(result,reference),max_abs_diff=(result.float()-reference.float()).abs().max().item())
                row['dense_checks']=[]
                for i in sorted({0,min(1,b-1),b-1}):
                    gold=dense_reference(c,c['k'],c['v'],i).to(torch.bfloat16)
                    torch.testing.assert_close(result[i],gold,atol=.005,rtol=.005)
                    row['dense_checks'].append(dict(row=i,passed=True,max_abs_diff=(result[i].float()-gold.float()).abs().max().item()))
            except Exception as error:row['error']=str(error)[:1200]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),records=records,service_installed=False),indent=2))
        after=graph_time(lambda:full_call(base,c,segments),repeats=15,unroll=4)[0]
        print(json.dumps(dict(batch=b,kv=kv,baseline_before_ms=before,baseline_after_ms=after)),flush=True)
        del c,reference
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
