"""Isolated FP16 KV/dot comparison for BF16 query inputs and BF16 outputs.

KV conversion is excluded from read timing. No serving hooks are installed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import torch
from biv150_benchmark_utils import load_module,inputs,full_call,graph_time
from bench_biv150_int8_kv_decode import dense_reference


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    torch.backends.cuda.matmul.allow_tf32=False
    args.out.parent.mkdir(parents=True,exist_ok=True)
    original=args.source.read_text()
    text=original.replace('def _gqa_decode_attention(','def _gqa_decode_attention_fp16_cache(',1)
    needle='    Q = tl.load(query_ptr + query_offsets, mask=row_valid[:, None], other=0.0)'
    assert text.count(needle)==1
    text=text.replace(needle,needle+'\n    Q = Q.to(tl.float16)')
    generated=args.out.parent/'fp16-gqa-generated.py';generated.write_text(text)
    baseline=load_module(args.source,'biv150_fp16_cache_baseline')
    candidate=load_module(generated,'biv150_fp16_cache_candidate')
    candidate._gqa_decode_attention=candidate._gqa_decode_attention_fp16_cache
    records=[]
    for batch,kv,mixed in [(64,4096,False),(32,16384,False),(31,16896,False),(8,1024,True),(1,100,False)]:
        c=inputs(batch,kv,mixed=mixed)
        segments=baseline._pick_segments(batch,2,16)
        full_call(baseline,c,segments);ref=c['out'].clone()
        before=graph_time(lambda:full_call(baseline,c,segments),repeats=15,unroll=4)[0]
        cc=dict(c,k=c['k'].to(torch.float16),v=c['v'].to(torch.float16))
        row=dict(batch=batch,kv=kv,mixed=mixed,segments=segments,bf16_ms=before,offline_cache_conversion=True)
        try:
            full_call(candidate,cc,segments);out=cc['out'].clone()
            torch.testing.assert_close(out,ref,atol=.005,rtol=.005)
            row['candidate_ms']=graph_time(lambda:full_call(candidate,cc,segments),repeats=15,unroll=4)[0]
            row['bf16_after_ms']=graph_time(lambda:full_call(baseline,c,segments),repeats=15,unroll=4)[0]
            row['speedup']=before/row['candidate_ms']
            row['max_abs_diff_from_bf16']=(out.float()-ref.float()).abs().max().item()
            row['dense_checks']=[]
            for i in sorted({0,min(1,batch-1),batch-1}):
                gold=dense_reference(c,c['k'],c['v'],i).to(torch.bfloat16)
                torch.testing.assert_close(out[i],gold,atol=.005,rtol=.005)
                row['dense_checks'].append(dict(row=i,passed=True,max_abs_diff=(out[i].float()-gold.float()).abs().max().item()))
        except Exception as error:row['error']=str(error)[:1000]
        records.append(row);print(json.dumps(row),flush=True)
        args.out.write_text(json.dumps(dict(source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            generated_sha256=hashlib.sha256(text.encode()).hexdigest(),records=records,service_installed=False),indent=2))
        del c,cc,ref
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
