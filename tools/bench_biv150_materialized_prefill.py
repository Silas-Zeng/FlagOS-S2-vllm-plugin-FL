"""Compare the actual paged prefill baseline with materialized FP32-score blocks."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import torch
from bench_biv150_prefill_tiles import measure


def main(args):
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    baseline=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    spec=importlib.util.spec_from_file_location('biv150_materialized_prefill_probe',args.source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    torch.manual_seed(20261023);records=[]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for qlen,kv in [(16,128),(2048,2048),(2048,4096),(2048,8192),(2048,16384)]:
        pages=(kv+15)//16
        q=torch.randn(qlen,16,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(pages,16,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(pages,device='cuda',dtype=torch.int32)[None,:]
        cu=torch.tensor([0,qlen],device='cuda',dtype=torch.int32);lens=torch.tensor([kv],device='cuda',dtype=torch.int32)
        ref=torch.empty_like(q);out=torch.empty_like(q)
        def old():
            baseline.unified_attention(q=q,k=k,v=v,out=ref,cu_seqlens_q=cu,max_seqlen_q=qlen,seqused_k=lens,max_seqlen_k=kv,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=1.,k_descale=1.,v_descale=1.)
        old();base_ms=measure(old)
        for chunk in [64,128,256,512]:
            row=dict(qlen=qlen,kv=kv,chunk=chunk,baseline_ms=base_ms)
            try:
                fn=lambda:module.attention(q,k,v,table,kv,out,chunk)
                fn();torch.testing.assert_close(out,ref,atol=.005,rtol=.005)
                row['max_abs_diff']=(out.float()-ref.float()).abs().max().item()
                if kv==128:
                    ids=table[0].long();kd=k[ids].reshape(-1,2,128)[:kv].float().repeat_interleave(8,1);vd=v[ids].reshape(-1,2,128)[:kv].float().repeat_interleave(8,1)
                    score=torch.einsum('qhd,khd->hqk',q.float(),kd)*128**-.5
                    valid=torch.arange(kv,device=q.device)[None,:]<=kv-qlen+torch.arange(qlen,device=q.device)[:,None]
                    score=score.masked_fill(~valid[None],-float('inf'))
                    dense=torch.einsum('hqk,khd->qhd',score.softmax(-1),vd).to(q.dtype)
                    torch.testing.assert_close(out,dense,atol=.005,rtol=.005)
                    row['independent_fp32']='passed'
                row['candidate_ms']=measure(fn);row['speedup']=base_ms/row['candidate_ms']
            except Exception as e:row['error']=str(e)[:500]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(baseline_source_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),records=records),indent=2))
        del q,k,v,table,cu,lens,ref,out
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
