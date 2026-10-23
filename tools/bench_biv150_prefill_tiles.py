"""Launch-only tuning of the pinned prefill kernel; no installed source edits."""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import socket
import statistics
import torch


def measure(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):fn()
    times=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record();end.synchronize()
        times.append(start.elapsed_time(end)/3)
    return statistics.median(times)


def main(args):
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1',9031))!=0,'Service running; do not compete for GPU'
    attention=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    source=inspect.getsource(attention.unified_attention)
    assert 'biv150_prefill_tile = 64' in source
    assert attention.current_platform.vendor_name=='iluvatar'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    metadata=dict(module_file=attention.__file__,source_sha256=hashlib.sha256(Path(attention.__file__).read_bytes()).hexdigest(),torch=torch.__version__)
    torch.manual_seed(20261023)
    configs=[(128,64,8,1),(128,64,8,2),(128,64,8,3),(128,64,4,1),(128,64,4,2),
             (64,64,4,1),(64,64,8,1),(256,64,8,1),(256,64,8,2),
             (128,128,8,1),(128,128,8,2),(256,128,8,1),(64,128,4,1)]
    for qlen,klen in [(2048,2048),(2048,4096),(2048,8192),(2048,16384)]:
        pages=(klen+15)//16
        q=torch.randn(qlen,16,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(pages,16,2,128,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        table=torch.randperm(pages,device='cuda',dtype=torch.int32)[None,:]
        cu=torch.tensor([0,qlen],device='cuda',dtype=torch.int32)
        lens=torch.tensor([klen],device='cuda',dtype=torch.int32)
        def call(fn,out):
            fn(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=qlen,seqused_k=lens,max_seqlen_k=klen,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=1.,k_descale=1.,v_descale=1.)
        reference=torch.empty_like(q);call(attention.unified_attention,reference)
        baseline=measure(lambda:call(attention.unified_attention,reference))
        for bm,tile,warps,stages in configs:
            row=dict(qlen=qlen,klen=klen,bm=bm,tile=tile,warps=warps,stages=stages,baseline_ms=baseline)
            try:
                candidate=source.replace('BLOCK_M = 128',f'BLOCK_M = {bm}').replace('BLOCK_Q = 16',f'BLOCK_Q = {bm//8}').replace('biv150_prefill_tile = 64',f'biv150_prefill_tile = {tile}').replace('launch_num_warps = 8',f'launch_num_warps = {warps}',1).replace('launch_num_stages = 1',f'launch_num_stages = {stages}',1)
                namespace=dict(vars(attention));exec(compile(candidate,'<isolated_prefill_launch>','exec'),namespace)
                fn=namespace['unified_attention'];out=torch.empty_like(q);call(fn,out)
                torch.testing.assert_close(out,reference,atol=.005,rtol=.005)
                row.update(max_abs_diff=(out.float()-reference.float()).abs().max().item(),bit_exact=torch.equal(out,reference),candidate_ms=measure(lambda:call(fn,out)))
                row['speedup']=baseline/row['candidate_ms']
            except Exception as e:
                row['error']=str(e)[:400]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(metadata=metadata,records=records),indent=2))
        del q,k,v,reference,table,cu,lens
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
