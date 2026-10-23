"""Audited prefill clones that skip causal masks on complete prefix tiles.

CPU source preparation is separate from GPU execution. No installed file changes.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import textwrap
import torch
from bench_biv150_prefill_tiles import measure

EXPECTED='54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f'
ELIGIBLE='USE_CAUSAL and not USE_PER_SEQ_CAUSAL and not SLIDING_WINDOW and not USE_MM_PREFIX and CHUNK_LOOKBACK < 0'


def sources(original):
    assert original.count('L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)')==1
    normalized=original.replace('L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)',
                                'L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    a=normalized.index('        query_abs_pos = context_len + query_pos[:, None]')
    b=normalized.index('        # S : (BLOCK_M, TILE_SIZE)',a)
    setup=normalized[a:b]
    masking='''        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )
'''
    assert normalized.count(masking)==1
    # j counts physical key tiles. This conservative bound guarantees every key
    # lies strictly before the earliest valid query; invalid query rows are not
    # stored and do not communicate with valid softmax rows.
    mask_code=setup.rstrip()+'\n'+masking
    conditional=normalized.replace(setup,'')
    replacement='        if '+ELIGIBLE+':\n'
    replacement+='            if (j + 1) * TILE_SIZE > context_len + q_block_local_idx * BLOCK_Q:\n'
    replacement+=textwrap.indent(mask_code,'        ')
    replacement+='        else:\n'+textwrap.indent(mask_code,'    ')
    conditional=conditional.replace(masking,replacement)
    loop='    for j in range(loop_lo, loop_hi):\n'
    a=normalized.index(loop);b=normalized.index('    # ---- Epilogue',a)
    body=normalized[a+len(loop):b]
    interior=body.replace(setup,'').replace(masking,'')
    prefix='    if '+ELIGIBLE+':\n'
    prefix+='        prefix_hi = tl.minimum(loop_hi, tl.maximum(loop_lo, (context_len + q_block_local_idx * BLOCK_Q) // TILE_SIZE))\n'
    prefix+='        for j in range(loop_lo, prefix_hi):\n'+textwrap.indent(interior,'    ')
    prefix+='        for j in range(prefix_hi, loop_hi):\n'+textwrap.indent(body,'    ')
    prefix+='    else:\n'+textwrap.indent(loop+body,'    ')
    split=normalized[:a]+prefix+normalized[b:]
    return {'normalized':normalized,'conditional':conditional,'split':split}


class Capture:
    def __init__(self,kernel):self.kernel=kernel;self.last=None
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            self.last=self.kernel[grid](*args,**kwargs)
            return self.last
        return launch


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    original=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    text=Path(original.__file__).read_text();assert hashlib.sha256(text.encode()).hexdigest()==EXPECTED
    args.out.parent.mkdir(parents=True,exist_ok=True)
    generated=args.out.parent/'prefix-generated';generated.mkdir(exist_ok=True)
    modules={}
    for name,source in sources(text).items():
        path=generated/(name+'.py');path.write_text(source)
        spec=importlib.util.spec_from_file_location('biv150_prefill_prefix_'+name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        capture=Capture(module.kernel_unified_attention);module.kernel_unified_attention=capture
        modules[name]=(module,capture)
    torch.manual_seed(20261023);torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    for qlen,klen in [(16,128),(127,257),(2048,2048),(2048,4096),(2048,8192),(2048,16384)]:
        pages=(klen+15)//16
        q=torch.randn(qlen,16,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(pages,16,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(pages,device='cuda',dtype=torch.int32)[None,:]
        cu=torch.tensor([0,qlen],device='cuda',dtype=torch.int32);lengths=torch.tensor([klen],device='cuda',dtype=torch.int32)
        kwargs=dict(q=q,k=k,v=v,cu_seqlens_q=cu,max_seqlen_q=qlen,seqused_k=lengths,
            max_seqlen_k=klen,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
            block_table=table,softcap=0.,q_descale=1.,k_descale=1.,v_descale=1.)
        reference=torch.empty_like(q);modules['normalized'][0].unified_attention(out=reference,**kwargs)
        baseline=measure(lambda:modules['normalized'][0].unified_attention(out=reference,**kwargs))
        gold=None
        if klen<=512:
            kd=k[table[0].long()].reshape(-1,2,128)[:klen].float().repeat_interleave(8,1)
            vd=v[table[0].long()].reshape(-1,2,128)[:klen].float().repeat_interleave(8,1)
            scores=torch.einsum('qhd,khd->hqk',q.float(),kd)*128**-.5
            valid=torch.arange(klen,device='cuda')[None,:]<=klen-qlen+torch.arange(qlen,device='cuda')[:,None]
            p=scores.masked_fill(~valid[None],-float('inf')).softmax(-1)
            gold=torch.einsum('hqk,khd->qhd',p,vd).to(q.dtype)
        for name,(module,capture) in modules.items():
            row=dict(qlen=qlen,klen=klen,variant=name,baseline_ms=baseline)
            try:
                out=torch.empty_like(q);module.unified_attention(out=out,**kwargs)
                torch.testing.assert_close(out,reference,atol=.005,rtol=.005)
                if gold is not None:torch.testing.assert_close(out,gold,atol=.005,rtol=.005)
                ms=measure(lambda:module.unified_attention(out=out,**kwargs))
                row.update(candidate_ms=ms,speedup=baseline/ms,bit_exact=torch.equal(out,reference),
                    max_abs_diff=(out.float()-reference.float()).abs().max().item(),
                    independent_fp32_short=gold is not None,n_regs=capture.last.n_regs,
                    n_spills=capture.last.n_spills,shared=capture.last.metadata.shared)
            except Exception as error:row['error']=str(error)[:1200]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(source_sha256=EXPECTED,records=records,service_installed=False),indent=2))
        del q,k,v,table,cu,lengths,kwargs,reference,gold
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
