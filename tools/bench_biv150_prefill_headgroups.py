"""Audited prefill clones that split GQA query heads while preserving query tiles.

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
import torch
from bench_biv150_prefill_tiles import measure

EXPECTED='54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f'
ELIGIBLE='USE_CAUSAL and not USE_PER_SEQ_CAUSAL and not SLIDING_WINDOW and not USE_MM_PREFIX and CHUNK_LOOKBACK < 0'


def sources(original):
    assert original.count('L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)') == 1
    normalized = original.replace('L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)',
                                  'L = tl.zeros([BLOCK_M], dtype=tl.float32)')
    variants = {'normalized': normalized}
    for groups, warps in [(2,4), (4,4), (2,8)]:
        text = normalized
        for stride in ['stride_k_cache_2', 'stride_v_cache_2']:
            old = '+ kv_head_idx * ' + stride
            assert text.count(old) == 1
            text = text.replace(old, f'+ (kv_head_idx // {groups}) * {stride}')
        variants[f'group{groups}_warp{warps}'] = text
    return variants


class Capture:
    def __init__(self, kernel, groups=1, warps=None):
        self.kernel, self.groups, self.warps = kernel, groups, warps
        self.last = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            assert not kwargs['IS_3D'] and not kwargs['USE_TD']
            assert not kwargs['USE_FP8']
            assert kwargs['num_queries_per_kv'] == 8
            kwargs['num_queries_per_kv'] //= self.groups
            kwargs['BLOCK_M'] //= self.groups
            if self.warps is not None:
                kwargs['num_warps'] = self.warps
            changed_grid = (grid[0], grid[1] * self.groups)
            self.last = self.kernel[changed_grid](*args, **kwargs)
            return self.last
        return launch


def main(args):
    with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    original=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    text=Path(original.__file__).read_text();assert hashlib.sha256(text.encode()).hexdigest()==EXPECTED
    args.out.parent.mkdir(parents=True,exist_ok=True)
    generated=args.out.parent/'headgroups-generated';generated.mkdir(exist_ok=True)
    modules={}
    for name,source in sources(text).items():
        path=generated/(name+'.py');path.write_text(source)
        spec=importlib.util.spec_from_file_location('biv150_prefill_headgroups_'+name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        if name == 'normalized':
            capture=Capture(module.kernel_unified_attention)
        else:
            group_part,warp_part=name.split('_')
            capture=Capture(module.kernel_unified_attention,int(group_part[5:]),int(warp_part[4:]))
        module.kernel_unified_attention=capture
        modules[name]=(module,capture)
    torch.manual_seed(20261023);torch.backends.cuda.matmul.allow_tf32=False
    records=[]
    cases=[([16],[128]),([127],[257]),([2048],[2048]),([2048],[4096]),([2048],[8192]),([2048],[16384]),
           ([1,16,40,0],[65,127,257,0]),([8,33,1],[16384,4097,100]),([16,1016,1016],[200,4097,16384])]
    for qlens,klens in cases:
        counts=[(n+15)//16 for n in klens];pages=sum(counts)
        q=torch.randn(sum(qlens),16,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(pages,16,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        packed=torch.stack([k,v],dim=1);k,v=packed.unbind(1)
        # The audited kernel reads page-table entries before masking KV loads.
        # Pad the table as real serving does, so tail-tile lookups stay in bounds.
        table=torch.zeros((len(qlens),((max(counts)+31)//32)*32),device='cuda',dtype=torch.int32)
        shuffled=torch.randperm(pages,device='cuda',dtype=torch.int32);offset=0
        for i,count in enumerate(counts):table[i,:count]=shuffled[offset:offset+count];offset+=count
        starts=[0]
        for n in qlens:starts.append(starts[-1]+n)
        cu=torch.tensor(starts,device='cuda',dtype=torch.int32);lengths=torch.tensor(klens,device='cuda',dtype=torch.int32)
        kwargs=dict(q=q,k=k,v=v,cu_seqlens_q=cu,max_seqlen_q=max(qlens),seqused_k=lengths,
            max_seqlen_k=max(klens),softmax_scale=128**-.5,causal=True,window_size=(-1,-1),
            block_table=table,softcap=0.,q_descale=1.,k_descale=1.,v_descale=1.)
        reference=torch.empty_like(q);modules['normalized'][0].unified_attention(out=reference,**kwargs)
        baseline=measure(lambda:modules['normalized'][0].unified_attention(out=reference,**kwargs))
        gold=[]
        for seq,(qlen,klen,count) in enumerate(zip(qlens,klens,counts)):
            if not qlen:continue
            ids=table[seq,:count].long()
            kd=k[ids].reshape(-1,2,128)[:klen].float()
            vd=v[ids].reshape(-1,2,128)[:klen].float()
            for start in sorted({0,max(0,qlen//2-8),max(0,qlen-16)}):
                stop=min(start+16,qlen)
                queries=q[starts[seq]+start:starts[seq]+stop].float().reshape(stop-start,2,8,128)
                scores=torch.einsum('qhgd,khd->hgqk',queries,kd)*128**-.5
                valid=torch.arange(klen,device='cuda')[None,:]<=klen-qlen+torch.arange(start,stop,device='cuda')[:,None]
                prob=scores.masked_fill(~valid[None,None],-float('inf')).softmax(-1)
                output=torch.einsum('hgqk,khd->qhgd',prob,vd).reshape(stop-start,16,128).to(q.dtype)
                gold.append((starts[seq]+start,starts[seq]+stop,output))
        for name,(module,capture) in modules.items():
            row=dict(qlens=qlens,klens=klens,variant=name,baseline_ms=baseline,block_table_stride=table.stride(0))
            try:
                out=torch.empty_like(q);module.unified_attention(out=out,**kwargs)
                torch.testing.assert_close(out,reference,atol=.005,rtol=.005)
                for lo,hi,expected in gold:torch.testing.assert_close(out[lo:hi],expected,atol=.005,rtol=.005)
                ms=measure(lambda:module.unified_attention(out=out,**kwargs))
                row.update(candidate_ms=ms,speedup=baseline/ms,bit_exact=torch.equal(out,reference),
                    max_abs_diff=(out.float()-reference.float()).abs().max().item(),
                    independent_fp32_windows=len(gold),n_regs=capture.last.n_regs,
                    n_spills=capture.last.n_spills,shared=capture.last.metadata.shared)
            except Exception as error:row['error']=str(error)[:1200]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(source_sha256=EXPECTED,records=records,service_installed=False),indent=2))
        del q,k,v,packed,table,cu,lengths,kwargs,reference,gold
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
