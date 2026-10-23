"""Probe prefill kernel liveness/cache/address changes without editing packages."""
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


class Capture:
    def __init__(self,kernel):self.kernel=kernel;self.last=None
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            self.last=self.kernel[grid](*args,**kwargs)
            return self.last
        return launch
    def metadata(self):
        k=self.last
        return dict(n_regs=getattr(k,'n_regs',None),n_spills=getattr(k,'n_spills',None),shared=getattr(getattr(k,'metadata',None),'shared',None))


def late_v(source):
    load='''            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )'''
    cast='        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)'
    anchor='        if SLIDING_WINDOW:\n            qpos_lo = q_block_local_idx * BLOCK_Q'
    assert source.count(load)==source.count(cast)==source.count(anchor)==1
    source=source.replace(load,'')
    source=source.replace(cast,'        if USE_TD:\n    '+cast)
    delayed='        if not USE_TD:\n'+load+'\n            V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)\n\n'
    return source.replace(anchor,delayed+anchor)


def cache_cg(source):
    for ptr,mask in [('key_cache_ptr + k_offset','dim_mask[:, None] & tile_mask[None, :]'),('value_cache_ptr + v_offset','dim_mask[None, :] & tile_mask[:, None]')]:
        old=ptr+',\n                mask='+mask+',\n                other=0.0,'
        assert source.count(old)==1
        source=source.replace(old,old+'\n                cache_modifier=".cg",')
    return source


def main(args):
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    original=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
    source=Path(original.__file__).read_text();assert hashlib.sha256(source.encode()).hexdigest()==EXPECTED
    args.out.parent.mkdir(parents=True,exist_ok=True)
    generated=args.out.parent/'liveness-generated';generated.mkdir(exist_ok=True)
    assert source.count(').to(tl.int64)\n\n        if USE_TD:')==1
    variants={'original':source,'late_v':late_v(source),'cg':cache_cg(source),'late_v_cg':cache_cg(late_v(source)),
              'i32_page':source.replace(').to(tl.int64)\n\n        if USE_TD:',').to(tl.int32)\n\n        if USE_TD:')}
    modules={}
    for name,text in variants.items():
        p=generated/(name+'.py');p.write_text(text)
        spec=importlib.util.spec_from_file_location('biv150_prefill_'+name,p)
        mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
        proxy=Capture(mod.kernel_unified_attention);mod.kernel_unified_attention=proxy
        modules[name]=(mod,proxy)
    records=[];torch.manual_seed(20261023)
    for qlen,klen in [(2048,2048),(2048,4096),(2048,8192),(2048,16384)]:
        pages=(klen+15)//16
        q=torch.randn(qlen,16,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(pages,16,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.randperm(pages,device='cuda',dtype=torch.int32)[None,:]
        cu=torch.tensor([0,qlen],device='cuda',dtype=torch.int32);lens=torch.tensor([klen],device='cuda',dtype=torch.int32)
        def call(fn,out):
            fn(q=q,k=k,v=v,out=out,cu_seqlens_q=cu,max_seqlen_q=qlen,seqused_k=lens,max_seqlen_k=klen,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=1.,k_descale=1.,v_descale=1.)
        reference=torch.empty_like(q);call(original.unified_attention,reference)
        baseline=measure(lambda:call(original.unified_attention,reference))
        for name,(mod,proxy) in modules.items():
            row=dict(qlen=qlen,klen=klen,variant=name,baseline_ms=baseline)
            try:
                out=torch.empty_like(q);call(mod.unified_attention,out)
                torch.testing.assert_close(out,reference,atol=.005,rtol=.005)
                row.update(max_abs_diff=(out.float()-reference.float()).abs().max().item(),bit_exact=torch.equal(out,reference),candidate_ms=measure(lambda:call(mod.unified_attention,out)),**proxy.metadata())
                row['speedup']=baseline/row['candidate_ms']
            except Exception as error:row['error']=str(error)[:500]
            records.append(row);print(json.dumps(row),flush=True)
            args.out.write_text(json.dumps(dict(source_sha256=EXPECTED,records=records),indent=2))
        del q,k,v,table,cu,lens,reference
        torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
