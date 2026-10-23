"""Isolate materialized-prefill correctness before changing any tolerance."""
import importlib,importlib.util,json,sys
from pathlib import Path
import torch
import socket
with socket.socket() as guard:assert guard.connect_ex(('127.0.0.1',9031))!=0,'Service running'
source=Path('vllm_fl/patches/biv150_materialized_prefill.py')
spec=importlib.util.spec_from_file_location('component_prefill',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
base=importlib.import_module('vllm.v1.attention.ops.triton_unified_attention')
torch.manual_seed(20261023)
qlen=16;kv=128
q=torch.randn(qlen,16,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn(kv//16,16,2,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
table=torch.randperm(kv//16,device='cuda',dtype=torch.int32)[None,:]
kp=k[table[0].long()].reshape(kv,2,128).permute(1,0,2).contiguous();vp=v[table[0].long()].reshape(kv,2,128).permute(1,0,2).contiguous()
qp=q.reshape(qlen,2,8,128).permute(1,0,2,3).reshape(2,qlen*8,128).contiguous()
s=torch.bmm(qp,kp.transpose(1,2),out_dtype=torch.float32);gold_s=torch.bmm(qp.float(),kp.float().transpose(1,2))

def report(name,x,y):
 print(json.dumps(dict(component=name,max_abs_diff=(x.float()-y.float()).abs().max().item(),x_min=x.min().item(),x_max=x.max().item(),finite=bool(torch.isfinite(x).all().item()))),flush=True)
report('QK_fp32_output',s,gold_s)
p=torch.empty_like(s,dtype=q.dtype)
mod._causal_softmax[(2*qlen*8,)](s,p,qlen*8,kv,kv-qlen,0,128**-.5,128,num_warps=8)
valid=torch.arange(kv,device=q.device)[None,:]<=kv-qlen+torch.arange(qlen,device=q.device).repeat_interleave(8)[:,None]
gold_p=(gold_s*128**-.5).masked_fill(~valid[None],-float('inf')).softmax(-1).to(q.dtype)
report('causal_softmax',p,gold_p)
product=torch.empty((2,qlen*8,128),device=q.device,dtype=q.dtype);torch.ops.aten.bmm.out(p,vp,out=product)
gold_product=torch.bmm(p.float(),vp.float()).to(q.dtype);report('PV_bf16_output',product,gold_product)
out=torch.empty_like(q);mod.attention(q,k,v,table,kv,out,128)
dense=torch.bmm(gold_p.float(),vp.float()).to(q.dtype).reshape(2,qlen,8,128).permute(1,0,2,3).reshape_as(q)
report('materialized_vs_dense',out,dense)
old=torch.empty_like(q)
base.unified_attention(q=q,k=k,v=v,out=old,cu_seqlens_q=torch.tensor([0,qlen],device=q.device,dtype=torch.int32),max_seqlen_q=qlen,seqused_k=torch.tensor([kv],device=q.device,dtype=torch.int32),max_seqlen_k=kv,softmax_scale=128**-.5,causal=True,window_size=(-1,-1),block_table=table,softcap=0.,q_descale=1.,k_descale=1.,v_descale=1.)
report('original_vs_dense',old,dense)
