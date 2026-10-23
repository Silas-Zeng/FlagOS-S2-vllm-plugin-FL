"""Isolate the pinned helper's first online-softmax update, including MMA layout."""
import json,math,socket
from pathlib import Path
import torch
import triton
import triton.language as tl
from vllm.v1.attention.ops.triton_attention_helpers import init_softmax_M,softmax_step


@triton.jit
def probe(Q,K,O,DOT:tl.constexpr):
    m=tl.arange(0,128);n=tl.arange(0,64);d=tl.arange(0,128)
    if DOT:
        q=tl.load(Q+m[:,None]*128+d[None,:])
        k=tl.load(K+d[:,None]*64+n[None,:])
        s=tl.dot(q,k)
    else:
        s=tl.full((128,64),2.,tl.float32)
    s=tl.where(n[None,:]==0,s,-float('inf'))
    initial=init_softmax_M(None,m,m<128,0,128,False,False)
    new,ls,p,alpha=softmax_step(s,initial,tl.full((128,),1.,tl.float32))
    tl.store(O+m*6+0,new);tl.store(O+m*6+1,ls);tl.store(O+m*6+2,alpha)
    tl.store(O+m*6+3,tl.sum(p,1));tl.store(O+m*6+4,tl.exp(-float('inf')))
    tl.store(O+m*6+5,initial)


if __name__=='__main__':
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0,'Service running'
    q=torch.zeros((128,128),device='cuda',dtype=torch.bfloat16);q[:,0]=1
    k=torch.zeros((128,64),device='cuda',dtype=torch.bfloat16);k[0,0]=2
    out=torch.empty((128,6),device='cuda',dtype=torch.float32)
    rows=[]
    for dot in [False,True]:
        probe[(1,)](q,k,out,dot,num_warps=8,num_stages=1)
        values=out[0].tolist()
        row=dict(mma=dot,fields=['new_max','new_sum','alpha','sum_probability','exp_neg_inf','initial_max'],values=[x if math.isfinite(x) else str(x) for x in values])
        rows.append(row);print(json.dumps(row),flush=True)
    Path('/workspace/experiments/minicpm-s2/prefill-diagnostics-20261023/softmax-component-probe.json').write_text(json.dumps(rows,indent=2))
