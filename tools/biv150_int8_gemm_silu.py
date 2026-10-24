"""Standalone INT8 GEMM + original BF16-boundary SiLU, no INT32 output tensor.

Targets the ordinary M>=17 gate path. Residual small-M is not implemented here.
The native Kx12288 quantized weight is read in place without repacking.
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


CONFIGS=((16,64,64,4),(32,64,64,4),(32,128,64,8),(32,64,128,4))


@triton.jit
def fused_gate(AQ,WQ,SA,SW,OUT,M:tl.constexpr,BM:tl.constexpr,
               BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    gate_acc=tl.zeros((BM,BN),tl.int32)
    up_acc=tl.zeros((BM,BN),tl.int32)
    for offset in range(0,2048,BK):
        k=offset+kk
        a=tl.load(AQ+row[:,None]*2048+k[None,:],row[:,None]<M,other=0)
        g=tl.load(WQ+k[:,None]*12288+col[None,:],col[None,:]<6144,other=0)
        u=tl.load(WQ+k[:,None]*12288+6144+col[None,:],col[None,:]<6144,other=0)
        gate_acc=tl.dot(a,g,gate_acc,out_dtype=tl.int32)
        up_acc=tl.dot(a,u,up_acc,out_dtype=tl.int32)
    rs=tl.load(SA+row,row<M,other=1)
    gs=tl.load(SW+col,col<6144,other=1)
    us=tl.load(SW+6144+col,col<6144,other=1)
    # Native epilogue groups row/column scale multiplication before applying
    # the INT32 accumulator and rounds each projection before activation.
    gate=(gate_acc.to(tl.float32)*(rs[:,None]*gs[None,:])).to(tl.bfloat16).to(tl.float32)
    up=(up_acc.to(tl.float32)*(rs[:,None]*us[None,:])).to(tl.bfloat16).to(tl.float32)
    activated=tl.div_rn(gate,1.0+libdevice.exp(-gate))
    tl.store(OUT+row[:,None]*6144+col[None,:],activated*up,
             (row[:,None]<M)&(col[None,:]<6144))


def from_quantized(aq,sa,wq,sw,rows,config):
    assert 17<=rows<=2048 and config in CONFIGS
    assert aq.shape==(max(32,rows),2048) and aq.dtype==torch.int8 and aq.is_contiguous()
    assert wq.shape==(2048,12288) and wq.dtype==torch.int8 and wq.is_contiguous()
    assert sa.shape==(max(32,rows),) and sw.shape==(12288,)
    assert sa.dtype==sw.dtype==torch.float32 and sa.is_contiguous() and sw.is_contiguous()
    assert aq.is_cuda and aq.device==sa.device==wq.device==sw.device
    output=torch.empty((rows,6144),device=aq.device,dtype=torch.bfloat16)
    bm,bn,bk,warps=config
    fused_gate[(triton.cdiv(rows,bm),triton.cdiv(6144,bn))](
        aq,wq,sa,sw,output,rows,bm,bn,bk,num_warps=warps,num_stages=1)
    return output


def full_gate(x,wq,sw,config):
    assert x.is_contiguous() and x.shape[1]==2048 and x.dtype==torch.bfloat16
    aq,sa=torch.ops.biv150_native.quantize_rows_probe(x,256)
    return from_quantized(aq,sa,wq,sw,x.shape[0],config)
