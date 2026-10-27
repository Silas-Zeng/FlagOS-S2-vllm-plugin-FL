"""Opt-in direct codec probe for the frozen helper's M1 and M8 instantiations."""
import triton
import triton.language as tl
from cjj_konly_decode_support import quantize_bf16_rne


@triton.jit
def probe_codec(source,codes,scales,N:tl.constexpr,M:tl.constexpr):
    tl.static_assert(M==1 or M==8)
    tl.static_assert(source.dtype.element_ty==tl.bfloat16)
    tl.static_assert(codes.dtype.element_ty==tl.int8)
    tl.static_assert(scales.dtype.element_ty==tl.float32)
    row=tl.program_id(0).to(tl.int64)*M+tl.arange(0,M)
    dim=tl.arange(0,128)
    value=tl.load(source+row[:,None]*128+dim[None,:],row[:,None]<N,other=0.)
    encoded,scale=quantize_bf16_rne(value)
    tl.store(codes+row[:,None]*128+dim[None,:],encoded,row[:,None]<N)
    tl.store(scales+row,scale,row<N)
