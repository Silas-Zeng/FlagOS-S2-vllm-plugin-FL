"""Explicit-device probe of the unchanged compact writer codec, M1/M8."""
import triton
import triton.language as tl
from vllm_fl.patches.biv150_konly_compact_writer import quantize_bf16_rne

@triton.jit
def probe(source,code,scale,N:tl.constexpr,M:tl.constexpr):
    r=tl.program_id(0)*M+tl.arange(0,M)
    d=tl.arange(0,128)
    x=tl.load(source+r[:,None]*128+d[None,:],r[:,None]<N,other=0.)
    c,s=quantize_bf16_rne(x)
    tl.store(code+r[:,None]*128+d[None,:],c,r[:,None]<N)
    tl.store(scale+r,s,r<N)
