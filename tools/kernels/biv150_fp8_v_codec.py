"""Experimental software E4M3 V codec; no FP8 device arithmetic is required.

Finite BF16 values use saturating nearest-even encoding at fixed scale1.
These helpers are not installed into a model or cache writer.
"""
import triton
import triton.language as tl


@triton.jit
def encode_bf16_e4m3(values):
    bits=values.to(tl.uint16,bitcast=True).to(tl.uint32)
    sign=(bits>>8)&128
    exponent=((bits>>7)&255).to(tl.int32)
    mantissa=(bits&127).to(tl.int32)
    normal=((exponent-120)<<3)|(mantissa>>4)
    remainder=mantissa&15
    normal+=((remainder>8)|((remainder==8)&((normal&1)!=0))).to(tl.int32)
    normal=tl.minimum(126,tl.maximum(0,normal)).to(tl.uint32)
    shift=tl.minimum(31,tl.maximum(1,125-exponent)).to(tl.uint32)
    significant=(128+mantissa).to(tl.uint32)
    one=tl.full((),1,tl.uint32)
    subnormal=significant>>shift
    leftover=significant&((one<<shift)-1)
    halfway=one<<(shift-1)
    subnormal+=((leftover>halfway)|((leftover==halfway)&((subnormal&1)!=0))).to(tl.uint32)
    encoded=sign|tl.where(exponent>=121,normal,subnormal)
    encoded=tl.where(exponent==255,sign|127,encoded)
    return encoded.to(tl.uint8)


@triton.jit
def decode_e4m3_bits(codes):
    code=codes.to(tl.uint32)
    sign=(code&128)<<8
    exponent=(code>>3)&15
    mantissa=code&7
    normal=sign|((exponent+120)<<7)|(mantissa<<4)
    leading=tl.where(mantissa>=4,2,tl.where(mantissa>=2,1,0)).to(tl.uint32)
    one=tl.full((),1,tl.uint32)
    subnormal=sign|((118+leading)<<7)|((mantissa-(one<<leading))<<(7-leading))
    subnormal=tl.where(mantissa==0,sign,subnormal)
    bits=tl.where(exponent==0,subnormal,normal)
    bits=tl.where((exponent==15)&(mantissa==7),sign|0x7fc0,bits)
    return bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)


@triton.jit
def decode_e4m3_lut(codes,table):
    return tl.load(table+codes.to(tl.int32))


@triton.jit(do_not_specialize=['N'])
def encode_vectors(source,encoded,N,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    values=tl.load(source+offsets,mask=offsets<N,other=0)
    code=encode_bf16_e4m3(values)
    tl.store(encoded+offsets,code,mask=offsets<N)


@triton.jit(do_not_specialize=['N'])
def decode_vectors(encoded,decoded,table,N,BLOCK:tl.constexpr,USE_LUT:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    codes=tl.load(encoded+offsets,mask=offsets<N,other=0)
    if USE_LUT:values=decode_e4m3_lut(codes,table)
    else:values=decode_e4m3_bits(codes)
    tl.store(decoded+offsets,values,mask=offsets<N)
