"""CPU-only E4M3 finite-saturating RNE references; fixed scale1 experiment.

Field interpretation follows Table1 of https://arxiv.org/pdf/2209.05433.
Saturating finite overflow and round-to-nearest-even are explicit choices.
This file imports neither Torch nor a device runtime.
"""
import bisect
import math
import struct


def bf16_value(bits):
    return struct.unpack('>f',struct.pack('>I',bits<<16))[0]


def decode_e4m3(code):
    sign=-1.0 if code&128 else 1.0
    exponent=(code>>3)&15
    mantissa=code&7
    if exponent==15 and mantissa==7:return math.copysign(float('nan'),sign)
    if exponent==0:return sign*math.ldexp(float(mantissa),-9)
    return sign*math.ldexp(1.0+mantissa/8.0,exponent-7)


POSITIVE_VALUES=tuple(decode_e4m3(code) for code in range(127))


def encode_nearest_reference(bits):
    """Independent oracle: enumerate mathematical FP8 values, not BF16 fields."""
    sign=(bits>>8)&128
    value=abs(bf16_value(bits))
    if not math.isfinite(value):return sign|127
    upper=bisect.bisect_left(POSITIVE_VALUES,value)
    if upper==0:return sign
    if upper==len(POSITIVE_VALUES):return sign|126
    lower=upper-1
    dl=value-POSITIVE_VALUES[lower]
    du=POSITIVE_VALUES[upper]-value
    if dl<du:result=lower
    elif du<dl:result=upper
    else:result=lower if lower%2==0 else upper
    return sign|result


def encode_integer_candidate(bits):
    """Integer-only candidate for BF16 source, independently checked above."""
    sign=(bits>>8)&128
    exponent=(bits>>7)&255
    mantissa=bits&127
    if exponent==255:return sign|127
    if exponent>=121:
        code=((exponent-120)<<3)|(mantissa>>4)
        rem=mantissa&15
        code+=int(rem>8 or (rem==8 and code&1))
        return sign|min(126,code)
    shift=min(31,max(1,125-exponent))
    significant=128+mantissa
    code=significant>>shift
    rem=significant&((1<<shift)-1)
    half=1<<(shift-1)
    code+=int(rem>half or (rem==half and code&1))
    return sign|code


def decode_integer_bf16_bits(code):
    sign=(code&128)<<8
    exponent=(code>>3)&15
    mantissa=code&7
    if exponent==15 and mantissa==7:return sign|0x7fc0
    if exponent:return sign|((exponent+120)<<7)|(mantissa<<4)
    if not mantissa:return sign
    leading=2 if mantissa>=4 else 1 if mantissa>=2 else 0
    return sign|((118+leading)<<7)|((mantissa-(1<<leading))<<(7-leading))


def bf16_bits_of_exact_value(value):
    return struct.unpack('>I',struct.pack('>f',value))[0]>>16
