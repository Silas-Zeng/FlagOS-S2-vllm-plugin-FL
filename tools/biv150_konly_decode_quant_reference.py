"""Independent codec v2: exact rational IEEE RN32 division followed by integer RNE."""
import struct

from biv150_konly_i8_qk_reference import f32, f32_bits, validate_vector

FORMAT='cjj-konly-rn32-rne-v2'


def rn32_div(numerator, denominator):
    """Integer arithmetic fixes RN32, including underflow, rather than double division."""
    if denominator<=0: raise ValueError('positive scale required')
    sign=0x80000000 if f32_bits(numerator)&0x80000000 else 0
    a,b=abs(numerator).as_integer_ratio();c,d=denominator.as_integer_ratio()
    n,z=a*d,b*c
    if n==0: return struct.unpack('<f',struct.pack('<I',sign))[0]
    exponent=n.bit_length()-z.bit_length()
    exponent_too_high=(n < z<<exponent) if exponent>=0 else (n<<-exponent < z)
    if exponent_too_high: exponent-=1
    exponent=max(exponent,-126)
    shift=23-exponent
    nn,dd=(n<<shift,z) if shift>=0 else (n,z<<-shift)
    q,r=divmod(nn,dd)
    q+=int(2*r>dd or (2*r==dd and q&1))
    if q>=2**24: q>>=1;exponent+=1
    if exponent>127: bits=sign|0x7f800000
    elif exponent==-126 and q<2**23: bits=sign|q
    else: bits=sign|((exponent+127)<<23)|(q-2**23)
    return struct.unpack('<f',struct.pack('<I',bits))[0]


def quantize_v2(values):
    values=validate_vector(values)
    peak=max(abs(x) for x in values)
    scale=1.0 if peak==0 else max(rn32_div(peak,127.0),2.**-126)
    codes=tuple(max(-127,min(127,round(rn32_div(v,scale)))) for v in values)
    return codes,scale
