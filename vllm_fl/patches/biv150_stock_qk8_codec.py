import triton
import triton.language as tl


@triton.jit
def quantize_bf16_rne(values):
    # Codec v2: fixed scale, IEEE RN32 quotient, then nearest/even integer code.
    wide = values.to(tl.float32)
    magnitude_bits = wide.to(tl.int32, bitcast=True) & 0x7fffffff
    peak_bits = tl.max(magnitude_bits, 1)
    peak = peak_bits.to(tl.float32, bitcast=True)
    scale = tl.where(peak_bits == 0, 1.0, tl.maximum(tl.div_rn(peak, 127.0), 1.1754943508222875e-38))
    quotient = tl.div_rn(wide, scale[:, None])
    magnitude = tl.abs(quotient)
    lower = tl.floor(magnitude)
    fraction = magnitude - lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    rounded = lower + increment.to(tl.float32)
    signed = tl.where(quotient < 0, -rounded, rounded)
    return tl.minimum(127.0, tl.maximum(-127.0, signed)).to(tl.int8), scale
