"""Private stride-aware K-only writer. Import only in explicit device mode."""
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



@triton.jit
def encode_append(key_source, value_source, slots, code_full, scale_full, value_full,
                  key_row_stride: tl.int64, value_row_stride: tl.int64,
                  key_rows: tl.int64, value_rows: tl.int64,
                  slot_count: tl.int64, cache_pages: tl.int64):
    # All three typed cache pointers point to the same cache-page0 byte address.
    tl.static_assert(key_source.dtype.element_ty == tl.bfloat16)
    tl.static_assert(value_source.dtype.element_ty == tl.bfloat16)
    tl.static_assert(slots.dtype.element_ty == tl.int64)
    tl.static_assert(code_full.dtype.element_ty == tl.int8)
    tl.static_assert(scale_full.dtype.element_ty == tl.float32)
    tl.static_assert(value_full.dtype.element_ty == tl.bfloat16)
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    if (token >= slot_count) | (token >= key_rows) | (token >= value_rows):
        return
    slot = tl.load(slots + token).to(tl.int64)
    if (slot < 0) | (slot >= cache_pages * 16):
        return
    page, column = slot // 16, slot % 16
    dim = tl.arange(0, 128)[None, :]
    keys = tl.load(key_source + token * key_row_stride + head * 128 + dim)
    values = tl.load(value_source + token * value_row_stride + head * 128 + dim)
    codes, scales = quantize_bf16_rne(keys)
    tl.store(code_full + page * 16384 + column * 256 + head * 128 + dim, codes)
    tl.store(scale_full + page * 4096 + 1024 + column * 2 + head + tl.arange(0, 1), scales)
    tl.store(value_full + page * 8192 + 4096 + column * 256 + head * 128 + dim, values)
