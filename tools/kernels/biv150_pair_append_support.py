"""Private CPU-prepared K-only codec/writers. Import only under explicit device mode."""
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
def prepare_q(query, query_code, query_scale, BATCH: tl.constexpr):
    tl.static_assert(query.dtype.element_ty == tl.bfloat16)
    tl.static_assert(query_code.dtype.element_ty == tl.int8)
    tl.static_assert(query_scale.dtype.element_ty == tl.float32)
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * 8 + tl.arange(0, 8)
    dim = tl.arange(0, 128)
    offsets = token * 2048 + head[:, None] * 128 + dim[None, :]
    values = tl.load(query + offsets)
    codes, scales = quantize_bf16_rne(values)
    tl.store(query_code + offsets, codes)
    tl.store(query_scale + token * 16 + head, scales)


@triton.jit
def encode_initial(source_k, source_v, code, scale, value, PAGES: tl.constexpr):
    # Caller must prove source/destination storage separation; no in-place compression.
    tl.static_assert(source_k.dtype.element_ty == tl.bfloat16)
    tl.static_assert(source_v.dtype.element_ty == tl.bfloat16)
    tl.static_assert(code.dtype.element_ty == tl.int8)
    tl.static_assert(scale.dtype.element_ty == tl.float32)
    tl.static_assert(value.dtype.element_ty == tl.bfloat16)
    vector = tl.program_id(0).to(tl.int64)
    page, slot, head = vector // 32, (vector // 2) % 16, vector % 2
    dim = tl.arange(0, 128)[None, :]
    source_offset = page * 8192 + slot * 256 + head * 128 + dim
    keys = tl.load(source_k + source_offset)
    values = tl.load(source_v + source_offset)
    codes, scales = quantize_bf16_rne(keys)
    tl.store(code + page * 16384 + slot * 256 + head * 128 + dim, codes)
    tl.store(scale + page * 4096 + 1024 + slot * 2 + head + tl.arange(0, 1), scales)
    tl.store(value + page * 8192 + slot * 256 + head * 128 + dim, values)


@triton.jit
def append_kv(new_k, new_v, slots, key, scale, value,
              TOTAL_SLOTS: tl.constexpr, ENCODE: tl.constexpr):
    tl.static_assert(new_k.dtype.element_ty == tl.bfloat16)
    tl.static_assert(new_v.dtype.element_ty == tl.bfloat16)
    tl.static_assert(value.dtype.element_ty == tl.bfloat16)
    tl.static_assert(slots.dtype.element_ty == tl.int64)
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(slots + token).to(tl.int64)
    if (slot < 0) | (slot >= TOTAL_SLOTS):
        return
    page, column = slot // 16, slot % 16
    dim = tl.arange(0, 128)[None, :]
    source_offset = token * 256 + head * 128 + dim
    keys = tl.load(new_k + source_offset)
    values = tl.load(new_v + source_offset)
    if ENCODE:
        tl.static_assert(key.dtype.element_ty == tl.int8)
        tl.static_assert(scale.dtype.element_ty == tl.float32)
        codes, scales = quantize_bf16_rne(keys)
        tl.store(key + page * 16384 + column * 256 + head * 128 + dim, codes)
        tl.store(scale + page * 4096 + 1024 + column * 2 + head + tl.arange(0, 1), scales)
    else:
        tl.static_assert(key.dtype.element_ty == tl.bfloat16)
        tl.store(key + page * 8192 + column * 256 + head * 128 + dim, keys)
    tl.store(value + page * 8192 + column * 256 + head * 128 + dim, values)
