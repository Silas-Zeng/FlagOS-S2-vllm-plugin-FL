"""Experimental self-developed INT8 token/head KV and fused decode consumption.

Diagnostic only: no serving registration, dtype override or model cache changes.
The decode loop derives from the shipping GQA kernel at main0915f01 (source
SHA2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1).
INT8 is unpacked/scaled in registers before its BF16 dots. The existing segment
reducer is reusable; no BF16 cache gather or materialized dequantized KV is used.
"""
import triton
import triton.language as tl

LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _quantized_values(values):
    wide = values.to(tl.float32)
    maximum = tl.max(tl.abs(wide), 0)
    # Min-normal keeps division finite for tiny BF16 inputs with FTZ enabled.
    scale = tl.where(maximum == 0, 1.0, tl.maximum(maximum * (1.0 / 127.0), 1.1754943508222875e-38))
    ratio = wide / scale
    rounded = tl.where(ratio >= 0, tl.floor(ratio + 0.5), -tl.floor(-ratio + 0.5))
    return tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8), scale


@triton.jit
def quantize_packed_cache(source, quantized, scales, DIM: tl.constexpr):
    # One token/head vector; input is contiguous [pages,2,16,2,128].
    vector = tl.program_id(0)
    dims = tl.arange(0, DIM)
    values = tl.load(source + vector * DIM + dims)
    encoded, scale = _quantized_values(values)
    tl.store(quantized + vector * DIM + dims, encoded)
    tl.store(scales + vector, scale)


@triton.jit
def append_token_kv(
    source_key, source_value, slot_mapping,
    cache_key, cache_value, key_scale, value_scale,
    source_stride_token: tl.constexpr, source_stride_head: tl.constexpr,
    cache_stride_block: tl.constexpr, cache_stride_slot: tl.constexpr,
    cache_stride_head: tl.constexpr, scale_stride_block: tl.constexpr,
    scale_stride_slot: tl.constexpr, TOTAL_SLOTS: tl.constexpr,
    DIM: tl.constexpr, QUANTIZE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_mapping + token)
    if slot < 0 or slot >= TOTAL_SLOTS:
        return
    dims = tl.arange(0, DIM)
    offset = token * source_stride_token + head * source_stride_head + dims
    key = tl.load(source_key + offset)
    value = tl.load(source_value + offset)
    target = (slot // 16) * cache_stride_block + (slot % 16) * cache_stride_slot + head * cache_stride_head + dims
    if QUANTIZE:
        encoded_key, scale_key = _quantized_values(key)
        encoded_value, scale_value = _quantized_values(value)
        scale_target = (slot // 16) * scale_stride_block + (slot % 16) * scale_stride_slot + head
        tl.store(key_scale + scale_target, scale_key)
        tl.store(value_scale + scale_target, scale_value)
        tl.store(cache_key + target, encoded_key)
        tl.store(cache_value + target, encoded_value)
    else:
        # Diagnostic BF16 copy control with the same input/slot mapping. This
        # is not claimed to replace or benchmark the framework's cache writer.
        tl.store(cache_key + target, key)
        tl.store(cache_value + target, value)


@triton.jit
def _gqa_decode_attention_int8(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    seq_lens_ptr,
    block_table_ptr,
    softmax_scale_log2e,
    queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_token: tl.int64,
    query_stride_head: tl.constexpr,
    k_stride_block: tl.int64,
    k_stride_slot: tl.int64,
    k_stride_head: tl.int64,
    v_stride_block: tl.int64,
    v_stride_slot: tl.int64,
    v_stride_head: tl.constexpr,
    scale_stride_block: tl.int64,
    scale_stride_slot: tl.constexpr,
    segment_stride_token: tl.constexpr,
    segment_stride_head: tl.constexpr,
    segment_stride_segment: tl.constexpr,
    max_stride_token: tl.constexpr,
    max_stride_head: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SEGMENTS: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segment_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    steps_per_segment = tl.cdiv(seq_len, SEGMENTS * PAGE_SIZE)
    lower = segment_idx * steps_per_segment * PAGE_SIZE
    if lower >= seq_len:
        return
    upper = tl.minimum(lower + steps_per_segment * PAGE_SIZE, seq_len)

    offs_row = tl.arange(0, ROWS)
    offs_dim = tl.arange(0, HEAD_SIZE)
    offs_step = tl.arange(0, PAGE_SIZE)
    # ROWS equals queries_per_kv, so every row is a real head; the padding that
    # the generic kernel carries does not exist here.
    row_valid = offs_row < queries_per_kv

    token_idx = seq_idx.to(tl.int64)
    head_idx = kv_head_idx * queries_per_kv + offs_row
    query_offsets = (
        token_idx * query_stride_token
        + head_idx[:, None] * query_stride_head
        + offs_dim[None, :]
    )
    Q = tl.load(query_ptr + query_offsets, mask=row_valid[:, None], other=0.0)

    block_table_row = block_table_ptr + seq_idx * block_table_stride
    column_slot = offs_step % PAGE_SIZE

    running_max = tl.full([ROWS], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([ROWS], dtype=tl.float32)
    accumulator = tl.zeros([ROWS, HEAD_SIZE], dtype=tl.float32)

    full_upper = lower + ((upper - lower) // PAGE_SIZE) * PAGE_SIZE
    position = lower
    while position < full_upper:
        # A step is exactly one page and positions advance page-aligned, so one
        # scalar block-table entry covers the whole tile.
        page = tl.load(block_table_row + position // PAGE_SIZE).to(tl.int64)
        k_offsets = (
            page * k_stride_block
            + column_slot[None, :] * k_stride_slot
            + kv_head_idx * k_stride_head
            + offs_dim[:, None]
        )
        v_offsets = (
            page * v_stride_block
            + column_slot[:, None] * v_stride_slot
            + kv_head_idx * v_stride_head
            + offs_dim[None, :]
        )
        scale_offsets = page * scale_stride_block + column_slot * scale_stride_slot + kv_head_idx
        key_scale = tl.load(key_scale_ptr + scale_offsets)
        K = (tl.load(key_cache_ptr + k_offsets).to(tl.float32) * key_scale[None, :]).to(Q.dtype)

        scores = tl.dot(Q, K) * softmax_scale_log2e
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        # Delay V reconstruction until K/scores are dead. INT8-to-FP32-to-
        # BF16 temporaries otherwise overlap both matrix operands.
        value_scale = tl.load(value_scale_ptr + scale_offsets)
        V = (tl.load(value_cache_ptr + v_offsets).to(tl.float32) * value_scale[:, None]).to(Q.dtype)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(V.dtype), V
        )
        running_max = new_max
        position += PAGE_SIZE

    if full_upper < upper:
        # Final partial page: bound the columns, and read the page only for the
        # slots that exist.
        page = tl.load(block_table_row + full_upper // PAGE_SIZE).to(tl.int64)
        column_valid = full_upper + offs_step < upper
        k_offsets = (
            page * k_stride_block
            + column_slot[None, :] * k_stride_slot
            + kv_head_idx * k_stride_head
            + offs_dim[:, None]
        )
        v_offsets = (
            page * v_stride_block
            + column_slot[:, None] * v_stride_slot
            + kv_head_idx * v_stride_head
            + offs_dim[None, :]
        )
        scale_offsets = page * scale_stride_block + column_slot * scale_stride_slot + kv_head_idx
        key_scale = tl.load(key_scale_ptr + scale_offsets, mask=column_valid, other=1.0)
        K = (tl.load(key_cache_ptr + k_offsets, mask=column_valid[None, :], other=0).to(tl.float32) * key_scale[None, :]).to(Q.dtype)

        scores = tl.dot(Q, K) * softmax_scale_log2e
        scores = tl.where(column_valid[None, :], scores, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        value_scale = tl.load(value_scale_ptr + scale_offsets, mask=column_valid, other=1.0)
        V = (tl.load(value_cache_ptr + v_offsets, mask=column_valid[:, None], other=0).to(tl.float32) * value_scale[:, None]).to(Q.dtype)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            probability.to(V.dtype), V
        )
        running_max = new_max

    output_offsets = (
        token_idx * segment_stride_token
        + head_idx[:, None] * segment_stride_head
        + segment_idx * segment_stride_segment
        + offs_dim[None, :]
    )
    tl.store(segm_output_ptr + output_offsets, accumulator, mask=row_valid[:, None])
    scalar_offsets = (
        token_idx * max_stride_token + head_idx * max_stride_head + segment_idx
    )
    # The scores were scaled by log2(e) so the loop is exp2-based; exp2 of a
    # log2-domain difference equals exp of the natural difference, so the sums and
    # the accumulator are already natural. reduce_segments however combines segments
    # with tl.exp, so hand back the maxima in natural units.
    tl.store(segm_max_ptr + scalar_offsets, running_max * LN2, mask=row_valid)
    tl.store(segm_expsum_ptr + scalar_offsets, running_sum, mask=row_valid)
