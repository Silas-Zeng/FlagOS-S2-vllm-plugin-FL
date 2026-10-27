"""Experimental V8 storage with BF16 weighted probabilities and BF16 PV.

The pinned GQA PAGE16 producer, BF16 QK, softmax and partial units stay intact.
Only V reads and the PV operands change: dot(bf16(P * V_scale), bf16(V_code)).
Signed codes [-127,127] are exactly representable in BF16; there is no P8
quantization, K32 tile, INT8 dot, or per-element wide V reconstruction.
No serving registration. The diagnostic allocation remains 16 KiB/page.
"""
import triton
import triton.language as tl

LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def gqa_decode_v8_bf16p(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
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
    scale_stride_block: tl.constexpr,
    scale_stride_slot: tl.constexpr,
    scale_stride_head: tl.constexpr,
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
        K = tl.load(key_cache_ptr + k_offsets)
        V = tl.load(value_cache_ptr + v_offsets)
        value_scale = tl.load(
            value_scale_ptr + page * scale_stride_block
            + column_slot * scale_stride_slot + kv_head_idx * scale_stride_head
        )

        scores = tl.dot(Q, K) * softmax_scale_log2e
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            (probability * value_scale[None, :]).to(tl.bfloat16),
            V.to(tl.bfloat16),
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
        K = tl.load(key_cache_ptr + k_offsets, mask=column_valid[None, :], other=0.0)
        V = tl.load(value_cache_ptr + v_offsets, mask=column_valid[:, None], other=0.0)
        value_scale = tl.load(
            value_scale_ptr + page * scale_stride_block
            + column_slot * scale_stride_slot + kv_head_idx * scale_stride_head,
            mask=column_valid, other=0.0,
        )

        scores = tl.dot(Q, K) * softmax_scale_log2e
        scores = tl.where(column_valid[None, :], scores, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
        accumulator = accumulator * rescale[:, None] + tl.dot(
            (probability * value_scale[None, :]).to(tl.bfloat16),
            V.to(tl.bfloat16),
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
def encode_cache_v(source,encoded,scales,
                   src_block:tl.constexpr,src_slot:tl.constexpr,src_head:tl.constexpr,
                   dst_block:tl.constexpr,dst_slot:tl.constexpr,dst_head:tl.constexpr,
                   SCALE_BLOCK:tl.constexpr,DIM:tl.constexpr):
    vector=tl.program_id(0);page=vector//32;slot=(vector//2)%16;head=vector%2
    dims=tl.arange(0,DIM)
    values=tl.load(source+page*src_block+slot*src_slot+head*src_head+dims)
    codes,scale=_quantized_values(values)
    tl.store(encoded+page*dst_block+slot*dst_slot+head*dst_head+dims,codes)
    tl.store(scales+page*SCALE_BLOCK+slot*2+head,scale)


@triton.jit
def append_token_kv(source_key,source_value,slot_mapping,cache_key,cache_value,value_scales,
                    source_token:tl.constexpr,source_head:tl.constexpr,
                    key_block:tl.constexpr,key_slot:tl.constexpr,key_head:tl.constexpr,
                    value_block:tl.constexpr,value_slot:tl.constexpr,value_head:tl.constexpr,
                    SCALE_BLOCK:tl.constexpr,TOTAL_SLOTS:tl.constexpr,DIM:tl.constexpr,ENCODE:tl.constexpr):
    token=tl.program_id(0);head=tl.program_id(1);slot=tl.load(slot_mapping+token)
    if slot<0 or slot>=TOTAL_SLOTS:return
    dims=tl.arange(0,DIM);src=token*source_token+head*source_head+dims
    key=tl.load(source_key+src);value=tl.load(source_value+src)
    koffset=(slot//16)*key_block+(slot%16)*key_slot+head*key_head+dims
    voffset=(slot//16)*value_block+(slot%16)*value_slot+head*value_head+dims
    tl.store(cache_key+koffset,key)
    if ENCODE:
        codes,scale=_quantized_values(value)
        tl.store(cache_value+voffset,codes)
        tl.store(value_scales+(slot//16)*SCALE_BLOCK+(slot%16)*2+head,scale)
    else:tl.store(cache_value+voffset,value)
