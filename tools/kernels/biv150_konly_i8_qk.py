"""Private primitive only. Never imported by a default CPU entry point."""
import triton
import triton.language as tl


@triton.jit
def konly_i8_qk_probe(
    query_code, query_scale, packed_code, packed_scale,
    physical_pages, kv_heads, valid_columns, integer_out, score_out,
):
    tl.static_assert(query_code.dtype.element_ty == tl.int8)
    tl.static_assert(packed_code.dtype.element_ty == tl.int8)
    tl.static_assert(query_scale.dtype.element_ty == tl.float32)
    tl.static_assert(packed_scale.dtype.element_ty == tl.float32)
    tl.static_assert(integer_out.dtype.element_ty == tl.int32)
    tl.static_assert(score_out.dtype.element_ty == tl.float32)
    tl.static_assert(physical_pages.dtype.element_ty == tl.int32)
    tl.static_assert(kv_heads.dtype.element_ty == tl.int32)
    tl.static_assert(valid_columns.dtype.element_ty == tl.int32)
    case = tl.program_id(0).to(tl.int64)
    page = tl.load(physical_pages + case).to(tl.int64)
    head = tl.load(kv_heads + case).to(tl.int64)
    valid = tl.load(valid_columns + case)
    rows = tl.arange(0, 8)
    dims = tl.arange(0, 128)
    columns = tl.arange(0, 16)
    Q = tl.load(query_code + case * 1024 + rows[:, None] * 128 + dims[None, :])
    key_offset = page * 16384 + columns[None, :] * 256 + head * 128 + dims[:, None]
    K = tl.load(packed_code + key_offset, mask=columns[None, :] < valid, other=0)
    zero = tl.zeros((8, 16), dtype=tl.int32)
    raw = tl.dot(Q, K, zero, out_dtype=tl.int32)
    qs = tl.load(query_scale + case * 8 + rows)
    scale_offset = page * 4096 + 1024 + columns * 2 + head
    ks = tl.load(packed_scale + scale_offset, mask=columns < valid, other=1.0)
    scaled_q = raw.to(tl.float32) * qs[:, None]
    score = scaled_q * ks[None, :]
    score = tl.where(columns[None, :] < valid, score, float('-inf'))
    offset = case * 128 + rows[:, None] * 16 + columns[None, :]
    tl.store(integer_out + offset, raw)
    tl.store(score_out + offset, score)
