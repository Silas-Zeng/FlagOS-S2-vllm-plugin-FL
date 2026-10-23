"""BF16 mixed attention: prefill plus offset-aware single-query GQA.

Only the operator dispatch changes. Query offsets stay on the GPU; no request
sorting, host synchronization, scheduler changes, or KV conversion is performed.
"""

import functools
import hashlib
import importlib
import inspect
import linecache
import os
import types

import torch
import triton
import triton.language as tl

from vllm_fl.patches.biv150_prefill_normalized import (
    _eligible as prefill_eligible,
    make_normalized,
)

LOG2E = 1.4426950408889634
LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _gqa_decode_attention(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    seq_lens_ptr,
    cu_query_ptr,
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

    query_start = tl.load(cu_query_ptr + seq_idx).to(tl.int64)
    query_end = tl.load(cu_query_ptr + seq_idx + 1).to(tl.int64)
    if query_end - query_start != 1:
        return
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
        query_start * query_stride_token
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

        scores = tl.dot(Q, K) * softmax_scale_log2e
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
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
        K = tl.load(key_cache_ptr + k_offsets, mask=column_valid[None, :], other=0.0)
        V = tl.load(value_cache_ptr + v_offsets, mask=column_valid[:, None], other=0.0)

        scores = tl.dot(Q, K) * softmax_scale_log2e
        scores = tl.where(column_valid[None, :], scores, float("-inf"))
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        probability = tl.math.exp2(scores - new_max[:, None])
        rescale = tl.math.exp2(running_max - new_max)
        running_sum = running_sum * rescale + tl.sum(probability, 1)
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


@triton.jit
def _reduce_partials(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    cu_query_ptr,
    segment_stride_token,
    segment_stride_head,
    segment_stride_segment,
    segment_stride_dim: tl.constexpr,
    max_stride_token,
    max_stride_head,
    max_stride_segment: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_head: tl.constexpr,
    page_size: tl.constexpr,
    num_segments: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    """Combine the per-segment partials for one (token, query head).

    vLLM's own reduction derives the head stride from its segment count, which is
    only valid when the buffers were allocated for exactly that width. This kernel
    takes strides, so a step may use fewer segments than the builder allocated by
    handing this reduction a prefix view.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    query_start = tl.load(cu_query_ptr + token_idx).to(tl.int64)
    query_end = tl.load(cu_query_ptr + token_idx + 1).to(tl.int64)
    if query_end - query_start != 1:
        return
    seq_len = tl.load(seq_lens_ptr + token_idx)
    # Each program covers ceil(seq_len / (num_segments * page)) tiles, so the number
    # of segments that actually hold data is derived from that span, not from
    # seq_len / (num_segments * page) directly. The difference matters for short
    # sequences: with 16 segments and a 100-token sequence the span is one page, so
    # 7 segments are live, while the naive formula would read 1 and silently drop
    # seven eighths of the keys and values.
    tiles_per_segment = tl.maximum(tl.cdiv(seq_len, num_segments * page_size), 1)
    active = tl.cdiv(seq_len, tiles_per_segment * page_size)

    offs_seg = tl.arange(0, num_segments)
    offs_dim = tl.arange(0, HEAD_SIZE)
    seg_valid = offs_seg < active

    max_base = segm_max_ptr + token_idx * max_stride_token + head_idx * max_stride_head
    segm_max = tl.load(
        max_base + offs_seg * max_stride_segment, mask=seg_valid, other=float("-inf")
    )
    # A CUDA-graph padded row can carry seq_len == 0, where no segment is live:
    # folding -inf through exp would give NaN, so the empty case reduces to 0.
    overall_max = tl.where(active > 0, tl.max(segm_max, 0), 0.0)

    expsum_base = (
        segm_expsum_ptr + token_idx * max_stride_token + head_idx * max_stride_head
    )
    segm_expsum = tl.load(
        expsum_base + offs_seg * max_stride_segment, mask=seg_valid, other=0.0
    )
    weights = tl.exp(tl.where(seg_valid, segm_max - overall_max, float("-inf")))
    overall_expsum = tl.sum(segm_expsum * weights, 0)

    out_base = (
        segm_output_ptr
        + token_idx * segment_stride_token
        + head_idx * segment_stride_head
    )
    partial_offsets = (
        offs_seg[:, None] * segment_stride_segment
        + offs_dim[None, :] * segment_stride_dim
    )
    partials = tl.load(out_base + partial_offsets, mask=seg_valid[:, None], other=0.0)
    acc = tl.sum(partials * weights[:, None], 0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc / overall_expsum)

    tl.store(
        output_ptr
        + query_start * output_stride_token
        + head_idx * output_stride_head
        + offs_dim,
        acc,
    )


def make_prefill(attention):
    normalized, metadata = make_normalized(attention)
    kernel = normalized.__globals__["kernel_unified_attention"]
    source = inspect.getsource(kernel.fn)
    anchor = (
        "    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:\n        return\n"
    )
    if source.count(anchor) != 1:
        raise RuntimeError("Unexpected prefill query-range guard")
    source = source.replace(
        anchor, anchor + "\n    if cur_batch_query_len == 1:\n        return\n"
    )
    source = source.replace(
        "def kernel_biv150_prefill_normalized(", "def kernel_biv150_mixed_prefill(", 1
    )
    digest = hashlib.sha256(source.encode()).hexdigest()
    filename = "<biv150-mixed-prefill-" + digest + ".py>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = dict(kernel.fn.__globals__)
    namespace["__name__"] = __name__
    exec(compile(source, filename, "exec"), namespace)
    globals_copy = dict(normalized.__globals__)
    globals_copy["kernel_unified_attention"] = namespace["kernel_biv150_mixed_prefill"]
    function = types.FunctionType(
        normalized.__code__,
        globals_copy,
        normalized.__name__,
        normalized.__defaults__,
        normalized.__closure__,
    )
    function.__kwdefaults__ = normalized.__kwdefaults__
    return function, dict(**metadata, mixed_kernel_sha256=digest)


def eligible(bound):
    if not prefill_eligible(bound):
        return False
    q, k, v, out = (bound[name] for name in ("q", "k", "v", "out"))
    cu, lengths, table = (
        bound.get(name) for name in ("cu_seqlens_q", "seqused_k", "block_table")
    )
    if any(value is None for value in (cu, lengths, table)):
        return False
    n = lengths.numel()
    return (
        8 <= n <= 64
        # Favor mixed calls dominated by one prefill; balanced all-prefill calls
        # have no useful decode work and would only pay for two extra launches.
        and (bound["max_seqlen_q"] > q.shape[0] // 2 or q.shape[0] <= 2 * n)
        and q.device.type == "cuda"
        and all(value.device == q.device for value in (k, v, out, cu, lengths, table))
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == out.stride(-1) == 1
        and k.shape == v.shape
        and cu.ndim == lengths.ndim == 1
        and cu.shape[0] == n + 1
        and table.ndim == 2
        and table.shape[0] >= n
        and table.stride(1) == 1
        and all(
            value.dtype in (torch.int32, torch.int64) for value in (cu, lengths, table)
        )
        and cu.stride(0) == lengths.stride(0) == 1
    )


_scratch = {}
_state = {"selected_calls": 0, "fallback_calls": 0, "shapes": set()}


def scratch_for(bound):
    n, q = bound["seqused_k"].numel(), bound["q"]
    tensors = tuple(
        bound.get(name)
        for name in ("softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum")
    )
    if all(value is not None for value in tensors):
        s, m, e = tensors
        if (
            s.ndim == 4
            and m.ndim == e.ndim == 3
            and s.shape[0] >= n
            and s.shape[1:] == (16, 16, 128)
            and m.shape == e.shape == s.shape[:3]
            and all(
                value.device == q.device and value.dtype == torch.float32
                for value in tensors
            )
            and s.stride(-1) == m.stride(-1) == e.stride(-1) == 1
            and m.stride() == e.stride()
        ):
            return tensors
    # One bounded allocation per stream, shared serially across model layers.
    key = (q.device, torch.cuda.current_stream(q.device).cuda_stream)
    if key not in _scratch:
        _scratch[key] = (
            torch.empty((64, 16, 16, 128), device=q.device, dtype=torch.float32),
            torch.empty((64, 16, 16), device=q.device, dtype=torch.float32),
            torch.empty((64, 16, 16), device=q.device, dtype=torch.float32),
        )
    return _scratch[key]


def run_decode(bound):
    q, k, v, out = (bound[name] for name in ("q", "k", "v", "out"))
    cu, lengths, table = (
        bound[name] for name in ("cu_seqlens_q", "seqused_k", "block_table")
    )
    n = lengths.numel()
    segments = min(16, triton.next_power_of_2(triton.cdiv(256, n * 2)))
    s, m, e = scratch_for(bound)
    _gqa_decode_attention[(n, 2, segments)](
        segm_output_ptr=s,
        segm_max_ptr=m,
        segm_expsum_ptr=e,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        seq_lens_ptr=lengths,
        cu_query_ptr=cu,
        block_table_ptr=table,
        softmax_scale_log2e=bound["softmax_scale"] * LOG2E,
        queries_per_kv=8,
        block_table_stride=table.stride(0),
        query_stride_token=q.stride(0),
        query_stride_head=q.stride(1),
        k_stride_block=k.stride(0),
        k_stride_slot=k.stride(1),
        k_stride_head=k.stride(2),
        v_stride_block=v.stride(0),
        v_stride_slot=v.stride(1),
        v_stride_head=v.stride(2),
        segment_stride_token=s.stride(0),
        segment_stride_head=s.stride(1),
        segment_stride_segment=s.stride(2),
        max_stride_token=m.stride(0),
        max_stride_head=m.stride(1),
        HEAD_SIZE=128,
        PAGE_SIZE=16,
        ROWS=8,
        SEGMENTS=segments,
        num_warps=1,
        num_stages=1,
    )
    _reduce_partials[(n, 16)](
        output_ptr=out,
        segm_output_ptr=s,
        segm_max_ptr=m,
        segm_expsum_ptr=e,
        seq_lens_ptr=lengths,
        cu_query_ptr=cu,
        segment_stride_token=s.stride(0),
        segment_stride_head=s.stride(1),
        segment_stride_segment=s.stride(2),
        segment_stride_dim=s.stride(3),
        max_stride_token=m.stride(0),
        max_stride_head=m.stride(1),
        max_stride_segment=m.stride(2),
        output_stride_token=out.stride(0),
        output_stride_head=out.stride(1),
        page_size=16,
        num_segments=segments,
        HEAD_SIZE=128,
        num_warps=4,
    )


def install():
    if os.environ.get("VLLM_FL_BIV150_MIXED_ATTENTION", "1") != "1":
        return {"installed": False, "reason": "disabled"}
    attention = importlib.import_module(
        "vllm.v1.attention.ops.triton_unified_attention"
    )
    backend = importlib.import_module("vllm.v1.attention.backends.triton_attn")
    marker = "_biv150_mixed_attention_installed"
    if getattr(attention, marker, False):
        return {"installed": True, "reason": "already installed"}
    if (
        attention.current_platform.vendor_name != "iluvatar"
        or attention.is_batch_invariant
    ):
        return {"installed": False, "reason": "unsupported platform"}
    original = attention.unified_attention
    if backend.unified_attention is not original or not getattr(
        attention, "_biv150_prefill_normalization_installed", False
    ):
        return {"installed": False, "reason": "expected normalized backend binding"}
    try:
        prefill, metadata = make_prefill(attention)
    except (RuntimeError, OSError, TypeError) as error:
        return {"installed": False, "reason": str(error)}
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        bound = kwargs
        if args:
            try:
                bound = signature.bind(*args, **kwargs).arguments
            except TypeError:
                return original(*args, **kwargs)
        if not eligible(bound):
            _state["fallback_calls"] += 1
            return original(*args, **kwargs)
        _state["selected_calls"] += 1
        shape = (bound["q"].shape[0], bound["max_seqlen_q"], bound["seqused_k"].numel())
        if shape not in _state["shapes"]:
            _state["shapes"].add(shape)
            print("BIV150_MIXED_ATTENTION_SELECTED", shape, flush=True)
        prefill(*args, **kwargs)
        run_decode(bound)
        return None

    attention.unified_attention = backend.unified_attention = wrapper
    setattr(attention, marker, True)
    return dict(
        installed=True,
        scope="BF16 dominant-prefill mixed attention, 8..64 sequences",
        **metadata,
    )
