"""BI-V150 GQA decode attention kernel for MiniCPM5-2B, installed as an override.

Why this exists: the generic vLLM kernel has to serve every shape, so on a pure
decode step it pays for structure this model does not use. ``BLOCK_M`` is 16 rows
because ``BLOCK_M / queries_per_kv`` must cover two query positions, but a decode
step has one position, so half of every Q, S, P and accumulator tile is masked
work. Grid dimension 0 is ``tokens // BLOCK_Q + num_seqs``, i.e. deliberately more
programs than there are (sequence, kv head) pairs, and each program then searches
``query_start_len`` to find its sequence. Every tile gathers a block-table entry
per token even though a decode tile of ``TILE_SIZE == block_size`` is exactly one
page, and every tile recomputes a full two-dimensional causal mask even though the
query is the last position, so causality degenerates to ``pos < seq_len``.

This kernel assumes the decode case outright: one query token per sequence, the
row block is exactly the ``queries_per_kv`` heads of one kv head, a step is a whole
number of pages with a page index loaded per page, the main loop is mask-free with
a separately clamped tail, and the softmax runs on exp2 with the scale folded in.
Partials use vLLM's own scratch layout, so ``reduce_segments`` finalizes them
unchanged; the maxima are converted back to natural-log units because that
reduction takes them with ``tl.exp``.

Measured against the generic kernel at the pinned one-warp launch, on this device:

    batch 64, kv 4096   0.648 ms vs 0.948 ms   1.46x   414 vs 283 GB/s
    batch 16, kv 16384  0.536 ms vs 0.981 ms   1.83x   501 vs 274 GB/s
    max abs difference against the generic kernel: 4.9e-4 (bf16 output)

Off unless VLLM_FL_BIV150_GQA_DECODE=1, and it only takes over calls that the
generic path would have run as 3D decode; anything else falls straight through.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
LN2 = tl.constexpr(0.6931471805599453)

# Fail-closed identity check: the gate below reads call-site kwargs by name, and
# the reduction assumes the audited scratch layout. Both hold only for these
# sources; the hashes are the same ones the merged launch-policy patches pin.
_AUDITED_SOURCES = {
    "vllm.v1.attention.ops.triton_unified_attention":
        "54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f",
    "vllm.v1.attention.ops.triton_attention_helpers":
        "59d9d34bbda988436593cfd9bcb05015d17db8ee85c50f6697e45f987acbf0b1",
}

# Every name the gate inspects, asserted against the audited signature once at
# install so an upstream rename turns the optimization off instead of letting a
# permissive default pass.
_GATE_KEYS = (
    "causal", "max_seqlen_q", "seqused_k", "seq_threshold_3D",
    "num_par_softmax_segments", "softmax_segm_output", "softmax_segm_max",
    "softmax_segm_expsum", "alibi_slopes", "sinks", "qq_bias",
    "mm_prefix_range", "output_scale", "k_scale_cache", "v_scale_cache",
    "softcap", "use_alibi_sqrt", "window_size", "chunk_lookback",
    "use_td", "kv_quant_mode",
)

# The audited triton_attn call passes 31 of the 32 parameters and omits qq_bias,
# which the callee default supplies; requiring it per call would reject every real
# decode step (measured: the whole probe fell back to the generic kernel).
_RUNTIME_KEYS = tuple(name for name in _GATE_KEYS if name != "qq_bias")

_ENABLE_ENV = "VLLM_FL_BIV150_GQA_DECODE"
_TARGET_ENV = "VLLM_FL_BIV150_GQA_TARGET_PROGRAMS"
_INSTALLED_ATTR = "_biv150_gqa_decode_installed"
TARGET_PROGRAMS = int(os.environ.get(_TARGET_ENV, "256"))

# Audited model geometry: MiniCPM5-2B, 16 query heads, 2 kv heads, head 128,
# paged block size 16, bf16.
_QUERY_HEADS = 16
_KV_HEADS = 2
_HEAD_SIZE = 128
_PAGE_SIZE = 16


@triton.jit
def _gqa_decode_attention(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
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
    segm_max = tl.load(max_base + offs_seg * max_stride_segment,
                       mask=seg_valid, other=float("-inf"))
    # A CUDA-graph padded row can carry seq_len == 0, where no segment is live:
    # folding -inf through exp would give NaN, so the empty case reduces to 0.
    overall_max = tl.where(active > 0, tl.max(segm_max, 0), 0.0)

    expsum_base = (
        segm_expsum_ptr + token_idx * max_stride_token + head_idx * max_stride_head
    )
    segm_expsum = tl.load(expsum_base + offs_seg * max_stride_segment,
                          mask=seg_valid, other=0.0)
    weights = tl.exp(tl.where(seg_valid, segm_max, 0.0) - overall_max)
    overall_expsum = tl.sum(segm_expsum * weights, 0)

    out_base = (
        segm_output_ptr
        + token_idx * segment_stride_token
        + head_idx * segment_stride_head
    )
    partial_offsets = (
        offs_seg[:, None] * segment_stride_segment + offs_dim[None, :] * segment_stride_dim
    )
    partials = tl.load(out_base + partial_offsets, mask=seg_valid[:, None], other=0.0)
    acc = tl.sum(partials * weights[:, None], 0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc / overall_expsum)

    tl.store(
        output_ptr + token_idx * output_stride_token + head_idx * output_stride_head
        + offs_dim,
        acc,
    )


_state = {"calls": 0, "fallbacks": 0, "reduce": None, "signature": None, "views": {}}


def _pick_segments(num_seqs, kv_heads, builder_segments):
    """Choose the split-K width for this step instead of inheriting the constant.

    The generic kernel always uses the builder's segment count, but measurement on
    this device shows the optimum is set by parallelism, not by sequence length:
    with one warp per program the kernel is fastest when the grid holds roughly
    ``target_programs`` programs and slows down on either side of it. At batch 64
    that is 2 segments (0.537 ms) against 0.648 at 16, while at batch 16 it is 8
    segments (0.531 ms) and 4 segments collapse to 0.90 ms because the grid is only
    128 programs. Fewer segments also mean less scratch traffic and a shorter
    reduction, so the target is deliberately modest.

    The result never exceeds the builder's count, because the partial buffers are
    allocated for it; taking a prefix view is safe since the reduction only reads
    as many segments as it is given.
    """
    # Measured packed-KV wave boundary; preserve every other shipping shape.
    if (num_seqs, kv_heads, builder_segments) == (40, 2, 16):
        return 16
    programs_per_segment = max(1, num_seqs * kv_heads)
    need = -(-TARGET_PROGRAMS // programs_per_segment)
    segments = 1
    while segments < need:
        segments <<= 1
    return max(1, min(segments, builder_segments))


def _eligible(q, k, v, out, kwargs):
    """True only for calls the generic path would have run as 3D decode."""
    causal = kwargs.get("causal")
    if any(key not in kwargs for key in _RUNTIME_KEYS) or (
        q.dtype is not torch.bfloat16
        or q.ndim != 3
        or tuple(q.shape[1:]) != (_QUERY_HEADS, _HEAD_SIZE)
        or k.ndim != 4
        or tuple(k.shape[1:]) != (_PAGE_SIZE, _KV_HEADS, _HEAD_SIZE)
        or k.shape[0] != v.shape[0]
        or kwargs.get("max_seqlen_q") != 1
        # The backend passes a plain bool here; a tensor means per-sequence
        # causality, which this kernel does not implement.
        or isinstance(causal, torch.Tensor)
        or bool(causal) is not True
        or q.shape[0] != kwargs["seqused_k"].shape[0]
        or kwargs.get("seq_threshold_3D") is None
        or kwargs.get("num_par_softmax_segments") is None
        or q.shape[0] > kwargs["seq_threshold_3D"]
        or kwargs.get("softmax_segm_output") is None
        or kwargs.get("softmax_segm_max") is None
        or kwargs.get("softmax_segm_expsum") is None
        or tuple(k.stride()) != tuple(v.stride())
    ):
        return False
    # q_descale / k_descale / v_descale are deliberately not checked: vLLM
    # expands them to per-(seq, kv head) tensors even on the bf16 path, and the
    # generic kernel only reads them under a quantized mode, which the two
    # conditions below already exclude.
    for name in (
        "alibi_slopes",
        "sinks",
        "qq_bias",
        "mm_prefix_range",
        "output_scale",
        "k_scale_cache",
        "v_scale_cache",
    ):
        if kwargs.get(name) is not None:
            return False
    return (
        kwargs.get("softcap", 0) == 0
        and kwargs.get("use_alibi_sqrt", False) is False
        and kwargs.get("window_size", (-1, -1)) == (-1, -1)
        and kwargs.get("chunk_lookback", -1) == -1
        and kwargs.get("use_td", False) is False
        and str(kwargs.get("kv_quant_mode")) in ("0", "KVQuantMode.NONE")
    )


def _make_wrapper(attention):
    reduce_segments = attention.reduce_segments
    original = attention.unified_attention

    @functools.wraps(original)
    def patched(*args, **kwargs):
        bound = kwargs
        if args:
            # The backend calls with keywords; bind positionals defensively so the
            # gate never reads a half-filled kwargs dict.
            names = list(_state["signature"].parameters)[: len(args)]
            bound = dict(zip(names, args))
            bound.update(kwargs)
        q = bound.get("q")
        try:
            eligible = q is not None and _eligible(
                q, bound["k"], bound["v"], bound["out"], bound
            )
        except Exception as error:  # a changed call site falls back, never raises
            eligible = False
            if _state["fallbacks"] == 0:
                print("BIV150_GQA_DECODE_GATE_ERROR", repr(error)[:200], flush=True)
        if not eligible:
            if _state["fallbacks"] == 0:
                print(
                    "BIV150_GQA_DECODE_FIRST_FALLBACK "
                    f"max_seqlen_q={bound.get('max_seqlen_q')} "
                    f"tokens={None if q is None else q.shape[0]} "
                    f"seqs={None if bound.get('seqused_k') is None else bound['seqused_k'].shape[0]}",
                    flush=True,
                )
            _state["fallbacks"] += 1
            return original(**bound)
        _state["calls"] += 1
        seq_lens = bound["seqused_k"]
        block_table = bound["block_table"]
        num_seqs = seq_lens.shape[0]
        builder_segments = bound["num_par_softmax_segments"]
        segments = _pick_segments(num_seqs, _KV_HEADS, builder_segments)
        if segments >= builder_segments:
            segm_output = bound["softmax_segm_output"]
            segm_max = bound["softmax_segm_max"]
            segm_expsum = bound["softmax_segm_expsum"]
        else:
            # The builder allocated the partials for its own segment count, so a
            # narrower step is a prefix view of the same memory. Views are cached
            # per buffer and width because this runs for every layer of every step.
            key = (bound["softmax_segm_output"].data_ptr(),
                   bound["softmax_segm_output"].stride(0), segments)
            views = _state["views"].get(key)
            if views is None:
                views = (
                    bound["softmax_segm_output"].narrow(2, 0, segments),
                    bound["softmax_segm_max"].narrow(2, 0, segments),
                    bound["softmax_segm_expsum"].narrow(2, 0, segments),
                )
                _state["views"][key] = views
            segm_output, segm_max, segm_expsum = views
        if _state["calls"] == 1:
            print(
                "BIV150_GQA_DECODE_FIRST_WIDTH "
                f"seqs={num_seqs} builder_segments={builder_segments} "
                f"segments={segments} target_programs={TARGET_PROGRAMS}",
                flush=True,
            )
        _gqa_decode_attention[(num_seqs, _KV_HEADS, segments)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=bound["k"],
            value_cache_ptr=bound["v"],
            seq_lens_ptr=seq_lens,
            block_table_ptr=block_table,
            softmax_scale_log2e=bound["softmax_scale"] * LOG2E,
            queries_per_kv=_QUERY_HEADS // _KV_HEADS,
            block_table_stride=block_table.stride(0),
            query_stride_token=q.stride(0),
            query_stride_head=q.stride(1),
            k_stride_block=bound["k"].stride(0),
            k_stride_slot=bound["k"].stride(1),
            k_stride_head=bound["k"].stride(2),
            v_stride_block=bound["v"].stride(0),
            v_stride_slot=bound["v"].stride(1),
            v_stride_head=bound["v"].stride(2),
            segment_stride_token=segm_output.stride(0),
            segment_stride_head=segm_output.stride(1),
            segment_stride_segment=segm_output.stride(2),
            max_stride_token=segm_max.stride(0),
            max_stride_head=segm_max.stride(1),
            HEAD_SIZE=_HEAD_SIZE,
            PAGE_SIZE=_PAGE_SIZE,
            ROWS=_QUERY_HEADS // _KV_HEADS,
            SEGMENTS=segments,
            num_warps=1,
            num_stages=1,
        )
        out = bound["out"]
        _reduce_partials[(num_seqs, _QUERY_HEADS)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seq_lens,
            segment_stride_token=segm_output.stride(0),
            segment_stride_head=segm_output.stride(1),
            segment_stride_segment=segm_output.stride(2),
            segment_stride_dim=segm_output.stride(3),
            max_stride_token=segm_max.stride(0),
            max_stride_head=segm_max.stride(1),
            max_stride_segment=segm_max.stride(2),
            output_stride_token=out.stride(0),
            output_stride_head=out.stride(1),
            page_size=_PAGE_SIZE,
            num_segments=segments,
            HEAD_SIZE=_HEAD_SIZE,
            num_warps=4,
        )
        return None

    return patched, original


def install():
    """Wrap the generic attention entry point.

    On by default, because the official serve command may not carry a switch that
    enables an optimization; the environment variable only ever turns it back off
    for an A/B arm.
    """
    if os.environ.get(_ENABLE_ENV, "1") != "1":
        return {"installed": False, "reason": _ENABLE_ENV + " disabled"}

    attention = importlib.import_module(
        "vllm.v1.attention.ops.triton_unified_attention"
    )
    if getattr(attention, _INSTALLED_ATTR, False):
        return {"installed": True, "scope": "already installed"}
    if attention.current_platform.vendor_name != "iluvatar":
        return {"installed": False, "reason": "non-iluvatar platform"}
    from vllm import envs as vllm_envs

    if vllm_envs.VLLM_BATCH_INVARIANT:
        return {"installed": False,
                "reason": "batch-invariant mode forces the non-3D path"}

    missing = [name for name in _GATE_KEYS
               if name not in inspect.signature(attention.unified_attention).parameters]
    if missing:
        return {"installed": False,
                "reason": "gate keys absent from signature: " + ",".join(missing)}

    audited = {}
    for name, expected in _AUDITED_SOURCES.items():
        module = importlib.import_module(name)
        actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if actual != expected:
            return {"installed": False,
                    "reason": "unaudited attention source " + name}
        audited[name] = actual[:12]

    backend = importlib.import_module("vllm.v1.attention.backends.triton_attn")
    wrapper, original = _make_wrapper(attention)
    _state["signature"] = inspect.signature(original)
    attention.unified_attention = wrapper
    # The backend module holds its own reference, and the prefill policy rebinds
    # it after importing, so replace both or the decode calls never reach here.
    if backend.unified_attention is original:
        backend.unified_attention = wrapper
    setattr(attention, _INSTALLED_ATTR, True)
    return {
        "installed": True,
        "scope": "BF16 MiniCPM pure-decode 3D only",
        "rows": _QUERY_HEADS // _KV_HEADS,
        "page_size": _PAGE_SIZE,
        "backend_rebound": backend.unified_attention is wrapper,
    }


def stats():
    return dict(calls=_state["calls"], fallbacks=_state["fallbacks"])
