# SPDX-License-Identifier: Apache-2.0

"""MetaX inference attention writing into the model runner's output buffer.

The MACA extension already accepts an output tensor, but its public Python
wrappers always request a fresh allocation. Keep those wrappers as fallbacks
for layouts or head sizes that need their normalization/padding behavior.
"""

import os

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from flash_attn.flash_attn_interface import flash_attn_cuda

_LOG_NATIVE_LAYOUT = os.environ.get("VLLM_FL_METAX_DECODE_LAYOUT_LOG", "0") == "1"
_native_layout_logged = False
_DENSE_PREFILL = os.environ.get("VLLM_FL_METAX_DENSE_PREFILL", "1") == "1"
_LONG_DENSE_PREFILL = os.environ.get("VLLM_FL_METAX_LONG_DENSE_PREFILL", "1") == "1"
_dense_layout_logged = False
_long_dense_layout_logged = False
_SPLIT_PREFILL = os.environ.get("VLLM_FL_METAX_SPLIT_PREFILL", "1") == "1"
_LONG_SPLIT_PREFILL = os.environ.get("VLLM_FL_METAX_LONG_SPLIT_PREFILL", "1") == "1"
_split_prefill_logged = False

def _can_write_directly(q, k, v, out):
    return (
        q.dtype in (torch.float16, torch.bfloat16)
        and q.dtype == k.dtype == v.dtype == out.dtype
        and q.device == k.device == v.device == out.device
        and q.shape == out.shape
        and q.shape[-1] % 8 == 0
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and out.is_contiguous()
        and not (q.requires_grad or k.requires_grad or v.requires_grad)
    )


def flash_attn_varlen_into(
    q, k, v, out, *, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, block_table, softmax_scale,
    causal, window_size, alibi_slopes, softcap, s_aux,
    cache_length_is_exact=False,
):
    """Prefill into the output buffer, optionally gathering BF16 pages first."""
    global _dense_layout_logged, _long_dense_layout_logged
    if not _can_write_directly(q, k, v, out):
        result = flash_attn_varlen_func(
            q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k, block_table=block_table,
            softmax_scale=softmax_scale, causal=causal,
            window_size=window_size, alibi_slopes=alibi_slopes,
            softcap=softcap, s_aux=s_aux,
        )
        out.copy_(result)
        return out
    if (_DENSE_PREFILL and q.ndim == 3 and q.shape[1:] == (16, 128)
            and (max_seqlen_k <= 4096 or _LONG_DENSE_PREFILL)
            and 1024 <= q.shape[0] <= 2048
            and q.shape[0] == max_seqlen_q <= max_seqlen_k
            and cu_seqlens_q is not None and cu_seqlens_q.shape == (2,)
            and causal and window_size == (-1, -1) and softcap == 0.0
            and alibi_slopes is None and s_aux is None):
        from flag_gems.runtime.backend._metax.ops.prefill_cache import gather_bf16_prefill_cache

        dense = gather_bf16_prefill_cache(
            k, v, block_table, cu_seqlens_k, max_seqlen_k,
            exact_length=cache_length_is_exact,
        )
        if dense is not None:
            k, v = dense
            block_table = None
            if cache_length_is_exact:
                from flag_gems.runtime.backend._metax.ops.bf16_prefill import try_bf16_prefill

                if try_bf16_prefill(q, k, v, out, softmax_scale):
                    return out
            if _LOG_NATIVE_LAYOUT and not _dense_layout_logged:
                print("MetaX dense prefill layout:", {
                    "q_shape": tuple(q.shape), "q_stride": q.stride(),
                    "k_shape": tuple(k.shape), "k_stride": k.stride(),
                    "max_seqlen_q": max_seqlen_q, "max_seqlen_k": max_seqlen_k,
                }, flush=True)
                _dense_layout_logged = True
            if (_LOG_NATIVE_LAYOUT and max_seqlen_k > 4096
                    and not _long_dense_layout_logged):
                print("MetaX long dense prefill selected:", {
                    "query_length": q.shape[0], "key_length": max_seqlen_k,
                    "k_stride": k.stride(),
                }, flush=True)
                _long_dense_layout_logged = True
    flash_attn_cuda.varlen_fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k,
        None, None, block_table, alibi_slopes,
        max_seqlen_q, max_seqlen_k, 0.0, softmax_scale,
        False, causal, window_size[0], window_size[1], softcap,
        False, None, s_aux, False,
    )
    return out


def try_split_prefill_into(
    q, k, v, out, *, query_lengths, key_lengths, cache_seqlens,
    cu_seqlens_k, block_table, softmax_scale, causal, window_size,
    alibi_slopes, softcap, s_aux,
):
    """Separate highly uneven C500 prefill requests with different histories.

    A tiny context query benefits from split-K; a large context query benefits
    from the existing dense BF16 gather. Longer histories also amortize the
    extra request launch for uneven batches. Nearly balanced query lengths
    retain the batched kernel because splitting can reduce their throughput.
    """
    global _split_prefill_logged
    if (not _SPLIT_PREFILL or len(query_lengths) != 2 or len(key_lengths) != 2
            or not _can_write_directly(q, k, v, out)
            or q.dtype != torch.bfloat16 or q.ndim != 3
            or q.shape[1:] != (16, 128) or not 1024 <= q.shape[0] <= 2048
            or sum(query_lengths) != q.shape[0] or min(query_lengths) < 2
            or not causal or window_size != (-1, -1) or softcap != 0.0
            or alibi_slopes is not None or s_aux is not None
            or cache_seqlens is None or cache_seqlens.shape != (2,)
            or cache_seqlens.dtype != torch.int32 or cache_seqlens.device != q.device
            or not cache_seqlens.is_contiguous()
            or cu_seqlens_k is None or cu_seqlens_k.shape != (3,)
            or block_table is None or block_table.ndim != 2 or block_table.shape[0] != 2):
        return False
    history = [i for i in range(2) if key_lengths[i] > query_lengths[i]]
    if len(history) != 1:
        return False
    context = history[0]
    fresh = 1 - context
    long_context = (
        _LONG_SPLIT_PREFILL and 8192 <= key_lengths[context] <= 16384
    )
    if ((key_lengths[context] != 4096 and not long_context)
            or key_lengths[fresh] != query_lengths[fresh]
            or (not long_context and not (
                query_lengths[context] <= 64 or query_lengths[context] >= 1408
            ))):
        return False
    large = 0 if query_lengths[0] >= query_lengths[1] else 1
    small = 1 - large
    if query_lengths[large] < 1408:
        return False
    from flag_gems.runtime.backend._metax.ops.prefill_cache import gather_bf16_prefill_cache

    dense = gather_bf16_prefill_cache(
        k, v, block_table[large:large + 1], cu_seqlens_k[large:large + 2],
        key_lengths[large], exact_length=True,
    )
    if dense is None:
        return False
    start = 0 if large == 0 else query_lengths[0]
    end = start + query_lengths[large]
    dk, dv = dense
    from flag_gems.runtime.backend._metax.ops.bf16_prefill import try_bf16_prefill

    if not try_bf16_prefill(q[start:end], dk, dv, out[start:end], softmax_scale):
        flash_attn_cuda.fwd_kvcache(
            q[start:end].unsqueeze(0), dk.unsqueeze(0), dv.unsqueeze(0),
            None, None, cache_seqlens[large:large + 1], None, None, None, None,
            None, None, out[start:end].unsqueeze(0), softmax_scale, True,
            -1, -1, 0.0, True, 1, None,
        )
    start = 0 if small == 0 else query_lengths[0]
    end = start + query_lengths[small]
    flash_attn_cuda.fwd_kvcache(
        q[start:end].unsqueeze(0), k, v, None, None,
        cache_seqlens[small:small + 1], None, None, None, None,
        block_table[small:small + 1], None, out[start:end].unsqueeze(0),
        softmax_scale, True, -1, -1, 0.0, True, 0, None,
    )
    if _LOG_NATIVE_LAYOUT and not _split_prefill_logged:
        print("MetaX split prefill selected:", {
            "query_lengths": query_lengths, "key_lengths": key_lengths,
            "q_stride": q.stride(), "dense_request": large,
        }, flush=True)
        _split_prefill_logged = True
    return True


def flash_attn_kvcache_into(
    q, k_cache, v_cache, out, *, block_table, cache_seqlens,
    softmax_scale, causal, window_size, alibi_slopes, softcap, s_aux,
):
    """Paged decode without an intermediate attention output and copy."""
    global _native_layout_logged
    if (
        not _can_write_directly(q, k_cache, v_cache, out)
        or q.shape[1] != 1
        or window_size != (-1, -1)
        or alibi_slopes is not None
        or s_aux is not None
    ):
        result = flash_attn_with_kvcache(
            q=q, k_cache=k_cache, v_cache=v_cache,
            block_table=block_table, cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale, causal=causal,
            window_size=window_size, alibi_slopes=alibi_slopes,
            softcap=softcap, s_aux=s_aux,
        )
        out.copy_(result)
        return out
    # MACA swaps the single-query GQA groups into the sequence dimension
    # before it validates an explicit output. Mirror that view, without
    # copying, so each head still writes to the runner's original layout.
    if q.dtype == torch.bfloat16 and softcap == 0.0:
        from flag_gems.runtime.backend._metax.ops.bf16_attention import try_bf16_decode

        selected = try_bf16_decode(q, k_cache, v_cache, out, block_table,
                                   cache_seqlens, softmax_scale)
        if _LOG_NATIVE_LAYOUT and not _native_layout_logged and 33 <= q.shape[0] <= 64:
            print("MetaX native decode layout:", {
                "selected": selected, "q_shape": tuple(q.shape),
                "q_stride": q.stride(), "kv_shape": tuple(k_cache.shape),
                "k_stride": k_cache.stride(), "v_stride": v_cache.stride(),
                "table_shape": tuple(block_table.shape),
                "table_stride": block_table.stride(),
                "lengths_shape": tuple(cache_seqlens.shape),
                "lengths_stride": cache_seqlens.stride(),
            }, flush=True)
            _native_layout_logged = True
        if selected:
            return out
    kernel_out = out
    kv_heads = k_cache.shape[2]
    if q.shape[2] > kv_heads:
        kernel_out = out.view(
            q.shape[0], kv_heads, q.shape[2] // kv_heads, q.shape[3]
        ).transpose(1, 2)
    flash_attn_cuda.fwd_kvcache(
        q, k_cache, v_cache, None, None, cache_seqlens,
        None, None, None, None, block_table, alibi_slopes, kernel_out,
        softmax_scale, causal, window_size[0], window_size[1], softcap,
        True, 0, s_aux,
    )
    return out
