# SPDX-License-Identifier: Apache-2.0

"""MetaX inference attention writing into the model runner's output buffer.

The MACA extension already accepts an output tensor, but its public Python
wrappers always request a fresh allocation. Keep those wrappers as fallbacks
for layouts or head sizes that need their normalization/padding behavior.
"""

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from flash_attn.flash_attn_interface import flash_attn_cuda


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
):
    """Paged prefill with the same kernel/arguments and an explicit output."""
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
    flash_attn_cuda.varlen_fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k,
        None, None, block_table, alibi_slopes,
        max_seqlen_q, max_seqlen_k, 0.0, softmax_scale,
        False, causal, window_size[0], window_size[1], softcap,
        False, None, s_aux, False,
    )
    return out


def flash_attn_kvcache_into(
    q, k_cache, v_cache, out, *, block_table, cache_seqlens,
    softmax_scale, causal, window_size, alibi_slopes, softcap, s_aux,
):
    """Paged decode without an intermediate attention output and copy."""
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
