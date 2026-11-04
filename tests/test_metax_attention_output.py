# SPDX-License-Identifier: Apache-2.0

"""Attention output reuse with vendor equality and CUDA Graph replay."""

import pytest
import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.flash_attn_out import (
    flash_attn_kvcache_into,
    flash_attn_varlen_into,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires MetaX GPU"
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "heads,kv_heads,qlen,window,strided_out",
    [
        (16, 2, 1, (-1, -1), False),
        (4, 4, 1, (-1, -1), False),
        (16, 2, 3, (-1, -1), False),
        (16, 2, 1, (64, 0), False),
        (16, 2, 1, (-1, -1), True),
    ],
)
def test_kvcache_output_reuse_graph(dtype, heads, kv_heads, qlen, window, strided_out):
    batch, dim, block, blocks = 3, 128, 16, 65
    q = torch.randn(batch, qlen, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch * blocks, block, kv_heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    table = torch.randperm(batch * blocks, device="cuda", dtype=torch.int32).view(batch, blocks)
    lengths = torch.tensor([129, 513, 1025], device="cuda", dtype=torch.int32)
    storage = torch.full(
        (batch + 2, qlen, heads, dim * (2 if strided_out else 1)),
        17, device="cuda", dtype=dtype,
    )
    out = storage[1:-1, :, :, ::2] if strided_out else storage[1:-1]
    kwargs = dict(
        block_table=table, cache_seqlens=lengths, softmax_scale=dim ** -0.5,
        causal=True, window_size=window, alibi_slopes=None, softcap=0.0, s_aux=None,
    )
    for _ in range(3):
        flash_attn_kvcache_into(q, k, v, out, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_kvcache_into(q, k, v, out, **kwargs)
    for _ in range(2):
        q.normal_()
        graph.replay()
        reference = flash_attn_with_kvcache(q, k, v, **kwargs)
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_varlen_output_reuse_graph(dtype):
    dim, block, blocks = 128, 16, 64
    q = torch.randn(387, 16, dim, device="cuda", dtype=dtype)
    k = torch.randn(3 * blocks, block, 2, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    table = torch.randperm(3 * blocks, device="cuda", dtype=torch.int32).view(3, blocks)
    storage = torch.full((389, 16, dim), 17, device="cuda", dtype=dtype)
    out = storage[1:-1]
    kwargs = dict(
        cu_seqlens_q=torch.tensor([0, 2, 131, 387], device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 513, 1313, 2337], device="cuda", dtype=torch.int32),
        max_seqlen_q=256, max_seqlen_k=1024, block_table=table,
        softmax_scale=dim ** -0.5, causal=True, window_size=(-1, -1),
        alibi_slopes=None, softcap=0.0, s_aux=None,
    )
    for _ in range(3):
        flash_attn_varlen_into(q, k, v, out, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_varlen_into(q, k, v, out, **kwargs)
    for _ in range(2):
        q.normal_()
        graph.replay()
        reference = flash_attn_varlen_func(q, k, v, **kwargs)
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        assert torch.all(storage[0] == 17) and torch.all(storage[-1] == 17)
