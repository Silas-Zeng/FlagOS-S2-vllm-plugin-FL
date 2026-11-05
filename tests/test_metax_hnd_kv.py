# SPDX-License-Identifier: Apache-2.0
"""Head-contiguous paged KV: layout guards, vendor parity and graph updates."""
from types import SimpleNamespace

import pytest
import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn as backend
from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.flash_attn_out import (
    flash_attn_kvcache_into, flash_attn_varlen_into,
)


@pytest.mark.parametrize('reason', ['supported', 'disabled', 'explicit', 'fp16',
                                   'hybrid', 'tp', 'quant', 'connector', 'heads',
                                   'noconfig'])
def test_hnd_layout_scope(monkeypatch, reason):
    monkeypatch.setenv('VLLM_FL_METAX_HND_KV', '1')
    monkeypatch.delenv('VLLM_KV_CACHE_LAYOUT', raising=False)
    text = SimpleNamespace(num_attention_heads=16, num_key_value_heads=2)
    model = SimpleNamespace(dtype=torch.bfloat16, is_hybrid=False,
                            hf_text_config=text, get_head_size=lambda: 128)
    config = SimpleNamespace(model_config=model,
                             parallel_config=SimpleNamespace(tensor_parallel_size=1),
                             cache_config=SimpleNamespace(cache_dtype='auto'),
                             kv_transfer_config=None)
    if reason == 'disabled': monkeypatch.setenv('VLLM_FL_METAX_HND_KV', '0')
    if reason == 'explicit': monkeypatch.setenv('VLLM_KV_CACHE_LAYOUT', 'NHD')
    if reason == 'fp16': model.dtype = torch.float16
    if reason == 'hybrid': model.is_hybrid = True
    if reason == 'tp': config.parallel_config.tensor_parallel_size = 2
    if reason == 'quant': config.cache_config.cache_dtype = 'fp8'
    if reason == 'connector': config.kv_transfer_config = object()
    if reason == 'heads': text.num_key_value_heads = 4
    def current_config():
        if reason == 'noconfig':
            raise AssertionError('no active engine config')
        return config
    monkeypatch.setattr(backend, 'get_current_vllm_config', current_config)
    expected = 'HND' if reason == 'supported' else None
    assert backend.MacaFlashAttentionBackend.get_required_kv_cache_layout() == expected
    monkeypatch.setattr(backend, 'get_kv_cache_layout', lambda: 'HND')
    order = backend.MacaFlashAttentionBackend.get_kv_cache_stride_order()
    assert order == ((1, 3, 0, 2, 4) if reason == 'supported' else (0, 1, 3, 2, 4))


def _head_contiguous_kv(k, v):
    # Physical [page, head, K/V, token, dim], logical [K/V,page,token,head,dim].
    raw = torch.empty((k.shape[0], k.shape[2], 2, k.shape[1], k.shape[3]),
                      device=k.device, dtype=k.dtype)
    kv = raw.permute(2, 0, 3, 1, 4)
    kv[0].copy_(k)
    kv[1].copy_(v)
    return kv[0], kv[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='MetaX GPU required')
@pytest.mark.parametrize('prefill', [False, True])
def test_hnd_vendor_graph_updates(prefill):
    torch.manual_seed(20261105)
    batch, pages, block, heads, dim = 3, 257, 16, 2, 128
    k = torch.randn(batch*pages, block, heads, dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kh, vh = _head_contiguous_kv(k, v)
    assert kh.stride() == (8192, 128, 4096, 1)
    table = torch.randperm(batch*pages, device='cuda', dtype=torch.int32).view(batch,pages)
    lengths = torch.tensor([129, 2049, 4097], device='cuda', dtype=torch.int32)
    qbase = torch.randn(2050 if prefill else batch, 2304, device='cuda', dtype=torch.bfloat16)
    q = qbase[:, :2048].view((-1,16,dim) if prefill else (batch,1,16,dim))
    storage = torch.full((q.shape[0]+2,*q.shape[1:]), 19, device='cuda', dtype=q.dtype)
    out = storage[1:-1]
    common = dict(block_table=table, softmax_scale=dim**-.5, causal=True,
                  window_size=(-1,-1), alibi_slopes=None, softcap=0., s_aux=None)
    if prefill:
        # One decode-sized query and two prefill chunks in the same paged call.
        kwargs = dict(common,cu_seqlens_q=torch.tensor([0,1,1025,2050],device='cuda',dtype=torch.int32),
                      cu_seqlens_k=torch.tensor([0,129,2178,6275],device='cuda',dtype=torch.int32),
                      max_seqlen_q=1025,max_seqlen_k=4097)
        run = lambda: flash_attn_varlen_into(q,kh,vh,out,**kwargs)
        ref = lambda: flash_attn_varlen_func(q,k,v,**kwargs)
    else:
        kwargs = dict(common,cache_seqlens=lengths)
        run = lambda: flash_attn_kvcache_into(q,kh,vh,out,**kwargs)
        ref = lambda: flash_attn_with_kvcache(q,k,v,**kwargs)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run()
    for _ in range(3):
        qbase.normal_();k.normal_();v.normal_();kh.copy_(k);vh.copy_(v)
        table.copy_(torch.randperm(batch*pages,device='cuda',dtype=torch.int32).view(batch,pages))
        lengths.sub_(1)
        if prefill:
            kwargs['cu_seqlens_k'][1:].copy_(lengths.cumsum(0))
        graph.replay()
        torch.testing.assert_close(out,ref(),atol=.002,rtol=.03)
        assert torch.all(storage[0]==19) and torch.all(storage[-1]==19)
