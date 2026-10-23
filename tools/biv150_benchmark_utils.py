import importlib.util
import math
import statistics
import torch

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def inputs(batch, kv, mixed=False):
    torch.manual_seed(20261023 + batch + kv)
    pages = math.ceil(kv / 16)
    q = torch.randn(batch, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * pages, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.randperm(batch * pages, device="cuda", dtype=torch.int32).reshape(batch, pages)
    lengths = torch.full((batch,), kv, device="cuda", dtype=torch.int32)
    if mixed:
        values = [0, 1, 15, 16, 17, 63, 100, max(1, kv - 1)]
        for i in range(min(batch, len(values))):
            lengths[i] = min(kv, values[i])
    scratch = torch.empty(batch, 16, 16, 128, device="cuda", dtype=torch.float32)
    maxima = torch.empty(batch, 16, 16, device="cuda", dtype=torch.float32)
    sums = torch.empty_like(maxima)
    return dict(q=q, k=k, v=v, table=table, lengths=lengths,
                scratch=scratch, maxima=maxima, sums=sums, out=torch.empty_like(q))

def main_call(module, c, segments):
    q, k, v = c["q"], c["k"], c["v"]
    s, m, e = c["scratch"], c["maxima"], c["sums"]
    return module._gqa_decode_attention[(q.shape[0], 2, segments)](
        segm_output_ptr=s, segm_max_ptr=m, segm_expsum_ptr=e,
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v,
        seq_lens_ptr=c["lengths"], block_table_ptr=c["table"],
        softmax_scale_log2e=128 ** -0.5 * module.LOG2E, queries_per_kv=8,
        block_table_stride=c["table"].stride(0),
        query_stride_token=q.stride(0), query_stride_head=q.stride(1),
        k_stride_block=k.stride(0), k_stride_slot=k.stride(1), k_stride_head=k.stride(2),
        v_stride_block=v.stride(0), v_stride_slot=v.stride(1), v_stride_head=v.stride(2),
        segment_stride_token=s.stride(0), segment_stride_head=s.stride(1),
        segment_stride_segment=s.stride(2), max_stride_token=m.stride(0),
        max_stride_head=m.stride(1), HEAD_SIZE=128, PAGE_SIZE=16, ROWS=8,
        SEGMENTS=segments, num_warps=1, num_stages=1)

def reduce_call(module, c, segments, warps):
    s, m, e, o = c["scratch"], c["maxima"], c["sums"], c["out"]
    return module._reduce_partials[(o.shape[0], 16)](
        output_ptr=o, segm_output_ptr=s, segm_max_ptr=m, segm_expsum_ptr=e,
        seq_lens_ptr=c["lengths"], segment_stride_token=s.stride(0),
        segment_stride_head=s.stride(1), segment_stride_segment=s.stride(2),
        segment_stride_dim=s.stride(3), max_stride_token=m.stride(0),
        max_stride_head=m.stride(1), max_stride_segment=m.stride(2),
        output_stride_token=o.stride(0), output_stride_head=o.stride(1),
        page_size=16, num_segments=segments, HEAD_SIZE=128, num_warps=warps)

def full_call(module, c, segments, warps=4):
    main_call(module, c, segments)
    reduce_call(module, c, segments, warps)

def graph_time(fn, repeats=15, unroll=8):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(unroll):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / unroll)
    return statistics.median(values), min(values), max(values)

def check_reference(c):
    """Independent dense FP32 attention for several short, shuffled-page rows."""
    lengths = c["lengths"].cpu().tolist()
    for i, length in enumerate(lengths[:8]):
        if length > 128:
            continue
        got = c["out"][i].float()
        if length == 0:
            assert torch.count_nonzero(got).item() == 0
            continue
        ids = c["table"][i, :math.ceil(length / 16)].long()
        k = c["k"][ids].reshape(-1, 2, 128)[:length].float().repeat_interleave(8, 1)
        v = c["v"][ids].reshape(-1, 2, 128)[:length].float().repeat_interleave(8, 1)
        q = c["q"][i].float()
        weights = torch.softmax(torch.einsum("hd,khd->hk", q, k) * 128 ** -0.5, -1)
        reference = torch.einsum("hk,khd->hd", weights, v).to(torch.bfloat16).float()
        torch.testing.assert_close(got, reference, atol=0.02, rtol=0.02)
