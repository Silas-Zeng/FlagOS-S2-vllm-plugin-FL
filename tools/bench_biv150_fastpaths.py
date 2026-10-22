"""Compare address arithmetic and reduction launches on the shipping GQA kernel.

Uses the same tensors for every variant, shuffled physical pages, unequal sequence
lengths and empty graph-padding rows. Timings replay a warmed CUDA graph. This is
an operator experiment, not the official end-to-end benchmark.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variants(source, out):
    text = source.read_text()
    a = text.index("@triton.jit\ndef _gqa_decode_attention")
    b = text.index("@triton.jit\ndef _reduce_partials", a)
    body = text[a:b]
    kinds = {"baseline": body}
    kinds["const64"] = body.replace(": tl.int64", ": tl.constexpr")
    kinds["index32"] = body.replace("tl.int64", "tl.int32")
    kinds["const32"] = kinds["const64"].replace("tl.int64", "tl.int32")
    result = {}
    for name, replacement in kinds.items():
        path = out / f"variant_{name}.py"
        path.write_text(text[:a] + replacement + text[b:])
        result[name] = load_module(path, f"biv150_{name}")
    return result


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


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    mods = variants(args.source, args.out)
    rows = []
    shapes = [(64, 4096), (64, 4608), (32, 16384), (16, 16896), (8, 32768), (1, 100)]
    for batch, kv in shapes:
        segments = mods["baseline"]._pick_segments(batch, 2, 16)
        c = inputs(batch, kv)
        full_call(mods["baseline"], c, segments)
        torch.cuda.synchronize()
        reference = c["out"].clone()
        # Warm the same working set before the comparisons.
        for _ in range(30):
            full_call(mods["baseline"], c, segments)
        baseline = None
        for name, module in mods.items():
            for warps in ([4, 1, 2] if name == "const32" else [4]):
                full_call(module, c, segments, warps)
                torch.cuda.synchronize()
                bit_exact = torch.equal(c["out"], reference)
                max_diff = (c["out"].float() - reference.float()).abs().max().item()
                if warps == 4:
                    assert bit_exact, (name, batch, kv, warps, max_diff)
                else:
                    torch.testing.assert_close(c["out"], reference, atol=0.005, rtol=0.005)
                median, low, high = graph_time(lambda: full_call(module, c, segments, warps))
                if name == "baseline":
                    baseline = median
                row = dict(batch=batch, kv=kv, segments=segments, variant=name,
                           reduce_warps=warps, median_ms=median, min_ms=low, max_ms=high,
                           speedup=baseline / median, bit_exact=bit_exact, max_abs_diff=max_diff)
                rows.append(row)
                print(json.dumps(row), flush=True)
                with (args.out / "results.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        # Address and reduction checks include short tails, shuffled pages and graph padding.
        c = inputs(batch, kv, mixed=True)
        full_call(mods["baseline"], c, segments)
        torch.cuda.synchronize()
        reference = c["out"].clone()
        check_reference(c)
        for name, module in mods.items():
            for warps in (1, 2, 4):
                full_call(module, c, segments, warps)
                torch.cuda.synchronize()
                if warps == 4:
                    assert torch.equal(c["out"], reference), ("mixed", name, batch, kv, warps)
                else:
                    torch.testing.assert_close(c["out"], reference, atol=0.005, rtol=0.005)
        print(json.dumps(dict(correctness="passed", batch=batch, kv=kv)), flush=True)
        del c, reference
        torch.cuda.empty_cache()
    with (args.out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
