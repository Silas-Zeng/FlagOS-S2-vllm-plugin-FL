"""Probe GPU-selected split widths with real interleaved NHD cache and graph replay."""

import argparse
import hashlib
import json
import socket
from functools import partial
from pathlib import Path

import torch
from bench_biv150_int8_kv_decode import dense_reference
from biv150_benchmark_utils import full_call, graph_time, inputs, load_module


def generate(original):
    old = "weights = tl.exp(tl.where(seg_valid, segm_max, 0.0) - overall_max)"
    new = 'weights = tl.exp(tl.where(seg_valid, segm_max - overall_max, float("-inf")))'
    assert original.count(old) == 1
    source = original.replace(old, new)
    # Every decision reads the live sequence-length tensor inside the GPU kernel.
    # Both passes must use the same expression, including graph-padded empty rows.
    for capacity, page in [("SEGMENTS", "PAGE_SIZE"), ("num_segments", "page_size")]:
        expression = (
            f"tl.minimum({capacity}, tl.where(seq_len <= 512, 16, "
            "tl.where(seq_len <= 2048, 32, tl.where(seq_len <= 8192, 64, 128))))"
        )
        if capacity == "SEGMENTS":
            before = "    steps_per_segment = tl.cdiv(seq_len, SEGMENTS * PAGE_SIZE)"
            after = f"    effective = {expression}\n    steps_per_segment = tl.cdiv(seq_len, effective * PAGE_SIZE)"
        else:
            before = "    tiles_per_segment = tl.maximum(tl.cdiv(seq_len, num_segments * page_size), 1)"
            after = f"    effective = {expression}\n    tiles_per_segment = tl.maximum(tl.cdiv(seq_len, effective * page_size), 1)"
        assert source.count(before) == 1
        source = source.replace(before, after)
    return source


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0, "Service running"
    torch.backends.cuda.matmul.allow_tf32 = False
    original = args.source.read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    path = args.out.parent / "adaptive-kernel.py"
    source = generate(original)
    path.write_text(source)
    adaptive = load_module(path, "adaptive_real_stride")
    baseline = load_module(args.source, "baseline_real_stride")
    result = dict(
        source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        service_installed=False,
        records=[],
    )
    for batch in [1, 2, 4, 8]:
        for length in [100, 1024, 4096, 16384, 32768]:
            c = inputs(batch, length)
            # Match vLLM's [blocks,2,page,kv_heads,head] allocation + unbind.
            packed = torch.stack([c["k"], c["v"]], dim=1)
            c["k"], c["v"] = packed.unbind(1)
            width = {1: 128, 2: 64, 4: 32, 8: 16}[batch]
            cc = dict(
                c,
                scratch=torch.empty(batch, 16, width, 128, device="cuda"),
                maxima=torch.empty(batch, 16, width, device="cuda"),
                sums=torch.empty(batch, 16, width, device="cuda"),
                out=torch.empty_like(c["out"]),
            )
            row = dict(
                batch=batch,
                length=length,
                capacity=width,
                k_stride=c["k"].stride(),
                v_stride=c["v"].stride(),
                checks=[],
            )
            try:
                full_call(baseline, c, 16)
                reference = c["out"].clone()
                full_call(adaptive, cc, width)
                torch.testing.assert_close(cc["out"], reference, atol=0.005, rtol=0.005)
                for i in sorted({0, batch - 1}):
                    gold = dense_reference(c, c["k"], c["v"], i).to(torch.bfloat16)
                    torch.testing.assert_close(
                        cc["out"][i], gold, atol=0.005, rtol=0.005
                    )
                before = graph_time(
                    partial(full_call, baseline, c, 16), repeats=15, unroll=4
                )[0]
                after = graph_time(
                    partial(full_call, adaptive, cc, width), repeats=15, unroll=4
                )[0]
                row.update(
                    baseline_ms=before, adaptive_ms=after, speedup=before / after
                )
                # Capture once, then change the live tensor across policy thresholds.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    full_call(adaptive, cc, width)
                for n in sorted(
                    {
                        0,
                        1,
                        min(length, 511),
                        min(length, 513),
                        min(length, 2049),
                        length,
                    }
                ):
                    c["lengths"].fill_(n)
                    if batch > 1:
                        c["lengths"][0] = 0
                    cc["out"].fill_(float("nan"))
                    cc["scratch"].fill_(float("nan"))
                    cc["maxima"].fill_(float("nan"))
                    cc["sums"].fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    for i in sorted({0, batch - 1}):
                        gold = dense_reference(c, c["k"], c["v"], i).to(torch.bfloat16)
                        torch.testing.assert_close(
                            cc["out"][i], gold, atol=0.005, rtol=0.005
                        )
                    row["checks"].append(dict(length=n, graph_replay=True, passed=True))
                row["passed"] = True
            except Exception as error:
                row["error"] = str(error)[:1500]
            result["records"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
            del c, cc, packed
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
