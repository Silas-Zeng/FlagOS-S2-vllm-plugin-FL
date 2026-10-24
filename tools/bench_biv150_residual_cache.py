"""Bit-exact residual quantization cache probe; no model dispatch change."""

import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
from bench_biv150_native_gate_residual import cpu_reference
from biv150_benchmark_utils import graph_time


def main(args):
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 9031)) != 0, "Service running"
    assert "No running processes found" in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ), "GPU busy"
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight

    load_library()
    torch.manual_seed(20261024)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    q, scales = quantize_weight(weight)
    result = dict(
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        records=[],
        service_installed=False,
    )
    for rows in (1, 2, 4, 8, 15, 16):
        for pitch in (2048, 2080):
            x = torch.randn(rows, pitch, device="cuda", dtype=weight.dtype)[:, :2048]
            old = partial(torch.ops.biv150_native.gate_silu_residual, x, q, scales)
            for threads in (64, 128, 256):
                new = partial(
                    torch.ops.biv150_native.gate_residual_cached_probe,
                    x,
                    q,
                    scales,
                    threads,
                )
                row = dict(m=rows, pitch=pitch, threads=threads)
                try:
                    for pattern in ("zero", "tiny", "normal", "large", "outlier"):
                        x.normal_()
                        if pattern == "zero":
                            x.zero_()
                        elif pattern == "tiny":
                            x.mul_(1e-9)
                        elif pattern == "large":
                            x.mul_(100)
                        elif pattern == "outlier":
                            x[:, 0] = 32
                        old_q, old_s = torch.ops.biv150_native.quantize_residual_probe(
                            x, 0
                        )
                        new_q, new_s = torch.ops.biv150_native.quantize_residual_probe(
                            x, threads
                        )
                        torch.testing.assert_close(new_q, old_q, atol=0, rtol=0)
                        torch.testing.assert_close(new_s, old_s, atol=0, rtol=0)
                        got = new()
                        torch.testing.assert_close(got, old(), atol=0, rtol=0)
                        cols, gold = cpu_reference(x, q, scales)
                        torch.testing.assert_close(
                            got[:, cols.to(x.device)].cpu(),
                            gold,
                            atol=0.005,
                            rtol=0.005,
                        )
                    x.normal_()
                    rounds = []
                    for iteration in range(3):
                        calls = [("baseline_ms", old), ("candidate_ms", new)]
                        if iteration % 2:
                            calls.reverse()
                        times = {
                            name: graph_time(fn, repeats=11, unroll=8)[0]
                            for name, fn in calls
                        }
                        times["speedup"] = times["baseline_ms"] / times["candidate_ms"]
                        rounds.append(times)
                    quant = {
                        str(width): graph_time(
                            partial(
                                torch.ops.biv150_native.quantize_residual_probe,
                                x,
                                width,
                            ),
                            repeats=11,
                            unroll=8,
                        )[0]
                        for width in (0, threads)
                    }
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = new()
                    for factor in (0.0, 0.5, 4.0):
                        x.normal_().mul_(factor)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(output, old(), atol=0, rtol=0)
                    row.update(
                        passed=True,
                        bit_exact=True,
                        sampled_cpu_reference=True,
                        changed_input_graph=True,
                        rounds=rounds,
                        median_speedup=statistics.median(t["speedup"] for t in rounds),
                        quantization_ms=quant,
                    )
                    del graph, output
                except Exception as error:
                    row.update(passed=False, error=repr(error))
                result["records"].append(row)
                (args.out / "result.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
    if len(result["records"]) != 36 or not all(x["passed"] for x in result["records"]):
        raise RuntimeError("Incomplete numeric gates; no service result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
