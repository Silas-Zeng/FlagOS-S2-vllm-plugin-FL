"""Full gate plus isolated quantization, cached row loads; no service installation."""

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
from biv150_benchmark_utils import graph_time


def main(args):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Service already running")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU busy")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight

    load_library()
    torch.manual_seed(20261024)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    q, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=torch.bfloat16).expand_as(weight)
    result = dict(
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        records=[],
        service_installed=False,
    )
    for rows in (17, 32, 40, 64, 256, 2048):
        for pitch in (2048, 2080):
            x = torch.randn(rows, pitch, device="cuda", dtype=torch.bfloat16)[:, :2048]
            original = partial(
                torch.ops.biv150_native.gate_silu, x, q, scales, marker, 17
            )
            for threads in (64, 128, 256):
                row = dict(m=rows, input_stride=list(x.stride()), threads=threads)
                candidate = partial(
                    torch.ops.biv150_native.gate_silu_cached_probe,
                    x,
                    q,
                    scales,
                    marker,
                    threads,
                )
                try:
                    for factor in (0.0, 1e-9, 1.0, 100.0):
                        x.normal_().mul_(factor)
                        old_q, old_s = torch.ops.biv150_native.quantize_rows_probe(x, 0)
                        new_q, new_s = torch.ops.biv150_native.quantize_rows_probe(
                            x, threads
                        )
                        torch.testing.assert_close(old_q, new_q, atol=0, rtol=0)
                        torch.testing.assert_close(old_s, new_s, atol=0, rtol=0)
                        host = x.cpu().float()
                        host_scale = host.abs().amax(1).clamp_min(1e-6) / 127
                        raw = host / host_scale[:, None]
                        gold_q = (
                            raw.abs()
                            .add(0.5)
                            .floor()
                            .copysign(raw)
                            .clamp(-127, 127)
                            .to(torch.int8)
                        )
                        torch.testing.assert_close(
                            new_q[:rows].cpu(), gold_q, atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            new_s[:rows].cpu(), host_scale, atol=0, rtol=0
                        )
                        torch.testing.assert_close(
                            candidate(), original(), atol=0, rtol=0
                        )
                    x.normal_()
                    rounds = []
                    for iteration in range(3):
                        calls = [("baseline_ms", original), ("candidate_ms", candidate)]
                        if iteration % 2:
                            calls.reverse()
                        times = {
                            name: graph_time(fn, repeats=11, unroll=8)[0]
                            for name, fn in calls
                        }
                        times["speedup"] = times["baseline_ms"] / times["candidate_ms"]
                        rounds.append(times)
                    quant = {}
                    for width in (0, threads):
                        quant[str(width)] = graph_time(
                            partial(
                                torch.ops.biv150_native.quantize_rows_probe, x, width
                            ),
                            repeats=11,
                            unroll=8,
                        )[0]
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = candidate()
                    for factor in (0.0, 0.5, 4.0):
                        x.normal_().mul_(factor)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(output, original(), atol=0, rtol=0)
                    row.update(
                        passed=True,
                        rounds=rounds,
                        median_speedup=statistics.median(t["speedup"] for t in rounds),
                        quantization_ms=quant,
                        changed_input_graph=True,
                        quantization_bit_exact=True,
                        gate_bit_exact=True,
                    )
                    del graph, output
                except Exception as error:
                    row.update(passed=False, error=repr(error))
                result["records"].append(row)
                (args.out / "result.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
    if len(result["records"]) != 36 or not all(
        x.get("passed") for x in result["records"]
    ):
        raise RuntimeError("Not all numeric cases passed; no service result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
