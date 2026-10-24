"""Compare unchanged and bounded INT32 LM-head paths, including peak allocation."""

import argparse
import gc
import hashlib
import json
import os
import socket
import subprocess
from functools import partial
from pathlib import Path

import torch

from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    values = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return sorted(values)[2]


def allocation_peak(fn):
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    live_output = output.numel() * output.element_size()
    del output
    return dict(peak_increment_bytes=peak, output_bytes=live_output)


def main(args):
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Existing service")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU occupied")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise RuntimeError("Never overwrite evidence")
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    load_library()
    torch.manual_seed(20261024)
    weight = torch.randn((130560, 2048), device="cuda", dtype=torch.bfloat16) * 0.02
    q, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=torch.bfloat16).expand_as(weight)
    del weight
    cols = torch.linspace(0, 130559, 32, device="cuda").long()
    qc = q[:, cols].cpu().to(torch.int64)
    sw = scales[cols].cpu().float()
    budget = 64 * 1024 * 1024
    result = dict(
        accumulator_budget_bytes=budget,
        service_installed=False,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=[],
    )
    for m in (0, 1, 8, 16, 32, 40, 64, 65, 127, 128, 129, 256, 2048):
        x = torch.randn((m, 2055), device="cuda", dtype=torch.bfloat16)[:, :2048]
        old = partial(torch.ops.biv150_native.linear, x, q, scales, marker, None, 17)
        new = partial(
            torch.ops.biv150_native.linear_bounded,
            x,
            q,
            scales,
            marker,
            None,
            17,
            budget,
        )
        row = dict(m=m, input_stride=list(x.stride()))
        try:
            expected = old()
            actual = new()
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            if m:
                rows = sorted({0, m // 2, m - 1})
                xc = x[rows].cpu().float()
                sa = xc.abs().amax(1).clamp_min(1e-6) / 127
                raw = xc / sa[:, None]
                qa = (raw.abs() + 0.5).floor().copysign(raw).clamp(-127, 127)
                gold = (
                    (qa.to(torch.int64) @ qc).float() * (sa[:, None] * sw)
                ).bfloat16()
                torch.testing.assert_close(
                    actual[rows][:, cols].cpu(), gold, atol=1e-5, rtol=0.008
                )
            del actual, expected
            rounds = []
            for index in range(3):
                calls = [("original_ms", old), ("bounded_ms", new)]
                if index % 2:
                    calls.reverse()
                rounds.append({name: timed(fn) for name, fn in calls})
            row["rounds"] = rounds
            row["allocation"] = {
                "original": allocation_peak(old),
                "bounded": allocation_peak(new),
            }
            if m == 2048:
                assert (
                    row["allocation"]["bounded"]["peak_increment_bytes"]
                    < 0.75 * row["allocation"]["original"]["peak_increment_bytes"]
                )
                row["large_m_peak_reduced"] = True
            try:
                torch.ops.biv150_native.linear_bounded(
                    x, q, scales, marker, None, 17, 1
                )
            except RuntimeError:
                row["invalid_budget_rejected"] = True
            else:
                raise AssertionError("Invalid accumulator budget accepted")
            row["passed"] = True
        except Exception as error:
            row["error"] = repr(error)
        result["records"].append(row)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
