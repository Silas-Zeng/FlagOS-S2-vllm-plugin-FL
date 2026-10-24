"""Isolated full-vocabulary projection probe; never installs a service hook.

Synthetic sampling rows measure a hypothesis, not end-to-end throughput or
model accuracy. BF16 and compressed INT8 copies coexist only for this A/B test.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import subprocess
from functools import partial
from pathlib import Path

import flag_gems
import torch

from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight


def graph_time(fn):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(4):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 4)
    return statistics.median(samples)


def sampled_reference(x, weight, q, scales, columns):
    # Transfer BEFORE converting to double: this CoreX GPU's FP64 conversion
    # previously produced zeros and invalidated an unrelated diagnostic.
    xc = x.detach().cpu().float()
    wc = weight[columns].detach().cpu().double()
    dense = xc.double() @ wc.t()
    sa = xc.abs().amax(1).clamp_min(1e-6) / 127
    raw = xc / sa[:, None]
    qa = (raw.abs() + 0.5).floor().copysign(raw).clamp(-127, 127)
    qc = q[:, columns].detach().cpu().to(torch.int64)
    acc = qa.to(torch.int64) @ qc
    sw = scales[columns].detach().cpu().float()
    # Match the native epilogue's FP32 multiply grouping and BF16 output.
    quant = (acc.float() * (sa[:, None] * sw[None, :])).bfloat16().double()
    return dense, quant


def main(args):
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Probe source must be committed and clean")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Existing service: refuse concurrent GPU probe")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU occupied: refuse concurrent probe")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise RuntimeError("Never overwrite probe evidence")
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    load_library()
    torch.manual_seed(20261024)
    n, k = 130560, 2048
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    weight.mul_(1 / math.sqrt(k))
    q, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=torch.bfloat16).expand_as(weight)
    columns = torch.linspace(0, n - 1, 32, device="cuda").long()
    result = dict(
        git_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        gpu=torch.cuda.get_device_name(),
        hypothesis="native INT8 full-vocabulary projection vs FlagGems BF16 mm",
        synthetic_weights=True,
        service_installed=False,
        model_accuracy_validated=False,
        shape=[n, k],
        library=str(args.library.resolve()),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        bf16_weight_bytes=weight.numel() * weight.element_size(),
        int8_weight_bytes=q.numel() + scales.numel() * scales.element_size(),
        packing_in_timing=False,
        duplicate_weights_for_diagnosis_only=True,
        records=[],
    )
    for m in (1, 8, 16, 24, 32, 40, 64):
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        old = partial(flag_gems.mm, x, weight.t())
        candidate = partial(
            torch.ops.biv150_native.linear, x, q, scales, marker, None, 17
        )
        row = dict(m=m, n=n, k=k)
        try:
            actual = candidate()
            baseline = old()
            dense, quant = sampled_reference(x, weight, q, scales, columns)
            actual_cpu = actual[:, columns].cpu().double()
            baseline_cpu = baseline[:, columns].cpu().double()
            torch.testing.assert_close(actual_cpu, quant, atol=1e-5, rtol=0.008)
            torch.testing.assert_close(baseline_cpu, dense, atol=0.02, rtol=0.02)
            row.update(
                primitive_passed=True,
                sampled_columns=columns.cpu().tolist(),
                quant_reference_max_abs=(actual_cpu - quant).abs().max().item(),
                candidate_relative_rmse=(
                    (actual_cpu - dense).square().mean().sqrt()
                    / dense.square().mean().sqrt().clamp_min(1e-12)
                ).item(),
                baseline_relative_rmse=(
                    (baseline_cpu - dense).square().mean().sqrt()
                    / dense.square().mean().sqrt().clamp_min(1e-12)
                ).item(),
            )
            rounds = []
            for index in range(3):
                order = [("baseline_us", old), ("candidate_us", candidate)]
                if index % 2:
                    order.reverse()
                timings = {name: graph_time(fn) for name, fn in order}
                timings["speedup"] = timings["baseline_us"] / timings["candidate_us"]
                rounds.append(timings)
            row["rounds"] = rounds
            row["median_speedup"] = statistics.median(r["speedup"] for r in rounds)
            # Verify replay reads current hidden states, including the zero case.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = candidate()
            mutations = []
            for zero in (False, True):
                if zero:
                    x.zero_()
                else:
                    x.normal_()
                graph.replay()
                torch.cuda.synchronize()
                _, expected = sampled_reference(x, weight, q, scales, columns)
                torch.testing.assert_close(
                    output[:, columns].cpu().double(), expected, atol=1e-5, rtol=0.008
                )
                mutations.append(dict(zero=zero, passed=True))
            row["graph_mutations"] = mutations
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
