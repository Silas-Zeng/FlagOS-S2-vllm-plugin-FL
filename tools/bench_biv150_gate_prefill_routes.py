"""Diagnose native gate versus transient BF16 dequantization, without a hook."""

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
import triton
import triton.language as tl

from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight


@triton.jit
def dequant(Q, S, W, SIZE: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    q = tl.load(Q + i, i < SIZE, other=0).to(tl.float32)
    s = tl.load(S + i % N, i < SIZE, other=0)
    tl.store(W + i, q * s, i < SIZE)


@triton.jit
def activate(Z, Y, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    r, c = i // N, i % N
    g = tl.load(Z + r * (2 * N) + c, i < M * N, other=0).to(tl.float32)
    u = tl.load(Z + r * (2 * N) + N + c, i < M * N, other=0).to(tl.float32)
    tl.store(Y + i, g * tl.sigmoid(g) * u, i < M * N)


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
    values = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / 4)
    return statistics.median(values)


def main(args):
    source = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Dirty source")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Existing service; refuse concurrent probe")
    if "No running processes found" not in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ):
        raise RuntimeError("GPU occupied")
    if args.out.exists():
        raise RuntimeError("Never overwrite evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    load_library()
    torch.manual_seed(20261024)
    n, k = 6144, 2048
    weight = torch.randn((2 * n, k), device="cuda", dtype=torch.bfloat16)
    weight.mul_(1 / math.sqrt(k))
    packed = weight.t().contiguous()
    q, scales = quantize_weight(weight)
    cached_dequant = torch.empty_like(q, dtype=torch.bfloat16)
    dequant[(triton.cdiv(q.numel(), 1024),)](
        q, scales, cached_dequant, q.numel(), 2 * n, 1024
    )
    torch.testing.assert_close(
        cached_dequant, (q.float() * scales[None, :]).bfloat16(), atol=0, rtol=0
    )
    marker = torch.zeros((), device="cuda", dtype=torch.bfloat16).expand_as(weight)
    columns = torch.linspace(0, n - 1, 32, device="cuda").long()
    # CPU transfer before FP64 conversion is required on this CoreX device.
    gate_cpu = cached_dequant[:, columns].cpu().double()
    up_cpu = cached_dequant[:, columns + n].cpu().double()
    result = dict(
        git_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        synthetic_weights=True,
        service_installed=False,
        model_accuracy_validated=False,
        cached_dequant_diagnostic_only=True,
        dynamic_dequant_in_timing=True,
        graph_pool_memory_is_not_serving_peak=True,
        records=[],
    )
    for m in (1, 8, 16, 31, 32, 40, 63, 64, 256, 1024, 2048):
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        scratch = {}

        def bf16(w, x=x, m=m):
            z = flag_gems.mm(x, w)
            y = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            activate[(triton.cdiv(m * n, 1024),)](z, y, m, n, 1024)
            return y

        def dynamic(bf16=bf16, scratch=scratch):
            w = torch.empty_like(q, dtype=torch.bfloat16)
            scratch["weight"] = w
            dequant[(triton.cdiv(q.numel(), 1024),)](
                q, scales, w, q.numel(), 2 * n, 1024
            )
            return bf16(w)

        routes = {
            "original_bf16": partial(bf16, weight.t()),
            "packed_bf16": partial(bf16, packed),
            "cached_dequant": partial(bf16, cached_dequant),
            "native_int8": partial(
                torch.ops.biv150_native.gate_silu_auto, x, q, scales, marker, 17
            ),
            "dynamic_dequant": dynamic,
        }
        row = dict(m=m, n=n, k=k)
        try:
            baseline = routes["original_bf16"]()
            native = routes["native_int8"]()
            cached = routes["cached_dequant"]()
            actual = dynamic()
            torch.testing.assert_close(actual, cached, atol=0, rtol=0)
            xc = x[: min(4, m)].cpu().double()
            g = (xc @ gate_cpu).bfloat16().float()
            u = (xc @ up_cpu).bfloat16().float()
            expected = (g * torch.sigmoid(g) * u).bfloat16()
            torch.testing.assert_close(
                actual[: len(xc), columns].cpu(), expected, atol=0.02, rtol=0.02
            )
            sampled = baseline[:, columns].float()
            denom = sampled.square().mean().sqrt().clamp_min(1e-12)
            row["native_relative_rmse_vs_original"] = (
                (native[:, columns].float() - sampled).square().mean().sqrt() / denom
            ).item()
            row["dequant_relative_rmse_vs_original"] = (
                (actual[:, columns].float() - sampled).square().mean().sqrt() / denom
            ).item()
            rounds = []
            for index in range(3):
                order = list(routes.items())
                if index % 2:
                    order.reverse()
                timings = {name: graph_time(fn) for name, fn in order}
                rounds.append(timings)
            row["rounds_us"] = rounds
            row["median_us"] = {
                name: statistics.median(t[name] for t in rounds) for name in routes
            }
            if m in (8, 31):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = dynamic()
                for zero in (True, False):
                    x.zero_() if zero else x.normal_()
                    scratch["weight"].fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        output, bf16(cached_dequant), atol=0, rtol=0
                    )
                row["changed_input_poisoned_weight_graph"] = True
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
