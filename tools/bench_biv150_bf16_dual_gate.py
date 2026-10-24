"""Operator-only dual-accumulator BF16 GEMM/SwiGLU layout experiment.

Compares original and transposed identical BF16 weights. Packing is setup only;
both weight copies are kept solely for A/B diagnosis, not installed in a model.
"""

import argparse
import hashlib
import json
import socket
import statistics
from functools import partial
from pathlib import Path

import flag_gems
import torch
import triton
import triton.language as tl


@triton.jit
def activate(Z, Y, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    r, c = i // N, i % N
    g = tl.load(Z + r * (2 * N) + c, i < M * N, other=0).to(tl.float32)
    u = tl.load(Z + r * (2 * N) + N + c, i < M * N, other=0).to(tl.float32)
    tl.store(Y + i, (g * tl.sigmoid(g)) * u, i < M * N)


@triton.jit
def dual_gate(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    WK: tl.constexpr,
    WN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + kk
        a = tl.load(
            X + r[:, None] * XS + k[None, :],
            (r[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        off = k[:, None] * WK + c[None, :] * WN
        mask = (k[:, None] < K) & (c[None, :] < N)
        g = tl.load(W + off, mask, other=0)
        u = tl.load(W + off + N * WN, mask, other=0)
        gate = tl.dot(a, g, gate)
        up = tl.dot(a, u, up)
    # Keep the service's BF16 projection rounding before FP32 activation.
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    tl.store(
        Y + r[:, None] * N + c[None, :],
        (gate * tl.sigmoid(gate)) * up,
        (r[:, None] < M) & (c[None, :] < N),
    )


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
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / 4)
    return statistics.median(values)


def cpu_reference(x, weight, rows, cols, n):
    # GPU FP64 conversion is not a valid independent reference on this stack.
    a = x[rows].cpu().double()
    w = weight[cols + [c + n for c in cols]].cpu().double()
    z = (a @ w.t()).to(torch.bfloat16).float()
    g, u = z.chunk(2, dim=1)
    return (g * torch.sigmoid(g) * u).to(torch.bfloat16)


def main(args):
    with socket.socket() as guard:
        if guard.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Service occupied; refuse GPU probe")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261024)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        service_installed=False,
        original_bf16_values=True,
        packing_in_timing=False,
        duplicate_weights_for_diagnosis=True,
        records=[],
    )
    variants = [
        (16, 32, 64, 4, 1),
        (16, 64, 64, 4, 1),
        (32, 64, 32, 4, 1),
        (32, 64, 64, 8, 1),
        (32, 128, 32, 8, 1),
        (64, 32, 64, 8, 1),
        (64, 64, 32, 8, 1),
        (64, 64, 64, 8, 2),
        (64, 128, 32, 8, 1),
        (128, 32, 32, 8, 1),
    ]
    for m in [1, 8, 16, 32, 40, 64, 2048]:
        n, k = 6144, 2048
        x = torch.randn(m, k + 32, device="cuda", dtype=torch.bfloat16)[:, :k]
        weight = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        packed = weight.t().contiguous()
        output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        def baseline(w=weight.t(), x=x, output=output, m=m, n=n):
            z = flag_gems.mm(x, w)
            activate[(triton.cdiv(m * n, 256),)](z, output, m, n, 256, num_warps=4)

        baseline()
        gold = output.clone()
        rows = sorted({0, m // 2, m - 1})
        cols = [0, 1, 17, 123, n // 2, n - 1]
        dense = cpu_reference(x, weight, rows, cols, n)
        # Validate the comparator independently too; a bad baseline invalidates M.
        torch.testing.assert_close(
            gold[rows][:, cols].cpu(), dense, atol=0.005, rtol=0.005
        )
        baseline(packed)
        torch.testing.assert_close(output, gold, atol=0.005, rtol=0.005)
        for layout, w in [("original_NK", weight.t()), ("packed_KN", packed)]:
            for bm, bn, bk, warps, stages in variants:
                y = torch.empty_like(output)
                row = dict(
                    m=m,
                    n=n,
                    k=k,
                    layout=layout,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    warps=warps,
                    stages=stages,
                )

                def candidate(
                    x=x,
                    w=w,
                    y=y,
                    m=m,
                    n=n,
                    k=k,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    warps=warps,
                    stages=stages,
                ):
                    return dual_gate[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                        x,
                        w,
                        y,
                        m,
                        n,
                        k,
                        x.stride(0),
                        w.stride(0),
                        w.stride(1),
                        bm,
                        bn,
                        bk,
                        num_warps=warps,
                        num_stages=stages,
                    )

                try:
                    compiled = candidate()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(y, gold, atol=0.005, rtol=0.005)
                    torch.testing.assert_close(
                        y[rows][:, cols].cpu(), dense, atol=0.005, rtol=0.005
                    )
                    rounds = []
                    for index in range(3):
                        calls = [
                            ("original_baseline_ms", baseline),
                            ("packed_baseline_ms", partial(baseline, packed)),
                            ("candidate_ms", candidate),
                        ]
                        if index % 2:
                            calls.reverse()
                        times = {name: graph_time(fn) for name, fn in calls}
                        times["speedup_vs_original"] = (
                            times["original_baseline_ms"] / times["candidate_ms"]
                        )
                        times["speedup_vs_best_layout"] = (
                            min(
                                times["original_baseline_ms"],
                                times["packed_baseline_ms"],
                            )
                            / times["candidate_ms"]
                        )
                        rounds.append(times)
                    row.update(
                        passed=True,
                        rounds=rounds,
                        median_speedup=statistics.median(
                            t["speedup_vs_original"] for t in rounds
                        ),
                        median_vs_best_layout=statistics.median(
                            t["speedup_vs_best_layout"] for t in rounds
                        ),
                        max_abs_diff=(y.float() - gold.float()).abs().max().item(),
                        n_regs=compiled.n_regs,
                        shared=compiled.metadata.shared,
                    )
                    if m in (8, 32) and (bm, bn, bk, warps, stages) == variants[0]:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            candidate()
                        saved = x.clone()
                        try:
                            for factor in [0.0, 0.5, 4.0]:
                                x.normal_().mul_(factor)
                                graph.replay()
                                baseline()
                                torch.cuda.synchronize()
                                torch.testing.assert_close(
                                    y, output, atol=0.005, rtol=0.005
                                )
                                ref = cpu_reference(x, weight, rows, cols, n)
                                torch.testing.assert_close(
                                    y[rows][:, cols].cpu(), ref, atol=0.005, rtol=0.005
                                )
                            row["graph_changed_input_passed"] = True
                        finally:
                            x.copy_(saved)
                except Exception as error:
                    row.update(passed=False, error=str(error)[:1800])
                result["records"].append(row)
                args.out.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
        del x, weight, packed, output, gold, y
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
