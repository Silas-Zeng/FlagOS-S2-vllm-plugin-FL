"""BF16 paired-column GEMM/SwiGLU fusion: diagnostic only, no model hook.

Original BF16 weights are loaded directly without repacking or quantization.
The activation preserves BF16 projection rounding and FP32 sigmoid/multiplies.
"""

import argparse
import json
import socket
from pathlib import Path

import flag_gems
import torch
import triton
import triton.language as tl
from biv150_benchmark_utils import graph_time


@triton.jit
def activation(x, out, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row, col = i // N, i % N
    g = tl.load(x + row * (2 * N) + col, i < M * N, other=0).to(tl.float32)
    u = tl.load(x + row * (2 * N) + N + col, i < M * N, other=0).to(tl.float32)
    tl.store(out + i, (g * tl.sigmoid(g)) * u, i < M * N)


@triton.jit
def paired_gate(
    X,
    W,
    Y,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    pair = tl.arange(0, 2 * BN)
    rn = tl.program_id(1) * BN + pair // 2
    wn = rn + (pair % 2) * N
    rk = tl.arange(0, BK)
    part = tl.program_id(2)
    acc = tl.zeros((BM, 2 * BN), tl.float32)
    for base in range(part * BK, K, BK * SPLIT):
        kk = base + rk
        a = tl.load(
            X + rm[:, None] * XS + kk[None, :],
            (rm[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        b = tl.load(
            W + wn[None, :] * K + kk[:, None],
            (rn[None, :] < N) & (kk[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    if SPLIT == 1:
        gate, up = tl.split(tl.reshape(acc, (BM, BN, 2)))
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        col = tl.program_id(1) * BN + tl.arange(0, BN)
        tl.store(
            Y + rm[:, None] * N + col[None, :],
            (gate * tl.sigmoid(gate)) * up,
            (rm[:, None] < M) & (col[None, :] < N),
        )
    else:
        tl.store(
            S + ((part * M + rm[:, None]) * N + rn[None, :]) * 2 + pair[None, :] % 2,
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def reduce_gate(
    S, Y, M: tl.constexpr, N: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.full((B,), 0, tl.float32)
    u = tl.full((B,), 0, tl.float32)
    for part in tl.static_range(SPLIT):
        g += tl.load(S + (part * M * N + i) * 2, i < M * N, other=0)
        u += tl.load(S + (part * M * N + i) * 2 + 1, i < M * N, other=0)
    g = g.to(tl.bfloat16).to(tl.float32)
    u = u.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, (g * tl.sigmoid(g)) * u, i < M * N)


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0, "Service active"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261023)
    result = dict(service_installed=False, original_weights=True, records=[])
    # A compact search; two split-K variants test occupancy without atomics.
    variants = [
        (16, 32, 64, 4, 1),
        (16, 64, 64, 4, 1),
        (32, 32, 64, 4, 1),
        (32, 64, 32, 4, 1),
        (32, 64, 64, 8, 1),
        (64, 32, 64, 8, 1),
        (16, 32, 64, 4, 2),
        (16, 32, 64, 4, 4),
    ]
    for m in [1, 8, 16, 32, 40, 64, 2048]:
        n, k = 6144, 2048
        # Pitched activations ensure the kernel honors input row stride.
        x = torch.randn(m, k + 32, device="cuda", dtype=torch.bfloat16)[:, :k]
        w = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        reference = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        def baseline(x=x, w=w, m=m, n=n, reference=reference):
            z = flag_gems.mm(x, w.t())
            activation[(triton.cdiv(m * n, 256),)](
                z, reference, m, n, B=256, num_warps=4
            )

        baseline()
        gold = reference.clone()
        # Independent CPU FP64 selected rows/channels: no tensor-core reference.
        rows = sorted(set([0, m // 2, m - 1]))
        cols = sorted(set([0, 1, 17, 123, n // 2, n - 1]))
        selected_w = w[cols + [c + n for c in cols]].double().cpu()
        linear = (x[rows].double().cpu() @ selected_w.t()).to(torch.bfloat16).float()
        g, u = linear.chunk(2, dim=1)
        dense = (g * torch.sigmoid(g) * u).to(torch.bfloat16)
        for bm, bn, bk, warps, split in variants:
            y = torch.empty_like(reference)
            scratch = (
                torch.empty((split, m, n, 2), device="cuda", dtype=torch.float32)
                if split > 1
                else y
            )
            record = dict(m=m, n=n, k=k, bm=bm, bn=bn, bk=bk, warps=warps, split=split)

            def candidate(
                x=x,
                w=w,
                y=y,
                scratch=scratch,
                m=m,
                n=n,
                k=k,
                bm=bm,
                bn=bn,
                bk=bk,
                warps=warps,
                split=split,
            ):
                compiled = paired_gate[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
                    x,
                    w,
                    y,
                    scratch,
                    m,
                    n,
                    k,
                    x.stride(0),
                    bm,
                    bn,
                    bk,
                    split,
                    num_warps=warps,
                    num_stages=1,
                )
                if split > 1:
                    reduce_gate[(triton.cdiv(m * n, 256),)](
                        scratch, y, m, n, split, B=256, num_warps=4
                    )
                return compiled

            try:
                kernel = candidate()
                torch.cuda.synchronize()
                torch.testing.assert_close(y, gold, atol=0.005, rtol=0.005)
                torch.testing.assert_close(
                    y[rows][:, cols].cpu(), dense, atol=0.005, rtol=0.005
                )
                rounds = []
                for number in range(3):
                    order = [("baseline", baseline), ("candidate", candidate)]
                    if number % 2:
                        order.reverse()
                    times = {
                        name: graph_time(fn, repeats=9, unroll=4)[0]
                        for name, fn in order
                    }
                    rounds.append(
                        dict(**times, speedup=times["baseline"] / times["candidate"])
                    )
                record.update(
                    passed=True,
                    rounds=rounds,
                    median_speedup=sorted(r["speedup"] for r in rounds)[1],
                    max_abs_diff=(y.float() - gold.float()).abs().max().item(),
                    n_regs=kernel.n_regs,
                    shared=kernel.metadata.shared,
                )
                if m == 8 and (bm, bn, bk, warps, split) == variants[0]:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        candidate()
                    original_x = x.clone()
                    try:
                        x.mul_(0.75)
                        graph.replay()
                        baseline()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(y, reference, atol=0.005, rtol=0.005)
                        record["graph_changed_input_passed"] = True
                    finally:
                        x.copy_(original_x)
            except Exception as error:
                record.update(passed=False, error=str(error)[:1500])
            result["records"].append(record)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(record), flush=True)
        del x, w, reference, gold, y, scratch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
