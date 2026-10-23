"""W8A16 operator probe: keep INT8 weights and avoid quantizing small-M inputs.

No serving hook. Precision reference is FP32 multiplication with the SAME
quantized weights, not a claim of equivalence to original BF16 model weights.
"""

import argparse
import hashlib
import json
import socket
from functools import partial
from pathlib import Path

import flag_gems
import torch
import triton
import triton.language as tl
from biv150_benchmark_utils import graph_time


@triton.jit
def kernel(
    X,
    W,
    S,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GATE: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    if GATE:
        upper = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        k = step * BK + kk
        a = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            0,
        )
        b = tl.load(
            W + k[:, None] * N + cols[None, :],
            (k[:, None] < K) & (cols[None, :] < (N // 2 if GATE else N)),
            0,
        ).to(a.dtype)
        acc = tl.dot(a, b, acc)
        if GATE:
            u = tl.load(
                W + k[:, None] * N + (cols[None, :] + N // 2),
                (k[:, None] < K) & (cols[None, :] < N // 2),
                0,
            ).to(a.dtype)
            upper = tl.dot(a, u, upper)
    valid = cols < (N // 2 if GATE else N)
    scale = tl.load(S + cols, valid, 0)
    value = (acc * scale[None, :]).to(tl.bfloat16)
    if GATE:
        other_scale = tl.load(S + cols + N // 2, valid, 0)
        other = (upper * other_scale[None, :]).to(tl.bfloat16).to(tl.float32)
        wide = value.to(tl.float32)
        value = (wide / (1 + tl.exp(-wide)) * other).to(tl.bfloat16)
    width: tl.constexpr = N // 2 if GATE else N
    tl.store(
        O + rows[:, None] * width + cols[None, :],
        value,
        (rows[:, None] < M) & valid[None, :],
    )


def invoke(x, wq, scales, gate, config):
    m, k = x.shape
    n = wq.shape[1]
    bm, bn, bk, warps = config
    out = torch.empty((m, n // 2 if gate else n), device=x.device, dtype=x.dtype)
    kernel[(triton.cdiv(m, bm), triton.cdiv(n // 2 if gate else n, bn))](
        x, wq, scales, out, m, n, k, bm, bn, bk, gate, num_warps=warps, num_stages=1
    )
    return out


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0, "Service active"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.ops.load_library(str(args.library))
    torch.manual_seed(20261023)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        service_installed=False,
        records=[],
    )
    for n, k, gate in [(2560, 2048, False), (2048, 2048, False), (12288, 2048, True)]:
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        scale = weight.float().abs().amax(1).clamp_min(1e-6) / 127
        wq = (
            (weight.float() / scale[:, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
            .t()
            .contiguous()
        )
        marker = torch.zeros((), device="cuda", dtype=weight.dtype).expand_as(weight)
        for m in [1, 8, 16, 32, 40, 64, 2048]:
            x = torch.randn(m, k, device="cuda", dtype=weight.dtype)
            original = flag_gems.mm(x, weight.t())
            if gate:
                native = partial(
                    torch.ops.biv150_native.gate_silu, x, wq, scale, marker, 17
                )
            else:
                native = partial(
                    torch.ops.biv150_native.linear, x, wq, scale, marker, None, 17
                )
            native_result = native()
            gold = ((x.float() @ wq.float()) * scale).to(torch.bfloat16)
            if gate:
                a, b = gold.float().chunk(2, -1)
                gold = (torch.nn.functional.silu(a) * b).to(torch.bfloat16)
                a, b = original.float().chunk(2, -1)
                original = (torch.nn.functional.silu(a) * b).to(torch.bfloat16)
            native_ms = graph_time(native, repeats=11, unroll=4)[0]
            for config in [(16, 64, 64, 4), (32, 128, 64, 4), (32, 128, 128, 4)]:
                row = dict(m=m, n=n, k=k, gate=gate, config=config, native_ms=native_ms)
                try:
                    fn = partial(invoke, x, wq, scale, gate, config)
                    out = fn()
                    torch.testing.assert_close(out, gold, atol=0.01, rtol=0.01)
                    ms = graph_time(fn, repeats=11, unroll=4)[0]
                    row.update(
                        passed=True,
                        candidate_ms=ms,
                        speedup=native_ms / ms,
                        max_abs_diff_fp32=(out.float() - gold.float())
                        .abs()
                        .max()
                        .item(),
                        rmse_vs_original=(out.float() - original.float())
                        .square()
                        .mean()
                        .sqrt()
                        .item(),
                        native_rmse_vs_original=(
                            native_result.float() - original.float()
                        )
                        .square()
                        .mean()
                        .sqrt()
                        .item(),
                    )
                except Exception as error:
                    row["error"] = str(error)[:1500]
                result["records"].append(row)
                args.out.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
            del x, original, gold, native_result
            torch.cuda.empty_cache()
        del weight, scale, wq, marker
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
