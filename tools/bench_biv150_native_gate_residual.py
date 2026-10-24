"""Validate residual activation quantization in the old 32-row padded GEMM.

Operator-only probe; not installed in the model and not a throughput score.
Reference uses independent CPU INT64 dot products at sampled output columns.
"""

import argparse
import hashlib
import json
import socket
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import graph_time


def quantized_parts(x):
    f = x.float()
    first_scale = f.abs().amax(1).clamp_min(1e-6) / 127
    normalized = f / first_scale[:, None]
    first = torch.where(
        normalized >= 0, (normalized + 0.5).floor(), (normalized - 0.5).ceil()
    ).clamp(-127, 127)
    second_scale = first_scale / 254
    residual = (f - first * first_scale[:, None]) / second_scale[:, None]
    second = torch.where(
        residual >= 0, (residual + 0.5).floor(), (residual - 0.5).ceil()
    ).clamp(-127, 127)
    return first, second, first_scale, second_scale


def cpu_reference(x, wq, scales):
    cols = torch.tensor([0, 1, 31, 127, 1023, 6143], dtype=torch.long)
    both = torch.cat((cols, cols + 6144))
    first, second, s1, s2 = quantized_parts(x)
    weights = wq[:, both.to(wq.device)].cpu().long()
    a = first.cpu().long() @ weights
    b = second.cpu().long() @ weights
    linear = (
        a.double() * s1.cpu().double()[:, None]
        + b.double() * s2.cpu().double()[:, None]
    ) * scales[both.to(scales.device)].cpu().double()
    gate, up = linear.float().to(torch.bfloat16).float().chunk(2, -1)
    return cols, (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0, (
            "Service active; refuse probe"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.ops.load_library(str(args.library))
    torch.manual_seed(20261023)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        service_installed=False,
        gemm_rows_baseline=32,
        gemm_rows_candidate=32,
        records=[],
    )
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    scales = weight.float().abs().amax(1).clamp_min(1e-6) / 127
    wq = (
        (weight.float() / scales[:, None])
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
        .t()
        .contiguous()
    )
    marker = torch.zeros((), device="cuda", dtype=weight.dtype).expand_as(weight)
    for m in [1, 2, 4, 8, 15, 16]:
        for pattern in ["zero", "normal", "large", "outlier"]:
            x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
            if pattern == "zero":
                x.zero_()
            if pattern == "large":
                x.mul_(4)
            if pattern == "outlier":
                x[:, 0] = 32
            old = partial(torch.ops.biv150_native.gate_silu, x, wq, scales, marker, 17)
            new = partial(torch.ops.biv150_native.gate_silu_residual, x, wq, scales)
            row = dict(m=m, pattern=pattern)
            try:
                before = old()
                after = new()
                cols, gold = cpu_reference(x, wq, scales)
                torch.testing.assert_close(
                    after[:, cols.to(x.device)].cpu(), gold, atol=0.005, rtol=0.005
                )
                linear = (x.float() @ wq.float()) * scales
                gate, up = linear.to(torch.bfloat16).float().chunk(2, -1)
                dense = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
                old_rmse = (
                    (before.float() - dense.float()).square().mean().sqrt().item()
                )
                new_rmse = (after.float() - dense.float()).square().mean().sqrt().item()
                assert new_rmse <= old_rmse + 1e-7, (old_rmse, new_rmse)
                assert torch.isfinite(after).all().item()
                rounds = []
                for index in range(3):
                    order = [("baseline", old), ("candidate", new)]
                    if index % 2:
                        order.reverse()
                    times = {
                        name: graph_time(fn, repeats=11, unroll=4)[0]
                        for name, fn in order
                    }
                    rounds.append(
                        dict(**times, speedup=times["baseline"] / times["candidate"])
                    )
                row.update(
                    passed=True,
                    old_rmse=old_rmse,
                    new_rmse=new_rmse,
                    sampled_cpu_columns=cols.tolist(),
                    rounds=rounds,
                    median_speedup=sorted(r["speedup"] for r in rounds)[1],
                )
            except Exception as error:
                row["error"] = str(error)[:2000]
            result["records"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)

    @torch.library.register_fake("biv150_native::gate_silu_residual")
    def fake(x, wq, scales):
        return torch.empty(
            (x.shape[0], wq.shape[1] // 2), device=x.device, dtype=x.dtype
        )

    x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16)
    fn = partial(torch.ops.biv150_native.gate_silu_residual, x, wq, scales)
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    saved = x.clone()
    for factor in [0.0, 1.0, 4.0]:
        x.copy_(saved * factor)
        graph.replay()
        torch.cuda.synchronize()
        cols, gold = cpu_reference(x, wq, scales)
        torch.testing.assert_close(
            captured[:, cols.to(x.device)].cpu(), gold, atol=0.005, rtol=0.005
        )
    result["changed_input_graph_replay"] = True
    compiled = torch.compile(torch.ops.biv150_native.gate_silu_residual, fullgraph=True)
    torch.testing.assert_close(compiled(x, wq, scales), fn(), atol=0, rtol=0)
    result["fullgraph_fake_tensor"] = True
    try:
        torch.ops.biv150_native.gate_silu_residual(
            torch.empty(17, 2048, device="cuda", dtype=x.dtype), wq, scales
        )
    except RuntimeError:
        result["unsupported_rows_rejected"] = True
    else:
        raise AssertionError("M>16 must reject")
    args.out.write_text(json.dumps(result, indent=2))
    assert all(r.get("passed") for r in result["records"]), (
        "Independent numerical check failed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
