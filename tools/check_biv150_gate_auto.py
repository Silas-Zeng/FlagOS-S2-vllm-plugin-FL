"""Validate the model's native small-M residual dispatch and unchanged large M."""

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import torch
from bench_biv150_native_gate_residual import cpu_reference


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library)
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight

    load_library()
    torch.manual_seed(20261023)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    wq, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=weight.dtype).expand_as(weight)
    result = dict(
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), records=[]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    auto = torch.ops.biv150_native.gate_silu_auto
    compiled = torch.compile(auto, fullgraph=True)
    for m in [0, 1, 8, 16, 17, 32, 40, 64, 2048]:
        x = torch.randn(m, 2080, device="cuda", dtype=weight.dtype)[:, :2048]
        got = auto(x, wq, scales, marker, 17)
        if 1 <= m <= 16:
            reference = torch.ops.biv150_native.gate_silu_residual(x, wq, scales)
            cols, gold = cpu_reference(x, wq, scales)
            torch.testing.assert_close(
                got[:, cols.to(x.device)].cpu(), gold, atol=0.005, rtol=0.005
            )
        else:
            reference = torch.ops.biv150_native.gate_silu(x, wq, scales, marker, 17)
        torch.testing.assert_close(got, reference, atol=0, rtol=0)
        assert torch.isfinite(got).all().item()
        row = dict(
            m=m, residual_selected=1 <= m <= 16, passed=True, bit_exact_route=True
        )
        if m in (8, 32):
            torch.testing.assert_close(
                compiled(x, wq, scales, marker, 17), reference, atol=0, rtol=0
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = auto(x, wq, scales, marker, 17)
            for factor in [0.0, 0.5, 4.0]:
                x.normal_().mul_(factor)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    replayed, auto(x, wq, scales, marker, 17), atol=0, rtol=0
                )
            row.update(fullgraph=True, graph_changed_inputs=True)
        result["records"].append(row)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)
    # Real BF16 weights must retain the original small-M BF16 fallback.
    x = torch.randn(8, 2048, device="cuda", dtype=weight.dtype)
    torch.testing.assert_close(
        auto(x, wq, scales, weight, 17),
        torch.ops.biv150_native.gate_silu(x, wq, scales, weight, 17),
        atol=0,
        rtol=0,
    )
    result["original_weight_fallback_unchanged"] = True
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
