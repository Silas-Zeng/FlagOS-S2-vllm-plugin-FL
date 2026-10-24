"""Validate the actual auto route across all residual rows before model loading."""

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

import torch
from audit_biv150_residual_reference import cpu_quantize, cpu_output


def main(args):
    with socket.socket() as guard:
        assert guard.connect_ex(("127.0.0.1", 9031)) != 0
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight

    load_library()
    torch.manual_seed(20261024)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    q, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=weight.dtype).expand_as(weight)
    cols = torch.tensor([0, 1, 31, 63, 127, 511, 1023, 2047, 4095, 6143])
    both = torch.cat((cols, cols + 6144)).to("cuda")
    cpu_weights, cpu_scales = q[:, both].cpu(), scales[both].cpu()
    auto = torch.ops.biv150_native.gate_silu_auto
    old = torch.ops.biv150_native.gate_silu_residual
    raw = torch.ops.biv150_native.quantize_residual_probe
    result = dict(library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  records=[], unchanged_routes=[], patterns_per_record=5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for m in range(1, 17):
        for pitch in (2048, 2080):
            x = torch.empty(m, pitch, device="cuda", dtype=weight.dtype)[:, :2048]
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
                original = old(x, q, scales)
                selected = auto(x, q, scales, marker, 17)
                torch.testing.assert_close(selected, original, atol=0, rtol=0)
                cached_q, cached_s = raw(x, 256)
                old_q, old_s = raw(x, 0)
                torch.testing.assert_close(cached_q, old_q, atol=0, rtol=0)
                torch.testing.assert_close(cached_s, old_s, atol=0, rtol=0)
                first, second, s1, s2 = cpu_quantize(x.cpu())
                torch.testing.assert_close(cached_q[:m].cpu(), first, atol=0, rtol=0)
                torch.testing.assert_close(cached_q[16:16+m].cpu(), second, atol=0, rtol=0)
                torch.testing.assert_close(cached_s[:m].cpu(), s1, atol=0, rtol=0)
                torch.testing.assert_close(cached_s[16:16+m].cpu(), s2, atol=0, rtol=0)
                assert not cached_q[m:16].count_nonzero().item()
                assert not cached_q[16+m:].count_nonzero().item()
                gold = cpu_output(first, second, s1, s2, cpu_weights, cpu_scales)
                for output in (original, selected):
                    torch.testing.assert_close(output[:, cols.to("cuda")].cpu(), gold,
                                               atol=0.005, rtol=0.005)
            # Every active row count/layout: capture, then change inputs in place.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                replayed = auto(x, q, scales, marker, 17)
            for factor in (0.0, 0.5, 4.0):
                x.normal_().mul_(factor)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(replayed, old(x, q, scales), atol=0, rtol=0)
            row = dict(m=m, pitch=pitch, patterns=5, raw_bit_exact=True,
                       full_gate_bit_exact=True, sampled_cpu_reference=True,
                       changed_input_graph=True, passed=True)
            result["records"].append(row)
            args.out.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
            del graph, replayed
    for m in (0, 1, 8, 16, 17, 32, 40, 64, 2048, 2049):
        x = torch.randn(m, 2080, device="cuda", dtype=weight.dtype)[:, :2048]
        # Real BF16 weights keep the old fallback for all rows, including small M.
        torch.testing.assert_close(auto(x, q, scales, weight, 17),
                                   torch.ops.biv150_native.gate_silu(x, q, scales, weight, 17),
                                   atol=0, rtol=0)
        if m == 0 or m >= 17:
            torch.testing.assert_close(auto(x, q, scales, marker, 17),
                                       torch.ops.biv150_native.gate_silu(x, q, scales, marker, 17),
                                       atol=0, rtol=0)
        result["unchanged_routes"].append(dict(m=m, bf16_fallback=True,
                                              outside_small_residual=(m==0 or m>=17)))
    result.update(completed=True, small_row_records=32, input_patterns=160)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
