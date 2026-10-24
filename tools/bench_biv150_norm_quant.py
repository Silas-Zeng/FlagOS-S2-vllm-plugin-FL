"""Audit residual RMSNorm+quantization fusion against the actual FlagGems path."""

import argparse
import hashlib
import importlib
import json
import os
import socket
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
from biv150_benchmark_utils import graph_time
from biv150_norm_quant_probe import norm_quant


def main(args):
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 9031)) != 0, "Service running"
    assert "No running processes found" in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    ), "GPU busy"
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    from flag_gems.config import use_c_extension
    from flag_gems.modules.normalization import gems_rms_forward

    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight

    pins = {
        "flag_gems.modules.normalization": "d00f40f876532204b496634a03687eab4b0c364d775ee6375f28f47257d356f4",
        "flag_gems.fused.fused_add_rms_norm": "192cbddc2283a564a9eaf90bfae1641a8a052398c0d7dd14f83f6a67f76ec3b9",
    }
    sources = {}
    for name, expected in pins.items():
        p = Path(importlib.import_module(name).__file__)
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == expected, (name, actual)
        sources[name] = dict(path=str(p), sha256=actual)
    load_library()
    config = json.loads(Path("/workspace/MiniCPM5-2B/config.json").read_text())
    eps = config["rms_norm_eps"]
    torch.manual_seed(20261024)
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    wq, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=weight.dtype).expand_as(weight)
    result = dict(
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        norm_sources=sources,
        use_c_extension=use_c_extension,
        epsilon=eps,
        service_installed=False,
        identical_input_reset_copies_per_call=2,
        records=[],
    )
    for rows in (1, 8, 16, 17, 32, 40, 64, 256, 2048):
        for wpattern in ("ones", "random"):
            norm_w = torch.ones(2048, device="cuda", dtype=weight.dtype)
            if wpattern == "random":
                norm_w.normal_(1.0, 0.2)
            x = torch.randn(rows, 2048, device="cuda", dtype=weight.dtype)
            residual = torch.randn_like(x)
            bx, br = torch.empty_like(x), torch.empty_like(x)
            nx, nr = torch.empty_like(x), torch.empty_like(x)

            def baseline(
                quant_only=False, bx=bx, br=br, x=x, residual=residual, norm_w=norm_w
            ):
                bx.copy_(x)
                br.copy_(residual)
                normalized, r = gems_rms_forward(bx, br, norm_w, eps)
                if quant_only:
                    return torch.ops.biv150_native.norm_quant_reference_probe(
                        normalized
                    ), r
                return torch.ops.biv150_native.gate_silu_auto(
                    normalized, wq, scales, marker, 17
                ), r

            def candidate(
                quant_only=False,
                nx=nx,
                nr=nr,
                x=x,
                residual=residual,
                norm_w=norm_w,
                rows=rows,
            ):
                nx.copy_(x)
                nr.copy_(residual)
                q, s, _ = norm_quant(nx, nr, norm_w, eps)
                if quant_only:
                    return (q, s), nr
                return torch.ops.biv150_native.gate_quantized_probe(
                    q, s, wq, scales, rows, rows <= 16
                ), nr

            row = dict(m=rows, norm_weight=wpattern)
            try:
                for pattern in ("zero", "tiny", "normal", "large", "cancel", "outlier"):
                    x.normal_()
                    residual.normal_()
                    if pattern == "zero":
                        x.zero_()
                        residual.zero_()
                    elif pattern == "tiny":
                        x.mul_(1e-9)
                        residual.mul_(1e-9)
                    elif pattern == "large":
                        x.mul_(100)
                        residual.mul_(100)
                    elif pattern == "cancel":
                        residual.copy_(-x)
                        residual[:, ::17] += 0.00390625
                    elif pattern == "outlier":
                        x[:, 0] = 32
                        residual[:, 7] = -64
                    dx, dr = x.clone(), residual.clone()
                    q, s, kernel = norm_quant(dx, dr, norm_w, eps, write_norm=True)
                    ox, ores = gems_rms_forward(
                        x.clone(), residual.clone(), norm_w, eps
                    )
                    row.update(last_pattern=pattern)
                    row["norm_mismatches"] = int((dx != ox).sum().item())
                    torch.testing.assert_close(dx, ox, atol=0, rtol=0)
                    torch.testing.assert_close(dr, ores, atol=0, rtol=0)
                    host_sum = x.cpu().double() + residual.cpu().double()
                    host_r = host_sum.to(torch.bfloat16)
                    host_y = (
                        host_sum
                        * torch.rsqrt(host_sum.square().mean(-1, keepdim=True) + eps)
                        * norm_w.cpu().double()
                    ).to(torch.bfloat16)
                    torch.testing.assert_close(dr.cpu(), host_r, atol=0, rtol=0)
                    torch.testing.assert_close(dx.cpu(), host_y, atol=0.005, rtol=0.005)
                    oq, oscale = torch.ops.biv150_native.norm_quant_reference_probe(ox)
                    row["quant_mismatches"] = int((q != oq).sum().item())
                    torch.testing.assert_close(q, oq, atol=0, rtol=0)
                    torch.testing.assert_close(s, oscale, atol=0, rtol=0)
                    got, got_r = candidate()
                    old, old_r = baseline()
                    torch.testing.assert_close(got, old, atol=0, rtol=0)
                    torch.testing.assert_close(got_r, old_r, atol=0, rtol=0)
                x.normal_()
                residual.normal_()
                _, _, measured_kernel = norm_quant(
                    nx.copy_(x), nr.copy_(residual), norm_w, eps
                )
                rounds = []
                for iteration in range(3):
                    calls = [("baseline_ms", baseline), ("candidate_ms", candidate)]
                    if iteration % 2:
                        calls.reverse()
                    times = {
                        name: graph_time(fn, repeats=11, unroll=8)[0]
                        for name, fn in calls
                    }
                    times["speedup"] = times["baseline_ms"] / times["candidate_ms"]
                    rounds.append(times)
                norm_quant_ms = {
                    name: graph_time(
                        partial(fn, quant_only=True), repeats=11, unroll=8
                    )[0]
                    for name, fn in (("baseline", baseline), ("candidate", candidate))
                }
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed, replayed_r = candidate()
                for factor in (0.0, 0.5, 4.0):
                    x.normal_().mul_(factor)
                    residual.normal_().mul_(factor)
                    graph.replay()
                    torch.cuda.synchronize()
                    old, old_r = baseline()
                    torch.testing.assert_close(replayed, old, atol=0, rtol=0)
                    torch.testing.assert_close(replayed_r, old_r, atol=0, rtol=0)
                row.update(
                    passed=True,
                    norm_bit_exact=True,
                    quant_bit_exact=True,
                    full_gate_bit_exact=True,
                    cpu_reference=True,
                    changed_input_graph=True,
                    rounds=rounds,
                    norm_quant_ms=norm_quant_ms,
                    median_speedup=statistics.median(t["speedup"] for t in rounds),
                    resources=dict(
                        n_regs=getattr(measured_kernel, "n_regs", None),
                        n_spills=getattr(measured_kernel, "n_spills", None),
                        metadata=str(measured_kernel.metadata),
                        asm_sha256={
                            name: hashlib.sha256(
                                measured_kernel.asm[name].encode()
                            ).hexdigest()
                            for name in ("ttir", "ttgir", "llir")
                            if isinstance(measured_kernel.asm.get(name), str)
                        },
                    ),
                )
                del graph, replayed, replayed_r
            except Exception as error:
                row.update(passed=False, error=repr(error))
            result["records"].append(row)
            (args.out / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)
    if len(result["records"]) != 18 or not all(x["passed"] for x in result["records"]):
        raise RuntimeError("Fusion numeric gates not passed; no service result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
