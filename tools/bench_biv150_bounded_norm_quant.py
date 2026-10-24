"""Bounded, non-mutating residual norm+quant with real layer norm weights.

Component only. No serving hook; model precision and throughput remain pending.
"""

import argparse
import hashlib
import importlib
import json
import os
import socket
import statistics
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from audit_biv150_residual_reference import cpu_quantize, cpu_output
from biv150_benchmark_utils import graph_time
from biv150_bounded_norm_quant import SELECTED_ROWS, norm_quant


def norm_weights(folder):
    files = sorted(folder.glob("*.safetensors"))
    result = []
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                if not key.endswith(".post_attention_layernorm.weight"):
                    continue
                layer = int(key.split("layers.")[1].split(".")[0])
                if 4 <= layer < 38:
                    weight = handle.get_tensor(key)
                    assert weight.shape == (2048,) and weight.dtype == torch.bfloat16
                    digest = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
                    result.append((layer, key, weight, digest))
    assert len(result) == 34 and len({x[0] for x in result}) == 34
    return sorted(result)


def main(args):
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 9031)) != 0
    assert "No running processes found" in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True
    )
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    from vllm.ir.ops.layernorm import fused_add_rms_norm
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight

    pins = {
        "vllm.ir.ops.layernorm": "d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910",
        "vllm.ir.op": "fd0355b398c0d96235bb6aba0c9c08f73c199429bd8ec2aed227178c297c1eb2",
        "vllm.model_executor.layers.layernorm": "696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c",
        "vllm.model_executor.models.llama": "1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384",
    }
    for module, expected in pins.items():
        path = Path(importlib.import_module(module).__file__)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, module
    config = json.loads((args.model / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"]
    epsilon = config["rms_norm_eps"]
    learned = norm_weights(args.model)
    load_library()
    torch.manual_seed(20261024)
    # Freeze a synthetic gate projection for kernel comparison; actual learned
    # norm weights cover every compressed layer. This is not model validation.
    weight = torch.randn(12288, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    q, scales = quantize_weight(weight)
    marker = torch.zeros((), device="cuda", dtype=weight.dtype).expand_as(weight)
    compiled_norm = torch.compile(fused_add_rms_norm.impls["native"].impl_fn,
                                  fullgraph=True, backend="inductor")
    cpu_columns = torch.tensor([0, 1, 31, 127, 1023, 6143])
    both = torch.cat((cpu_columns, cpu_columns + 6144)).to("cuda")
    cpu_q, cpu_scales = q[:, both].cpu(), scales[both].cpu()
    result = dict(component_only=True, service_installed=False, inputs_mutated=False,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  model_architecture=config["architectures"], selected_rows=list(SELECTED_ROWS),
                  reference="pinned native IR, torch.compile fullgraph/inductor",
                  learned_norm_weights=[dict(layer=i, key=k, sha256=h) for i,k,_,h in learned],
                  records=[], performance=[])

    def save():
        (args.out / "result.json").write_text(json.dumps(result, indent=2))

    for rows in SELECTED_ROWS:
        x = torch.empty(rows, 2048, device="cuda", dtype=torch.bfloat16)
        residual = torch.empty_like(x)
        for layer, key, host_weight, digest in learned:
            norm_weight = host_weight.to("cuda")
            row = dict(m=rows, layer=layer, weight_sha256=digest, patterns=[])
            try:
                for pattern in ("zero", "tiny", "normal", "large", "cancel", "outlier"):
                    x.normal_()
                    residual.normal_()
                    if pattern == "zero":
                        x.zero_(); residual.zero_()
                    elif pattern == "tiny":
                        x.mul_(1e-9); residual.mul_(1e-9)
                    elif pattern == "large":
                        x.mul_(100); residual.mul_(100)
                    elif pattern == "cancel":
                        residual.copy_(-x); residual[:, ::17] += 0.00390625
                    elif pattern == "outlier":
                        x[:, 0] = 32; residual[:, 7] = -64
                    x_before, r_before = x.clone(), residual.clone()
                    original_norm, original_r = compiled_norm(x, residual, norm_weight, epsilon)
                    aq, act_scales, new_r, new_norm, _ = norm_quant(
                        x, residual, norm_weight, epsilon, write_norm=True)
                    torch.testing.assert_close(x, x_before, atol=0, rtol=0)
                    torch.testing.assert_close(residual, r_before, atol=0, rtol=0)
                    assert new_r.data_ptr() not in (x.data_ptr(), residual.data_ptr())
                    assert new_norm.data_ptr() not in (x.data_ptr(), residual.data_ptr())
                    torch.testing.assert_close(new_r, original_r, atol=0, rtol=0)
                    torch.testing.assert_close(new_norm, original_norm, atol=0.005, rtol=0.005)
                    own_q, own_s = torch.ops.biv150_native.norm_quant_reference_probe(new_norm)
                    torch.testing.assert_close(aq, own_q, atol=0, rtol=0)
                    torch.testing.assert_close(act_scales, own_s, atol=0, rtol=0)
                    selected = torch.ops.biv150_native.gate_quantized_probe(
                        aq, act_scales, q, scales, rows, rows <= 16)
                    original = torch.ops.biv150_native.gate_silu_auto(
                        original_norm, q, scales, marker, 17)
                    torch.testing.assert_close(selected, original, atol=0.005, rtol=0.005)
                    # Independent CPU arithmetic, including BF16 norm boundary.
                    host_sum = x.cpu().float() + residual.cpu().float()
                    host_r = host_sum.to(torch.bfloat16)
                    host_norm = (host_sum.double() * torch.rsqrt(
                        host_sum.double().square().mean(-1,keepdim=True) + epsilon)
                        * host_weight.double()).to(torch.bfloat16)
                    torch.testing.assert_close(new_r.cpu(), host_r, atol=0, rtol=0)
                    torch.testing.assert_close(new_norm.cpu(), host_norm, atol=0.005, rtol=0.005)
                    if rows <= 16:
                        first, second, s1, s2 = cpu_quantize(new_norm.cpu())
                        gold = cpu_output(first, second, s1, s2, cpu_q, cpu_scales)
                        torch.testing.assert_close(selected[:,cpu_columns.to("cuda")].cpu(),
                                                   gold, atol=0.005, rtol=0.005)
                    row["patterns"].append(dict(name=pattern,
                        norm_bit_differences=int((new_norm != original_norm).sum().item()),
                        gate_bit_differences=int((selected != original).sum().item()),
                        own_quant_bit_exact=True, cpu_reference=True, inputs_unchanged=True))
                # Address-stable Graph, new input values; no input reset copies
                # in either measured path because both are non-mutating.
                def candidate():
                    aq, act_scales, out_r, _, _ = norm_quant(x, residual, norm_weight, epsilon)
                    return torch.ops.biv150_native.gate_quantized_probe(
                        aq, act_scales, q, scales, rows, rows <= 16), out_r
                def baseline():
                    out_norm, out_r = compiled_norm(x, residual, norm_weight, epsilon)
                    return torch.ops.biv150_native.gate_silu_auto(
                        out_norm, q, scales, marker, 17), out_r
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual, actual_r = candidate()
                for factor in (0.0, 0.5, 4.0):
                    x.normal_().mul_(factor); residual.normal_().mul_(factor)
                    graph.replay(); torch.cuda.synchronize()
                    expected, expected_r = baseline()
                    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
                    torch.testing.assert_close(actual_r, expected_r, atol=0, rtol=0)
                del graph, actual, actual_r
                row.update(passed=True, changed_input_graph=True)
                if layer in (4, 20, 37):
                    rounds=[]
                    for iteration in range(3):
                        calls=[("baseline_ms",baseline),("candidate_ms",candidate)]
                        if iteration % 2: calls.reverse()
                        times={name:graph_time(fn,repeats=11,unroll=8)[0] for name,fn in calls}
                        times["speedup"]=times["baseline_ms"]/times["candidate_ms"]
                        rounds.append(times)
                    result["performance"].append(dict(m=rows,layer=layer,rounds=rounds,
                        median_speedup=statistics.median(t["speedup"] for t in rounds)))
            except Exception as error:
                row.update(passed=False, error=repr(error))
                fixture=args.out/f"failed-m{rows}-layer{layer}.pt"
                torch.save(dict(x=x.cpu(),residual=residual.cpu(),weight=host_weight,
                                epsilon=epsilon,pattern=pattern),fixture)
                row.update(fixture=fixture.name, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
            result["records"].append(row)
            save()
            print(json.dumps({k:v for k,v in row.items() if k!='patterns'}),flush=True)
    result["summary"]={str(m):dict(passed=sum(r["passed"] for r in result["records"] if r["m"]==m),
                                       total=34) for m in SELECTED_ROWS}
    save()
    if len(result["records"]) != 238 or not all(r["passed"] for r in result["records"]):
        raise RuntimeError("Bounded candidate has failures: retain per-shape evidence, no service promotion")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model",type=Path,default=Path("/workspace/MiniCPM5-2B"))
    main(parser.parse_args())
