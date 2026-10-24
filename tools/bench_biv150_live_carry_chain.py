"""Joint norm/quant/gate/down/next-norm diagnostic; no service installation.

The candidate leaves the original transparent residual expression in the graph.
Learned weights, actual worker MM policy and complete-call costs are retained.
Strict failures are recorded even when finite, reproducible cases are timed.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import traceback

import torch
from safetensors import safe_open
from biv150_benchmark_utils import graph_time
from biv150_bounded_norm_quant import norm_quant


@torch.library.custom_op("biv150_joint_probe::gate_without_carry_store", mutates_args=())
def gate_without_carry_store(o: torch.Tensor, residual: torch.Tensor,
                            norm_weight: torch.Tensor, wq: torch.Tensor,
                            scales: torch.Tensor, eps: float) -> torch.Tensor:
    assert o.shape[0] in (32, 64)
    q, s, carry, _, _ = norm_quant(o, residual, norm_weight, eps,
                                   materialize_residual=False)
    assert carry is None
    return torch.ops.biv150_native.gate_quantized_probe(q, s, wq, scales,
                                                       o.shape[0], False)


@gate_without_carry_store.register_fake
def fake_gate(o, residual, norm_weight, wq, scales, eps):
    return torch.empty((o.shape[0], 6144), dtype=o.dtype, device=o.device)


def compare(actual, expected):
    a, b = actual.cpu().float(), expected.cpu().float()
    error = (a-b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    bad = (error > .005 + .005*b.abs()) | ~torch.isfinite(a)
    return dict(passed=finite and not bool(bad.any()), finite=finite,
                violations=int(bad.sum()), elements=a.numel(),
                bit_differences=int((a != b).sum()), max_abs=float(error.max()))


def learned(model, layer):
    keys = [f"model.layers.{layer}.{p}" for p in (
        "post_attention_layernorm.weight", "mlp.gate_proj.weight",
        "mlp.up_proj.weight", "mlp.down_proj.weight")]
    keys.append(f"model.layers.{layer+1}.input_layernorm.weight")
    found = {}
    for path in sorted(model.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in keys:
                if key in handle.keys():
                    found[key] = handle.get_tensor(key)
    assert set(found) == set(keys)
    identities = {k: hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()
                  for k, v in found.items()}
    return [found[k].to("cuda") for k in keys], identities


def main(args):
    with socket.socket() as port:
        assert port.connect_ex(("127.0.0.1", 9031)) != 0
    assert "No running processes found" in subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi"], text=True)
    root = Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    # Reuse the known private probe ABI, never modify site-packages or rebuild
    # an active service's library. Its source must match this exact probe source.
    for name in ("native_w8a8.cpp", "native_w8a8.cu"):
        assert (args.library_source / "csrc/iluvatar" / name).read_bytes() == (root / "csrc/iluvatar" / name).read_bytes()
    assert hashlib.sha256(args.library.read_bytes()).hexdigest() == args.library_sha256
    args.out.mkdir(parents=True, exist_ok=False)
    assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).resolve() == args.out / "inductor-cache"
    os.environ["BIV150_NATIVE_W8A8_LIBRARY"] = str(args.library.resolve())
    import flag_gems
    from vllm_fl.utils import use_flaggems, get_flag_gems_whitelist_blacklist
    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert use_flaggems() and not whitelist and not blacklist
    mm_path = Path(inspect.getsourcefile(inspect.unwrap(flag_gems.mm_out)))
    assert hashlib.sha256(mm_path.read_bytes()).hexdigest() == "4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090"
    flag_gems.enable(record=False, once=True)
    from vllm.ir.ops.layernorm import fused_add_rms_norm
    from vllm_fl.patches.biv150_native_w8a8 import load_library, quantize_weight
    load_library()
    native_norm = fused_add_rms_norm.impls["native"].impl_fn
    config = json.loads((args.model / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"]
    eps = config["rms_norm_eps"]

    def baseline(o, residual, nw, wq, scales, marker, down_weight, next_weight):
        normal, carry = native_norm(o, residual, nw, eps)
        gate = torch.ops.biv150_native.gate_silu_auto(normal, wq, scales, marker, 17)
        down = torch.nn.functional.linear(gate, down_weight)
        next_normal, next_residual = native_norm(down, carry, next_weight, eps)
        return next_normal, next_residual, gate

    def candidate(o, residual, nw, wq, scales, marker, down_weight, next_weight):
        # Keep the original graph expression instead of returning a BF16 carry
        # from the opaque fused operator. Dead norm work should be eliminated;
        # generated code is preserved to check whether that actually happened.
        _, carry = native_norm(o, residual, nw, eps)
        gate = gate_without_carry_store(o, residual, nw, wq, scales, eps)
        down = torch.nn.functional.linear(gate, down_weight)
        next_normal, next_residual = native_norm(down, carry, next_weight, eps)
        return next_normal, next_residual, gate

    bf = torch.compile(baseline, fullgraph=True, dynamic=True)
    cf = torch.compile(candidate, fullgraph=True, dynamic=True)
    result = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  component_only=True, service_installed=False, synthetic_activations=True,
                  preserves_original_transparent_carry_expression=True,
                  shipping_cpp_cuda_and_quantizer_unchanged=True,
                  library_sha256=args.library_sha256,
                  native_mm_source_sha256=hashlib.sha256(mm_path.read_bytes()).hexdigest(),
                  dispatch_table=torch._C._dispatch_dump_table("aten::mm.out"),
                  original_strict_tolerance=dict(atol=.005, rtol=.005), records=[])
    save = lambda: (args.out / "result.json").write_text(json.dumps(result, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    for layer in (4, 20, 37):
        (nw, gate_w, up_w, down_w, next_w), identities = learned(args.model, layer)
        combined = torch.cat((gate_w, up_w))
        wq, scales = quantize_weight(combined)
        marker = torch.zeros((), device="cuda", dtype=torch.bfloat16).expand_as(combined)
        del combined, gate_w, up_w
        for m in (32, 64):
            o = torch.empty((m, 2048), device="cuda", dtype=torch.bfloat16)
            residual = torch.empty_like(o)
            inputs = (o, residual, nw, wq, scales, marker, down_w, next_w)
            row = dict(layer=layer, m=m, learned_weight_sha256=identities, patterns=[],
                       original_strict_passed=True, graph_eager_bit_exact=True)
            result["records"].append(row)
            try:
                for pattern in ("zero", "tiny", "normal", "large", "cancel", "outlier"):
                    o.normal_(); residual.normal_()
                    if pattern == "zero": o.zero_(); residual.zero_()
                    elif pattern == "tiny": o.mul_(1e-9); residual.mul_(1e-9)
                    elif pattern == "large": o.mul_(100); residual.mul_(100)
                    elif pattern == "cancel": residual.copy_(-o); residual[:, ::17] += .00390625
                    elif pattern == "outlier": o[:, 0] = 32; residual[:, 7] = -64
                    before = (o.cpu(), residual.cpu())
                    expected, actual = bf(*inputs), cf(*inputs)
                    checked = {name: compare(a, b) for name, a, b in zip(
                        ("next_norm", "next_residual", "gate"), actual, expected)}
                    row["original_strict_passed"] &= all(v["passed"] for v in checked.values())
                    assert torch.equal(o.cpu(), before[0]) and torch.equal(residual.cpu(), before[1])
                    row["patterns"].append(dict(name=pattern, checks=checked, inputs_unchanged=True))
                    if not all(v["passed"] for v in checked.values()) and "fixture" not in row:
                        fixture = args.out / f"failed-l{layer}-m{m}.pt"
                        torch.save(dict(o=o.cpu(), residual=residual.cpu(),
                                        expected=[v.cpu() for v in expected],
                                        candidate=[v.cpu() for v in actual], pattern=pattern), fixture)
                        row.update(fixture=fixture.name, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): captured = cf(*inputs)
                for factor in (0., .5, 4.):
                    o.normal_().mul_(factor); residual.normal_().mul_(factor)
                    before = (o.cpu(), residual.cpu())
                    graph.replay(); torch.cuda.synchronize()
                    eager = cf(*inputs)
                    row["graph_eager_bit_exact"] &= all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(captured, eager))
                    expected = bf(*inputs)
                    checked = {name: compare(a, b) for name, a, b in zip(
                        ("next_norm", "next_residual", "gate"), captured, expected)}
                    row.setdefault("changed_input_graph_checks", []).append(dict(factor=factor, checks=checked))
                    row["original_strict_passed"] &= all(v["passed"] for v in checked.values())
                    assert torch.equal(o.cpu(), before[0]) and torch.equal(residual.cpu(), before[1])
                    if not all(v["passed"] for v in checked.values()) and "fixture" not in row:
                        fixture = args.out / f"failed-l{layer}-m{m}.pt"
                        torch.save(dict(o=o.cpu(), residual=residual.cpu(),
                                        expected=[v.cpu() for v in expected],
                                        candidate=[v.cpu() for v in captured], pattern=f"graph-factor-{factor}"), fixture)
                        row.update(fixture=fixture.name, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
                del graph, captured
                finite = all(v["finite"] for p in row["patterns"] for v in p["checks"].values())
                finite &= all(v["finite"] for p in row["changed_input_graph_checks"] for v in p["checks"].values())
                if finite and row["graph_eager_bit_exact"]:
                    rounds = []
                    for iteration in range(3):
                        calls = [("baseline_ms", bf), ("candidate_ms", cf)]
                        if iteration % 2: calls.reverse()
                        values = {name: graph_time(lambda fn=fn: fn(*inputs), repeats=9, unroll=4)[0] for name, fn in calls}
                        values["speedup"] = values["baseline_ms"] / values["candidate_ms"]
                        rounds.append(values)
                    row.update(rounds=rounds, min_round_speedup=min(v["speedup"] for v in rounds),
                               median_speedup=statistics.median(v["speedup"] for v in rounds),
                               diagnostic_timing_despite_strict_failure=not row["original_strict_passed"])
            except Exception as error:
                row.update(error=repr(error), traceback=traceback.format_exc(), original_strict_passed=False)
                result["completed"] = False
                save()
                raise
            save(); print(json.dumps({k: v for k, v in row.items() if k not in ("patterns", "learned_weight_sha256", "traceback")}), flush=True)
    result["completed"] = len(result["records"]) == 6
    result["summary"] = dict(records=len(result["records"]),
        original_strict_passed=sum(row["original_strict_passed"] for row in result["records"]),
        graph_eager_bit_exact=sum(row["graph_eager_bit_exact"] for row in result["records"]),
        timed=sum("rounds" in row for row in result["records"]))
    result["generated_python"] = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                   for p in sorted((args.out / "inductor-cache").rglob("*.py"))]
    save()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/workspace/MiniCPM5-2B"))
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--library-source", type=Path, required=True)
    p.add_argument("--library-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    with torch.inference_mode(): main(p.parse_args())
