"""Isolated allocation evidence; never a model/KV-capacity or timing verdict."""

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


CASES = (("gate", 32), ("gate", 256), ("gate", 2048),
         ("bf16_head", 256), ("historical_int8_head", 256))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def footprint(kind, rows):
    """Explicit tensor bytes, excluding inputs/weights and vendor workspaces."""
    if kind == "bf16_head":
        return {"output": rows * 130560 * 2}
    columns = 12288 if kind == "gate" else 130560
    padded = max(32, rows)
    return dict(output=rows * (columns // 2 if kind == "gate" else columns) * 2,
                aq=padded * 2048, row_scales=padded * 4,
                int32_acc=padded * columns * 4)


def require_idle():
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", 9031)) == 0:
            raise RuntimeError("Service port occupied; no probe started")
    if "No running processes found" not in subprocess.check_output(
            ["/usr/local/corex/bin/ixsmi"], text=True):
        raise RuntimeError("GPU occupied; no probe started")


def worker(args):
    # Parent and each child check idle before importing/initializing CUDA.
    if (args.kind, args.rows) not in CASES:
        raise ValueError("Only the declared measurement matrix is supported")
    require_idle()
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.ops.load_library(str(args.library))
    torch.cuda.init()
    torch.cuda.synchronize()
    args.out.mkdir(parents=True, exist_ok=False)
    quantizer_path = Path(__file__).resolve().parents[1] / "vllm_fl/patches/biv150_native_w8a8.py"
    function = next(n for n in ast.parse(quantizer_path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "quantize_weight")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(quantizer_path), "exec"), scope)
    result = dict(kind=args.kind, rows=args.rows, pid=os.getpid(),
                  quantizer_source_sha256=sha(quantizer_path),
                  quantizer_ast_sha256=hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
                  library_sha256=sha(args.library), torch_version=torch.__version__,
                  probe_sha256=sha(__file__),
                  weight_index_sha256=sha(args.weights / "model.safetensors.index.json"),
                  explicit_tensor_bytes=footprint(args.kind, args.rows), stages=[],
                  limits=["Standalone primitive, not production graph or KV sizing",
                          "mem_get_info samples retained bytes, not non-Torch transient peak",
                          "No graph capture, latency, model accuracy or gain claim"])

    def save():
        (args.out / "result.json").write_text(json.dumps(result, indent=2))

    def sample(stage):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        result["stages"].append(dict(stage=stage, allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(), device_used_sample=total-free,
            device_total=total, active_bytes=stats.get("active_bytes.all.current"),
            inactive_split_bytes=stats.get("inactive_split_bytes.all.current")))
        save()

    def storage(tensor):
        s = tensor.untyped_storage()
        return dict(pointer=s.data_ptr(), storage_nbytes=s.nbytes(),
                    shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype))

    sample("context")
    index = json.loads((args.weights / "model.safetensors.index.json").read_text())["weight_map"]
    keys = (["model.layers.20.mlp.gate_proj.weight", "model.layers.20.mlp.up_proj.weight"]
            if args.kind == "gate" else ["lm_head.weight"])
    tensors = []
    weight_sources = []
    for key in keys:
        path = args.weights / index[key]
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        assert tensor.dtype == torch.bfloat16
        weight_sources.append(dict(key=key, shape=list(tensor.shape),
            tensor_sha256=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()))
        tensors.append(tensor)
    weight = torch.cat(tensors) if len(tensors) > 1 else tensors[0]
    del tensors, tensor
    assert tuple(weight.shape) == ((12288, 2048) if args.kind == "gate" else (130560, 2048))
    weight = weight.cuda()
    result["weight_sources"] = weight_sources
    if args.kind == "bf16_head":
        operands = {"weight": weight}
    else:
        wq, scales = scope["quantize_weight"](weight)
        marker = torch.zeros((), device="cuda", dtype=torch.bfloat16).expand_as(weight)
        operands = {"weight_int8": wq, "scales": scales, "marker": marker}
        del weight
    gc.collect()
    torch.cuda.empty_cache()
    sample("weights_prepared_and_unused_tensors_released")
    x = torch.randn((args.rows, 2048), device="cuda", dtype=torch.bfloat16)
    result["retained_storages"] = {name: storage(t) for name, t in dict(operands, input=x).items()}
    # Count physical storages once; broadcast marker is a scalar allocation.
    unique = {v["pointer"]: v["storage_nbytes"] for v in result["retained_storages"].values()}
    result["retained_unique_storage_bytes"] = sum(unique.values())
    torch.cuda.empty_cache()
    sample("input_ready")
    # All allocated model/input operands are outside the measured call delta.
    torch.cuda.memory._record_memory_history(enabled="all", max_entries=12000)
    try:
        for iteration in range(3):
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            if args.kind == "bf16_head":
                output = torch.nn.functional.linear(x, operands["weight"])
            elif args.kind == "gate":
                output = torch.ops.biv150_native.gate_silu_auto(
                    x, operands["weight_int8"], operands["scales"], operands["marker"], 17)
            else:
                output = torch.ops.biv150_native.linear(
                    x, operands["weight_int8"], operands["scales"], operands["marker"], None, 17)
            assert output.dtype == torch.bfloat16
            assert tuple(output.shape) == (args.rows, 6144 if args.kind == "gate" else 130560)
            sample(f"call_{iteration}_output_live")
            result["stages"][-1].update(output=storage(output),
                call_peak_allocated_delta=torch.cuda.max_memory_allocated()-before)
            (args.out / f"snapshot_{iteration}.json").write_text(
                json.dumps(torch.cuda.memory._snapshot()))
            del output
            gc.collect()
            sample(f"call_{iteration}_output_released")
            if iteration == 0:
                torch.cuda.empty_cache()
                sample("cold_call_inactive_cache_released")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
        save()
    result["completed"] = True
    save()


def main(args):
    if args.kind:
        worker(args)
        return
    require_idle()
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, cases=[], purpose="memory attribution only")
    for kind, rows in CASES:
        out = args.out / f"{kind}-{rows}"
        command = [sys.executable, str(Path(__file__).resolve()), "--library", str(args.library),
                   "--weights", str(args.weights), "--out", str(out), "--kind", kind, "--rows", str(rows)]
        with (args.out / f"{kind}-{rows}.log").open("x") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        result["cases"].append(dict(kind=kind, rows=rows, exit_code=code, output=str(out)))
        (args.out / "summary.json").write_text(json.dumps(result, indent=2))
        if code:
            raise SystemExit(code)
    result["completed"] = True
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=["gate", "bf16_head", "historical_int8_head"])
    parser.add_argument("--rows", type=int, default=256)
    main(parser.parse_args())
