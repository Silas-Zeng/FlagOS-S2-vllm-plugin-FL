"""Check every packed INT8/scale element before allocation-order service tests."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from probe_biv150_memory_ownership import require_idle, sha


BASELINE_AST = "16133e9295cea198c7edd73a865f59c0cb5fd57c1d29a2d448050a5df37ecc3e"


def function(path, torch):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "quantize_weight")
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["quantize_weight"], digest


def main(args):
    require_idle()
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    source = Path(__file__).resolve().parents[1] / "vllm_fl/patches/biv150_native_w8a8.py"
    baseline, baseline_hash = function(args.baseline_source, torch)
    candidate, candidate_hash = function(source, torch)
    assert baseline_hash == BASELINE_AST and candidate_hash != BASELINE_AST
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert not args.out.exists()
    index = json.loads((args.weights / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [(f"layer{layer}", [f"model.layers.{layer}.mlp.{part}_proj.weight" for part in ("gate", "up")])
            for layer in range(4, 38)]
    keys.append(("historical_head_not_production", ["lm_head.weight"]))
    result = dict(baseline_source_sha256=sha(args.baseline_source), baseline_ast_sha256=baseline_hash,
                  candidate_source_sha256=sha(source), candidate_ast_sha256=candidate_hash,
                  records=[], model_accuracy_pending=True, completed=False)
    for label, names in keys:
        parts = []
        identities = []
        for name in names:
            with safe_open(args.weights / index[name], framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            identities.append(dict(key=name, sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()))
            parts.append(value)
        host = torch.cat(parts) if len(parts) > 1 else parts[0]
        weight = host.cuda()
        old_q, old_scales = baseline(weight)
        new_q, new_scales = candidate(weight)
        row = dict(label=label, weights=identities, shape=list(weight.shape),
            int8_bitwise_equal=bool(torch.equal(old_q, new_q)),
            scales_bitwise_equal=bool(torch.equal(old_scales.view(torch.int32), new_scales.view(torch.int32))),
            packed_stride=list(new_q.stride()), packed_contiguous=new_q.is_contiguous(),
            input_unchanged=bool(torch.equal(weight.cpu().view(torch.int16), host.view(torch.int16))),
            expected_dtypes=(new_q.dtype == torch.int8 and new_scales.dtype == torch.float32))
        row["passed"] = all(row[k] for k in ["int8_bitwise_equal", "scales_bitwise_equal", "packed_contiguous", "input_unchanged", "expected_dtypes"])
        result["records"].append(row)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(label=label, passed=row["passed"])), flush=True)
        if not row["passed"]:
            raise RuntimeError("Packing contract changed")
        del weight, old_q, new_q, old_scales, new_scales
    result.update(completed=True, passed=len(result["records"]) == 35)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
