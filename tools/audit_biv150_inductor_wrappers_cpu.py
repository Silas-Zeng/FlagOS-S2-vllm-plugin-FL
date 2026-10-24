"""Read generated Python ASTs without importing Torch or executing cached code.

Embedded tuning examples/comments are excluded. Structural counts prove no
runtime frequency or device cost; casts inside fused Triton code are separate.
"""
import argparse
import ast
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path


def audit(path):
    source = path.read_text()
    module = ast.parse(source)
    names = {n.name for n in module.body if isinstance(n, ast.FunctionDef)}
    module_kind = ("kernel_benchmark_harness" if {"get_args", "benchmark_all_configs"} <= names
                   else "module_call_unproven_runtime")
    functions = [(n, module_kind) for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "call"]
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Runner":
            functions.extend((n, "model_runner_method") for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "call")
    wrappers = []
    for function, kind in functions:
        calls = sorted((n for n in ast.walk(function) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
        operations = [dict(line=n.lineno, target=ast.unparse(n.func), expression=ast.unparse(n)) for n in calls]
        copies = [n for n in operations if any(term in n["target"] for term in ("clone", "copy_", "_to_copy", "contiguous", "convert_element_type"))]
        wrappers.append(dict(kind=kind, line=function.lineno, operations=operations,
                             explicit_copy_or_conversion_calls=copies,
                             counts=dict(Counter(n["target"] for n in operations)),
                             source=ast.get_source_segment(source, function)))
    kernels = []
    for node in module.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if ast.unparse(call.func) != "async_compile.triton":
            continue
        assert len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)
        code = call.args[1].value
        parsed = ast.parse(code)
        funcs = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name.startswith("triton_")]
        kernels.append(dict(name=ast.literal_eval(call.args[0]), line=node.lineno,
                            source_sha256=hashlib.sha256(code.encode()).hexdigest(),
                            functions=[dict(name=f.name, calls=dict(Counter(ast.unparse(c.func) for c in ast.walk(f) if isinstance(c, ast.Call)))) for f in funcs]))
    return dict(path=str(path), sha256=hashlib.sha256(source.encode()).hexdigest(),
                bytes=len(source.encode()), wrappers=wrappers, fused_kernels=kernels)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [audit(path) for path in sorted(args.cache.rglob("*.py"))]
    result = dict(cpu_only=True, imports_or_executes_cached_code=False,
                  snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  cache=str(args.cache), files=rows,
                  total_wrappers=sum(len(row["wrappers"]) for row in rows),
                  wrapper_kinds=dict(Counter(w["kind"] for row in rows for w in row["wrappers"])),
                  explicit_copy_or_conversion_call_sites=sum(len(w["explicit_copy_or_conversion_calls"]) for row in rows for w in row["wrappers"]),
                  runtime_frequency_or_cost_measured=False,
                  example_shapes_are_not_runtime_coverage=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}))
