"""Check immutable-input kernel ABI and preserve existing production functions."""

import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path


def main(args):
    root=Path(__file__).resolve().parents[1]
    base="0bf9d2efeaa10144dd7f0ff4597298df029f9328"
    old=subprocess.check_output(["git","-C",str(root),"show",
                                 base+":csrc/iluvatar/native_w8a8.cpp"],text=True)
    current=(root/"csrc/iluvatar/native_w8a8.cpp").read_text()
    begin=current.index("// Standalone fusion probe:")
    end=current.index("\n}\n\nTORCH_LIBRARY",begin)
    remaining=current[:begin]+current[end:]
    for line in list(remaining.splitlines(keepends=True)):
        if any(value in line for value in (
            'm.def("norm_quant_reference_probe', 'm.def("gate_quantized_probe',
            'm.impl("norm_quant_reference_probe', 'm.impl("gate_quantized_probe',
        )):
            remaining=remaining.replace(line,"")
    assert [s for s in remaining.splitlines() if s.strip()] == [s for s in old.splitlines() if s.strip()]
    module=ast.parse((root/"tools/biv150_bounded_norm_quant.py").read_text())
    kernel=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=="_norm_quant")
    assert [a.arg for a in kernel.args.args] == [
        "X","R","R_OUT","W","Q","S","NORM_OUT","M","EPS","RESIDUAL","WRITE_NORM","WRITE_CARRY"]
    prior=ast.parse(subprocess.check_output(["git","-C",str(root),"show",
        "459432d:tools/biv150_bounded_norm_quant.py"],text=True))
    old_kernel=next(n for n in prior.body if isinstance(n,ast.FunctionDef) and n.name=="_norm_quant")
    restored=copy.deepcopy(kernel)
    restored.args.args.pop()
    class RestoreCarry(ast.NodeTransformer):
        def visit_If(self,node):
            if isinstance(node.test,ast.Name) and node.test.id=="WRITE_CARRY":
                assert not node.orelse and len(node.body)==1
                return node.body
            return self.generic_visit(node)
    restored=RestoreCarry().visit(restored)
    assert ast.dump(restored,include_attributes=False)==ast.dump(old_kernel,include_attributes=False)
    stores=[ast.unparse(n.args[0]) for n in ast.walk(kernel)
            if isinstance(n,ast.Call) and ast.unparse(n.func)=="tl.store"]
    assert not any(s.startswith("X +") or s.startswith("R +") for s in stores)
    assert any(s.startswith("R_OUT +") for s in stores)
    assert any(s.startswith("NORM_OUT +") for s in stores)
    assert not subprocess.check_output(["git","-C",str(root),"diff",base,"--",
                                        "vllm_fl","csrc/iluvatar/native_w8a8.cu"],text=True)
    bench=ast.parse((root/"tools/bench_biv150_bounded_norm_quant.py").read_text())
    local_modules=[n.module for n in ast.walk(bench) if isinstance(n,ast.ImportFrom)
                   and n.module and n.module.startswith(("biv150_","audit_biv150_","bench_biv150_"))]
    assert all((root/"tools"/(name+".py")).is_file() for name in local_modules)
    original=ast.parse(subprocess.check_output(["git","-C",str(root),"show",
        "8a1fc9bb05689b4b781ba8b065416bdfb459bcb5:tools/audit_biv150_residual_reference.py"],text=True))
    extracted=ast.parse((root/"tools/biv150_residual_cpu_reference.py").read_text())
    for name in ("cpu_quantize","cpu_output"):
        before=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        after=next(n for n in extracted.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    fused = ast.parse((root/"tools/biv150_live_carry_fused_gemm.py").read_text())
    raw_f09=(root/"docs/evidence/bounded-norm-quant-20261024/f09-provenance/source-3756f22.py.txt").read_bytes()
    assert hashlib.sha1(b"blob "+str(len(raw_f09)).encode()+b"\0"+raw_f09).hexdigest()=="d86b7fa9abf97f6027e3004252ee7f700a4deff3"
    f09 = ast.parse(raw_f09)
    for name in ("fused_gate","from_quantized"):
        before=next(n for n in f09.body if isinstance(n,ast.FunctionDef) and n.name==name)
        after=next(n for n in fused.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
    result=dict(cpu_only=True,base=base,cpp_shipping_functions_unchanged=True,
                production_cuda_and_python_unchanged=True,
                triton_inputs_not_store_targets=True,kernel_abi_checked=True,
                previous_norm_quant_math_ast_unchanged=True,
                residual_store_guarded_by_compile_time_flag=True,
                f09_from_quantized_math_ast_unchanged=True,
                local_probe_imports_present=local_modules,cpu_oracle_math_ast_unchanged=True,
                real_alias_and_device_checks_pending=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    main(parser.parse_args())
