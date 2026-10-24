"""Check immutable-input kernel ABI and preserve existing production functions."""

import argparse
import ast
import json
import subprocess
from pathlib import Path


def main(args):
    root=Path(__file__).resolve().parents[1]
    base="31a0a7f9205d90904afac0ba19e9e65856c79867"
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
        "X","R","R_OUT","W","Q","S","NORM_OUT","M","EPS","RESIDUAL","WRITE_NORM"]
    stores=[ast.unparse(n.args[0]) for n in ast.walk(kernel)
            if isinstance(n,ast.Call) and ast.unparse(n.func)=="tl.store"]
    assert not any(s.startswith("X +") or s.startswith("R +") for s in stores)
    assert any(s.startswith("R_OUT +") for s in stores)
    assert any(s.startswith("NORM_OUT +") for s in stores)
    assert not subprocess.check_output(["git","-C",str(root),"diff",base,"--",
                                        "vllm_fl","csrc/iluvatar/native_w8a8.cu"],text=True)
    result=dict(cpu_only=True,base=base,cpp_shipping_functions_unchanged=True,
                production_cuda_and_python_unchanged=True,
                triton_inputs_not_store_targets=True,kernel_abi_checked=True,
                real_alias_and_device_checks_pending=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    main(parser.parse_args())
