"""Bound and store-target checks for the new fused INT8 component."""

import ast
import json
import subprocess
from pathlib import Path


root=Path(__file__).resolve().parents[1]
module=ast.parse((root/'tools/biv150_int8_gemm_silu.py').read_text())
kernel=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='fused_gate')
assert [a.arg for a in kernel.args.args]==['AQ','WQ','SA','SW','OUT','M','BM','BN','BK']
stores=[n for n in ast.walk(kernel) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store']
assert len(stores)==1 and ast.unparse(stores[0].args[0]).startswith('OUT +')
assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.dot' for n in ast.walk(kernel))==2
assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to'
           and any(ast.unparse(a)=='tl.bfloat16' for a in n.args) for n in ast.walk(kernel))==2
assert 2048*127*127 < 2**31
assert 2048*12288 < 2**31
assert not subprocess.check_output(['git','-C',str(root),'diff','0bf9d2efeaa10144dd7f0ff4597298df029f9328','--','csrc','vllm_fl'],text=True)
print(json.dumps(dict(cpu_only=True,only_output_store=True,no_int32_global_output=True,
    two_integer_accumulators=True,int32_dot_bound=2048*127*127,
    bf16_boundary_before_activation=True,production_unchanged=True,
    device_numeric_and_performance_pending=True)))
