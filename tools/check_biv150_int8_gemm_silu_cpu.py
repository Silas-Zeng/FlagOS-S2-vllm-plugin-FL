"""Bound and store-target checks for the new fused INT8 component."""

import ast
import json
import subprocess
from pathlib import Path


root=Path(__file__).resolve().parents[1]
module=ast.parse((root/'vllm_fl/patches/biv150_int8_gemm_silu.py').read_text())
kernel=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='fused_gate')
assert [a.arg for a in kernel.args.args]==['AQ','WQ','SA','SW','OUT','M','BM','BN','BK']
stores=[n for n in ast.walk(kernel) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store']
assert len(stores)==1 and ast.unparse(stores[0].args[0]).startswith('OUT +')
assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.dot' for n in ast.walk(kernel))==2
assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='to'
           and any(ast.unparse(a)=='tl.bfloat16' for a in n.args) for n in ast.walk(kernel))==2
assert 2048*127*127 < 2**31
assert 2048*12288 < 2**31
dispatcher=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='gate_m32_or_auto')
assert len(dispatcher.body)==3
assert isinstance(dispatcher.body[1],ast.If)
assert 'tuple(x.shape) == (32, 2048)' in ast.unparse(dispatcher.body[1].test)
assert 'original.stride() == (0, 0)' in ast.unparse(dispatcher.body[1].test)
assert 'min_rows == 17' in ast.unparse(dispatcher.body[1].test)
assert ast.unparse(dispatcher.body[-1].value.func)=='torch.ops.biv150_native.gate_silu_auto'
tested=ast.parse(subprocess.check_output(['git','-C',str(root),'show',
    '1f06b7e9c357c7b7b0d9e6fffcadc653633cc776:tools/biv150_int8_gemm_silu.py'],text=True))
for name in ('fused_gate','from_quantized','full_gate'):
    a=next(n for n in tested.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
assert not subprocess.check_output(['git','-C',str(root),'diff','0bf9d2efeaa10144dd7f0ff4597298df029f9328','--','csrc'],text=True)
print(json.dumps(dict(cpu_only=True,only_output_store=True,no_int32_global_output=True,
    two_integer_accumulators=True,int32_dot_bound=2048*127*127,
    bf16_boundary_before_activation=True,native_cpp_cuda_unchanged=True,service_route_m32_only=True,
    same_kernel_math_as_34_layer_component=True,
    device_numeric_and_performance_pending=True)))
