"""Generate an independent 2-row/16-warp precision-repair probe, no model hook.

The original two reduction/output loops are copied from the exact service
kernel. Only the final BF16 output store is made optional and extended with
quantization. Matching topology does not prove matching emitted instructions;
GPU bitwise comparisons against the original kernel are mandatory.
"""
import ast
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'docs/evidence/live-carry-service-20261025/postcompile-reference-norm.py.txt'
REFERENCE_SHA256='a80edc5a687ebceb25c78d2c1b2cf45093f6501955c5b485a38565e90a121026'
NAME='triton_red_fused_fused_add_rms_norm_gate_silu_auto_0'


def generated_source():
    source=REFERENCE.read_bytes()
    assert hashlib.sha256(source).hexdigest()==REFERENCE_SHA256
    original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==NAME)
    node=copy.deepcopy(original)
    node.name='original_topology_norm_quant'
    node.decorator_list=ast.parse('@triton.jit\ndef unused(): pass').body[0].decorator_list
    for name in ('Q','S','WRITE_NORM'):
        node.args.args.append(ast.arg(arg=name,annotation=ast.parse('tl.constexpr',mode='eval').body if name=='WRITE_NORM' else None))
    assert isinstance(node.body[-1],ast.For)
    store=node.body[-1].body[-1]
    assert isinstance(store,ast.Expr) and ast.unparse(store.value.func)=='tl.store'
    quant=ast.parse('''
if WRITE_NORM:
    tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp22, r0_mask & xmask)
norm_bf16 = tmp22.to(tl.bfloat16)
norm_f32 = norm_bf16.to(tl.float32)
peak = tl.max(tl.abs(norm_f32), 1)[:, None]
scale = tl.div_rn(tl.maximum(peak, 1e-6), 127.0)
scaled = tl.div_rn(norm_f32, scale)
rounded = tl.where(scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
quantized = tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8)
tl.store(Q + (r0_1 + 2048*x0), quantized, r0_mask & xmask)
tl.store(S + x0, scale, xmask)
''').body
    assert ast.dump(quant[0].body[0],include_attributes=False)==ast.dump(store,include_attributes=False)
    node.body[-1].body[-1:]=quant
    # The quantization max must span the entire row; chunked variants are not
    # admitted. This corresponds to the saved service configuration, not a sweep.
    guards=ast.parse('tl.static_assert(XBLOCK == 2)\ntl.static_assert(R0_BLOCK == 2048)').body
    node.body[:0]=guards
    node=ast.fix_missing_locations(node)
    # Restore exactly the store and remove only the new guards/signature. Prove
    # every original load, cast, addition, loop and variance expression survives.
    restored=copy.deepcopy(node)
    restored.name=original.name;restored.decorator_list=original.decorator_list
    restored.args=original.args;restored.body=restored.body[2:]
    restored.body[-1].body[-len(quant):]=[copy.deepcopy(store)]
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    header='import torch\nimport triton\nimport triton.language as tl\nfrom torch._inductor.runtime.triton_helpers import libdevice\n\n'
    wrapper='''
def norm_quant(x, residual, weight, write_norm=False):
    assert x.shape == residual.shape == (64,2048) and weight.shape == (2048,)
    assert x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.device == residual.device == weight.device
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    q=torch.empty((64,2048),device=x.device,dtype=torch.int8)
    s=torch.empty((64,),device=x.device,dtype=torch.float32)
    y=torch.empty_like(x) if write_norm else x
    kernel=original_topology_norm_quant[(32,)](x,residual,weight,y,64,2048,2,2048,q,s,
        WRITE_NORM=write_norm,num_warps=16,num_stages=1,enable_fp_fusion=True)
    return q,s,y,kernel
'''
    return header+ast.unparse(node)+'\n'+wrapper


def load_file(path,source,module_name):
    """New artifact file only; Triton's inspect can read its physical source."""
    assert module_name not in sys.modules
    path=Path(path)
    with path.open('x') as file:file.write(source)
    spec=importlib.util.spec_from_file_location(module_name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=module
    spec.loader.exec_module(module)
    return module
