"""M64 MiniCPM norm/quant + F09 gate; original carry stays in the FX graph."""
import ast
import hashlib
import inspect
import json
import linecache
import os
from pathlib import Path
import textwrap

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

CONFIGS = ((64,64,128,8),)
SELECTED_ROWS = (64,)
_ACTIVATION_TYPE = None
_SELECTED = False
EXPECTED_LLAMA_SHA256 = "1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384"

@triton.jit
def _norm_quant(
    X,
    R,
    R_OUT,
    W,
    Q,
    S,
    NORM_OUT,
    M: tl.constexpr,
    EPS,
    RESIDUAL: tl.constexpr,
    WRITE_NORM: tl.constexpr,
    WRITE_CARRY: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 2048)
    if row < M:
        x = tl.load(X + row * 2048 + cols).to(tl.float32)
        r = tl.load(R + row * 2048 + cols).to(tl.float32)
        x += r
        # Preserve BF16 residual storage but normalize the unrounded FP32 sum.
        if WRITE_CARRY:
            tl.store(R_OUT + row * 2048 + cols, x)
        var = tl.sum(x * x, axis=0) / 2048
        rrms = libdevice.rsqrt(var + EPS)
        w = tl.load(W + cols)
        # The selected generated native kernel keeps FP32 until after weight.
        unweighted = x * rrms
        y = (unweighted * w.to(tl.float32)).to(tl.bfloat16)
        if WRITE_NORM:
            tl.store(NORM_OUT + row * 2048 + cols, y)
        y32 = y.to(tl.float32)
        peak = tl.max(tl.abs(y32), axis=0)
        scale = tl.div_rn(tl.maximum(peak, 1e-6), 127.0)
        f = tl.div_rn(y32, scale)
        rounded = tl.where(f >= 0, tl.floor(f + 0.5), tl.ceil(f - 0.5))
        first = tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8)
        tl.store(Q + row * 2048 + cols, first)
        tl.store(S + row, scale)
        if RESIDUAL:
            second_scale = tl.div_rn(scale, 254.0)
            residual = tl.div_rn(y32 - first.to(tl.float32) * scale, second_scale)
            second = tl.where(
                residual >= 0, tl.floor(residual + 0.5), tl.ceil(residual - 0.5)
            )
            second = tl.minimum(127.0, tl.maximum(-127.0, second)).to(tl.int8)
            tl.store(Q + (row + 16) * 2048 + cols, second)
            tl.store(S + row + 16, second_scale)
    else:
        tl.store(Q + row * 2048 + cols, 0)
        tl.store(S + row, 1.0)
        if RESIDUAL:
            tl.store(Q + (row + 16) * 2048 + cols, 0)
            tl.store(S + row + 16, 1.0)

def norm_quant(x, residual, weight, eps, write_norm=False, materialize_residual=True):
    """Non-mutating component; unknown row counts are not an eligible route."""
    assert x.is_cuda and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
    assert x.shape == residual.shape and x.ndim == 2 and x.shape[1] == 2048
    assert x.device == residual.device == weight.device and weight.shape == (2048,)
    assert x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    rows = x.shape[0]
    assert rows in SELECTED_ROWS
    padded = max(32, rows)
    q = torch.empty((padded, 2048), device=x.device, dtype=torch.int8)
    scales = torch.empty((padded,), device=x.device, dtype=torch.float32)
    residual_out = torch.empty_like(residual) if materialize_residual else None
    normalized_out = torch.empty_like(x) if write_norm else x
    # WRITE_NORM writes to a separate output too: inputs and aliases stay intact.
    kernel = _norm_quant[(16 if rows <= 16 else padded,)](
        x, residual, residual_out if materialize_residual else x,
        weight, q, scales, normalized_out, rows, eps,
        RESIDUAL=rows <= 16, WRITE_NORM=write_norm,
        WRITE_CARRY=materialize_residual, num_warps=4,
    )
    return q, scales, residual_out, normalized_out, kernel

@triton.jit
def fused_gate(AQ,WQ,SA,SW,OUT,M:tl.constexpr,BM:tl.constexpr,
               BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    gate_acc=tl.zeros((BM,BN),tl.int32)
    up_acc=tl.zeros((BM,BN),tl.int32)
    for offset in range(0,2048,BK):
        k=offset+kk
        a=tl.load(AQ+row[:,None]*2048+k[None,:],row[:,None]<M,other=0)
        g=tl.load(WQ+k[:,None]*12288+col[None,:],col[None,:]<6144,other=0)
        u=tl.load(WQ+k[:,None]*12288+6144+col[None,:],col[None,:]<6144,other=0)
        gate_acc=tl.dot(a,g,gate_acc,out_dtype=tl.int32)
        up_acc=tl.dot(a,u,up_acc,out_dtype=tl.int32)
    rs=tl.load(SA+row,row<M,other=1)
    gs=tl.load(SW+col,col<6144,other=1)
    us=tl.load(SW+6144+col,col<6144,other=1)
    # Native epilogue groups row/column scale multiplication before applying
    # the INT32 accumulator and rounds each projection before activation.
    gate=(gate_acc.to(tl.float32)*(rs[:,None]*gs[None,:])).to(tl.bfloat16).to(tl.float32)
    up=(up_acc.to(tl.float32)*(rs[:,None]*us[None,:])).to(tl.bfloat16).to(tl.float32)
    activated=tl.div_rn(gate,1.0+libdevice.exp(-gate))
    tl.store(OUT+row[:,None]*6144+col[None,:],activated*up,
             (row[:,None]<M)&(col[None,:]<6144))

def from_quantized(aq,sa,wq,sw,rows,config):
    assert 17<=rows<=2048 and config in CONFIGS
    assert aq.shape==(max(32,rows),2048) and aq.dtype==torch.int8 and aq.is_contiguous()
    assert wq.shape==(2048,12288) and wq.dtype==torch.int8 and wq.is_contiguous()
    assert sa.shape==(max(32,rows),) and sw.shape==(12288,)
    assert sa.dtype==sw.dtype==torch.float32 and sa.is_contiguous() and sw.is_contiguous()
    assert aq.is_cuda and aq.device==sa.device==wq.device==sw.device
    output=torch.empty((rows,6144),device=aq.device,dtype=torch.bfloat16)
    bm,bn,bk,warps=config
    fused_gate[(triton.cdiv(rows,bm),triton.cdiv(6144,bn))](
        aq,wq,sa,sw,output,rows,bm,bn,bk,num_warps=warps,num_stages=1)
    return output


@torch.library.custom_op("biv150_live_carry::gate_m64", mutates_args=())
def gate_m64(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
             wq: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    global _SELECTED
    assert tuple(x.shape) == (64,2048)
    if not _SELECTED:
        print("BIV150_LIVE_CARRY_M64_SELECTED", flush=True)
        _SELECTED = True
    q, s, carry, _, _ = norm_quant(x,residual,weight,1e-6,materialize_residual=False)
    assert carry is None
    return from_quantized(q,s,wq,scales,64,(64,64,128,8))


@gate_m64.register_fake
def fake_gate_m64(x,residual,weight,wq,scales):
    return torch.empty((x.shape[0],6144),dtype=x.dtype,device=x.device)


def _supported(layer,x,residual):
    if (residual is None or x.ndim!=2 or tuple(x.shape)!=(64,2048)
        or not x.is_cuda or x.dtype!=torch.bfloat16 or not x.is_contiguous()
        or tuple(residual.shape)!=tuple(x.shape) or residual.dtype!=x.dtype
        or residual.device!=x.device or not residual.is_contiguous()):
        return False
    norm=layer.post_attention_layernorm
    mlp=layer.mlp
    linear=mlp.gate_up_proj
    if (not isinstance(mlp.act_fn,_ACTIVATION_TYPE) or linear.bias is not None
        or norm.variance_epsilon!=1e-6 or norm.variance_size_override is not None
        or not norm.has_weight or not hasattr(linear,"_biv150_native_wq")):
        return False
    nw=norm.weight;wq=linear._biv150_native_wq;sw=linear._biv150_native_sw
    return (tuple(nw.shape)==(2048,) and nw.dtype==torch.bfloat16 and nw.is_contiguous()
        and nw.device==x.device and tuple(wq.shape)==(2048,12288)
        and wq.dtype==torch.int8 and wq.is_contiguous() and wq.device==x.device
        and tuple(sw.shape)==(12288,) and sw.dtype==torch.float32
        and sw.is_contiguous() and sw.device==x.device
        and tuple(linear.weight.shape)==(12288,2048)
        and linear.weight.dtype==torch.bfloat16 and linear.weight.stride()==(0,0)
        and linear.weight.device==x.device)


def _post_norm_mlp(layer,hidden_states,residual):
    if not _supported(layer,hidden_states,residual):
        hidden_states,residual=layer.post_attention_layernorm(hidden_states,residual)
        return layer.mlp(hidden_states),residual
    # The unused norm output is eliminated by Inductor. Keep the original
    # transparent residual expression; never substitute a fused BF16 carry.
    _,carry=layer.post_attention_layernorm(hidden_states,residual)
    linear=layer.mlp.gate_up_proj
    gate=gate_m64(hidden_states,residual,layer.post_attention_layernorm.weight,
                  linear._biv150_native_wq,linear._biv150_native_sw)
    output,_=layer.mlp.down_proj(gate)
    return output,carry


def _rewrite_forward_ast(source):
    module=ast.parse(textwrap.dedent(source))
    assert len(module.body)==1 and isinstance(module.body[0],ast.FunctionDef)
    function=module.body[0]
    assert function.name=="forward" and not function.decorator_list
    pair=ast.parse("hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)\nhidden_states = self.mlp(hidden_states)").body
    indices=[i for i in range(len(function.body)-1)
        if all(ast.dump(function.body[i+j],include_attributes=False)==ast.dump(pair[j],include_attributes=False) for j in (0,1))]
    assert len(indices)==1, "Revalidate changed Llama norm/MLP structure"
    index=indices[0]
    replacement=ast.parse("hidden_states, residual = _biv150_post_norm_mlp(self, hidden_states, residual)").body
    function.body[index:index+2]=replacement
    return ast.fix_missing_locations(module)


def install():
    global _ACTIVATION_TYPE
    if os.environ.get("VLLM_FL_BIV150_LIVE_CARRY_GATE","1")!="1":
        return {"installed":False,"reason":"disabled"}
    from vllm.model_executor.models.llama import LlamaDecoderLayer
    from vllm.model_executor.layers.activation import SiluAndMul
    _ACTIVATION_TYPE=SiluAndMul
    if getattr(LlamaDecoderLayer,"_biv150_live_carry",False):
        return {"installed":True,"already":True}
    original=LlamaDecoderLayer.forward
    path=Path(inspect.getsourcefile(original))
    if (original.__qualname__!="LlamaDecoderLayer.forward"
        or original.__code__.co_filename!=str(path)
        or hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_LLAMA_SHA256):
        return {"installed":False,"reason":"unaudited or already wrapped decoder forward"}
    rewritten=_rewrite_forward_ast(inspect.getsource(original))
    generated=str(Path(__file__).resolve())+".generated_forward.py"
    source=ast.unparse(rewritten)+"\n"
    linecache.cache[generated]=(len(source),None,source.splitlines(keepends=True),generated)
    # Dynamo resolves global guards through the function's real module. A copied
    # globals dict can execute eagerly but leaves that module binding missing.
    namespace=original.__globals__
    key="_biv150_post_norm_mlp"
    if key in namespace and namespace[key] is not _post_norm_mlp:
        return {"installed":False,"reason":"foreign helper binding"}
    namespace[key]=_post_norm_mlp
    local={}
    exec(compile(source,generated,"exec"),namespace,local)
    forward=local["forward"]
    forward.__qualname__=original.__qualname__
    LlamaDecoderLayer.forward=forward
    LlamaDecoderLayer._biv150_live_carry=True
    return dict(installed=True,exact_rows=[64],epsilon=1e-6,
                norm_weight_shape=[2048],compressed_gate_shape=[2048,12288],
                original_transparent_carry=True,persistent_weight_copy=False,
                original_llama_sha256=EXPECTED_LLAMA_SHA256,
                generated_forward_sha256=hashlib.sha256(source.encode()).hexdigest(),
                generated_forward_file=generated,
                hook_file=str(Path(__file__).resolve()),
                hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
