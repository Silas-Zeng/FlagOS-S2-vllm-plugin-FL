"""Opt-in BF16 small-row MiniCPM projections; original linear above8 rows."""
import os

import torch
import triton
import triton.language as tl

_DEFINITION = None
_IMPLEMENTATION = None
_INSTALLED = False
_OBSERVED_ROWS = set()
_MM_SHA = '4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'


def _install_mm_route():
    """Replace only the selected FlagGems vendor launch seam, keeping mm/out API."""
    import hashlib
    import importlib
    import inspect
    from pathlib import Path
    import flag_gems
    module=importlib.import_module('flag_gems.runtime.backend._iluvatar.ops.mm')
    path=Path(inspect.getsourcefile(module))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=_MM_SHA:
        return {'installed':False,'reason':'unsupported FlagGems MM source'}
    if not all(inspect.unwrap(fn).__globals__ is module.__dict__
               for fn in (flag_gems.mm,flag_gems.mm_out)):
        return {'installed':False,'reason':'MM symbols do not use audited launch globals'}
    previous=module._launch_mm
    def launch(a,b,c,m,n,k):
        if not (1<=m<=8 and (n,k) in ((2048,2048),(2560,2048),(12288,2048),(2048,6144))
                and tuple(a.shape)==(m,k) and tuple(b.shape)==(k,n) and tuple(c.shape)==(m,n)
                and a.dtype==b.dtype==c.dtype==torch.bfloat16
                and a.is_cuda and b.is_cuda and c.is_cuda and a.device==b.device==c.device
                and a.stride(1)==b.stride(0)==c.stride(1)==1
                and a.stride(0)>=k and b.stride(1)>=k and c.stride(0)>=n):
            return previous(a,b,c,m,n,k)
        with torch.cuda.device(a.device):
            key=(m,n,k)
            if key not in _OBSERVED_ROWS and not torch.cuda.is_current_stream_capturing():
                _OBSERVED_ROWS.add(key)
                print('BIV150_STABLE_SMALL_O_SELECTED', {
                    'rows':m,'output_columns':n,'input_columns':k,'binding':'mm',
                    'input_stride':tuple(a.stride()),'rhs_stride':tuple(b.stride()),
                    'output_stride':tuple(c.stride()),'warps':4,'stages':1,
                },flush=True)
            # B is KxN; the original kernel views weights as NxK. Swap only
            # pointer strides, without a transpose/copy operation or allocation.
            _small_o_kernel[(n//64,)](a,b,c,m,K=k,
                XS0=a.stride(0),XS1=a.stride(1),
                WS0=b.stride(1),WS1=b.stride(0),
                YS0=c.stride(0),YS1=c.stride(1),num_warps=4,num_stages=1)
        return c
    module._launch_mm=launch
    return {'installed':True,'binding':'mm','scope':'BF16 TP1 MiniCPM MM/out M1-8',
            'block_m':16,'block_n':64,'block_k':128,'warps':4,'stages':1,
            'original_mm_sha256':_MM_SHA,'larger_rows':'original vendor launch unchanged'}


@triton.jit(do_not_specialize=['M'])
def _small_o_kernel(X, W, Y, M, K: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
                    WS0: tl.constexpr, WS1: tl.constexpr,
                    YS0: tl.constexpr, YS1: tl.constexpr):
    # All accepted output widths are multiples of64; only M needs masking.
    row = tl.arange(0,16)
    col = tl.program_id(0)*64 + tl.arange(0,64)
    inner = tl.arange(0,128)
    accumulator = tl.zeros((16,64),tl.float32)
    for block in range(K // 128):
        k = block*128 + inner
        a = tl.load(X + row[:,None]*XS0 + k[None,:]*XS1,
                    mask=row[:,None]<M,other=0)
        b = tl.load(W + col[None,:]*WS0 + k[:,None]*WS1)
        accumulator = tl.dot(a,b,accumulator)
    tl.store(Y + row[:,None]*YS0 + col[None,:]*YS1,
             accumulator.to(tl.bfloat16),mask=row[:,None]<M)


def _small_o_into(x, weight, output):
    _small_o_kernel[(weight.shape[0] // 64,)](x,weight,output,x.shape[0],K=x.shape[1],
                           XS0=x.stride(0),XS1=x.stride(1),
                           WS0=weight.stride(0),WS1=weight.stride(1),
                           YS0=output.stride(0),YS1=output.stride(1),
                           num_warps=4,num_stages=1)
    return output


def _linear(x, weight):
    if not (x.ndim==weight.ndim==2 and x.shape[1]==weight.shape[1]
            and tuple(weight.shape) in ((2048,2048),(12288,2048),(2560,2048),(2048,6144))
            and x.dtype==weight.dtype==torch.bfloat16
            and x.device==weight.device and x.device.type=='cuda'
            and 0<x.shape[0]<=8 and x.stride(1)==weight.stride(1)==1
            and x.stride(0)>=x.shape[1] and weight.stride(0)>=weight.shape[1]):
        # On the supported CUDA platform the original framework dispatch is
        # default_unquantized_gemm -> F.linear. Preserve that exact route.
        return torch.nn.functional.linear(x,weight)
    with torch.cuda.device(x.device):
        key=(x.shape[0],weight.shape[0],weight.shape[1])
        if key not in _OBSERVED_ROWS and not torch.cuda.is_current_stream_capturing():
            _OBSERVED_ROWS.add(key)
            print('BIV150_STABLE_SMALL_O_SELECTED', {
                'rows': x.shape[0], 'output_columns': weight.shape[0], 'input_columns': weight.shape[1], 'input_stride': tuple(x.stride()),
                'weight_stride': tuple(weight.stride()), 'warps': 4, 'stages': 1,
            }, flush=True)
        output=torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
        return _small_o_into(x,weight,output)


def _register():
    global _DEFINITION,_IMPLEMENTATION
    if _DEFINITION is not None:return
    definition=torch.library.Library('biv150_stable_small_o','DEF')
    definition.define('linear(Tensor input, Tensor weight) -> Tensor')
    implementation=torch.library.Library('biv150_stable_small_o','IMPL','CUDA')
    implementation.impl('linear',_linear)
    @torch.library.register_fake('biv150_stable_small_o::linear')
    def fake(x,weight):
        return torch.empty((x.shape[0],weight.shape[0]),dtype=x.dtype,device=x.device)
    _DEFINITION,_IMPLEMENTATION=definition,implementation


def install():
    global _INSTALLED
    if os.environ.get('VLLM_FL_BIV150_STABLE_SMALL_O','0')!='1':
        return {'installed':False,'reason':'disabled'}
    if _INSTALLED:return {'installed':True,'reason':'already installed'}
    from vllm_fl.patches.biv150_native_w8a8 import _eligible_model
    if not _eligible_model():return {'installed':False,'reason':'unsupported model'}
    from vllm.distributed import get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size()!=1:
        return {'installed':False,'reason':'TP1 required'}
    from vllm import envs
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm, default_unquantized_gemm
    if envs.VLLM_BATCH_INVARIANT or dispatch_unquantized_gemm() is not default_unquantized_gemm:
        return {'installed':False,'reason':'unsupported original GEMM dispatch'}
    binding=os.environ.get('VLLM_FL_BIV150_STABLE_SMALL_O_BINDING','linear')
    if binding=='mm':
        result=_install_mm_route()
        _INSTALLED=bool(result.get('installed'))
        return result
    if binding!='linear':return {'installed':False,'reason':'unknown small GEMM binding'}
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    _register()
    previous=UnquantizedLinearMethod.apply
    def apply(self,layer,x,bias=None):
        projection=(layer.prefix.endswith('.self_attn.o_proj') and tuple(layer.weight.shape)==(2048,2048)
                    or layer.prefix.endswith('.mlp.gate_up_proj') and tuple(layer.weight.shape)==(12288,2048)
                    or layer.prefix.endswith('.self_attn.qkv_proj') and tuple(layer.weight.shape)==(2560,2048)
                    or layer.prefix.endswith('.mlp.down_proj') and tuple(layer.weight.shape)==(2048,6144))
        if (bias is not None or not projection or hasattr(layer,'_biv150_native_wq')
                or layer.weight.dtype!=torch.bfloat16):
            return previous(self,layer,x,bias)
        # M dispatch is inside the opaque CUDA implementation, keeping vLLM's
        # dynamic row dimension out of Python/Dynamo shape branches.
        output=torch.ops.biv150_stable_small_o.linear(x.reshape(-1,layer.weight.shape[1]),layer.weight)
        return output.reshape(*x.shape[:-1],layer.weight.shape[0])
    UnquantizedLinearMethod.apply=apply
    _INSTALLED=True
    return {'installed':True,'scope':'BF16 TP1 QKV/O/down and uncompressed edge-gate projections M1-8',
            'block_m':16,'block_n':64,'block_k':128,'warps':4,'stages':1,
            'larger_rows':'original default_unquantized_gemm F.linear route inside the opaque operation'}
