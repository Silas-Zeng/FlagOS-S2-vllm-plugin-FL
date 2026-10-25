"""Opt-in BF16 small-row O projection, retaining the original route above8 rows."""
import os

import torch
import triton
import triton.language as tl

_DEFINITION = None
_IMPLEMENTATION = None
_INSTALLED = False


@triton.jit(do_not_specialize=['M'])
def _small_o_kernel(X, W, Y, M, XS0: tl.constexpr, XS1: tl.constexpr,
                    WS0: tl.constexpr, WS1: tl.constexpr,
                    YS0: tl.constexpr, YS1: tl.constexpr):
    # Masked16-row tile fits a four-warp layout. The measured original M2/M8
    # implementation uses a much larger warp allocation and is unstable.
    row = tl.arange(0,16)
    col = tl.program_id(0)*64 + tl.arange(0,64)
    inner = tl.arange(0,128)
    accumulator = tl.zeros((16,64),tl.float32)
    for block in range(16):
        k = block*128 + inner
        a = tl.load(X + row[:,None]*XS0 + k[None,:]*XS1,
                    mask=row[:,None]<M,other=0)
        b = tl.load(W + col[None,:]*WS0 + k[:,None]*WS1)
        accumulator = tl.dot(a,b,accumulator)
    tl.store(Y + row[:,None]*YS0 + col[None,:]*YS1,
             accumulator.to(tl.bfloat16),mask=row[:,None]<M)


def _small_o_into(x, weight, output):
    _small_o_kernel[(32,)](x,weight,output,x.shape[0],
                           XS0=x.stride(0),XS1=x.stride(1),
                           WS0=weight.stride(0),WS1=weight.stride(1),
                           YS0=output.stride(0),YS1=output.stride(1),
                           num_warps=4,num_stages=1)
    return output


def _linear(x, weight):
    if not (x.ndim==weight.ndim==2 and x.shape[1]==2048
            and tuple(weight.shape)==(2048,2048)
            and x.dtype==weight.dtype==torch.bfloat16
            and x.device==weight.device and x.device.type=='cuda'
            and 0<x.shape[0]<=8 and x.stride(1)==weight.stride(1)==1
            and x.stride(0)>=2048 and weight.stride(0)>=2048):
        return torch.mm(x,weight.t())
    with torch.cuda.device(x.device):
        output=torch.empty((x.shape[0],2048),device=x.device,dtype=x.dtype)
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
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    _register()
    previous=UnquantizedLinearMethod.apply
    def apply(self,layer,x,bias=None):
        if (bias is not None or not layer.prefix.endswith('.self_attn.o_proj')
                or tuple(layer.weight.shape)!=(2048,2048)
                or layer.weight.dtype!=torch.bfloat16):
            return previous(self,layer,x,bias)
        # M dispatch is inside the opaque CUDA implementation, keeping vLLM's
        # dynamic row dimension out of Python/Dynamo shape branches.
        output=torch.ops.biv150_stable_small_o.linear(x.reshape(-1,2048),layer.weight)
        return output.reshape(*x.shape[:-1],2048)
    UnquantizedLinearMethod.apply=apply
    _INSTALLED=True
    return {'installed':True,'scope':'BF16 TP1 O projection M1-8',
            'block_m':16,'block_n':64,'block_k':128,'warps':4,'stages':1,
            'larger_rows':'original FlagGems mm inside the opaque operation'}
