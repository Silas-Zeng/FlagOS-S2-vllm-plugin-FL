"""Inert access to the pinned SDK linear kernel with caller output.

Equivalent bias-free launch ABI to the installed Apache-2.0 FlagGems Iluvatar
linear wrapper (SHA below). No SDK symbol replacement; initialized only by an opted-in scoped installer.
Rows1-8 are intentionally rejected: the existing own BF16 protection is required.
"""
import hashlib
import importlib
import inspect
from pathlib import Path

LINEAR_SHA='d431a4fa144ed67dc3deab46dd5beace3612858075f85c39e602447ba3bf768a'
LIBENTRY_SHA='414e8a3f74e2522e40659e7fccd84e301f4517220837a87af0311f050bbf3025'
GEOMETRIES=((2048,2048),(2560,2048),(12288,2048),(2048,6144))
_KERNEL=None
_PROVENANCE=None


def initialize():
    global _KERNEL,_PROVENANCE
    if _KERNEL is not None:return _PROVENANCE
    import flag_gems
    function=inspect.unwrap(flag_gems.linear)
    module_name=getattr(function,'__module__','')
    if module_name not in ('_iluvatar.ops.linear','flag_gems.runtime.backend._iluvatar.ops.linear'):
        raise RuntimeError('Unsupported selected linear provider: '+module_name)
    module=importlib.import_module(module_name)
    source=Path(inspect.getsourcefile(module))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=LINEAR_SHA:
        raise RuntimeError('Selected linear SDK source changed')
    if function is not module.linear or function.__globals__ is not module.__dict__:
        raise RuntimeError('Selected linear wrapper globals changed')
    kernel=module.linear_kernel;entry_source=Path(inspect.getsourcefile(type(kernel)))
    if hashlib.sha256(entry_source.read_bytes()).hexdigest()!=LIBENTRY_SHA:
        raise RuntimeError('Selected linear LibEntry changed')
    _PROVENANCE=dict(module=module_name,linear_sha256=LINEAR_SHA,libentry_sha256=LIBENTRY_SHA,
                     kernel_type=type(kernel).__module__+'.'+type(kernel).__qualname__,
                     bias=False,group_m=8,split_k=1,sdk_mutated=False)
    _KERNEL=kernel
    return _PROVENANCE


def linear_into(x,weight,out):
    import torch
    if not (x.ndim==weight.ndim==out.ndim==2 and x.shape[0]>8
            and tuple(weight.shape) in GEOMETRIES and x.shape[1]==weight.shape[1]
            and tuple(out.shape)==(x.shape[0],weight.shape[0])
            and x.dtype==weight.dtype==out.dtype==torch.bfloat16
            and x.device==weight.device==out.device and x.device.type=='cuda'
            and x.stride(1)==weight.stride(1)==out.stride(1)==1
            and x.stride(0)>=x.shape[1] and weight.stride(0)==weight.shape[1]
            and out.stride(0)>=out.shape[1]
            and out.untyped_storage().data_ptr() not in (x.untyped_storage().data_ptr(),weight.untyped_storage().data_ptr())
            and not x.requires_grad and not weight.requires_grad and not out.requires_grad):
        raise ValueError('Unsupported provided-output linear scope; keep protected fallback')
    initialize()
    import triton
    m,k=x.shape;n=weight.shape[0]
    grid=lambda meta:(triton.cdiv(m,meta['BLOCK_M'])*triton.cdiv(n,meta['BLOCK_N']),meta.get('SPLIT_K',1))
    with torch.cuda.device(x.device):
        # Exactly the pinned bias-free SDK wrapper, including its unused bias
        # pointer alias to weight; only the output allocation is supplied.
        _KERNEL[grid](x,weight,weight,out,m,n,k,x.stride(0),x.stride(1),
                      weight.stride(0),weight.stride(1),out.stride(0),out.stride(1),0,
                      BIAS=False,GROUP_M=8,SPLIT_K=1)
    return out
