# SPDX-License-Identifier: Apache-2.0
"""One-pass add/RMSNorm for compatible compiled BF16 layers."""
import os
import types
import torch
from vllm.model_executor.layers.layernorm import RMSNorm


@torch.library.custom_op("vllm_fl::metax_residual_rms",mutates_args=(),device_types="cuda")
def _norm(x:torch.Tensor,residual:torch.Tensor,weight:torch.Tensor,epsilon:float
          )->tuple[torch.Tensor,torch.Tensor]:
    from flag_gems.runtime.backend._metax.ops.residual_rms import residual_rms
    return residual_rms(x,residual,weight,epsilon)


@_norm.register_fake
def _norm_fake(x,residual,weight,epsilon):
    return torch.empty_like(x),torch.empty_like(residual)


def _forward(self,x,residual=None):
    if (residual is None or x.ndim!=2 or x.shape[-1]!=2048 or residual.shape!=x.shape
            or x.dtype!=torch.bfloat16 or residual.dtype!=x.dtype
            or x.device.type!='cuda' or residual.device!=x.device
            or not x.is_contiguous() or not residual.is_contiguous()
            or x.requires_grad or residual.requires_grad):
        return RMSNorm.forward_native(self,x,residual)
    return _norm(x,residual,self.weight.data,self.variance_epsilon)


def install(model):
    if os.environ.get('VLLM_FL_METAX_RESIDUAL_RMS','1')!='1':
        return 0
    from vllm_fl.ops.layernorm import RMSNormFL
    count=0
    for layer in model.modules():
        if (type(layer) in (RMSNorm,RMSNormFL) and layer.hidden_size==2048
                and layer.variance_size_override is None and layer.pass_weight_add
                and layer.weight.dtype==torch.bfloat16 and layer.weight.is_contiguous()
                and getattr(getattr(layer,'_forward_method',None),'__func__',None)
                    is RMSNorm.forward_native):
            layer._forward_method=types.MethodType(_forward,layer)
            count+=1
    return count
