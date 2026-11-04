# SPDX-License-Identifier: Apache-2.0
"""Transposed weight caches for BF16 QKV/O prefill GEMM on MetaX."""
import functools
import os

import torch
from vllm.model_executor.layers.linear import (
    QKVParallelLinear, RowParallelLinear, UnquantizedLinearMethod,
)
from vllm.model_executor.models.llama import LlamaForCausalLM

from flag_gems.runtime.backend._metax.ops.prefill_linear import prefill_linear_nn

_original_apply = UnquantizedLinearMethod.apply
_original_load_weights = None
_odd_split = os.environ.get("VLLM_FL_METAX_ODD_SPLITK", "1") == "1"


@torch.library.custom_op("vllm_fl::metax_prefill_linear_nn", mutates_args=())
def _linear(a: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    return prefill_linear_nn(a, weight, packed, odd_split=_odd_split)


@_linear.register_fake
def _linear_fake(a, weight, packed):
    return a.new_empty((a.shape[0], weight.shape[0]))


@functools.wraps(_original_apply)
def _apply(self, layer, x, bias=None):
    packed = getattr(layer, "_metax_prefill_nn_weight", None)
    if (packed is None or bias is not None or x.ndim != 2
            or x.dtype != torch.bfloat16 or x.requires_grad):
        return _original_apply(self, layer, x, bias)
    return _linear(x, layer.weight, packed)


def refresh(model):
    for layer in model.modules():
        packed = getattr(layer, "_metax_prefill_nn_weight", None)
        if packed is not None:
            if packed.ndim == 4:
                from flag_gems.runtime.backend._metax.ops.packed_mlp import pack_mlp_weight
                pack_mlp_weight(layer.weight, out=packed)
            else:
                packed.copy_(layer.weight.t())


def _load_weights(self, weights):
    assert _original_load_weights is not None
    result = _original_load_weights(self, weights)
    refresh(self)
    return result


def install(model):
    global _original_load_weights
    if os.environ.get("VLLM_FL_METAX_PREFILL_NN", "1") != "1":
        return 0
    count = 0
    for layer in model.modules():
        weight = getattr(layer, "weight", None)
        if (
            isinstance(layer, (QKVParallelLinear, RowParallelLinear))
            and type(getattr(layer, "quant_method", None)) is UnquantizedLinearMethod
            and getattr(layer, "tp_size", 1) == 1
            and getattr(layer, "bias", None) is None
            and weight is not None and weight.dtype == torch.bfloat16
            and weight.device.type == "cuda" and weight.is_contiguous()
            and weight.shape in ((2560, 2048), (2048, 2048))
        ):
            if not hasattr(layer, "_metax_prefill_nn_weight"):
                if os.environ.get("VLLM_FL_METAX_PROJECTION_PACK", "1") == "1":
                    from flag_gems.runtime.backend._metax.ops.packed_mlp import pack_mlp_weight
                    cache = pack_mlp_weight(weight)
                else:
                    cache = weight.t().contiguous()
                layer.register_buffer("_metax_prefill_nn_weight", cache, persistent=False)
            count += 1
    if count:
        if not getattr(UnquantizedLinearMethod, "_metax_prefill_nn_installed", False):
            UnquantizedLinearMethod.apply = _apply
            UnquantizedLinearMethod._metax_prefill_nn_installed = True
        if not getattr(LlamaForCausalLM, "_metax_prefill_nn_reload", False):
            _original_load_weights = LlamaForCausalLM.load_weights
            LlamaForCausalLM.load_weights = functools.wraps(_original_load_weights)(_load_weights)
            LlamaForCausalLM._metax_prefill_nn_reload = True
    return count
