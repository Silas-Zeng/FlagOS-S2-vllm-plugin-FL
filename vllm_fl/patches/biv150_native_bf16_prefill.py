"""Opt-in native BF16 projection for the measured 2048-token prefill shape."""
import os

import torch

from vllm_fl.compile_semantics import native_bf16_library_path
from vllm_fl.patches.biv150_native_w8a8 import _eligible_model

_LOADED = False
_GEOMETRIES = {(2560, 2048), (2048, 2048), (2048, 6144), (12288, 2048)}


def load_library():
    global _LOADED
    if _LOADED:
        return
    library = native_bf16_library_path()
    torch.ops.load_library(str(library))

    @torch.library.register_fake("biv150_native_bf16_probe::linear")
    def fake(x, weight):
        return x.new_empty((x.shape[0], weight.shape[0]))

    _LOADED = True


def install():
    if os.environ.get("VLLM_FL_BIV150_NATIVE_BF16_PREFILL", "0") != "1":
        return {"installed": False}
    if not _eligible_model():
        return {"installed": False, "reason": "unsupported model geometry"}
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if get_tensor_model_parallel_world_size() != 1:
        raise RuntimeError("Native BF16 prefill experiment requires TP=1")
    if getattr(UnquantizedLinearMethod, "_biv150_native_bf16_prefill", False):
        return {"installed": True, "already": True}
    load_library()
    original = UnquantizedLinearMethod.apply

    def apply(self, layer, x, bias=None):
        weight = layer.weight
        if (
            not hasattr(layer, "_biv150_native_wq")
            and bias is None
            and x.ndim == 2
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.is_contiguous()
            and weight.is_contiguous()
            and tuple(weight.shape) in _GEOMETRIES
        ):
            return torch.ops.biv150_native_bf16_probe.linear(x, weight)
        return original(self, layer, x, bias)

    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._biv150_native_bf16_prefill = True
    return {"installed": True, "rows": 2048, "library": str(native_bf16_library_path())}
