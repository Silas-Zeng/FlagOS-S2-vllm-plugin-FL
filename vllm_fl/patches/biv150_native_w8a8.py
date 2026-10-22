"""Optional native operator experiment; only enabled explicitly for evaluation."""
import os
import re
from pathlib import Path

import torch

_LOADED = False
_STATE = {"quantized_layers": 0, "compressed_weight_bytes_saved": 0}


def _eligible_model():
    try:
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config().model_config.hf_config
        expected = dict(hidden_size=2048, intermediate_size=6144, num_hidden_layers=42,
                        num_attention_heads=16, num_key_value_heads=2, vocab_size=130560)
        return all(getattr(config, k, None) == v for k, v in expected.items())
    except (AttributeError, AssertionError, RuntimeError):
        return False


def load_library():
    global _LOADED
    if _LOADED:
        return
    library = os.environ.get("BIV150_NATIVE_W8A8_LIBRARY")
    if not library:
        raise RuntimeError("Build native_w8a8.cu and set BIV150_NATIVE_W8A8_LIBRARY")
    torch.ops.load_library(library)

    @torch.library.register_fake("biv150_native::linear")
    def fake(x, wq, scales, original, bias, min_rows):
        return torch.empty((*x.shape[:-1], wq.shape[1]), dtype=x.dtype, device=x.device)

    _LOADED = True


def quantize_weight(weight):
    wide = weight.float()
    scales = wide.abs().amax(1).clamp_min(1.e-6) / 127.0
    q = torch.round(wide / scales[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.t().contiguous(), scales.contiguous()


def install():
    if os.environ.get("VLLM_FL_BIV150_NATIVE_W8A8", "0") != "1":
        return {"installed": False}
    load_library()
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    if getattr(UnquantizedLinearMethod, "_biv150_native_w8a8", False):
        return {"installed": True, "already": True}
    after_loading = UnquantizedLinearMethod.process_weights_after_loading
    apply_original = UnquantizedLinearMethod.apply
    selected = tuple(os.environ.get("BIV150_NATIVE_W8A8_LAYERS", "gate_up_proj").split(","))
    min_rows = int(os.environ.get("BIV150_NATIVE_W8A8_MIN_ROWS", "1024"))
    compressed = os.environ.get("BIV150_NATIVE_W8A8_COMPRESSED_ONLY", "0") == "1"
    skip_edges = int(os.environ.get("BIV150_NATIVE_W8A8_SKIP_EDGE_LAYERS", "0"))

    def after(self, layer):
        after_loading(self, layer)
        index = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.prefix)
        if index is not None and skip_edges and (
            int(index.group(1)) < skip_edges or int(index.group(1)) >= 42-skip_edges
        ):
            return
        weight = getattr(layer, "weight", None)
        if _eligible_model() and weight is not None and weight.dtype == torch.bfloat16 and tuple(weight.shape) in (
            (2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)
        ) and any(
            layer.prefix.endswith("." + name) for name in selected
        ):
            wq, scales = quantize_weight(weight.data)
            layer.register_buffer("_biv150_native_wq", wq, persistent=False)
            layer.register_buffer("_biv150_native_sw", scales, persistent=False)
            if compressed:
                # Preserve the public weight shape without retaining its storage.
                marker = torch.zeros((), dtype=weight.dtype, device=weight.device).expand_as(weight)
                layer.weight = torch.nn.Parameter(marker, requires_grad=False)
                _STATE["compressed_weight_bytes_saved"] += weight.numel() * weight.element_size() - wq.numel()
            _STATE["quantized_layers"] += 1
            if _STATE["quantized_layers"] == 1:
                print("BIV150_NATIVE_W8A8_FIRST_WEIGHT", layer.prefix,
                      tuple(wq.shape), "compressed_only", compressed, flush=True)

    def apply(self, layer, x, bias=None):
        if not hasattr(layer, "_biv150_native_wq"):
            return apply_original(self, layer, x, bias)
        y = torch.ops.biv150_native.linear(
            x.reshape(-1, x.shape[-1]), layer._biv150_native_wq,
            layer._biv150_native_sw, layer.weight, bias, min_rows)
        return y.reshape(*x.shape[:-1], layer._biv150_native_wq.shape[1])

    UnquantizedLinearMethod.process_weights_after_loading = after
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._biv150_native_w8a8 = True
    return {"installed": True, "layers": selected, "min_rows": min_rows,
            "compressed_only": compressed, "skip_edge_layers": skip_edges}
