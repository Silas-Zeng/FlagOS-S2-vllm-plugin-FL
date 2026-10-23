"""Native BI-V150 linear/SwiGLU path for the audited MiniCPM model geometry."""

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
        expected = dict(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=42,
            num_attention_heads=16,
            num_key_value_heads=2,
            vocab_size=130560,
        )
        return all(getattr(config, k, None) == v for k, v in expected.items())
    except (AttributeError, AssertionError, RuntimeError):
        return False


def load_library():
    global _LOADED
    if _LOADED:
        return
    library = os.environ.get(
        "BIV150_NATIVE_W8A8_LIBRARY",
        str(
            Path(__file__).resolve().parents[2]
            / "build-native-w8a8/biv150_native_w8a8.so"
        ),
    )
    if not Path(library).is_file():
        raise RuntimeError(
            "Build the BI-V150 operator with python tools/build_biv150_native_w8a8.py"
        )
    torch.ops.load_library(library)

    @torch.library.register_fake("biv150_native::linear")
    def fake(x, wq, scales, original, bias, min_rows):
        return torch.empty((*x.shape[:-1], wq.shape[1]), dtype=x.dtype, device=x.device)

    if hasattr(torch.ops.biv150_native, "gate_silu"):

        @torch.library.register_fake("biv150_native::gate_silu")
        def fake_gate_silu(x, wq, scales, original, min_rows):
            return torch.empty(
                (*x.shape[:-1], wq.shape[1] // 2), dtype=x.dtype, device=x.device
            )

    _LOADED = True


def quantize_weight(weight):
    wide = weight.float()
    scales = wide.abs().amax(1).clamp_min(1.0e-6) / 127.0
    q = torch.round(wide / scales[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.t().contiguous(), scales.contiguous()


def install():
    if os.environ.get("VLLM_FL_BIV150_NATIVE_W8A8", "1") != "1":
        return {"installed": False}
    if not _eligible_model():
        return {
            "installed": False,
            "reason": "unsupported or unavailable model geometry",
        }
    load_library()
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(UnquantizedLinearMethod, "_biv150_native_w8a8", False):
        return {"installed": True, "already": True}
    after_loading = UnquantizedLinearMethod.process_weights_after_loading
    apply_original = UnquantizedLinearMethod.apply
    selected = ("gate_up_proj",)
    min_rows = 17
    compressed = True
    skip_edges = 4
    fused_silu = True

    def after(self, layer):
        after_loading(self, layer)
        index = re.search(r"(?:^|\.)layers\.(\d+)\.", layer.prefix)
        if (
            index is not None
            and skip_edges
            and (
                int(index.group(1)) < skip_edges
                or int(index.group(1)) >= 42 - skip_edges
            )
        ):
            return
        weight = getattr(layer, "weight", None)
        if (
            _eligible_model()
            and weight is not None
            and weight.dtype == torch.bfloat16
            and tuple(weight.shape)
            in ((2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144))
            and any(layer.prefix.endswith("." + name) for name in selected)
        ):
            wq, scales = quantize_weight(weight.data)
            layer.register_buffer("_biv150_native_wq", wq, persistent=False)
            layer.register_buffer("_biv150_native_sw", scales, persistent=False)
            if compressed:
                # Preserve the public weight shape without retaining its storage.
                marker = torch.zeros(
                    (), dtype=weight.dtype, device=weight.device
                ).expand_as(weight)
                layer.weight = torch.nn.Parameter(marker, requires_grad=False)
                _STATE["compressed_weight_bytes_saved"] += (
                    weight.numel() * weight.element_size() - wq.numel()
                )
            _STATE["quantized_layers"] += 1
            if _STATE["quantized_layers"] == 1:
                print(
                    "BIV150_NATIVE_W8A8_FIRST_WEIGHT",
                    layer.prefix,
                    tuple(wq.shape),
                    "compressed_only",
                    compressed,
                    flush=True,
                )

    def apply(self, layer, x, bias=None):
        if not hasattr(layer, "_biv150_native_wq"):
            return apply_original(self, layer, x, bias)
        y = torch.ops.biv150_native.linear(
            x.reshape(-1, x.shape[-1]),
            layer._biv150_native_wq,
            layer._biv150_native_sw,
            layer.weight,
            bias,
            min_rows,
        )
        return y.reshape(*x.shape[:-1], layer._biv150_native_wq.shape[1])

    UnquantizedLinearMethod.process_weights_after_loading = after
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._biv150_native_w8a8 = True
    if fused_silu:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.models.llama import LlamaMLP

        if get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError("Native fused gate/SwiGLU currently requires TP=1")
        forward_original = LlamaMLP.forward

        def forward(self, x):
            layer = self.gate_up_proj
            if (
                not hasattr(layer, "_biv150_native_wq")
                or layer.bias is not None
                or not isinstance(self.act_fn, SiluAndMul)
                or tuple(layer.weight.shape) != (12288, 2048)
            ):
                return forward_original(self, x)
            # Operator fusion only: replace gate/up linear + activation; leave
            # down projection and all framework scheduling/dispatch intact.
            y = torch.ops.biv150_native.gate_silu(
                x.reshape(-1, x.shape[-1]),
                layer._biv150_native_wq,
                layer._biv150_native_sw,
                layer.weight,
                min_rows,
            )
            y = y.reshape(*x.shape[:-1], 6144)
            y, _ = self.down_proj(y)
            return y

        LlamaMLP.forward = forward
    return {
        "installed": True,
        "layers": selected,
        "min_rows": min_rows,
        "compressed_only": compressed,
        "skip_edge_layers": skip_edges,
        "fused_silu": fused_silu,
    }
