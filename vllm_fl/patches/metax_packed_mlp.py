# SPDX-License-Identifier: Apache-2.0

"""Blocked BF16 weights and fused SwiGLU for supported unquantized MLPs."""

import os

import torch

_large_down_prefill = os.environ.get("VLLM_FL_METAX_LARGE_PREFILL_DOWN", "1") == "1"
_large_prefill = os.environ.get("VLLM_FL_METAX_LARGE_PREFILL_MLP", "1") == "1"
_small_split = 6 if os.environ.get("VLLM_FL_METAX_ODD_SPLITK", "1") == "1" else 4


@torch.library.custom_op(
    "vllm_fl::metax_packed_gate_up_silu", mutates_args=(), device_types="cuda"
)
def _gate_up(x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    from flag_gems.runtime.backend._metax.ops.packed_mlp import packed_gate_up_silu

    return packed_gate_up_silu(x, weight, packed, large_prefill=_large_prefill)


@_gate_up.register_fake
def _gate_up_fake(x, weight, packed):
    return x.new_empty((x.shape[0], weight.shape[0] // 2))


@torch.library.custom_op(
    "vllm_fl::metax_packed_down", mutates_args=(), device_types="cuda"
)
def _down(x: torch.Tensor, weight: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    from flag_gems.runtime.backend._metax.ops.packed_mlp import packed_down

    return packed_down(x, weight, packed, large_prefill=_large_down_prefill,
                       small_split=_small_split)


@_down.register_fake
def _down_fake(x, weight, packed):
    return x.new_empty((x.shape[0], weight.shape[0]))


def _refresh(model):
    from flag_gems.runtime.backend._metax.ops.packed_mlp import pack_mlp_weight

    for layer in model.modules():
        if hasattr(layer, "_metax_gate_pack"):
            pack_mlp_weight(
                layer.gate_up_proj.weight,
                interleave=True,
                out=layer._metax_gate_pack,
            )
            pack_mlp_weight(layer.down_proj.weight, out=layer._metax_down_pack)


def install(model, *, lora_enabled=False, compiled=True):
    if os.environ.get("VLLM_FL_METAX_PACKED_MLP", "1") != "1":
        return {"installed": False, "reason": "disabled"}
    if lora_enabled or not compiled:
        return {"installed": False, "reason": "requires compiled inference without LoRA"}

    from flag_gems.runtime.backend._metax.ops.packed_mlp import pack_mlp_weight
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaMLP

    count = 0
    for layer in model.modules():
        if not isinstance(layer, LlamaMLP) or not isinstance(layer.act_fn, SiluAndMul):
            continue
        gate, down = layer.gate_up_proj, layer.down_proj
        if not all(
            isinstance(proj.quant_method, UnquantizedLinearMethod)
            and getattr(proj, "tp_size", 1) == 1
            and getattr(proj, "bias", None) is None
            and proj.weight.dtype == torch.bfloat16
            and proj.weight.is_contiguous()
            for proj in (gate, down)
        ):
            continue
        if gate.weight.shape != (12288, 2048) or down.weight.shape != (2048, 6144):
            continue
        if hasattr(layer, "_metax_gate_pack"):
            continue
        # Pack at model load, inside memory profiling, so KV sizing accounts for
        # these buffers. Preserve the original weights for large-batch GEMMs.
        layer.register_buffer(
            "_metax_gate_pack", pack_mlp_weight(gate.weight, interleave=True),
            persistent=False,
        )
        layer.register_buffer(
            "_metax_down_pack", pack_mlp_weight(down.weight), persistent=False
        )
        count += 1

    if count and not getattr(LlamaMLP, "_metax_packed_mlp_installed", False):
        original_forward = LlamaMLP.forward

        def forward(self, x):
            if not hasattr(self, "_metax_gate_pack"):
                return original_forward(self, x)
            x = _gate_up(x, self.gate_up_proj.weight, self._metax_gate_pack)
            return _down(x, self.down_proj.weight, self._metax_down_pack)

        LlamaMLP.forward = forward
        LlamaMLP._metax_packed_mlp_installed = True

        # Update buffers in place after reloads, keeping captured graph pointers
        # valid. The operator itself has no cache keyed by tensor addresses.
        original_load_weights = LlamaForCausalLM.load_weights

        def load_weights(self, weights):
            loaded = original_load_weights(self, weights)
            _refresh(self)
            return loaded

        LlamaForCausalLM.load_weights = load_weights
    return {"installed": count > 0, "layers": count, "layout": [128, 64]}
