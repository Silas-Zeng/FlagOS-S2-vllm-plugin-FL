# SPDX-License-Identifier: Apache-2.0
"""Fuse compatible decoder RoPE with the existing BF16 cache update."""
import functools
import os

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.llama import LlamaAttention
from vllm.v1.attention.backend import AttentionType

from flag_gems.runtime.backend._metax.ops.rope_qk import rope_qk
from flag_gems.runtime.backend._metax.ops.rope_cache import rope_qk_cache

_original_forward = LlamaAttention.forward
_qkv_fusion = os.environ.get("VLLM_FL_METAX_QKV_ROPE_FUSION", "1") == "1"


def _state(layer_name):
    try:
        context = get_forward_context()
    except AssertionError:
        return None
    raw = context.attn_metadata
    metadata = raw.get(layer_name) if isinstance(raw, dict) else raw
    if metadata is None or isinstance(metadata, list):
        return None
    layer = context.no_compile_layers.get(layer_name)
    if layer is None:
        return None
    kv = layer.kv_cache
    slots = getattr(metadata, "slot_mapping", None)
    if (not getattr(layer, "_metax_rope_cache_enabled", False)
            or getattr(layer, "kv_cache_dtype", None) != "auto"
            or getattr(layer, "kv_sharing_target_layer_name", None) is not None
            or getattr(layer, "calculate_kv_scales", False)
            or not isinstance(kv, torch.Tensor) or kv.ndim != 5
            or kv.shape[0] != 2 or kv.shape[-2:] != (2, 128)
            or kv.dtype != torch.bfloat16 or kv.stride(-1) != 1
            or slots is None or slots.ndim != 1 or not slots.is_contiguous()
            or slots.dtype not in (torch.int32, torch.int64) or slots.device != kv.device
            or slots.numel() == 0):
        return None
    return context, layer, kv, slots


def consume_marker(layer_name):
    done = getattr(get_forward_context(), "_metax_rope_cache_done", None)
    if done is not None and layer_name in done:
        done.remove(layer_name)
        return True
    return False


def _mark_done(context, layer_name):
    done = getattr(context, "_metax_rope_cache_done", None)
    if done is None:
        done = set()
        context._metax_rope_cache_done = done
    done.add(layer_name)


@torch.library.custom_op("vllm_fl::metax_rope_cache", mutates_args=(), device_types="cuda")
def _rope_cache(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, cache: torch.Tensor, layer_name: str) -> torch.Tensor:
    state = _state(layer_name)
    if state is None or state[3].numel() > query.shape[0]:
        return rope_qk(positions, query, key, cache)
    context, layer, kv, slots = state
    output = rope_qk_cache(positions, query, key, value, cache, kv[0], kv[1], slots)
    _mark_done(context, layer_name)
    return output


@_rope_cache.register_fake
def _rope_cache_fake(positions, query, key, value, cache, layer_name):
    return query.new_empty((query.shape[0], 2304))


@torch.library.custom_op("vllm_fl::metax_qkv_rope_cache", mutates_args=(), device_types="cuda")
def _qkv_rope_cache(positions: torch.Tensor, hidden: torch.Tensor,
                    weight: torch.Tensor, packed: torch.Tensor,
                    cache: torch.Tensor, layer_name: str
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    state = _state(layer_name)
    if (33 <= hidden.shape[0] <= 64 and hidden.is_contiguous()
            and state is not None and state[3].numel() <= hidden.shape[0]):
        from flag_gems.runtime.backend._metax.ops.qkv_rope_cache import qkv_rope_cache

        context, _, kv, slots = state
        result = qkv_rope_cache(hidden, packed, positions, cache, kv[0], kv[1], slots)
        _mark_done(context, layer_name)
        return result
    from flag_gems.runtime.backend._metax.ops.prefill_linear import prefill_linear_nn

    qkv = prefill_linear_nn(hidden, weight, packed)
    q, k, value = qkv.split([2048, 256, 256], dim=-1)
    if state is None or state[3].numel() > hidden.shape[0]:
        return rope_qk(positions, q, k, cache), value
    context, _, kv, slots = state
    output = rope_qk_cache(positions, q, k, value, cache, kv[0], kv[1], slots)
    _mark_done(context, layer_name)
    return output, value


@_qkv_rope_cache.register_fake
def _qkv_rope_cache_fake(positions, hidden, weight, packed, cache, layer_name):
    return hidden.new_empty((hidden.shape[0], 2304)), hidden.new_empty(
        (hidden.shape[0], 2560)
    )[:, 2304:]


@functools.wraps(_original_forward)
def _forward(self, positions, hidden_states):
    if (not getattr(self, "_metax_rope_cache_enabled", False)
            or hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2
            or hidden_states.requires_grad):
        return _original_forward(self, positions, hidden_states)
    packed = getattr(self.qkv_proj, "_metax_prefill_nn_weight", None)
    if (_qkv_fusion and packed is not None and packed.shape == (20, 32, 128, 64)):
        cache = self.rotary_emb._match_cos_sin_cache_dtype(hidden_states)
        output, v = _qkv_rope_cache(
            positions.flatten(), hidden_states, self.qkv_proj.weight, packed,
            cache, self.attn.layer_name,
        )
    else:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        cache = self.rotary_emb._match_cos_sin_cache_dtype(q)
        output = _rope_cache(positions.flatten(), q, k, v, cache, self.attn.layer_name)
    q, k = output[:, :2048], output[:, 2048:]
    attn_output = self.attn(q, k, v)
    result, _ = self.o_proj(attn_output)
    return result


def install(model):
    if os.environ.get("VLLM_FL_METAX_ROPE_CACHE", "1") != "1":
        return 0
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import FlashAttentionImpl
    count = 0
    for module in model.modules():
        if not isinstance(module, LlamaAttention):
            continue
        rope, attn = module.rotary_emb, module.attn
        if (module.num_heads == 16 and module.num_kv_heads == 2 and module.head_dim == 128
                and type(rope) in (RotaryEmbedding, RotaryEmbeddingFL)
                and rope.head_size == rope.rotary_dim == 128 and rope.is_neox_style
                and isinstance(attn.impl, FlashAttentionImpl)
                and attn.attn_type == AttentionType.DECODER and attn.kv_cache_dtype == "auto"
                and not attn.calculate_kv_scales and attn.query_quant is None
                and attn.kv_sharing_target_layer_name is None):
            module._metax_rope_cache_enabled = True
            attn._metax_rope_cache_enabled = True
            count += 1
    if count and not getattr(LlamaAttention, "_metax_rope_cache_installed", False):
        LlamaAttention.forward = _forward
        LlamaAttention._metax_rope_cache_installed = True
    return count
