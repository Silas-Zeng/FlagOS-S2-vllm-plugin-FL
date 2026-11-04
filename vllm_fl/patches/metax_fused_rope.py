# SPDX-License-Identifier: Apache-2.0
"""Fused RoPE for compatible compiled BF16 decoder modules."""
import os
import types
import logging

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


@torch.library.custom_op("vllm_fl::metax_fused_qk_rope", mutates_args=(), device_types="cuda")
def _rope(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
          cache: torch.Tensor) -> torch.Tensor:
    from flag_gems.runtime.backend._metax.ops.rope_qk import rope_qk
    return rope_qk(positions, query, key, cache)


@_rope.register_fake
def _rope_fake(positions, query, key, cache):
    return query.new_empty((query.shape[0], 2304))


def _forward(self, positions, query, key=None):
    if (key is None or query.ndim != 2 or key.ndim != 2
            or query.shape[1] != 2048 or key.shape[1] != 256
            or query.shape[0] != key.shape[0] or positions.numel() != query.shape[0]
            or query.dtype != torch.bfloat16 or key.dtype != query.dtype
            or query.device.type != "cuda" or query.device != key.device
            or query.device != positions.device
            or positions.dtype not in (torch.int32, torch.int64)
            or not positions.is_contiguous() or query.stride(-1) != 1 or key.stride(-1) != 1
            or query.requires_grad or key.requires_grad):
        return RotaryEmbedding.forward_native(self, positions, query, key)
    cache = self._match_cos_sin_cache_dtype(query)
    if not cache.is_contiguous() or cache.shape[1] != 128:
        return RotaryEmbedding.forward_native(self, positions, query, key)
    result = _rope(positions.flatten(), query, key, cache)
    return result[:, :2048], result[:, 2048:]


def install(model):
    if os.environ.get("VLLM_FL_METAX_FUSED_ROPE", "1") != "1":
        return 0
    from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

    count = 0
    for layer in model.modules():
        if (type(layer) in (RotaryEmbedding, RotaryEmbeddingFL)
                and layer.head_size == layer.rotary_dim == 128
                and layer.is_neox_style and layer.dtype == torch.bfloat16
                and getattr(getattr(layer, "_forward_method", None), "__func__", None)
                is RotaryEmbedding.forward_native):
            layer._forward_method = types.MethodType(_forward, layer)
            count += 1
    logging.getLogger(__name__).info("Installed fused Q/K RoPE on %d compatible modules", count)
    return count
