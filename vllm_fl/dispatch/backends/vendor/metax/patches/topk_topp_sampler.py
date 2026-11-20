# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). Patch apply_top_k_top_p
# to always use the PyTorch fallback.
# TODO: remove once FlagGems provides a MetaX-compatible topk_topp kernel.

import os

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler


_USE_TOP_P_KERNEL = (
    os.environ.get("VLLM_FL_TOP_P_KERNEL", "1").lower()
    not in ("0", "false")
)
_original_apply_top_k_top_p_pytorch = (
    topk_topp_sampler.apply_top_k_top_p_pytorch
)
_original_forward_native = topk_topp_sampler.TopKTopPSampler.forward_native


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return _original_apply_top_k_top_p_pytorch(logits, k, p)


def _top_p_prefix_probs(
    logits: torch.Tensor, p: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort logits and return the exact top-p sampling prefix.

    The sort and cumsum order intentionally match vLLM's reference
    implementation exactly, including its floating-point rounding behavior.
    Keeping the sorted probabilities here avoids scattering the masked logits
    back to vocabulary order and computing a second full softmax.
    """
    sorted_logits, sorted_indices = torch.sort(
        logits, dim=-1, descending=False
    )
    probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
    cumulative_mass = torch.cumsum(probs, dim=-1)
    mask = cumulative_mass <= 1 - p.unsqueeze(-1)
    mask[:, -1] = False
    probs.masked_fill_(mask, 0.0)
    return probs, sorted_indices


def _forward_native_top_p_fast(
    self,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused MetaX top-p sampling for the common top-p-only path."""
    if (
        k is not None
        or p is None
        or logits.shape[0] < 8
        or logits.dtype != torch.float32
        or self.logprobs_mode in ("processed_logits", "processed_logprobs")
    ):
        return _original_forward_native(self, logits, generators, k, p)

    if _USE_TOP_P_KERNEL and not generators:
        from flag_gems.runtime.backend._metax.ops.top_p_sample import (
            top_p_sample,
        )

        return top_p_sample(logits, p), None

    probs, sorted_indices = _top_p_prefix_probs(logits, p)
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    for row, generator in generators.items():
        q[row].exponential_(generator=generator)

    sampled_positions = (probs / q).argmax(dim=-1, keepdim=True)
    sampled = sorted_indices.gather(1, sampled_positions).view(-1)
    return sampled, None


# Replace the dispatch function with one that skips Triton
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton
topk_topp_sampler.TopKTopPSampler.forward_native = _forward_native_top_p_fast
