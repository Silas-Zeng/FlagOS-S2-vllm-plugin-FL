# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_fl.dispatch.backends.vendor.metax.patches import (
    topk_topp_sampler as metax_sampler,
)


def _sorted_top_p_support(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs, indices = metax_sampler._top_p_prefix_probs(
        logits, torch.tensor([top_p], dtype=torch.float32)
    )
    support = torch.zeros_like(logits, dtype=torch.bool)
    return support.scatter(1, indices, probs > 0)


def _reference_top_p_support(
    logits: torch.Tensor, top_p: float
) -> torch.Tensor:
    masked = metax_sampler._original_apply_top_k_top_p_pytorch(
        logits.clone(), None, torch.tensor([top_p], dtype=torch.float32)
    )
    return torch.isfinite(masked)


def test_fused_top_p_uses_same_support_as_reference():
    torch.manual_seed(0)
    logits = torch.randn(8, 4096, dtype=torch.float32)
    support = _sorted_top_p_support(logits, 0.95)
    reference = _reference_top_p_support(logits, 0.95)
    assert torch.equal(support, reference)


def test_fused_top_p_support_handles_distinct_row_thresholds():
    torch.manual_seed(1)
    logits = torch.randn(4, 2048, dtype=torch.float32)
    p = torch.tensor([0.5, 0.8, 0.95, 1.0], dtype=torch.float32)
    probs, indices = metax_sampler._top_p_prefix_probs(logits, p)
    support = torch.zeros_like(logits, dtype=torch.bool)
    support.scatter_(1, indices, probs > 0)
    reference = torch.isfinite(
        metax_sampler._original_apply_top_k_top_p_pytorch(
            logits.clone(), None, p
        )
    )
    assert torch.equal(support, reference)
