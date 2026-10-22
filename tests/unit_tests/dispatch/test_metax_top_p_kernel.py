# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.metax.impl.top_p_sample import (
    top_p_sample,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_top_p_kernel_peaked_distribution():
    logits = torch.full((8, 4096), -5.0, device="cuda")
    logits[:, 123] = 50.0
    top_p = torch.full((8,), 0.95, device="cuda")
    sampled = top_p_sample(logits, top_p)
    assert torch.equal(sampled, torch.full((8,), 123, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_top_p_kernel_samples_inside_nucleus():
    torch.manual_seed(4)
    logits = torch.randn(8, 4096, device="cuda")
    top_p = torch.full((8,), 0.95, device="cuda")

    masked = logits.clone()
    sorted_logits, sorted_indices = masked.sort(dim=-1)
    probs = sorted_logits.softmax(dim=-1, dtype=torch.float32)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative <= 1 - top_p[0]
    remove[:, -1] = False
    sorted_logits.masked_fill_(remove, float("-inf"))
    masked.scatter_(1, sorted_indices, sorted_logits)
    support = torch.isfinite(masked)

    sampled = top_p_sample(logits, top_p)
    assert torch.all(support.gather(1, sampled.unsqueeze(1)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_top_p_kernel_matches_reference_distribution():
    torch.manual_seed(123)
    rows = 16384
    active_logits = torch.tensor(
        [
            3.0,
            2.6,
            2.2,
            1.8,
            1.4,
            1.0,
            0.6,
            0.2,
            -0.2,
            -0.6,
            -1.0,
            -1.4,
            -1.8,
            -2.2,
            -2.6,
            -3.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )
    logits = torch.full(
        (rows, 4096), -100.0, device="cuda", dtype=torch.float32
    )
    logits[:, : active_logits.numel()] = active_logits
    top_p = torch.full((rows,), 0.8, device="cuda")

    probs = torch.softmax(logits[0], dim=0, dtype=torch.float32)
    sorted_probs, sorted_indices = torch.sort(probs)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    remove = cumulative <= 1 - top_p[0]
    remove[-1] = False
    sorted_probs.masked_fill_(remove, 0.0)
    expected = torch.zeros_like(probs)
    expected.scatter_(0, sorted_indices, sorted_probs)
    expected /= expected.sum()

    sampled = top_p_sample(logits, top_p)
    expected_support = expected > 0
    assert torch.all(expected_support[sampled])
    assert torch.unique(sampled).numel() > 1

    observed = torch.bincount(
        sampled, minlength=logits.shape[1]
    ).float()
    observed /= observed.sum()
    assert torch.max(torch.abs(observed - expected)).item() < 0.03
