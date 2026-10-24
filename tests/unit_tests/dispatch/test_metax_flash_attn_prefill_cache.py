# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import (
    _cumulative_seq_lens,
)


def test_cumulative_seq_lens():
    seq_lens = torch.tensor([4, 2, 7], dtype=torch.int32)
    result = _cumulative_seq_lens(seq_lens)
    assert torch.equal(
        result, torch.tensor([0, 4, 6, 13], dtype=torch.int32)
    )
