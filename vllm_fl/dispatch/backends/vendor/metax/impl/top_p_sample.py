# SPDX-License-Identifier: Apache-2.0

"""GPU-only top-p masking and sampling for MetaX.

The first kernel scans each row once to compute max/sum-exp. Rows whose maximum
token already covers top_p record that token directly; broader rows use a fixed
number of binary-search passes and write masked logits. A second kernel samples
the broader rows with vLLM's standard exponential-noise method. No host
synchronization is performed.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _top_p_sample_kernel(
    LOGITS,
    TOP_P,
    OUT,
    DIRECT,
    ROW_MAX,
    LOGIT_STRIDE_0: tl.constexpr,
    LOGIT_STRIDE_1: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = LOGITS + row * LOGIT_STRIDE_0
    p = tl.load(TOP_P + row)

    row_max = -float("inf")
    row_min = float("inf")
    row_argmax = 0
    sum_exp = 0.0

    for tile in range(NUM_TILES):
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < VOCAB_SIZE
        values = tl.load(
            row_start + offsets * LOGIT_STRIDE_1,
            mask=mask,
            other=-float("inf"),
        )
        tile_max = tl.max(values, axis=0)
        tile_argmax = tl.argmax(values, axis=0)
        new_max = tl.maximum(row_max, tile_max)
        sum_exp = (
            sum_exp * tl.exp(row_max - new_max)
            + tl.sum(tl.exp(values - new_max), axis=0)
        )
        take_tile = tile_max > row_max
        row_argmax = tl.where(
            take_tile,
            tile * BLOCK_SIZE + tile_argmax,
            row_argmax,
        )
        row_max = new_max
        row_min = tl.minimum(
            row_min,
            tl.min(tl.where(mask, values, float("inf")), axis=0),
        )

    max_prob = 1.0 / sum_exp
    tl.store(ROW_MAX + row, row_max)
    if max_prob >= p:
        tl.store(DIRECT + row, row_argmax)
        return

    tl.store(DIRECT + row, -1)
    lower = row_min
    upper = row_max
    for _ in range(SEARCH_STEPS):
        threshold = (lower + upper) * 0.5
        kept_mass = 0.0
        for tile in range(NUM_TILES):
            offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < VOCAB_SIZE
            values = tl.load(
                row_start + offsets * LOGIT_STRIDE_1,
                mask=mask,
                other=-float("inf"),
            )
            kept_mass += tl.sum(
                tl.where(
                    mask & (values >= threshold),
                    tl.exp(values - row_max),
                    0.0,
                ),
                axis=0,
            )
        covers_top_p = kept_mass >= p * sum_exp
        lower = tl.where(covers_top_p, threshold, lower)
        upper = tl.where(covers_top_p, upper, threshold)

    for tile in range(NUM_TILES):
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < VOCAB_SIZE
        values = tl.load(
            row_start + offsets * LOGIT_STRIDE_1,
            mask=mask,
            other=-float("inf"),
        )
        tl.store(
            OUT + row * LOGIT_STRIDE_0 + offsets * LOGIT_STRIDE_1,
            tl.where((values >= lower) & mask, values, -float("inf")),
            mask=mask,
        )


@triton.jit
def _top_p_masked_sample_kernel(
    MASKED_LOGITS,
    NOISE,
    DIRECT,
    ROW_MAX,
    SAMPLED,
    LOGIT_STRIDE_0: tl.constexpr,
    LOGIT_STRIDE_1: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    direct = tl.load(DIRECT + row)
    if direct >= 0:
        tl.store(SAMPLED + row, direct)
        return

    row_max = tl.load(ROW_MAX + row)
    row_start = MASKED_LOGITS + row * LOGIT_STRIDE_0
    noise_start = NOISE + row * LOGIT_STRIDE_0
    best_ratio = -float("inf")
    best_token = 0
    for tile in range(NUM_TILES):
        offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < VOCAB_SIZE
        values = tl.load(
            row_start + offsets * LOGIT_STRIDE_1,
            mask=mask,
            other=-float("inf"),
        )
        noise = tl.load(
            noise_start + offsets * LOGIT_STRIDE_1,
            mask=mask,
            other=1.0,
        )
        ratios = tl.where(
            mask,
            tl.exp(values - row_max) / noise,
            -float("inf"),
        )
        tile_best = tl.max(ratios, axis=0)
        tile_argmax = tl.argmax(ratios, axis=0)
        take_tile = tile_best > best_ratio
        best_token = tl.where(
            take_tile,
            tile * BLOCK_SIZE + tile_argmax,
            best_token,
        )
        best_ratio = tl.maximum(best_ratio, tile_best)
    tl.store(SAMPLED + row, best_token)


def top_p_sample(
    logits: torch.Tensor,
    top_p: torch.Tensor,
    *,
    search_steps: int = 20,
) -> torch.Tensor:
    """Sample one token per row from a top-p-truncated distribution."""
    batch_size, vocab_size = logits.shape
    if top_p.numel() == 1:
        top_p = top_p.expand(batch_size)
    assert top_p.numel() == batch_size

    # A wider tile amortizes the per-tile loop and reduction overhead on
    # MetaX C500. The 4 KB/thread private-memory limit is reached above this
    # tile size for the 8-warp launch.
    block_size = 32768
    num_tiles = triton.cdiv(vocab_size, block_size)
    masked_logits = torch.empty_like(logits)
    direct_tokens = torch.empty(
        batch_size, device=logits.device, dtype=torch.long
    )
    row_max = torch.empty(
        batch_size, device=logits.device, dtype=logits.dtype
    )
    _top_p_sample_kernel[(batch_size,)](
        logits,
        top_p,
        masked_logits,
        direct_tokens,
        row_max,
        logits.stride(0),
        logits.stride(1),
        vocab_size,
        block_size,
        num_tiles,
        search_steps,
        num_warps=8,
    )

    noise = torch.empty_like(logits)
    noise.exponential_()
    sampled = torch.empty_like(direct_tokens)
    _top_p_masked_sample_kernel[(batch_size,)](
        masked_logits,
        noise,
        direct_tokens,
        row_max,
        sampled,
        masked_logits.stride(0),
        masked_logits.stride(1),
        vocab_size,
        block_size,
        num_tiles,
        num_warps=8,
    )
    return sampled
