# SPDX-License-Identifier: Apache-2.0
"""Top-p search reusing masses for unchanged BF16 threshold partitions.

Inputs to the reduction remain FP32 to preserve the original reduction layout.
At unit temperature they lie on the BF16 grid. Equal upward-rounded BF16
thresholds select exactly the same values, so their mass need only be computed
once. Search stops early only when adjacent BF16 levels prove that the
remaining twenty-pass updates cannot change the selected set. Ties and RNG
draws are preserved. Other temperatures retain the original full search.
"""
import os
import torch
from vllm.triton_utils import tl, triton
from flag_gems.utils.triton_lang_extension import div_rn
from .top_p_sample import _top_p_masked_sample_kernel

_PAIR_STEPS = os.environ.get("VLLM_FL_METAX_PAIR_TOP_P", "1") == "1"
_CUTOFF_SAMPLE = os.environ.get("VLLM_FL_METAX_CUTOFF_SAMPLE", "1") == "1"

@triton.jit
def _top_p_grid_kernel(LOGITS, TEMPERATURE, TOP_P, MASKED_LOGITS, DIRECT, ROW_MAX, VOCAB_SIZE: tl.constexpr, ROW_STRIDE: tl.constexpr, TEMP_STRIDE: tl.constexpr, TOP_P_STRIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_TILES: tl.constexpr, SEARCH_STEPS: tl.constexpr, PAIR_STEPS: tl.constexpr = False, CUTOFF_ONLY: tl.constexpr = False):
    row = tl.program_id(0)
    p = tl.load(TOP_P + row * TOP_P_STRIDE)
    temperature = tl.load(TEMPERATURE + row * TEMP_STRIDE)
    maximum = -float('inf')
    minimum = float('inf')
    argmax = 0
    total = 0.0
    for tile in range(NUM_TILES):
        i = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = i < VOCAB_SIZE
        x = _load_scaled(LOGITS + row * ROW_STRIDE, i, valid, temperature)
        tile_max = tl.max(x, 0)
        new_max = tl.maximum(maximum, tile_max)
        total = total * tl.exp(maximum - new_max) + tl.sum(tl.exp(x - new_max), 0)
        argmax = tl.where(tile_max > maximum, tile * BLOCK_SIZE + tl.argmax(x, 0), argmax)
        maximum = new_max
        minimum = tl.minimum(minimum, tl.min(tl.where(valid, x, float('inf')), 0))
    tl.store(ROW_MAX + row, maximum)
    if 1.0 / total >= p:
        tl.store(DIRECT + row, argmax)
        return
    tl.store(DIRECT + row, -1)
    lower = minimum
    upper = maximum
    iterations = 0
    if (temperature == 1.0) & (tl.abs(maximum) < 1e+30) & (tl.abs(minimum) < 1e+30):
        has_failed = False
        active = (minimum != maximum) if PAIR_STEPS else True
        while (iterations < SEARCH_STEPS) & active:
            threshold = (lower + upper) * 0.5
            kept = 0.0
            if PAIR_STEPS:
                left = (lower + threshold) * 0.5
                right = (threshold + upper) * 0.5
                kept_left = 0.0
                kept_right = 0.0
            for tile in range(NUM_TILES):
                i = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                valid = i < VOCAB_SIZE
                x = _load_scaled(LOGITS + row * ROW_STRIDE, i, valid, temperature)
                keep = valid & (x >= threshold)
                weight = tl.exp(x - maximum)
                kept += tl.sum(tl.where(keep, weight, 0.0), 0)
                if PAIR_STEPS:
                    kept_left += tl.sum(tl.where(valid & (x >= left), weight, 0.0), 0)
                    kept_right += tl.sum(tl.where(valid & (x >= right), weight, 0.0), 0)
            covers = kept >= p * total
            lower = tl.where(covers, threshold, lower)
            upper = tl.where(covers, upper, threshold)
            iterations += 1
            has_failed = has_failed | ~covers
            lower_key = _threshold_key(lower)
            upper_key = _threshold_key(upper)
            next_key = _next_bf16(lower_key)
            active = (lower_key != upper_key) & (~has_failed | (upper_key > next_key))
            if PAIR_STEPS:
                if active & (iterations < SEARCH_STEPS):
                    # The second threshold and its FP32 mass were computed
                    # with the same arithmetic and reduction order above.
                    second = tl.where(covers, right, left)
                    second_mass = tl.where(covers, kept_right, kept_left)
                    second_covers = second_mass >= p * total
                    lower = tl.where(second_covers, second, lower)
                    upper = tl.where(second_covers, upper, second)
                    iterations += 1
                    has_failed = has_failed | ~second_covers
                    lower_key = _threshold_key(lower)
                    upper_key = _threshold_key(upper)
                    next_key = _next_bf16(lower_key)
                    active = (lower_key != upper_key) & (~has_failed | (upper_key > next_key))
    else:
        for _ in range(SEARCH_STEPS):
            threshold = (lower + upper) * 0.5
            kept = 0.0
            for tile in range(NUM_TILES):
                i = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                valid = i < VOCAB_SIZE
                x = _load_scaled(LOGITS + row * ROW_STRIDE, i, valid, temperature)
                kept += tl.sum(tl.where(valid & (x >= threshold), tl.exp(x - maximum), 0.0), 0)
            covers = kept >= p * total
            lower = tl.where(covers, threshold, lower)
            upper = tl.where(covers, upper, threshold)
    if CUTOFF_ONLY:
        tl.store(MASKED_LOGITS + row, lower)
    else:
        for tile in range(NUM_TILES):
            i = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            valid = i < VOCAB_SIZE
            x = _load_scaled(LOGITS + row * ROW_STRIDE, i, valid, temperature)
            tl.store(MASKED_LOGITS + row * ROW_STRIDE + i, tl.where(valid & (x >= lower), x, -float('inf')), valid)

@triton.jit
def _threshold_key(threshold):
    rounded = threshold.to(tl.bfloat16).to(tl.float32)
    bits = rounded.to(tl.uint32, bitcast=True)
    next_bits = tl.where(rounded == 0.0, 65536, tl.where(rounded > 0.0, bits + 65536, bits - 65536)).to(tl.uint32)
    upward = next_bits.to(tl.float32, bitcast=True)
    return tl.where(rounded >= threshold, rounded, upward)

@triton.jit
def _load_scaled(LOGITS, offsets, valid, temperature):
    value = tl.load(LOGITS + offsets, valid, -float('inf')).to(tl.float32)
    return value

@triton.jit
def _convert_scale_kernel(X, T, O, N: tl.constexpr, TS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    temperature = tl.load(T + row * TS)
    value = tl.load(X + row * N + i, i < N, 0.0).to(tl.float32)
    if temperature != 1.0:
        value = div_rn(value, temperature)
    tl.store(O + row * N + i, value, i < N)

def sample_bf16_top_p(logits, temperature, top_p):
    batch, vocab = logits.shape
    if logits.dtype != torch.bfloat16 or not logits.is_contiguous() or temperature.dtype != torch.float32 or (top_p.dtype != torch.float32) or (temperature.ndim != 1) or (top_p.ndim != 1) or (temperature.numel() not in (1, batch)) or (top_p.numel() not in (1, batch)) or (logits.device != temperature.device) or (logits.device != top_p.device):
        raise ValueError('unsupported BF16 sampler operands')
    shape = (batch,) if _CUTOFF_SAMPLE else (batch, vocab)
    masked = torch.empty(shape, device=logits.device, dtype=torch.float32)
    scaled = torch.empty((batch, vocab), device=logits.device, dtype=torch.float32)
    _convert_scale_kernel[triton.cdiv(vocab, 2048), batch](logits, temperature, scaled, vocab, 0 if temperature.numel() == 1 else temperature.stride(0), BLOCK=2048, num_warps=4)
    direct = torch.empty(batch, device=logits.device, dtype=torch.int32)
    row_max = torch.empty(batch, device=logits.device, dtype=torch.float32)
    block = 32768
    tiles = triton.cdiv(vocab, block)
    _top_p_grid_kernel[batch,](scaled, temperature, top_p, masked, direct, row_max, vocab, vocab, 0 if temperature.numel() == 1 else temperature.stride(0), 0 if top_p.numel() == 1 else top_p.stride(0), BLOCK_SIZE=block, NUM_TILES=tiles, SEARCH_STEPS=20, PAIR_STEPS=_PAIR_STEPS, CUTOFF_ONLY=_CUTOFF_SAMPLE, num_warps=8)
    noise = torch.empty_like(scaled)
    noise.exponential_()
    sampled = torch.empty_like(direct)
    if _CUTOFF_SAMPLE:
        _sample_cutoff_kernel[batch,](scaled, noise, masked, direct, row_max, sampled,
                                      vocab, 1, vocab, block, tiles, num_warps=8)
    else:
        _top_p_masked_sample_kernel[batch,](masked, noise, direct, row_max, sampled,
                                           vocab, 1, vocab, block, tiles, num_warps=8)
    return sampled

@triton.jit
def _next_bf16(value):
    bits = value.to(tl.uint32, bitcast=True)
    next_bits = tl.where(value == 0.0, 65536, tl.where(value > 0.0, bits + 65536, bits - 65536)).to(tl.uint32)
    return next_bits.to(tl.float32, bitcast=True)


@triton.jit
def _sample_cutoff_kernel(
    MASKED_LOGITS,
    NOISE,
    CUTOFF,
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
    cutoff = tl.load(CUTOFF + row)
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
        values = tl.where(values >= cutoff, values, -float("inf"))
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
