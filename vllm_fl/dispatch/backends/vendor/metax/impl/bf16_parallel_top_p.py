# Copyright (c) 2026 FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
"""BF16 top-p search with per-tile FP32 masses on C500.

Expose four vocabulary tiles to separate CTAs, retaining each tile's original
reduction layout and summing tile masses in the original order. Row updates
perform the same twenty binary-search steps, including the original BF16
partition stopping rule. A direct row's cutoff is unused by sampling.
"""
import torch
from vllm.triton_utils import triton, tl
from .bf16_top_p_sample import _threshold_key, _next_bf16, _sample_cutoff_kernel

@triton.jit(do_not_specialize=["B"])
def _init_top_p_rows(X, P, STATE, DIRECT, ROWMAX, CUTOFF, B, V: tl.constexpr, PS: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr):
    row = tl.program_id(0)
    maximum = -float('inf')
    minimum = float('inf')
    argmax = 0
    total = 0.0
    for tile in range(TILES):
        i = tile * BLOCK + tl.arange(0, BLOCK)
        valid = i < V
        x = tl.load(X + row * V + i, valid, -float('inf')).to(tl.float32)
        tilemax = tl.max(x, 0)
        newmax = tl.maximum(maximum, tilemax)
        total = total * tl.exp(maximum - newmax) + tl.sum(tl.exp(x - newmax), 0)
        argmax = tl.where(tilemax > maximum, tile * BLOCK + tl.argmax(x, 0), argmax)
        maximum = newmax
        minimum = tl.minimum(minimum, tl.min(tl.where(valid, x, float('inf')), 0))
    p = tl.load(P + row * PS)
    direct = 1.0 / total >= p
    tl.store(DIRECT + row, tl.where(direct, argmax, -1))
    tl.store(ROWMAX + row, maximum)
    tl.store(CUTOFF + row, minimum)
    tl.store(STATE + row, minimum)
    tl.store(STATE + B + row, maximum)
    tl.store(STATE + 2 * B + row, total)
    tl.store(STATE + 3 * B + row, maximum)
    finite = (tl.abs(maximum) < 1e+30) & (tl.abs(minimum) < 1e+30)
    tl.store(STATE + 4 * B + row, 0.0)
    tl.store(STATE + 5 * B + row, ((~finite | (minimum != maximum)) & ~direct).to(tl.float32))
    tl.store(STATE + 6 * B + row, finite.to(tl.float32))

@triton.jit(do_not_specialize=["B"])
def _tile_top_p_mass(X, STATE, PART, B, V: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    active = tl.load(STATE + 5 * B + row) > 0.0
    if active:
        lower = tl.load(STATE + row)
        upper = tl.load(STATE + B + row)
        maximum = tl.load(STATE + 3 * B + row)
        threshold = (lower + upper) * 0.5
        left = (lower + threshold) * 0.5
        right = (threshold + upper) * 0.5
        i = tile * BLOCK + tl.arange(0, BLOCK)
        valid = i < V
        x = tl.load(X + row * V + i, valid, -float('inf')).to(tl.float32)
        weight = tl.exp(x - maximum)
        middle_mass = tl.sum(tl.where(valid & (x >= threshold), weight, 0.0), 0)
        left_mass = tl.sum(tl.where(valid & (x >= left), weight, 0.0), 0)
        right_mass = tl.sum(tl.where(valid & (x >= right), weight, 0.0), 0)
        tl.store(PART + tile * B + row, middle_mass)
        tl.store(PART + (TILES + tile) * B + row, left_mass)
        tl.store(PART + (2 * TILES + tile) * B + row, right_mass)

@triton.jit(do_not_specialize=["B"])
def _update_top_p_rows(STATE, PART, P, CUTOFF, B, PS: tl.constexpr, TILES: tl.constexpr, ROWS: tl.constexpr):
    row = tl.arange(0, ROWS)
    valid = row < B
    active = tl.load(STATE + 5 * B + row, valid, 0) > 0.0
    lower = tl.load(STATE + row, valid, 0)
    upper = tl.load(STATE + B + row, valid, 0)
    total = tl.load(STATE + 2 * B + row, valid, 1)
    finite = tl.load(STATE + 6 * B + row, valid, 0) > 0.0
    failed = tl.load(STATE + 4 * B + row, valid, 0) > 0.0
    p = tl.load(P + row * PS, valid, 1)
    kept = tl.full((ROWS,), 0, tl.float32)
    leftmass = tl.full((ROWS,), 0, tl.float32)
    rightmass = tl.full((ROWS,), 0, tl.float32)
    for tile in range(TILES):
        kept += tl.load(PART + tile * B + row, valid & active, 0)
        leftmass += tl.load(PART + (TILES + tile) * B + row, valid & active, 0)
        rightmass += tl.load(PART + (2 * TILES + tile) * B + row, valid & active, 0)
    threshold = (lower + upper) * 0.5
    left = (lower + threshold) * 0.5
    right = (threshold + upper) * 0.5
    covers = kept >= p * total
    lo = tl.where(active & covers, threshold, lower)
    hi = tl.where(active & ~covers, threshold, upper)
    failed = failed | active & ~covers
    lk = _threshold_key(lo)
    uk = _threshold_key(hi)
    nextkey = _next_bf16(lk)
    active2 = active & (~finite | (lk != uk) & (~failed | (uk > nextkey)))
    second = tl.where(covers, right, left)
    second_mass = tl.where(covers, rightmass, leftmass)
    covers2 = second_mass >= p * total
    lo = tl.where(active2 & covers2, second, lo)
    hi = tl.where(active2 & ~covers2, second, hi)
    failed = failed | active2 & ~covers2
    lk = _threshold_key(lo)
    uk = _threshold_key(hi)
    nextkey = _next_bf16(lk)
    remaining = active2 & (~finite | (lk != uk) & (~failed | (uk > nextkey)))
    tl.store(STATE + row, lo, valid)
    tl.store(STATE + B + row, hi, valid)
    tl.store(STATE + 4 * B + row, failed.to(tl.float32), valid)
    tl.store(STATE + 5 * B + row, remaining.to(tl.float32), valid)
    tl.store(CUTOFF + row, lo, valid)

def _parallel_cutoff(logits, top_p):
    b, v = logits.shape
    block = 32768
    tiles = triton.cdiv(v, block)
    ps = 0 if top_p.numel() == 1 else top_p.stride(0)
    state = torch.empty((7, b), device=logits.device, dtype=torch.float32)
    part = torch.empty((3, tiles, b), device=logits.device, dtype=torch.float32)
    direct = torch.empty(b, device=logits.device, dtype=torch.int32)
    rowmax = torch.empty(b, device=logits.device, dtype=torch.float32)
    cutoff = torch.empty_like(rowmax)
    _init_top_p_rows[b,](logits, top_p, state, direct, rowmax, cutoff, b, v, ps, block, tiles, num_warps=8)
    for _ in range(10):
        _tile_top_p_mass[tiles, b](logits, state, part, b, v, block, tiles, num_warps=8)
        _update_top_p_rows[1,](state, part, top_p, cutoff, b, ps, tiles, triton.next_power_of_2(b), num_warps=1)
    return (cutoff, direct, rowmax)

def sample_parallel_bf16_top_p(logits, top_p):
    b, v = logits.shape
    block = 32768
    tiles = triton.cdiv(v, block)
    cutoff, direct, rowmax = _parallel_cutoff(logits, top_p)
    noise = torch.empty((b, v), device=logits.device, dtype=torch.float32)
    noise.exponential_()
    out = torch.empty_like(direct)
    _sample_cutoff_kernel[b,](logits, noise, cutoff, direct, rowmax, out, v, 1, v, block, tiles, num_warps=8)
    return out
