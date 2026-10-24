"""Standalone BF16 narrow GEMM: unsplit or deterministic FP32 partial reduction.

Weights retain original contiguous [N,K] storage. No packing, quantization,
atomic BF16 accumulation or service hook is introduced.
"""

import torch
import triton
import triton.language as tl


# Structural choices, not a general autotuning sweep. The reduction stage and
# scratch allocation are included in each split variant's full call timing.
CONFIGS = (
    (16, 128, 64, 1, 4),
    (32, 128, 64, 1, 4),
    (16, 256, 64, 1, 4),
    (32, 256, 64, 1, 8),
    (16, 128, 64, 2, 4),
    (16, 128, 64, 4, 4),
    (32, 128, 64, 2, 4),
    (16, 256, 64, 2, 4),
    (64, 64, 64, 2, 4),
    (64, 64, 64, 4, 4),
)


def chunk_intervals(k, split):
    chunk = triton.cdiv(k, split)
    return [(min(i * chunk,k), min((i + 1) * chunk, k)) for i in range(split)]


@triton.jit
def partial_mm(
    X, W, PARTIAL, OUTPUT,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, XS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    # Keep loop bounds as Python constexpr arithmetic on the Corex frontend.
    chunk: tl.constexpr = (K + SPLIT - 1) // SPLIT
    start = part * chunk
    finish = tl.minimum(start + chunk, K)
    kk = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for step in range((chunk + BK - 1) // BK):
        k = start + step * BK + kk
        a = tl.load(X + row[:, None] * XS + k[None, :],
                    (row[:, None] < M) & (k[None, :] < finish), other=0)
        b = tl.load(W + column[None, :] * K + k[:, None],
                    (column[None, :] < N) & (k[:, None] < finish), other=0)
        accum = tl.dot(a, b, accum, out_dtype=tl.float32, input_precision=None)
    valid = (row[:, None] < M) & (column[None, :] < N)
    offset = row[:, None] * N + column[None, :]
    if SPLIT == 1:
        tl.store(OUTPUT + offset, accum.to(tl.bfloat16), valid)
    else:
        # Every split owns a disjoint FP32 slice. Never round a partial to BF16.
        tl.store(PARTIAL + part * M * N + offset, accum, valid)


@triton.jit
def reduce_partials(PARTIAL, OUTPUT, TOTAL: tl.constexpr, SPLIT: tl.constexpr,
                    BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    part = tl.arange(0, SPLIT)
    values = tl.load(PARTIAL + part[:, None] * TOTAL + offset[None, :],
                     offset[None, :] < TOTAL, other=0)
    final = tl.sum(values, axis=0)
    tl.store(OUTPUT + offset, final.to(tl.bfloat16), offset < TOTAL)


def allocate(x, weight, config):
    m, k = x.shape
    n = weight.shape[0]
    split = config[3]
    output = torch.empty((m,n), device=x.device, dtype=torch.bfloat16)
    partial = (torch.empty((split,m,n), device=x.device, dtype=torch.float32)
               if split > 1 else output)
    return output, partial


def run_into(x, weight, output, partial, config):
    assert config in CONFIGS
    bm, bn, bk, split, warps = config
    m, k = x.shape
    n = weight.shape[0]
    assert x.is_cuda and x.dtype == weight.dtype == torch.bfloat16
    assert x.device == weight.device == output.device == partial.device
    assert x.ndim == weight.ndim == 2 and weight.shape[1] == k
    assert x.stride(1) == 1 and x.stride(0) >= k and weight.is_contiguous()
    assert output.shape == (m,n) and output.is_contiguous()
    assert output.dtype == torch.bfloat16 and 0 < m and 0 < k and 0 < n
    if split > 1:
        assert partial.shape == (split,m,n) and partial.is_contiguous()
        assert partial.dtype == torch.float32
    kernel = partial_mm[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
        x, weight, partial, output, m,n,k,x.stride(0),bm,bn,bk,split,
        num_warps=warps,num_stages=1,
    )
    if split > 1:
        reduce_partials[(triton.cdiv(m*n,256),)](
            partial,output,m*n,split,256,num_warps=4,
        )
    return kernel


def matmul(x, weight, config):
    output, partial = allocate(x,weight,config)
    run_into(x,weight,output,partial,config)
    return output
