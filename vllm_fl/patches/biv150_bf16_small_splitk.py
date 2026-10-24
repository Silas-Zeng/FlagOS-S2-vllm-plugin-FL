"""Deterministic BF16 split-K and bounded worker MM launcher integration."""

import torch
import triton
import triton.language as tl


# Structural choices, not a general autotuning sweep. The reduction stage and
# scratch allocation are included in each split variant's full call timing.
CONFIGS = ((16,128,64,4,4),)


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


# The model eligibility check is owned by native_w8a8.install before this hook.
# Preserve the registered mm/mm.out functions and their common original
# launcher for every unsupported shape/type/layout, including dominant M32/64.
_EXACT_ROWS=(1,8,16)
_GEOMETRIES={(2048,2560),(2048,2048),(6144,2048)}
_SERVICE_CONFIG=(16,128,64,4,4)
_PROVIDER_SHA='4c2e23da2a6b65b89cbb8170967f14c5327d143e21137387f3907ccb4fcde090'
_SELECTED=set()


def _supported(a,b,c,M,N,K):
    return (M in _EXACT_ROWS and (K,N) in _GEOMETRIES
        and torch.is_inference_mode_enabled() and a.is_cuda
        and a.dtype==b.dtype==c.dtype==torch.bfloat16
        and a.device==b.device==c.device
        and tuple(a.shape)==(M,K) and tuple(b.shape)==(K,N) and tuple(c.shape)==(M,N)
        and a.is_contiguous() and c.is_contiguous() and b.stride()==(1,K))


def install_worker_launcher():
    import hashlib
    import inspect
    from pathlib import Path
    import flag_gems
    entry=inspect.unwrap(flag_gems.mm_out)
    provider=Path(inspect.getsourcefile(entry)).resolve()
    assert hashlib.sha256(provider.read_bytes()).hexdigest()==_PROVIDER_SHA
    namespace=entry.__globals__
    original=namespace['_launch_mm']
    if getattr(original,'_biv150_bf16_small',False):return original._biv150_metadata
    assert Path(inspect.getsourcefile(original)).resolve()==provider
    assert list(inspect.signature(original).parameters)==['a','b','c','M','N','K']
    def launch(a,b,c,M,N,K):
        if not _supported(a,b,c,M,N,K):return original(a,b,c,M,N,K)
        key=(M,K,N)
        if key not in _SELECTED:
            print('BIV150_BF16_SMALL_SELECTED',key,flush=True)
            _SELECTED.add(key)
        # b is the existing same-storage [K,N] weight view; transpose only
        # changes metadata. c remains the caller-provided output allocation.
        weight=b.transpose(0,1)
        partial=torch.empty((4,M,N),device=a.device,dtype=torch.float32)
        run_into(a,weight,c,partial,_SERVICE_CONFIG)
        return c
    metadata=dict(installed=True,provider_file=str(provider),provider_sha256=_PROVIDER_SHA,
        hook_file=__file__,hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_launcher_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        exact_rows=list(_EXACT_ROWS),geometries_k_n=[list(x) for x in sorted(_GEOMETRIES)],
        config=list(_SERVICE_CONFIG),caller_output_preserved=True,
        persistent_weight_copy=False,original_dispatch_registration_preserved=True)
    launch._biv150_bf16_small=True
    launch._biv150_metadata=metadata
    namespace['_launch_mm']=launch
    return metadata
