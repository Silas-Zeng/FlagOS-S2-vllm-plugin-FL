"""仅原BF16 head store前融合温度；原dot/寻址/流水代码逐字保留。"""
import triton
import triton.language as tl

@triton.jit
def head_temperature_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_im,
    stride_ik,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_bn,
    temperature_ptr,
    BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)

    A = input_ptr + (ram[:, None] * stride_im + rk[None, :] * stride_ik)
    W = weight_ptr + (rk[:, None] * stride_wk + rbn[None, :] * stride_wn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for _ in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            a = tl.load(A)
            b = tl.load(W)
            acc += tl.dot(a, b, allow_tf32=False)
            A += BLOCK_K * SPLIT_K * stride_ik
            W += BLOCK_K * SPLIT_K * stride_wk
    else:
        loop_num = tl.cdiv(K, BLOCK_K * SPLIT_K) - 1
        for _ in range(0, loop_num):
            a = tl.load(A)
            b = tl.load(W)
            acc += tl.dot(a, b, allow_tf32=False)
            A += BLOCK_K * SPLIT_K * stride_ik
            W += BLOCK_K * SPLIT_K * stride_wk

        _0 = tl.zeros((1, 1), dtype=output_ptr.dtype.element_ty)
        k_remaining = K - loop_num * (BLOCK_K * SPLIT_K)
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
        b = tl.load(W, mask=rk[:, None] < k_remaining, other=_0)
        acc += tl.dot(a, b, allow_tf32=False)

    if BIAS:
        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
        acc = acc + bias[None, :]

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        temperature = tl.load(temperature_ptr + rm, mask=rm < M, other=1.0)
        scaled = acc.to(tl.bfloat16).to(tl.float32) / temperature[:, None]
        tl.store(C, scaled, mask=mask)
    else:
        tl.atomic_add(C, acc.to(output_ptr.dtype.element_ty), mask=mask)
