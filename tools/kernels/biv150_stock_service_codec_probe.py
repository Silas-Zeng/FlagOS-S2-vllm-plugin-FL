"""M16-only direct probe for the frozen codec used by the stock diagnostic."""
import triton
import triton.language as tl
from vllm_fl.patches.biv150_stock_qk8_codec import quantize_bf16_rne


@triton.jit
def probe_stock_m16(source, codes, scales, N: tl.constexpr):
    tl.static_assert(source.dtype.element_ty == tl.bfloat16)
    tl.static_assert(codes.dtype.element_ty == tl.int8)
    tl.static_assert(scales.dtype.element_ty == tl.float32)
    row = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    dim = tl.arange(0, 128)
    values = tl.load(source + row[:, None] * 128 + dim[None, :], row[:, None] < N, other=0.)
    encoded, scale = quantize_bf16_rne(values)
    tl.store(codes + row[:, None] * 128 + dim[None, :], encoded, row[:, None] < N)
    tl.store(scales + row, scale, row < N)
