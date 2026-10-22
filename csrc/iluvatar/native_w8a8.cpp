#include <ATen/ATen.h>
#include <ATen/ops/_int_mm.h>
#include <ATen/ops/linear.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <algorithm>
#include <optional>

extern "C" void biv150_quantize(const void*, int8_t*, float*, int, int, int, cudaStream_t);
extern "C" void biv150_epilogue(const int32_t*, const float*, const float*,
                                 const void*, void*, int, int, cudaStream_t);

namespace {
at::Tensor linear(const at::Tensor& x, const at::Tensor& wq,
                  const at::Tensor& scales, const at::Tensor& original,
                  const std::optional<at::Tensor>& bias, int64_t min_rows) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2);
  TORCH_CHECK(wq.scalar_type() == at::kChar && wq.dim() == 2 && wq.is_contiguous());
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous());
  const auto m = x.size(0), k = x.size(1), n = wq.size(1);
  TORCH_CHECK(k == wq.size(0) && k % 16 == 0 && n % 16 == 0);
  // The compressed-only path keeps a broadcast scalar as a shape/dtype marker;
  // no full BF16 weight storage is retained. Small batches are padded for int_mm.
  const bool compressed_only = original.stride(0) == 0 && original.stride(1) == 0;
  if (!compressed_only && m < std::max<int64_t>(17, min_rows))
    return at::linear(x, original, bias);
  c10::cuda::CUDAGuard guard(x.device());
  auto input = x.contiguous();
  auto output = at::empty({m, n}, x.options());
  const int64_t chunks = (m + 8191) / 8192;
  const int64_t span = (m + chunks - 1) / chunks;
  const auto stream = c10::cuda::getCurrentCUDAStream();
  for (int64_t start = 0; start < m; start += span) {
    const auto rows = std::min<int64_t>(span, m - start);
    if (rows <= 16 && !compressed_only) {
      output.narrow(0, start, rows).copy_(at::linear(input.narrow(0, start, rows), original, bias));
      continue;
    }
    const auto padded_rows = std::max<int64_t>(32, rows);
    auto aq = at::empty({padded_rows, k}, x.options().dtype(at::kChar));
    auto sa = at::empty({padded_rows}, x.options().dtype(at::kFloat));
    biv150_quantize(input.data_ptr<at::BFloat16>() + start * k,
                   aq.data_ptr<int8_t>(), sa.data_ptr<float>(), int(rows), int(padded_rows), int(k), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto acc = at::_int_mm(aq, wq);
    biv150_epilogue(acc.data_ptr<int32_t>(), sa.data_ptr<float>(), scales.data_ptr<float>(),
                    bias.has_value() ? bias->data_ptr<at::BFloat16>() : nullptr,
                    output.data_ptr<at::BFloat16>() + start * n, int(rows), int(n), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
}

TORCH_LIBRARY(biv150_native, m) {
  m.def("linear(Tensor x, Tensor weight_int8, Tensor scales, Tensor original, Tensor? bias, int min_rows) -> Tensor");
}
TORCH_LIBRARY_IMPL(biv150_native, CUDA, m) { m.impl("linear", &linear); }
