#include <ATen/ATen.h>
#include <ATen/ops/_int_mm.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/silu.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <algorithm>
#include <optional>

extern "C" void biv150_quantize(const void*, int8_t*, float*, int, int, int, cudaStream_t);
extern "C" void biv150_epilogue(const int32_t*, const float*, const float*,
                                 const void*, void*, int, int, cudaStream_t);
extern "C" void biv150_silu_epilogue(const int32_t*, const float*, const float*,
                                      void*, int, int, cudaStream_t);

namespace {
at::Tensor linear_impl(const at::Tensor& x, const at::Tensor& wq,
                  const at::Tensor& scales, const at::Tensor& original,
                  const std::optional<at::Tensor>& bias, int64_t min_rows, bool fused_silu) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2);
  TORCH_CHECK(wq.scalar_type() == at::kChar && wq.dim() == 2 && wq.is_contiguous());
  TORCH_CHECK(scales.scalar_type() == at::kFloat && scales.is_contiguous());
  const auto m = x.size(0), k = x.size(1), n = wq.size(1);
  TORCH_CHECK(!fused_silu || (!bias.has_value() && n == 12288 && k == 2048));
  TORCH_CHECK(k == wq.size(0) && k % 16 == 0 && n % 16 == 0);
  // The compressed-only path keeps a broadcast scalar as a shape/dtype marker;
  // no full BF16 weight storage is retained. Small batches are padded for int_mm.
  const bool compressed_only = original.stride(0) == 0 && original.stride(1) == 0;
  const auto output_n = fused_silu ? n/2 : n;
  auto fallback = [&](const at::Tensor& input) {
    auto y = at::linear(input, original, bias);
    if (!fused_silu) return y;
    auto wide = y.to(at::kFloat);
    return (at::silu(wide.narrow(1,0,n/2))*wide.narrow(1,n/2,n/2)).to(x.scalar_type());
  };
  if (!compressed_only && m < std::max<int64_t>(17, min_rows)) return fallback(x);
  c10::cuda::CUDAGuard guard(x.device());
  auto input = x.contiguous();
  auto output = at::empty({m, output_n}, x.options());
  if (m == 0) return output;
  const int64_t chunks = (m + 8191) / 8192;
  const int64_t span = (m + chunks - 1) / chunks;
  const auto stream = c10::cuda::getCurrentCUDAStream();
  for (int64_t start = 0; start < m; start += span) {
    const auto rows = std::min<int64_t>(span, m - start);
    if (rows <= 16 && !compressed_only) {
      output.narrow(0, start, rows).copy_(fallback(input.narrow(0, start, rows)));
      continue;
    }
    const auto padded_rows = std::max<int64_t>(32, rows);
    auto aq = at::empty({padded_rows, k}, x.options().dtype(at::kChar));
    auto sa = at::empty({padded_rows}, x.options().dtype(at::kFloat));
    biv150_quantize(input.data_ptr<at::BFloat16>() + start * k,
                   aq.data_ptr<int8_t>(), sa.data_ptr<float>(), int(rows), int(padded_rows), int(k), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto acc = at::_int_mm(aq, wq);
    if (fused_silu) {
      biv150_silu_epilogue(acc.data_ptr<int32_t>(),sa.data_ptr<float>(),scales.data_ptr<float>(),
                           output.data_ptr<at::BFloat16>()+start*output_n,int(rows),int(n),stream);
    } else {
      biv150_epilogue(acc.data_ptr<int32_t>(), sa.data_ptr<float>(), scales.data_ptr<float>(),
                    bias.has_value() ? bias->data_ptr<at::BFloat16>() : nullptr,
                    output.data_ptr<at::BFloat16>() + start * n, int(rows), int(n), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor linear(const at::Tensor& x, const at::Tensor& wq,
                  const at::Tensor& scales, const at::Tensor& original,
                  const std::optional<at::Tensor>& bias, int64_t min_rows) {
  return linear_impl(x,wq,scales,original,bias,min_rows,false);
}

at::Tensor gate_silu(const at::Tensor& x, const at::Tensor& wq,
                     const at::Tensor& scales, const at::Tensor& original,
                     int64_t min_rows) {
  return linear_impl(x,wq,scales,original,std::nullopt,min_rows,true);
}
}

TORCH_LIBRARY(biv150_native, m) {
  m.def("linear(Tensor x, Tensor weight_int8, Tensor scales, Tensor original, Tensor? bias, int min_rows) -> Tensor");
  m.def("gate_silu(Tensor x, Tensor weight_int8, Tensor scales, Tensor original, int min_rows) -> Tensor");
}
TORCH_LIBRARY_IMPL(biv150_native, CUDA, m) {
  m.impl("linear", &linear);
  m.impl("gate_silu", &gate_silu);
}
