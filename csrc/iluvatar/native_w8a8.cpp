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

extern "C" void biv150_quantize_cached(const void*,int8_t*,float*,int,int,int,int,cudaStream_t);

extern "C" void biv150_quantize_residual(const void*,int8_t*,float*,int,int,cudaStream_t);
extern "C" void biv150_quantize_residual_cached(const void*,int8_t*,float*,int,int,cudaStream_t);
extern "C" void biv150_residual_silu_epilogue(const int32_t*,const float*,const float*,void*,int,int,cudaStream_t);

namespace {
at::Tensor linear_impl(const at::Tensor& x, const at::Tensor& wq,
                  const at::Tensor& scales, const at::Tensor& original,
                  const std::optional<at::Tensor>& bias, int64_t min_rows, bool fused_silu, int quant_threads = 0) {
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
    if (quant_threads) {
      biv150_quantize_cached(input.data_ptr<at::BFloat16>() + start * k,
          aq.data_ptr<int8_t>(), sa.data_ptr<float>(), int(rows), int(padded_rows), int(k), quant_threads, stream);
    } else {
    biv150_quantize(input.data_ptr<at::BFloat16>() + start * k,
                   aq.data_ptr<int8_t>(), sa.data_ptr<float>(), int(rows), int(padded_rows), int(k), stream);
    }
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

std::vector<at::Tensor> quantize_rows_probe(const at::Tensor& x, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(1)==2048);
  TORCH_CHECK(threads==0 || threads==64 || threads==128 || threads==256);
  c10::cuda::CUDAGuard guard(x.device());
  auto input=x.contiguous();
  const int m=int(x.size(0)), padded=std::max(32,m);
  auto q=at::empty({padded,2048},x.options().dtype(at::kChar));
  auto scale=at::empty({padded},x.options().dtype(at::kFloat));
  auto stream=c10::cuda::getCurrentCUDAStream();
  if (threads) biv150_quantize_cached(input.data_ptr<at::BFloat16>(),q.data_ptr<int8_t>(),scale.data_ptr<float>(),m,padded,2048,int(threads),stream);
  else biv150_quantize(input.data_ptr<at::BFloat16>(),q.data_ptr<int8_t>(),scale.data_ptr<float>(),m,padded,2048,stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q,scale};
}

at::Tensor gate_silu_cached_probe(const at::Tensor& x, const at::Tensor& wq,
    const at::Tensor& scales, const at::Tensor& marker, int64_t threads) {
  TORCH_CHECK(threads==64 || threads==128 || threads==256);
  TORCH_CHECK(x.dim()==2 && x.size(1)==2048 && x.size(0)>=17);
  TORCH_CHECK(wq.dim()==2 && wq.size(0)==2048 && wq.size(1)==12288);
  TORCH_CHECK(scales.dim()==1 && scales.numel()==12288);
  TORCH_CHECK(marker.dim()==2 && marker.size(0)==12288 && marker.size(1)==2048);
  TORCH_CHECK(marker.stride(0)==0 && marker.stride(1)==0);
  TORCH_CHECK(x.device()==wq.device() && x.device()==scales.device() && x.device()==marker.device());
  return linear_impl(x,wq,scales,marker,std::nullopt,17,true,int(threads));
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

at::Tensor gate_silu_residual_impl(const at::Tensor& x, const at::Tensor& wq,
                             const at::Tensor& scales, int threads) {
  TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.scalar_type()==at::kBFloat16);
  TORCH_CHECK(wq.is_cuda() && wq.dim()==2 && wq.scalar_type()==at::kChar && wq.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dim()==1 && scales.scalar_type()==at::kFloat && scales.is_contiguous());
  TORCH_CHECK(x.device()==wq.device() && x.device()==scales.device());
  const auto m=x.size(0), k=x.size(1), n=wq.size(1);
  TORCH_CHECK(m>=1 && m<=16 && k==2048 && wq.size(0)==k && n==12288 && scales.numel()==n);
  c10::cuda::CUDAGuard guard(x.device());
  auto input=x.contiguous();
  auto aq=at::empty({32,k},x.options().dtype(at::kChar));
  auto row_scales=at::empty({32},x.options().dtype(at::kFloat));
  auto output=at::empty({m,n/2},x.options());
  const auto stream=c10::cuda::getCurrentCUDAStream();
  if (threads)
    biv150_quantize_residual_cached(input.data_ptr<at::BFloat16>(),aq.data_ptr<int8_t>(),
                          row_scales.data_ptr<float>(),int(m),threads,stream);
  else
    biv150_quantize_residual(input.data_ptr<at::BFloat16>(),aq.data_ptr<int8_t>(),
                          row_scales.data_ptr<float>(),int(m),int(k),stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto acc=at::_int_mm(aq,wq);
  biv150_residual_silu_epilogue(acc.data_ptr<int32_t>(),row_scales.data_ptr<float>(),
      scales.data_ptr<float>(),output.data_ptr<at::BFloat16>(),int(m),int(n),stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor gate_silu_residual(const at::Tensor& x, const at::Tensor& wq,
                             const at::Tensor& scales) {
  return gate_silu_residual_impl(x,wq,scales,0);
}

at::Tensor gate_residual_cached_probe(const at::Tensor& x,const at::Tensor& wq,
                                    const at::Tensor& scales,int64_t threads) {
  TORCH_CHECK(threads==64 || threads==128 || threads==256);
  return gate_silu_residual_impl(x,wq,scales,int(threads));
}

std::vector<at::Tensor> quantize_residual_probe(const at::Tensor& x,int64_t threads) {
  TORCH_CHECK(threads==0 || threads==64 || threads==128 || threads==256);
  TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.scalar_type()==at::kBFloat16);
  TORCH_CHECK(x.size(0)>=1 && x.size(0)<=16 && x.size(1)==2048);
  c10::cuda::CUDAGuard guard(x.device());
  auto input=x.contiguous();
  auto q=at::empty({32,2048},x.options().dtype(at::kChar));
  auto scales=at::empty({32},x.options().dtype(at::kFloat));
  auto stream=c10::cuda::getCurrentCUDAStream();
  if (threads) biv150_quantize_residual_cached(input.data_ptr<at::BFloat16>(),q.data_ptr<int8_t>(),scales.data_ptr<float>(),int(x.size(0)),int(threads),stream);
  else biv150_quantize_residual(input.data_ptr<at::BFloat16>(),q.data_ptr<int8_t>(),scales.data_ptr<float>(),int(x.size(0)),2048,stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q,scales};
}

at::Tensor gate_silu_auto(const at::Tensor& x, const at::Tensor& wq,
                          const at::Tensor& scales, const at::Tensor& original,
                          int64_t min_rows) {
  // Only the compressed MiniCPM path has no original BF16 fallback. Reuse its
  // padded GEMM rows for the already-tested activation residual at small M.
  const bool compressed = original.dim()==2 && original.stride(0)==0 && original.stride(1)==0;
  if (compressed && x.dim()==2 && x.size(0)>=1 && x.size(0)<=16)
    return gate_silu_residual_impl(x,wq,scales,256);
  // The cached 256-thread quantizer is bit-exact to the original quantizer.
  // Bound it to the compressed MiniCPM geometry and measured service row range.
  if (compressed && x.dim()==2 && x.size(0)>=17 && x.size(0)<=2048 &&
      x.size(1)==2048 && wq.dim()==2 && wq.size(0)==2048 && wq.size(1)==12288)
    return linear_impl(x,wq,scales,original,std::nullopt,min_rows,true,256);
  return gate_silu(x,wq,scales,original,min_rows);
}

}

TORCH_LIBRARY(biv150_native, m) {
  m.def("quantize_residual_probe(Tensor x, int threads) -> Tensor[]");
  m.def("gate_residual_cached_probe(Tensor x, Tensor weight_int8, Tensor scales, int threads) -> Tensor");
  m.def("quantize_rows_probe(Tensor x, int threads) -> Tensor[]");
  m.def("gate_silu_cached_probe(Tensor x, Tensor weight_int8, Tensor scales, Tensor marker, int threads) -> Tensor");
  m.def("gate_silu_auto(Tensor x, Tensor weight_int8, Tensor scales, Tensor original, int min_rows) -> Tensor");
  m.def("gate_silu_residual(Tensor x, Tensor weight_int8, Tensor scales) -> Tensor");
  m.def("linear(Tensor x, Tensor weight_int8, Tensor scales, Tensor original, Tensor? bias, int min_rows) -> Tensor");
  m.def("gate_silu(Tensor x, Tensor weight_int8, Tensor scales, Tensor original, int min_rows) -> Tensor");
}
TORCH_LIBRARY_IMPL(biv150_native, CUDA, m) {
  m.impl("quantize_residual_probe", &quantize_residual_probe);
  m.impl("gate_residual_cached_probe", &gate_residual_cached_probe);
  m.impl("quantize_rows_probe", &quantize_rows_probe);
  m.impl("gate_silu_cached_probe", &gate_silu_cached_probe);
  m.impl("gate_silu_auto", &gate_silu_auto);
  m.impl("gate_silu_residual", &gate_silu_residual);
  m.impl("linear", &linear);
  m.impl("gate_silu", &gate_silu);
}
