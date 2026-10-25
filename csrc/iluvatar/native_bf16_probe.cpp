// Standalone, UNTESTED Corex BF16 linear-out candidate. No GPU work was run
// while preparing this file. Keep M32/M64 production routing unchanged.
// Inputs: contiguous row-major X[M,K], W[N,K]; caller-owned C[M,N].
// Computes C = X @ W.T with BF16 input/output and FP32 compute.
//
// Use a Torch C++ extension (is_python_module=False); no .cu file required.
// Add /usr/local/corex/include to includes and link:
//   -L/usr/local/corex/lib64 -lcublas -ltorch_cuda -lc10_cuda
// plus normal Torch extension libraries/ABI flags. Do not load this file into
// the running baseline service. Compile/link, stream, numerical and performance
// validation remain the owner's work in an isolated diagnostic process.

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <IX/ixblas/ixblas_api.h>

#include <limits>

namespace {

inline void check_ixblas(ixblasStatus_t status, const char* operation) {
  TORCH_CHECK(status == IXBLAS_STATUS_SUCCESS, operation,
              " failed, ixblas status=", static_cast<int>(status));
}

at::Tensor bf16_linear_out(const at::Tensor& x, const at::Tensor& w,
                           at::Tensor out) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(),
              "x, w and out must be CUDA/Corex tensors");
  TORCH_CHECK(x.device() == w.device() && x.device() == out.device(),
              "x, w and out must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
              w.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16,
              "x, w and out must be BF16");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2,
              "x[M,K], w[N,K] and out[M,N] must be matrices");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous(),
              "x, w and out must be contiguous row-major matrices");
  const int64_t m64 = x.size(0), k64 = x.size(1), n64 = w.size(0);
  TORCH_CHECK(w.size(1) == k64 && out.size(0) == m64 && out.size(1) == n64,
              "incompatible linear-out shapes");
  constexpr int64_t limit = std::numeric_limits<int>::max();
  TORCH_CHECK(m64 > 0 && n64 > 0 && k64 > 0 &&
              m64 <= limit && n64 <= limit && k64 <= limit,
              "this candidate requires nonempty dimensions fitting BLAS int");
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, x);
  at::assert_no_overlap(out, w);

  const int m = static_cast<int>(m64);
  const int n = static_cast<int>(n64);
  const int k = static_cast<int>(k64);
  const c10::cuda::CUDAGuard device_guard(x.device());

  // Installed libtorch_cuda.so disassembly verifies this getter sets the
  // handle to c10::cuda::getCurrentCUDAStream(). It does not call SetMathMode.
  // The Torch thread/device handle pool retains ownership; do not destroy it.
  const ixblasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  ixStream_t handle_stream = nullptr;
  check_ixblas(ixblasGetStream_v2(handle, &handle_stream), "ixblasGetStream_v2");
  TORCH_CHECK(handle_stream == c10::cuda::getCurrentCUDAStream().stream(),
              "Torch BLAS handle is not bound to the current stream");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  static_assert(IX_R_16BF == 14 && IX_R_32F == 0,
                "revalidate this candidate against the installed Corex ABI");
  {
    // Existing ATen RAII implementation saves/restores pointer mode.
    // It neither modifies math mode nor changes global precision settings.
    const at::cuda::blas::PointerModeGuard pointer_mode(
        handle, CUBLAS_POINTER_MODE_HOST);

    // Row-major C[M,N] is column-major C^T[N,M]. Stored W[N,K] is
    // column-major W^T[K,N], so transpose it; stored X[M,K] is already
    // column-major X^T[K,M]. Thus C^T = op_T(W_storage) * op_N(X_storage).
    // The installed legacy ABI takes ixDataType computeType (0), NOT the
    // separate ixblasComputeType_t IXBLAS_COMPUTE_32F enum (68).
    check_ixblas(ixblasGemmEx(
        handle, IXBLAS_OP_T, IXBLAS_OP_N,
        n, m, k,
        &alpha,
        w.const_data_ptr<at::BFloat16>(), IX_R_16BF, k,
        x.const_data_ptr<at::BFloat16>(), IX_R_16BF, k,
        &beta,
        out.mutable_data_ptr<at::BFloat16>(), IX_R_16BF, n,
        IX_R_32F, IXBLAS_GEMM_DEFAULT), "ixblasGemmEx(BF16)");
  }
  // No explicit sync, tensor allocation, aten mm/linear dispatch, or host
  // transfer here. The first handle/library use may initialize private state;
  // warm this path outside capture and timed measurement.
  return out;
}

}  // namespace

TORCH_LIBRARY(biv150_native_bf16_probe, m) {
  m.def("linear_out(Tensor x, Tensor w, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(biv150_native_bf16_probe, CUDA, m) {
  m.impl("linear_out", TORCH_FN(bf16_linear_out));
}
