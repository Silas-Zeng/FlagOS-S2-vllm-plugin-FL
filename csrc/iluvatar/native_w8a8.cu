// Operator-side experiment: quantize -> vendor int8 GEMM -> fused epilogue.
// Native dispatch keeps the Python interpreter out of the captured operator.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace {
using BF16 = __ix_bfloat16;

__global__ void quantize_rows(const BF16* x, int8_t* q, float* scales, int m, int k) {
  const int row = blockIdx.x;
  if (row >= m) {
    for (int col = threadIdx.x; col < k; col += blockDim.x)
      q[int64_t(row) * k + col] = 0;
    if (threadIdx.x == 0) scales[row] = 1.f;
    return;
  }
  __shared__ float maximum[4];
  float value = 0.f;
  for (int col = threadIdx.x; col < k; col += blockDim.x)
    value = fmaxf(value, fabsf(__bfloat162float(x[int64_t(row) * k + col])));
  for (int shift = 32; shift > 0; shift >>= 1)
    value = fmaxf(value, __shfl_xor(value, shift, 64));
  if ((threadIdx.x & 63) == 0) maximum[threadIdx.x >> 6] = value;
  __syncthreads();
  const float peak = fmaxf(fmaxf(maximum[0], maximum[1]), fmaxf(maximum[2], maximum[3]));
  const float scale = fmaxf(peak, 1.e-6f) / 127.f;
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < k; col += blockDim.x) {
    const float f = __bfloat162float(x[int64_t(row) * k + col]) / scale;
    const float rounded = f >= 0.f ? floorf(f + 0.5f) : ceilf(f - 0.5f);
    q[int64_t(row) * k + col] = int8_t(fminf(127.f, fmaxf(-127.f, rounded)));
  }
}

__global__ void epilogue(const int32_t* acc, const float* rows,
                         const float* cols, const BF16* bias,
                         BF16* output, int n) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t index = int64_t(row) * n + col;
  if (col < n) {
    float value = float(acc[index]) * (rows[row] * cols[col]);
    if (bias) value += __bfloat162float(bias[col]);
    output[index] = __float2bfloat16_rn(value);
  }
}

__global__ void silu_epilogue(const int32_t* acc, const float* rows,
                              const float* cols, BF16* output, int n) {
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = n / 2;
  if (col < half) {
    const int64_t index = int64_t(row) * n + col;
    // Reproduce the BF16 linear output rounding before the FP32 activation.
    const float gate = __bfloat162float(__float2bfloat16_rn(float(acc[index]) * (rows[row] * cols[col])));
    const float up = __bfloat162float(__float2bfloat16_rn(float(acc[index+half]) * (rows[row] * cols[col+half])));
    output[int64_t(row)*half+col] = __float2bfloat16_rn((gate / (1.f + expf(-gate))) * up);
  }
}

}  // namespace

extern "C" void biv150_quantize(const void* x, int8_t* q, float* scales,
                                 int m, int padded_m, int k, cudaStream_t stream) {
  quantize_rows<<<padded_m, 256, 0, stream>>>(static_cast<const BF16*>(x), q, scales, m, k);
}

extern "C" void biv150_epilogue(const int32_t* acc, const float* rows,
                                 const float* cols, const void* bias,
                                 void* output, int m, int n, cudaStream_t stream) {
  epilogue<<<dim3((n + 255) / 256, m), 256, 0, stream>>>(
      acc, rows, cols, static_cast<const BF16*>(bias), static_cast<BF16*>(output),
      n);
}

extern "C" void biv150_silu_epilogue(const int32_t* acc, const float* rows,
                                      const float* cols, void* output,
                                      int m, int n, cudaStream_t stream) {
  silu_epilogue<<<dim3((n/2+255)/256,m),256,0,stream>>>(
      acc, rows, cols, static_cast<BF16*>(output), n);
}
