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

namespace {
// Reuse the old GEMM's 32 rows: main quantization in [0,16), residual in [16,32).
// Only M<=16 is eligible, so no additional GEMM rows or weight reads are needed.
__global__ void quantize_residual_rows(const BF16* x, int8_t* q, float* scales,
                                      int m, int k) {
  const int row = blockIdx.x;
  if (row >= m) {
    for (int col=threadIdx.x; col<k; col+=blockDim.x) {
      q[int64_t(row)*k+col]=0;
      q[int64_t(row+16)*k+col]=0;
    }
    if (threadIdx.x==0) { scales[row]=1.f; scales[row+16]=1.f; }
    return;
  }
  __shared__ float maximum[4];
  float peak=0.f;
  for (int col=threadIdx.x; col<k; col+=blockDim.x)
    peak=fmaxf(peak,fabsf(__bfloat162float(x[int64_t(row)*k+col])));
  for (int shift=32; shift>0; shift>>=1)
    peak=fmaxf(peak,__shfl_xor(peak,shift,64));
  if ((threadIdx.x&63)==0) maximum[threadIdx.x>>6]=peak;
  __syncthreads();
  peak=fmaxf(fmaxf(maximum[0],maximum[1]),fmaxf(maximum[2],maximum[3]));
  const float main_scale=fmaxf(peak,1.e-6f)/127.f;
  const float residual_scale=main_scale/254.f;
  if (threadIdx.x==0) { scales[row]=main_scale; scales[row+16]=residual_scale; }
  for (int col=threadIdx.x; col<k; col+=blockDim.x) {
    const float value=__bfloat162float(x[int64_t(row)*k+col]);
    const float f=value/main_scale;
    const float rounded=f>=0.f?floorf(f+0.5f):ceilf(f-0.5f);
    const int8_t first=int8_t(fminf(127.f,fmaxf(-127.f,rounded)));
    const float residual=(value-float(first)*main_scale)/residual_scale;
    const float second=residual>=0.f?floorf(residual+0.5f):ceilf(residual-0.5f);
    q[int64_t(row)*k+col]=first;
    q[int64_t(row+16)*k+col]=int8_t(fminf(127.f,fmaxf(-127.f,second)));
  }
}

__global__ void residual_silu_epilogue(const int32_t* acc, const float* rows,
                                     const float* cols, BF16* output, int n) {
  const int row=blockIdx.y;
  const int col=blockIdx.x*blockDim.x+threadIdx.x;
  const int half=n/2;
  if (col<half) {
    const int64_t index=int64_t(row)*n+col;
    const int64_t residual_index=int64_t(row+16)*n+col;
    const float gate_raw=(float(acc[index])*rows[row]+float(acc[residual_index])*rows[row+16])*cols[col];
    const float up_raw=(float(acc[index+half])*rows[row]+float(acc[residual_index+half])*rows[row+16])*cols[col+half];
    const float gate=__bfloat162float(__float2bfloat16_rn(gate_raw));
    const float up=__bfloat162float(__float2bfloat16_rn(up_raw));
    output[int64_t(row)*half+col]=__float2bfloat16_rn((gate/(1.f+expf(-gate)))*up);
  }
}
}  // namespace

extern "C" void biv150_quantize_residual(const void* x, int8_t* q, float* scales,
                                         int m, int k, cudaStream_t stream) {
  quantize_residual_rows<<<16,256,0,stream>>>(static_cast<const BF16*>(x),q,scales,m,k);
}

extern "C" void biv150_residual_silu_epilogue(const int32_t* acc, const float* rows,
                                              const float* cols, void* output,
                                              int m, int n, cudaStream_t stream) {
  residual_silu_epilogue<<<dim3((n/2+255)/256,m),256,0,stream>>>(
      acc,rows,cols,static_cast<BF16*>(output),n);
}

// Retain each original BF16 activation value between max and
// quantization. Finite-input math, divisions, rounding and clamping stay exact.
namespace {
template<int THREADS>
__global__ void quantize_cached_rows(const BF16* x, int8_t* q, float* scales, int m) {
  const int row=blockIdx.x;
  constexpr int K=2048, ITEMS=K/THREADS;
  if (row>=m) {
    for (int col=threadIdx.x;col<K;col+=THREADS) q[int64_t(row)*K+col]=0;
    if (threadIdx.x==0) scales[row]=1.f;
    return;
  }
  float values[ITEMS];
  float peak=0.f;
  #pragma unroll
  for (int item=0;item<ITEMS;++item) {
    values[item]=__bfloat162float(x[int64_t(row)*K+threadIdx.x+item*THREADS]);
    peak=fmaxf(peak,fabsf(values[item]));
  }
  for (int shift=32;shift>0;shift>>=1) peak=fmaxf(peak,__shfl_xor(peak,shift,64));
  if constexpr (THREADS>64) {
    __shared__ float maxima[THREADS/64];
    if ((threadIdx.x&63)==0) maxima[threadIdx.x>>6]=peak;
    __syncthreads();
    peak=0.f;
    #pragma unroll
    for (int warp=0;warp<THREADS/64;++warp) peak=fmaxf(peak,maxima[warp]);
  }
  const float scale=fmaxf(peak,1.e-6f)/127.f;
  if (threadIdx.x==0) scales[row]=scale;
  #pragma unroll
  for (int item=0;item<ITEMS;++item) {
    const float value=values[item]/scale;
    const float rounded=value>=0.f?floorf(value+0.5f):ceilf(value-0.5f);
    q[int64_t(row)*K+threadIdx.x+item*THREADS]=int8_t(fminf(127.f,fmaxf(-127.f,rounded)));
  }
}
}
extern "C" void biv150_quantize_cached(const void* x,int8_t* q,float* scales,
    int m,int padded_m,int k,int threads,cudaStream_t stream) {
  // Both public probe and bounded service dispatch require K2048 geometry.
  if (threads==64) quantize_cached_rows<64><<<padded_m,64,0,stream>>>(static_cast<const BF16*>(x),q,scales,m);
  else if (threads==128) quantize_cached_rows<128><<<padded_m,128,0,stream>>>(static_cast<const BF16*>(x),q,scales,m);
  else quantize_cached_rows<256><<<padded_m,256,0,stream>>>(static_cast<const BF16*>(x),q,scales,m);
}

// Component-only: two adjacent columns, preserving every original BF16 round.
#include <cstring>
namespace {
__device__ __forceinline__ BF16 vec2_scalar(int32_t acc_g, int32_t acc_u,
                                          float row, float col_g, float col_u) {
  const float gate = __bfloat162float(__float2bfloat16_rn(float(acc_g) * (row * col_g)));
  const float up = __bfloat162float(__float2bfloat16_rn(float(acc_u) * (row * col_u)));
  return __float2bfloat16_rn((gate / (1.f + expf(-gate))) * up);
}
__global__ void silu_vec2_epilogue(const int32_t* acc, const float* rows,
                                  const float* cols, BF16* output, int n) {
  const int row = blockIdx.y;
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = n / 2;
  const int col = pair * 2;
  if (col + 1 < half) {
    const int64_t index = int64_t(row) * n + col;
    const int2 gate_acc = *reinterpret_cast<const int2*>(acc + index);
    const int2 up_acc = *reinterpret_cast<const int2*>(acc + index + half);
    const float2 gate_cols = *reinterpret_cast<const float2*>(cols + col);
    const float2 up_cols = *reinterpret_cast<const float2*>(cols + col + half);
    const float row_scale = rows[row];
    const BF16 left = vec2_scalar(gate_acc.x, up_acc.x, row_scale, gate_cols.x, up_cols.x);
    const BF16 right = vec2_scalar(gate_acc.y, up_acc.y, row_scale, gate_cols.y, up_cols.y);
    ushort2 packed;
    static_assert(sizeof(BF16) == sizeof(unsigned short), "BF16 size");
    memcpy(&packed.x, &left, sizeof(left));
    memcpy(&packed.y, &right, sizeof(right));
    *reinterpret_cast<ushort2*>(output + int64_t(row) * half + col) = packed;
  }
}
}
extern "C" void biv150_silu_vec2(const int32_t* acc, const float* rows,
                                const float* cols, void* output,
                                int m, int n, cudaStream_t stream) {
  silu_vec2_epilogue<<<dim3((n/4+255)/256,m),256,0,stream>>>(
      acc, rows, cols, static_cast<BF16*>(output), n);
}
