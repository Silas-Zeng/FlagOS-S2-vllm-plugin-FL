#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_functions.h>
#include <cstdint>

// Header compatibility and actual RN32/FTZ lowering require target compilation
// and IR inspection. No ordinary division / reciprocal / half-away substitute.
#if defined(__IX_ARCH__)
__device__ __forceinline__ float cjj_k8_qreload_scale_rn32(float peak){
 return peak==0.0f?1.0f:fmaxf(__fdiv_rn(peak,127.0f),1.1754943508222875e-38f);
}
__device__ __forceinline__ int cjj_k8_qreload_code_rne(float value,float scale){
 const int code=__float2int_rn(__fdiv_rn(value,scale));
 return max(-127,min(127,code));
}
#endif
