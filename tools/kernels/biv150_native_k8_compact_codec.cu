#include "biv150_native_k8_codec.cuh"

// Diagnostic codec only; not a cache encoder. Finite, BF16-owned input is an
// explicit fixture prerequisite. The compile-only frontend never calls it.
__global__ void cjj_native_k8_compact_codec_probe(const __nv_bfloat16* input,
 signed char* codes,float* scales,int rows){
#if defined(__IX_ARCH__)
 const unsigned lane=threadIdx.x,row=blockIdx.x;
 if(row>=static_cast<unsigned>(rows))return;
 const float a=__bfloat162float(input[static_cast<std::int64_t>(row)*128+lane]);
 const float b=__bfloat162float(input[static_cast<std::int64_t>(row)*128+lane+64]);
 float peak=fmaxf(fabsf(a),fabsf(b));
 #pragma unroll
 for(unsigned mask=32;mask;mask>>=1)peak=fmaxf(peak,__shfl(peak,lane^mask,64));
 const float scale=cjj_k8_scale_rn32(peak);
 codes[static_cast<std::int64_t>(row)*128+lane]=static_cast<signed char>(cjj_k8_code_rne(a,scale));
 codes[static_cast<std::int64_t>(row)*128+lane+64]=static_cast<signed char>(cjj_k8_code_rne(b,scale));
 if(lane==0)scales[row]=scale;
#endif
}
extern "C" void cjj_launch_native_k8_compact_codec(const void* input,void* codes,float* scales,int rows,cudaStream_t stream){
 cjj_native_k8_compact_codec_probe<<<rows,64,0,stream>>>(static_cast<const __nv_bfloat16*>(input),static_cast<signed char*>(codes),scales,rows);
}
