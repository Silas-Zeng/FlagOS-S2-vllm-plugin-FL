// Standalone, unqualified M2048 gate component. No production dispatcher.
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#if defined(__IX_ARCH__) && (!defined(__ivcore11__) || __IVCORE_ARCH__ != 11)
#error "This prototype is bound to BI-V150 ivcore11"
#endif

#if defined(__IX_ARCH__)
namespace {
// One CTA owns 64x64 output; four warps each own 16x64. A, gate B and up B
// each have four 1024-byte panels, reused across both K32 instructions.
using Shape = nvcuda::wmma::MMBaseInfo<16,16,32,16,16,64>;
using A = nvcuda::wmma::FragmentA<Shape,signed char,nvcuda::wmma::row_major>;
using B = nvcuda::wmma::FragmentB<Shape,signed char,nvcuda::wmma::col_major>;
using C = nvcuda::wmma::FragmentC<Shape,signed int>;
static_assert(sizeof(A)==8 && sizeof(B)==8 && sizeof(C)==16,"SDK fragment ABI changed");

// Exact CoordToOffset<8,row/col> definitions from the saved CoreX SDK.
// Coordinates are local to a 16x64 A panel or a 64x16 B panel.
__device__ __forceinline__ unsigned a_byte(unsigned row,unsigned column) {
  const unsigned i=column/16,c=column%16,r=row/4;
  return i*256+((r*64+c*4+(row&3)+64*i)&255);
}
__device__ __forceinline__ unsigned b_byte(unsigned row,unsigned column) {
  const unsigned r=row/4;
  return (column/4)*256+(r&3)*64+((r/4)^(column/4))*16+(column&3)*4+(row&3);
}
__device__ __forceinline__ unsigned pack4(const signed char* p,unsigned stride) {
  unsigned word=0;
  #pragma unroll
  for(unsigned b=0;b<4;++b)word|=(static_cast<unsigned>(p[b*stride])&255u)<<(8*b);
  return word;
}
__device__ __forceinline__ int signed_word(unsigned bits) {
  union { unsigned u; int i; } v;v.u=bits;return v.i;
}
} // namespace
#endif

__global__ void cjj_native_i8_gate2048_kernel(
    const signed char* aq,const signed char* wq,const float* rows,
    const float* cols,__nv_bfloat16* output) {
#if defined(__IX_ARCH__)
  const unsigned tid=threadIdx.x,lane=__ivcorex_lane_id(),warp=tid/64;
  const unsigned row_base=blockIdx.y*64,col_base=blockIdx.x*64;
  __shared__ __align__(16) unsigned stage[3072]; // Exactly 12 KiB; one stage.
  C gate[4],up[4]; // SDK signed-I32 FragmentC constructor zeroes all words.
  #pragma unroll 1
  for(unsigned k_base=0;k_base<2048;k_base+=64) {
    // Every 32-bit shared word has one writer. Each packed byte is read from
    // the original AQ / Kx12288 WQ; there is no persistent weight repacking.
    for(unsigned word=tid;word<1024;word+=256) {
      const unsigned panel=word/256,within=word%256;
      const unsigned ar=(within/64)*4,ak=within%64;
      stage[panel*256+a_byte(ar,ak)/4]=pack4(aq+(row_base+panel*16+ar)*2048+k_base+ak,2048);
      const unsigned bk=(within/16)*4,bn=within%16;
      const auto* p=wq+(k_base+bk)*12288+col_base+panel*16+bn;
      stage[1024+panel*256+b_byte(bk,bn)/4]=pack4(p,12288);
      stage[2048+panel*256+b_byte(bk,bn)/4]=pack4(p+6144,12288);
    }
    __syncthreads(); // All A/G/U ordinary stores precede fragment loads.
    #pragma unroll
    for(unsigned kh=0;kh<2;++kh) {
      A a;nvcuda::wmma::load_matrix_sync_tcu(a,stage+warp*256,0,kh);
      #pragma unroll
      for(unsigned n=0;n<4;++n) {
        B g,u;
        nvcuda::wmma::load_matrix_sync_tcu(g,stage+1024+n*256,0,kh);
        nvcuda::wmma::load_matrix_sync_tcu(u,stage+2048+n*256,0,kh);
        nvcuda::wmma::mma_sync_tcu(gate[n],a,g,gate[n]);
        nvcuda::wmma::mma_sync_tcu(up[n],a,u,up[n]);
      }
    }
    __syncthreads(); // No next-stage overwrite before any warp's last read.
  }
  #pragma unroll
  for(unsigned n=0;n<4;++n) {
    const unsigned col=col_base+n*16+lane%16;
    #pragma unroll
    for(unsigned j=0;j<4;++j) {
      const unsigned row=row_base+warp*16+lane/16+4*j;
      // Keep original scale grouping, independent BF16 projection rounding,
      // FP32 expf/div/multiply and final BF16 rounding, in that order.
      const float g=__bfloat162float(__float2bfloat16_rn(float(signed_word(gate[n][j]))*(rows[row]*cols[col])));
      const float u=__bfloat162float(__float2bfloat16_rn(float(signed_word(up[n][j]))*(rows[row]*cols[col+6144])));
      output[row*6144+col]=__float2bfloat16_rn((g/(1.f+expf(-g)))*u);
    }
  }
#endif
}

extern "C" void launch_cjj_native_i8_gate2048(
    const signed char* aq,const signed char* wq,const float* rows,
    const float* cols,void* output,cudaStream_t stream) {
  cjj_native_i8_gate2048_kernel<<<dim3(96,32),256,0,stream>>>(
      aq,wq,rows,cols,static_cast<__nv_bfloat16*>(output));
}
