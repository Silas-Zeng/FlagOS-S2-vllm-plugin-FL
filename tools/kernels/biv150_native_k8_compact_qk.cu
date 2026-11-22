// CPU-only draft of original S2 producer. Not frozen/compiled/device-qualified.
#include <cstdint>
#include <type_traits>
#include <cmath>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "biv150_native_k8_codec.cuh"

#if defined(__IX_ARCH__) && (!defined(__ivcore11__) || __IVCORE_ARCH__ != 11)
#error "Bound to actual BI-V150/native14f ivcore11 target"
#endif

#if defined(__IX_ARCH__)
namespace {
using QKShape=nvcuda::wmma::MMBaseInfo<16,16,16,16,16,128>;
using PVShape=nvcuda::wmma::MMBaseInfo<16,16,16,16,128,16>;
template<class S>using A=nvcuda::wmma::FragmentA<S,__nv_bfloat16,nvcuda::wmma::row_major>;
template<class S>using BCol=nvcuda::wmma::FragmentB<S,__nv_bfloat16,nvcuda::wmma::col_major>;
template<class S>using BRow=nvcuda::wmma::FragmentB<S,__nv_bfloat16,nvcuda::wmma::row_major>;
template<class S>using C=nvcuda::wmma::FragmentC<S,float>;
using QKI8Shape=nvcuda::wmma::MMBaseInfo<16,16,32,16,16,128>;
using AI8=nvcuda::wmma::FragmentA<QKI8Shape,signed char,nvcuda::wmma::row_major>;
using BI8=nvcuda::wmma::FragmentB<QKI8Shape,signed char,nvcuda::wmma::col_major>;
using CI8=nvcuda::wmma::FragmentC<QKI8Shape,signed int>;
static_assert(sizeof(AI8)==8 && sizeof(BI8)==8 && sizeof(CI8)==16,"I8 fragment ABI changed");
using Desc=int __attribute__((ext_vector_type(4)));
static_assert(sizeof(A<QKShape>)==8 && sizeof(BCol<QKShape>)==8 && sizeof(C<QKShape>)==16,"SDK fragment changed");

__device__ __forceinline__ unsigned raw16(__nv_bfloat16 x){union{__nv_bfloat16 b;unsigned short u;}v;v.b=x;return v.u;}
__device__ __forceinline__ unsigned pair_bits(__nv_bfloat16 a,__nv_bfloat16 b){return raw16(a)|(raw16(b)<<16);}
__device__ __forceinline__ __nv_bfloat16 from_bits(unsigned x){union{__nv_bfloat16 b;unsigned short u;}v;v.u=static_cast<unsigned short>(x);return v.b;}
__device__ __forceinline__ Desc descriptor(const __nv_bfloat16* base,int page,unsigned head){
 const auto p=reinterpret_cast<std::uintptr_t>(base+static_cast<std::uint64_t>(page)*6272+head*128);
 return Desc{static_cast<int>(p),static_cast<int>(p>>32),-1,512};
}
__device__ __forceinline__ float shuffle(float x,unsigned index){return __shfl(x,index,64);}
__device__ __forceinline__ unsigned shuffle_bits(unsigned x,unsigned index){return static_cast<unsigned>(__shfl(static_cast<int>(x),index,64));}

// CPU draft only. Fixed original NHD page16/D128/8Q, no production installer.
__device__ __forceinline__ unsigned k_half_index(unsigned slot,unsigned dim){
 const unsigned r=dim%32,pair=r/2,c=slot;
 return (dim/32)*512+(c/4)*128+(pair&3)*32+((pair/8)^(c/8))*16+(((pair/4)&1)^((c/4)&1))*8+(c&3)*2+(r&1);
}
__device__ __forceinline__ unsigned v_half_index(unsigned slot,unsigned dim){
 const unsigned panel=dim/16,c=dim%16,r=slot/2;
 return panel*256+(r/4)*128+((r*32+c*2+(slot&1)+panel*64)&127);
}
__device__ __forceinline__ void clear_k_tail(unsigned* stage,unsigned valid,unsigned lane){
 // Each physical shared word has exactly one writer; K pairs adjacent D.
 for(unsigned i=lane;i<1024;i+=64){
  const unsigned slot=i/64,dim=(i%64)*2;
  if(slot>=valid)stage[k_half_index(slot,dim)/2]=0;
 }
}
__device__ __forceinline__ void clear_v_tail(unsigned* stage,unsigned valid,unsigned lane){
 // V pairs adjacent slots; preserve the valid low half if the cut is odd.
 for(unsigned i=lane;i<1024;i+=64){
  const unsigned slot=(i/128)*2,dim=i%128,index=1024+v_half_index(slot,dim)/2;
  if(slot>=valid)stage[index]=0;
  else if(slot+1>=valid)stage[index]&=0xffffu;
 }
}
// Own independently checked I8 col image; no BF16 fragment/SME alias.
__device__ __forceinline__ unsigned k_i8_index(unsigned slot,unsigned dim){
 const unsigned panel=dim/64,within=dim%64,quartet=within/4,byte=within%4,bank=slot/4,subslot=slot%4;
 return panel*1024+bank*256+(quartet%4)*64+((quartet/4)^bank)*16+subslot*4+byte;
}
__device__ __forceinline__ void issue_k_i8(unsigned* stage,const signed char* key,int page,unsigned head,unsigned valid,unsigned lane){
 const auto* base=key+static_cast<std::int64_t>(page)*12544+head*128;
 // 512 unique aligned words. Invalid slots do not perform a source read.
 for(unsigned word=lane;word<512;word+=64){
  const unsigned slot=word/32,dim=(word%32)*4;
  unsigned bits=0;
  if(slot<valid)bits=*reinterpret_cast<const unsigned*>(base+slot*256+dim);
  stage[k_i8_index(slot,dim)/4]=bits;
 }
}
__device__ __forceinline__ int signed_i32_word(unsigned bits){
 union{unsigned bits;int signed_value;}value;value.bits=bits;return value.signed_value;
}
__device__ __forceinline__ void issue_v(unsigned dst,Desc desc){
 __ivcorex_sme_load_16x1b64_rowxfb16(dst+4096,desc,0,0);
 __ivcorex_sme_load_16x1b64_rowxfb16(dst+5120,desc,64,0);
 __ivcorex_sme_load_16x1b64_rowxfb16(dst+6144,desc,128,0);
 __ivcorex_sme_load_16x1b64_rowxfb16(dst+7168,desc,192,0);
}

template<bool OverlapV>
__device__ __forceinline__ void consume_page(
 const AI8 (&q)[4],float qscale0,float qscale1,const signed char* key,const float* key_scale_full,const __nv_bfloat16* value,
 int page,unsigned head,unsigned valid,unsigned lane,unsigned col,float scale,
 unsigned* stage,float& running_m0,float& running_m1,float& running_l0,float& running_l1,
 C<PVShape> (&acc)[8]){
 // Original ASM (not just its LLVM) fuses L*rho+page_sum with ml_fma_f32.
 const unsigned dst=static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(stage));
 // K ordinary global loads complete through dependent shared stores/barrier;
 // V retains original SME issue/wait/layout. No assumed SME8 instruction.
 issue_k_i8(stage,key,page,head,valid,lane);__syncthreads();
 static_assert(!OverlapV,"Only original overlap=false is admitted");
 issue_v(dst,descriptor(value,page,head));
 __ivcorex_sl_waitcnt(8ULL);__syncthreads();
 if(valid<16){clear_v_tail(stage,valid,lane);__syncthreads();}
 CI8 scores;
 #pragma unroll
 for(unsigned k=0;k<4;++k){
  BI8 b;nvcuda::wmma::load_matrix_sync_tcu(b,stage,0,k);
  nvcuda::wmma::mma_sync_tcu(scores,q[k],b,scores);
 }
 // Full layer-page0 F32 view; internal +1024 elements is added exactly once.
 const float kscale=col<valid?key_scale_full[static_cast<std::int64_t>(page)*3136+1024+col*2+head]:1.0f;
 const float s0=col<valid?static_cast<float>(signed_i32_word(scores[0]))*qscale0*kscale*scale:-INFINITY;
 const float s1=col<valid?static_cast<float>(signed_i32_word(scores[1]))*qscale1*kscale*scale:-INFINITY;
 float page_m0=s0,page_m1=s1;
 #pragma unroll
 for(unsigned mask=8;mask;mask>>=1){page_m0=fmaxf(page_m0,shuffle(page_m0,lane^mask));page_m1=fmaxf(page_m1,shuffle(page_m1,lane^mask));}
 const float new_m0=fmaxf(running_m0,page_m0),new_m1=fmaxf(running_m1,page_m1);
 const float p0=exp2f(s0-new_m0),p1=exp2f(s1-new_m1);
 const float rho0=exp2f(running_m0-new_m0),rho1=exp2f(running_m1-new_m1);
 float page_l0=p0,page_l1=p1;
 #pragma unroll
 for(unsigned mask=8;mask;mask>>=1){page_l0=page_l0+shuffle(page_l0,lane^mask);page_l1=page_l1+shuffle(page_l1,lane^mask);}
 running_l0=fmaf(running_l0,rho0,page_l0);running_l1=fmaf(running_l1,rho1,page_l1);
 const auto pb0=__float2bfloat16_rn(p0),pb1=__float2bfloat16_rn(p1);
 const unsigned pa=(lane&16)?raw16(pb1):raw16(pb0),pb=(lane&16)?raw16(pb0):raw16(pb1);
 const unsigned index=((lane&16)<<1)|((lane&32)>>1)|(lane&15);
 const unsigned u=shuffle_bits(pa,index),v=shuffle_bits(pb,index^16);
 const unsigned packed=(lane&32)?pair_bits(from_bits(v),from_bits(u)):pair_bits(from_bits(u),from_bits(v));
 A<PVShape> probability;probability[0]=packed;probability[1]=packed;
 if constexpr(OverlapV){
  // Wait-moving is a future variant only, not current numeric/perf baseline.
  // Actual compiler may force this wait earlier; LLVM/ASM must decide.
  __ivcorex_sl_waitcnt(8ULL);
  if(valid<16)clear_v_tail(stage,valid,lane);
  __syncthreads();
 }
 #pragma unroll
 for(unsigned n=0;n<8;++n){
  BRow<PVShape> b;nvcuda::wmma::load_matrix_sync_tcu(b,stage+1024,n,0);
  // Preserve every original C component, including the duplicate outputs.
  acc[n][0]=acc[n][0]*rho0;acc[n][1]=acc[n][1]*rho1;
  acc[n][2]=acc[n][2]*rho0;acc[n][3]=acc[n][3]*rho1;
  nvcuda::wmma::mma_sync_tcu(acc[n],probability,b,acc[n]);
 }
 running_m0=new_m0;running_m1=new_m1;
 __syncthreads(); // All shared fragment reads precede the next page overwrite.
}
}
#endif


#ifndef COREX_NATIVE_GQA_V_OVERLAP
#define COREX_NATIVE_GQA_V_OVERLAP 0
#endif
static_assert(COREX_NATIVE_GQA_V_OVERLAP==0,"Private K8 prototype requires original overlap=false");
template<unsigned Segments>
__global__ void cjj_native_k8_compact_producer(
 const __nv_bfloat16* query,const signed char* key,const float* key_scale_full,const __nv_bfloat16* value,
 const int* seq_lens,const int* block_table,float scale,
 std::int64_t table_stride,std::int64_t q_token_stride,std::int64_t q_head_stride,
 std::int64_t acc_token_stride,std::int64_t acc_head_stride,std::int64_t acc_segment_stride,
 std::int64_t scalar_token_stride,std::int64_t scalar_head_stride,
 float* acc_out,float* max_out,float* sum_out){
#if defined(__IX_ARCH__)
 const unsigned lane=__ivcorex_lane_id(),seq=blockIdx.x,h=blockIdx.y,segment=blockIdx.z,g=lane/16,col=lane%16;
 static_assert(Segments==2 || Segments==4,"Only separately qualified S2/S4 planned");
 const int length=seq_lens[seq],steps=(length+Segments*16-1)/(Segments*16),lower=segment*steps*16;
 if(lower>=length)return; // Exact original empty/padded-row no-write contract.
 const int upper=min(lower+steps*16,length),full_upper=lower+((upper-lower)/16)*16;
 AI8 q[4];
 float qscale0,qscale1;
 float values0[8],values1[8],peak0=0.0f,peak1=0.0f;
 #pragma unroll
 for(unsigned k=0;k<8;++k){
  const auto* base=query+static_cast<std::int64_t>(seq)*q_token_stride;
  values0[k]=__bfloat162float(base[(h*8+2*g)*q_head_stride+16*k+col]);
  values1[k]=__bfloat162float(base[(h*8+2*g+1)*q_head_stride+16*k+col]);
  peak0=fmaxf(peak0,fabsf(values0[k]));peak1=fmaxf(peak1,fabsf(values1[k]));
 }
 #pragma unroll
 for(unsigned mask=8;mask;mask>>=1){
  peak0=fmaxf(peak0,shuffle(peak0,lane^mask));peak1=fmaxf(peak1,shuffle(peak1,lane^mask));
 }
 const float input_scale0=cjj_k8_scale_rn32(peak0),input_scale1=cjj_k8_scale_rn32(peak1);
 // C row g/g+4 differs from original input pair 2*g/2*g+1.
 // All lane collectives execute unconditionally before the local selection.
 const float c00=shuffle(input_scale0,(g/2)*16+col),c01=shuffle(input_scale1,(g/2)*16+col);
 const float c10=shuffle(input_scale0,(g/2+2)*16+col),c11=shuffle(input_scale1,(g/2+2)*16+col);
 qscale0=(g&1)?c01:c00;qscale1=(g&1)?c11:c10;
 #pragma unroll
 for(unsigned k=0;k<4;++k){
  #pragma unroll
  for(unsigned component=0;component<2;++component){
   const unsigned i=2*k+component;
   const unsigned pair=(static_cast<unsigned>(cjj_k8_code_rne(values0[i],input_scale0))&255u)
      |((static_cast<unsigned>(cjj_k8_code_rne(values1[i],input_scale1))&255u)<<8);
   const unsigned a=shuffle_bits(pair,((2*g)%4)*16+col);
   const unsigned b=shuffle_bits(pair,((2*g+1)%4)*16+col);
   q[k][component]=(a&65535u)|((b&65535u)<<16);
  }
 }
 __shared__ __align__(16) unsigned stage[2048];
 C<PVShape> acc[8]; // All 32 FP32 components are zero-initialized.
 float m0=-INFINITY,m1=-INFINITY,l0=0.0f,l1=0.0f;
 const auto* row=block_table+static_cast<std::int64_t>(seq)*table_stride;
 int position=lower;
 while(position<full_upper){
  consume_page<COREX_NATIVE_GQA_V_OVERLAP!=0>(q,qscale0,qscale1,key,key_scale_full,value,row[position/16],h,16,lane,col,scale,stage,m0,m1,l0,l1,acc);
  position+=16;
 }
 if(full_upper<upper){
  consume_page<COREX_NATIVE_GQA_V_OVERLAP!=0>(q,qscale0,qscale1,key,key_scale_full,value,row[full_upper/16],h,upper-full_upper,lane,col,scale,stage,m0,m1,l0,l1,acc);
 }
 const std::int64_t row0=h*8+g,row1=row0+4;
 const std::int64_t a0=static_cast<std::int64_t>(seq)*acc_token_stride+row0*acc_head_stride+segment*acc_segment_stride;
 const std::int64_t a1=static_cast<std::int64_t>(seq)*acc_token_stride+row1*acc_head_stride+segment*acc_segment_stride;
 #pragma unroll
 for(unsigned n=0;n<8;++n){acc_out[a0+n*16+col]=acc[n][2];acc_out[a1+n*16+col]=acc[n][3];}
 if(col==0){
  const auto s0=static_cast<std::int64_t>(seq)*scalar_token_stride+row0*scalar_head_stride+segment;
  const auto s1=static_cast<std::int64_t>(seq)*scalar_token_stride+row1*scalar_head_stride+segment;
  max_out[s0]=m0*0.6931471805599453f;max_out[s1]=m1*0.6931471805599453f;sum_out[s0]=l0;sum_out[s1]=l1;
 }
#endif
}

// First draft exposes only S2; S4 remains the same source template, not an
// enabled/qualified path. A future R32 component must instantiate/validate S4.
extern "C" void launch_cjj_native_k8_compact_s2(
 const void* query,const void* key,const void* key_scale_full,const void* value,const int* seq_lens,
 const int* block_table,int batch,float scale,
 std::int64_t table_stride,std::int64_t q_token_stride,std::int64_t q_head_stride,
 std::int64_t acc_token_stride,std::int64_t acc_head_stride,std::int64_t acc_segment_stride,
 std::int64_t scalar_token_stride,std::int64_t scalar_head_stride,
 float* acc,float* maximum,float* summation,cudaStream_t stream){
 cjj_native_k8_compact_producer<2><<<dim3(batch,2,2),64,0,stream>>>(
  static_cast<const __nv_bfloat16*>(query),static_cast<const signed char*>(key),static_cast<const float*>(key_scale_full),
  static_cast<const __nv_bfloat16*>(value),seq_lens,block_table,scale,
  table_stride,q_token_stride,q_head_stride,acc_token_stride,acc_head_stride,
  acc_segment_stride,scalar_token_stride,scalar_head_stride,acc,maximum,summation);
}

extern "C" void launch_cjj_native_k8_compact_s4(
 const void* query,const void* key,const void* key_scale_full,const void* value,const int* seq_lens,
 const int* block_table,int batch,float scale,
 std::int64_t table_stride,std::int64_t q_token_stride,std::int64_t q_head_stride,
 std::int64_t acc_token_stride,std::int64_t acc_head_stride,std::int64_t acc_segment_stride,
 std::int64_t scalar_token_stride,std::int64_t scalar_head_stride,
 float* acc,float* maximum,float* summation,cudaStream_t stream){
 cjj_native_k8_compact_producer<4><<<dim3(batch,2,4),64,0,stream>>>(
  static_cast<const __nv_bfloat16*>(query),static_cast<const signed char*>(key),static_cast<const float*>(key_scale_full),
  static_cast<const __nv_bfloat16*>(value),seq_lens,block_table,scale,
  table_stride,q_token_stride,q_head_stride,acc_token_stride,acc_head_stride,
  acc_segment_stride,scalar_token_stride,scalar_head_stride,acc,maximum,summation);
}
