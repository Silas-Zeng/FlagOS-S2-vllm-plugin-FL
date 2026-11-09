#include <cstdint>
#include <cmath>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

using Launch=void(*)(const void*,const void*,const void*,const void*,const int*,const int*,int,float,
 std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,
 float*,float*,float*,cudaStream_t);
extern "C" void launch_corex_native_k8_qreload_gqa_s2_draft(const void*,const void*,const void*,const void*,const int*,const int*,int,float,
 std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,
 float*,float*,float*,cudaStream_t);
extern "C" void launch_corex_native_k8_qreload_gqa_s4_draft(const void*,const void*,const void*,const void*,const int*,const int*,int,float,
 std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,
 float*,float*,float*,cudaStream_t);
extern "C" void cjj_launch_native_k8_qreload_codec(const void*,void*,float*,int,cudaStream_t);

namespace {
std::uintptr_t backing_begin(const at::Tensor& t){return reinterpret_cast<std::uintptr_t>(t.storage().data_ptr().get());}
std::uintptr_t backing_end(const at::Tensor& t){return backing_begin(t)+t.storage().nbytes();}
bool backing_overlap(const at::Tensor& a,const at::Tensor& b){
 return backing_begin(a)<backing_end(b) && backing_begin(b)<backing_end(a);
}
void same_device(const at::Tensor& t,const at::Tensor& q){TORCH_CHECK(t.is_cuda() && t.device()==q.device(),"one CUDA device required");}

void produce(const at::Tensor& q,const at::Tensor& k,const at::Tensor& scale_full,const at::Tensor& v,
 const at::Tensor& lengths,const at::Tensor& table,const at::Tensor& partial,
 const at::Tensor& maximum,const at::Tensor& summation,double scale,std::int64_t segments){
 TORCH_CHECK(segments==2 || segments==4,"S2/S4 only");
 const std::int64_t rows=segments==2?64:32;
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.sizes()==at::IntArrayRef({rows,16,128}),"Q BF16 shape");
 TORCH_CHECK(q.strides()==at::IntArrayRef({2048,128,1}) || q.strides()==at::IntArrayRef({2560,128,1}),"original Q stride");
 for(const auto& t:{k,scale_full,v,lengths,table,partial,maximum,summation})same_device(t,q);
 TORCH_CHECK(k.scalar_type()==at::kChar && k.dim()==4 && k.size(0)>0 && k.size(1)==16 && k.size(2)==2 && k.size(3)==128,"signed K code role");
 TORCH_CHECK(k.strides()==at::IntArrayRef({16384,256,128,1}),"16KiB K code strides only");
 const std::int64_t pages=k.size(0);
 TORCH_CHECK(pages<=2147483647,"int32 page IDs");
 TORCH_CHECK(v.scalar_type()==at::kBFloat16 && v.sizes()==k.sizes() && v.strides()==at::IntArrayRef({8192,256,128,1}),"already-offset original BF16 V role");
 TORCH_CHECK(scale_full.scalar_type()==at::kFloat && scale_full.dim()==1 && scale_full.is_contiguous() && scale_full.numel()>=pages*4096,"FULL layer-page0 F32 scale role");
 const auto kb=reinterpret_cast<std::uintptr_t>(k.data_ptr());
 TORCH_CHECK(kb%4==0 && reinterpret_cast<std::uintptr_t>(scale_full.data_ptr())==kb,"scale must point to same FULL base, not +4096B");
 TORCH_CHECK(reinterpret_cast<std::uintptr_t>(v.data_ptr())==kb+8192,"V must already point to +8192B once");
 TORCH_CHECK(backing_begin(k)==backing_begin(scale_full) && backing_begin(k)==backing_begin(v),"same backing storage");
 TORCH_CHECK(kb>=backing_begin(k) && kb+pages*16384<=backing_end(k),"complete canonical cache allocation");
 TORCH_CHECK(lengths.scalar_type()==at::kInt && lengths.sizes()==at::IntArrayRef({rows}) && lengths.is_contiguous(),"lengths I32");
 TORCH_CHECK(table.scalar_type()==at::kInt && table.dim()==2 && table.size(0)==rows && table.size(1)>0 && table.stride(1)==1 && table.stride(0)>=table.size(1),"table I32 layout");
 TORCH_CHECK(partial.scalar_type()==at::kFloat && partial.dim()==4 && partial.size(0)>=rows && partial.size(1)==16 && partial.size(2)==segments && partial.size(3)==128,"partial F32 shape");
 TORCH_CHECK(partial.stride(3)==1 && partial.stride(2)==128 && partial.stride(1)>=segments*128 && partial.stride(0)>=16*partial.stride(1),"partial nonoverlap strides");
 for(const auto& t:{maximum,summation}){
  TORCH_CHECK(t.scalar_type()==at::kFloat && t.dim()==3 && t.size(0)>=rows && t.size(1)==16 && t.size(2)==segments,"scalar F32 shape");
  TORCH_CHECK(t.stride(2)==1 && t.stride(1)>=segments && t.stride(0)>=16*t.stride(1),"scalar nonoverlap strides");
 }
 TORCH_CHECK(maximum.strides()==summation.strides(),"scalar common stride");
 for(const auto& src:{q,lengths,table})TORCH_CHECK(!backing_overlap(src,k),"readonly sources cannot alias live cache");
 for(const auto& out:{partial,maximum,summation}){
  for(const auto& src:{q,k,scale_full,v,lengths,table})TORCH_CHECK(!backing_overlap(out,src),"scratch/readonly cache alias");
 }
 TORCH_CHECK(!backing_overlap(partial,maximum) && !backing_overlap(partial,summation) && !backing_overlap(maximum,summation),"scratch outputs need separate backing in this prototype");
 TORCH_CHECK(std::isfinite(scale) && scale>0.0,"positive finite scale_log2e");
 // Content prerequisite, not certified by tensor dtype/shape: component-owned
 // CPU preupload lengths/table must fit table extent/page range; valid K/Q and
 // per-token/head scales must be finite. This is NOT a live service entry.
 c10::cuda::CUDAGuard guard(q.device());
 Launch launch=segments==2?launch_corex_native_k8_qreload_gqa_s2_draft:launch_corex_native_k8_qreload_gqa_s4_draft;
 launch(q.data_ptr(),k.data_ptr(),scale_full.data_ptr(),v.data_ptr(),lengths.data_ptr<int>(),table.data_ptr<int>(),static_cast<int>(rows),static_cast<float>(scale),
  table.stride(0),q.stride(0),q.stride(1),partial.stride(0),partial.stride(1),partial.stride(2),maximum.stride(0),maximum.stride(1),
  partial.data_ptr<float>(),maximum.data_ptr<float>(),summation.data_ptr<float>(),c10::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void codec(const at::Tensor& input,const at::Tensor& codes,const at::Tensor& scales){
 TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kBFloat16 && input.dim()==2 && input.size(0)>0 && input.size(0)<=2147483647 && input.size(1)==128 && input.is_contiguous(),"finite owned BF16 codec fixture [rows,128]");
 same_device(codes,input);same_device(scales,input);
 TORCH_CHECK(codes.scalar_type()==at::kChar && codes.sizes()==input.sizes() && codes.is_contiguous(),"I8 codec output");
 TORCH_CHECK(scales.scalar_type()==at::kFloat && scales.dim()==1 && scales.size(0)==input.size(0) && scales.is_contiguous(),"F32 codec scales");
 TORCH_CHECK(!backing_overlap(input,codes) && !backing_overlap(input,scales) && !backing_overlap(codes,scales),"codec backing alias");
 c10::cuda::CUDAGuard guard(input.device());
 cjj_launch_native_k8_qreload_codec(input.data_ptr(),codes.data_ptr(),scales.data_ptr<float>(),static_cast<int>(input.size(0)),c10::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
TORCH_LIBRARY(cjj_native_k8_qreload_probe,m){
 m.def("produce(Tensor query, Tensor key_code, Tensor scale_full, Tensor value_offset, Tensor lengths, Tensor table, Tensor(a!) partial, Tensor(b!) maximum, Tensor(c!) summation, float scale_log2e, int segments) -> ()");
 m.def("codec(Tensor input, Tensor(a!) codes, Tensor(b!) scales) -> ()");
}
TORCH_LIBRARY_IMPL(cjj_native_k8_qreload_probe,CUDA,m){m.impl("produce",&produce);m.impl("codec",&codec);}
