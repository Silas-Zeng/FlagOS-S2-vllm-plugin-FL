#include <cstdint>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
extern "C" void launch_corex_native_gqa_s8_component(
 const void*,const void*,const void*,const int*,const int*,float,
 std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,std::int64_t,
 float*,float*,float*,cudaStream_t);

void produce_n30_s8(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
 const at::Tensor& lengths,const at::Tensor& table,const at::Tensor& partial,
 const at::Tensor& maximum,const at::Tensor& summation,double scale){
 TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.sizes()==at::IntArrayRef({30,16,128}) &&
   q.strides()==at::IntArrayRef({2048,128,1}), "unsupported query");
 for(const auto& t:{k,v,lengths,table,partial,maximum,summation})
  TORCH_CHECK(t.is_cuda() && t.device()==q.device(), "all tensors must share query device");
 for(const auto& t:{k,v}){
  TORCH_CHECK(t.scalar_type()==at::kBFloat16 && t.dim()==4 && t.size(1)==16 && t.size(2)==2 && t.size(3)==128);
  TORCH_CHECK(t.strides()==at::IntArrayRef({8192,256,128,1}), "unsupported KV layout");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr())%64==0, "KV base must be 64-byte aligned");
 }
 TORCH_CHECK(k.size(0)==v.size(0) && k.size(0)>0, "KV physical page pools must be positive and equal");
 TORCH_CHECK(lengths.scalar_type()==at::kInt && lengths.sizes()==at::IntArrayRef({30}) && lengths.is_contiguous());
 TORCH_CHECK(table.scalar_type()==at::kInt && table.dim()==2 && table.size(0)==30 && table.size(1)>=1024 && table.stride(1)==1);
 TORCH_CHECK(partial.scalar_type()==at::kFloat && partial.dim()==4 && partial.size(0)>=30 &&
   partial.size(1)==16 && partial.size(2)==8 && partial.size(3)==128 && partial.stride(3)==1 && partial.stride(2)==128);
 for(const auto& t:{maximum,summation})
  TORCH_CHECK(t.scalar_type()==at::kFloat && t.dim()==3 && t.size(0)>=30 && t.size(1)==16 && t.size(2)==8 && t.stride(2)==1);
 TORCH_CHECK(maximum.strides()==summation.strides());
 c10::cuda::CUDAGuard guard(q.device());
 launch_corex_native_gqa_s8_component(q.data_ptr(),k.data_ptr(),v.data_ptr(),lengths.data_ptr<int>(),table.data_ptr<int>(),static_cast<float>(scale),
  table.stride(0),q.stride(0),q.stride(1),partial.stride(0),partial.stride(1),partial.stride(2),maximum.stride(0),maximum.stride(1),
  partial.data_ptr<float>(),maximum.data_ptr<float>(),summation.data_ptr<float>(),c10::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
TORCH_LIBRARY(corex_native_gqa_s8_component,m){m.def("produce(Tensor query, Tensor key, Tensor value, Tensor lengths, Tensor table, Tensor(a!) partial, Tensor(b!) maximum, Tensor(c!) summation, float scale) -> ()");}
TORCH_LIBRARY_IMPL(corex_native_gqa_s8_component,CUDA,m){m.impl("produce",&produce_n30_s8);}
