#include <cstdint>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

extern "C" void launch_cjj_native_i8_gate2048(
    const signed char*,const signed char*,const float*,const float*,void*,cudaStream_t);

namespace {
bool storage_overlaps(const at::Tensor& a,const at::Tensor& b) {
  const auto ap=reinterpret_cast<std::uintptr_t>(a.storage().data_ptr().get());
  const auto bp=reinterpret_cast<std::uintptr_t>(b.storage().data_ptr().get());
  return ap<bp+b.storage().nbytes() && bp<ap+a.storage().nbytes();
}
void gate_out(const at::Tensor& aq,const at::Tensor& wq,const at::Tensor& rows,
              const at::Tensor& cols,const at::Tensor& out) {
  TORCH_CHECK(aq.is_cuda(),"This explicit component requires CUDA tensors");
  for(const auto& t:{aq,wq,rows,cols,out}) {
    TORCH_CHECK(t.is_cuda() && t.device()==aq.device() && t.is_contiguous(),"one device and contiguous tensors required");
  }
  TORCH_CHECK(aq.scalar_type()==at::kChar && aq.sizes()==at::IntArrayRef({2048,2048}),"AQ signed I8 [2048,2048]");
  TORCH_CHECK(wq.scalar_type()==at::kChar && wq.sizes()==at::IntArrayRef({2048,12288}),"WQ signed I8 [2048,12288]");
  TORCH_CHECK(rows.scalar_type()==at::kFloat && rows.sizes()==at::IntArrayRef({2048}),"row scales F32 [2048]");
  TORCH_CHECK(cols.scalar_type()==at::kFloat && cols.sizes()==at::IntArrayRef({12288}),"column scales F32 [12288]");
  TORCH_CHECK(out.scalar_type()==at::kBFloat16 && out.sizes()==at::IntArrayRef({2048,6144}),"provided output BF16 [2048,6144]");
  for(const auto& t:{aq,wq,rows,cols})TORCH_CHECK(!storage_overlaps(t,out),"output must not share any input backing allocation");
  // AQ/row scales come from the unchanged production quantizer; content
  // finiteness belongs to the caller's numeric fixture, not a device sync here.
  c10::cuda::CUDAGuard guard(aq.device());
  launch_cjj_native_i8_gate2048(aq.data_ptr<signed char>(),wq.data_ptr<signed char>(),
      rows.data_ptr<float>(),cols.data_ptr<float>(),out.data_ptr(),c10::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
TORCH_LIBRARY(cjj_native_i8_gate2048_probe,m) {
  m.def("gate_out(Tensor aq, Tensor wq, Tensor row_scales, Tensor col_scales, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(cjj_native_i8_gate2048_probe,CUDA,m) {m.impl("gate_out",&gate_out);}
