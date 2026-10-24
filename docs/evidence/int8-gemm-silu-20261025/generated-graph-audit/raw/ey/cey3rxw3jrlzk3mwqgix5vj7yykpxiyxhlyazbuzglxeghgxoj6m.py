
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=71, major=7, regs_per_multiprocessor=262144, max_threads_per_multi_processor=8192, max_threads_per_block=4096, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'backend_hash': 'F5FD294A91E70FCAFA2471F47F488BC998971E650BB152BC40EEE4B822EE2FAB', 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 128)
        x1 = ((xindex // 128) % 2)
        x2 = xindex // 256
        x4 = xindex
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 64, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (2048 + 128*x1 + 2560*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full([XBLOCK], 131072, tl.int32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp6 < 0
        tmp10 = tl.where(tmp9, tmp8, tmp6)
        tmp11 = tl.load(in_ptr2 + (128*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tmp5 * tmp11
        tmp13 = tl.load(in_ptr0 + (2112 + 128*x1 + 2560*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (64 + 128*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 - tmp15
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp4, tmp16, tmp17)
        tmp19 = tmp0 >= tmp3
        tmp20 = tl.full([1], 128, tl.int64)
        tmp21 = tmp0 < tmp20
        tmp22 = tl.load(in_ptr0 + (2112 + 128*x1 + 2560*x2 + ((-64) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (x2), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.full([XBLOCK], 131072, tl.int32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp23 < 0
        tmp27 = tl.where(tmp26, tmp25, tmp23)
        tmp28 = tl.load(in_ptr2 + (128*tmp27 + ((-64) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tmp22 * tmp28
        tmp30 = tl.load(in_ptr0 + (2048 + 128*x1 + 2560*x2 + ((-64) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr2 + (64 + 128*tmp27 + ((-64) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tmp30 * tmp31
        tmp33 = tmp29 + tmp32
        tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
        tmp35 = tl.where(tmp19, tmp33, tmp34)
        tmp36 = tl.where(tmp4, tmp18, tmp35)
        tl.store(out_ptr0 + (x4), tmp36, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 128)
        x6 = ((xindex // 128) % 16)
        x7 = xindex // 2048
        x9 = xindex
        tmp37 = x5
        tmp38 = tl.full([1], 0, tl.int64)
        tmp39 = tmp37 >= tmp38
        tmp40 = tl.full([1], 64, tl.int64)
        tmp41 = tmp37 < tmp40
        tmp42 = tl.load(in_ptr0 + (128*x6 + 2560*x7 + (x5)), tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp43 = tl.load(in_ptr1 + (x7), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
        tmp44 = tl.full([XBLOCK], 131072, tl.int32)
        tmp45 = tmp43 + tmp44
        tmp46 = tmp43 < 0
        tmp47 = tl.where(tmp46, tmp45, tmp43)
        tmp48 = tl.load(in_ptr2 + (128*tmp47 + (x5)), tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp49 = tmp42 * tmp48
        tmp50 = tl.load(in_ptr0 + (64 + 128*x6 + 2560*x7 + (x5)), tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp51 = tl.load(in_ptr2 + (64 + 128*tmp47 + (x5)), tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp52 = tmp50 * tmp51
        tmp53 = tmp49 - tmp52
        tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
        tmp55 = tl.where(tmp41, tmp53, tmp54)
        tmp56 = tmp37 >= tmp40
        tmp57 = tl.full([1], 128, tl.int64)
        tmp58 = tmp37 < tmp57
        tmp59 = tl.load(in_ptr0 + (64 + 128*x6 + 2560*x7 + ((-64) + x5)), tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp60 = tl.load(in_ptr1 + (x7), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
        tmp61 = tl.full([XBLOCK], 131072, tl.int32)
        tmp62 = tmp60 + tmp61
        tmp63 = tmp60 < 0
        tmp64 = tl.where(tmp63, tmp62, tmp60)
        tmp65 = tl.load(in_ptr2 + (128*tmp64 + ((-64) + x5)), tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp66 = tmp59 * tmp65
        tmp67 = tl.load(in_ptr0 + (128*x6 + 2560*x7 + ((-64) + x5)), tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp68 = tl.load(in_ptr2 + (64 + 128*tmp64 + ((-64) + x5)), tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp69 = tmp67 * tmp68
        tmp70 = tmp66 + tmp69
        tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
        tmp72 = tl.where(tmp56, tmp70, tmp71)
        tmp73 = tl.where(tmp41, tmp55, tmp72)
        tl.store(out_ptr1 + (x9), tmp73, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((2048, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((131072, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((2048, 2, 128), (256, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((2048, 16, 128), (2048, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 524288, 4194304,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
