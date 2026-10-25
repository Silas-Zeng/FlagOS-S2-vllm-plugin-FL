# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /workspace/experiments/minicpm-s2/live-carry-codegen-v2-20261025/device-inductor-cache/kb/ckbdavm3xbqhmqrhfe7cvddkufzfqbpvat7puge72jkpg43zqlay.py
# Topologically Sorted Source Nodes: [x, to_1, x_1, pow_1, variance, add_1, rsqrt, x_2, to_3, x_3, gate_silu_auto], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, biv150_native.gate_silu_auto]
# Source node to ATen node mapping:
#   add_1 => add_19
#   gate_silu_auto => gate_silu_auto
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_1 => convert_element_type_1
#   to_3 => convert_element_type_3
#   variance => mean
#   x => convert_element_type
#   x_1 => add_6
#   x_2 => mul_13
#   x_3 => mul_18
# Graph fragment:
#   %arg3_1 : Tensor "bf16[s26, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg1_1 : Tensor "bf16[s26, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf0 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=buf0]
#   %arg2_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %convert_element_type : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %add_6 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %pow_1 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 2), kwargs = {})
#   %mean : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_19 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %mul_13 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %rsqrt), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   %mul_18 : Tensor "bf16[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %arg2_1), kwargs = {})
#   %gate_silu_auto : Tensor "bf16[s26, 6144][6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.biv150_native.gate_silu_auto.default](args = (%mul_18, %arg4_1, %arg5_1, %arg6_1, 17), kwargs = {})
#   return %buf0,%buf1
triton_red_fused__to_copy_add_gate_silu_auto_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_gate_silu_auto_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=71, major=7, regs_per_multiprocessor=262144, max_threads_per_multi_processor=8192, max_threads_per_block=4096, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_gate_silu_auto_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F5FD294A91E70FCAFA2471F47F488BC998971E650BB152BC40EEE4B822EE2FAB', 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_gate_silu_auto_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = 2048.0
        tmp15 = (tmp7 / tmp14)
        tmp16 = 1e-06
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp22, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/experiments/minicpm-s2/live-carry-codegen-v2-20261025/device-inductor-cache/kp/ckpz724a2x4uinylj75ydelfiiilwacxw4unnwpmpmverwmxdgto.py
# Topologically Sorted Source Nodes: [x, to_1, x_1, x_4, x_5, pow_2, variance_1, add_3, rsqrt_1, x_6, to_8, x_7, x_residual_1], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_3 => add_60
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to_1 => convert_element_type_1
#   to_8 => convert_element_type_9
#   variance_1 => mean_1
#   x => convert_element_type
#   x_1 => add_6
#   x_4 => convert_element_type_6
#   x_5 => add_47
#   x_6 => mul_38
#   x_7 => mul_43
#   x_residual_1 => convert_element_type_8
# Graph fragment:
#   %mm : Tensor "bf16[s26, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg3_1 : Tensor "bf16[s26, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg1_1 : Tensor "bf16[s26, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %buf5 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=buf5]
#   %arg8_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %convert_element_type : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.float32), kwargs = {})
#   %add_6 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %convert_element_type_6 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %add_47 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, %add_6), kwargs = {})
#   %pow_2 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_47, 2), kwargs = {})
#   %mean_1 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_60 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_60,), kwargs = {})
#   %mul_38 : Tensor "f32[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %rsqrt_1), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_38, torch.bfloat16), kwargs = {})
#   %mul_43 : Tensor "bf16[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, %arg8_1), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[s26, 2048][2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_47, torch.bfloat16), kwargs = {})
#   return %buf5,%mul_43,%convert_element_type_8
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=16, cc=71, major=7, regs_per_multiprocessor=262144, max_threads_per_multi_processor=8192, max_threads_per_block=4096, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'F5FD294A91E70FCAFA2471F47F488BC998971E650BB152BC40EEE4B822EE2FAB', 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp1 + tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 2048*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp19 = tmp13 + tmp18
        tmp20 = 2048.0
        tmp21 = (tmp10 / tmp20)
        tmp22 = 1e-06
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp26 * tmp27
        tmp29 = tmp19.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2048*x0), tmp28, r0_mask & xmask)
        tl.store(out_ptr2 + (r0_1 + 2048*x0), tmp29, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
        args.clear()
        s26 = arg0_1
        assert_size_stride(arg1_1, (s26, 2048), (2048, 1))
        assert_size_stride(arg2_1, (2048, ), (1, ))
        assert_size_stride(arg3_1, (s26, 2048), (2048, 1))
        assert_size_stride(arg4_1, (2048, 12288), (12288, 1))
        assert_size_stride(arg5_1, (12288, ), (1, ))
        assert_size_stride(arg6_1, (12288, 2048), (0, 0))
        assert_size_stride(arg7_1, (2048, 6144), (6144, 1))
        assert_size_stride(arg8_1, (2048, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((s26, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, to_1, x_1, pow_1, variance, add_1, rsqrt, x_2, to_3, x_3, gate_silu_auto], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, biv150_native.gate_silu_auto]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_gate_silu_auto_mean_mul_pow_rsqrt_0.run(arg3_1, arg1_1, arg2_1, buf1, s26, 2048, stream=stream0)
            del arg2_1
            # Topologically Sorted Source Nodes: [x, to_1, x_1, pow_1, variance, add_1, rsqrt, x_2, to_3, x_3, gate_silu_auto], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, biv150_native.gate_silu_auto]
            buf2 = torch.ops.biv150_native.gate_silu_auto.default(buf1, arg4_1, arg5_1, arg6_1, 17)
            del arg4_1
            del arg5_1
            del arg6_1
            buf3 = buf2
            assert_size_stride(buf3, (s26, 6144), (6144, 1), 'torch.ops.biv150_native.gate_silu_auto.default')
            assert_alignment(buf3, 16, 'torch.ops.biv150_native.gate_silu_auto.default')
            del buf2
            buf4 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_1], Original ATen: [aten.linear]
            extern_kernels.mm(buf3, reinterpret_tensor(arg7_1, (6144, 2048), (1, 6144), 0), out=buf4)
            del arg7_1
            del buf3
            buf6 = empty_strided_cuda((s26, 2048), (2048, 1), torch.bfloat16)
            buf7 = empty_strided_cuda((s26, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [x, to_1, x_1, x_4, x_5, pow_2, variance_1, add_3, rsqrt_1, x_6, to_8, x_7, x_residual_1], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1.run(buf4, arg3_1, arg1_1, arg8_1, buf6, buf7, s26, 2048, stream=stream0)
            del arg1_1
            del arg3_1
            del arg8_1
        return (buf6, buf7, buf4, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2048
    arg1_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((2048, 12288), (12288, 1), device='cuda:0', dtype=torch.int8)
    arg5_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((12288, 2048), (0, 0), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
