def call(self, args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    s26 = arg0_1
    assert_size_stride(arg1_1, (s26, 2048), (2048, 1))
    assert_size_stride(arg2_1, (2048,), (1,))
    assert_size_stride(arg3_1, (s26, 2048), (2048, 1))
    assert_size_stride(arg4_1, (2048, 12288), (12288, 1))
    assert_size_stride(arg5_1, (12288,), (1,))
    assert_size_stride(arg6_1, (12288, 2048), (0, 0))
    assert_size_stride(arg7_1, (2048, 6144), (6144, 1))
    assert_size_stride(arg8_1, (2048,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((s26, 2048), (2048, 1), torch.bfloat16)
        stream0 = get_raw_stream(0)
        buf2 = _biv150_generated_norm_gate_pair(triton_red_fused__to_copy_add_gate_silu_auto_mean_mul_pow_rsqrt_0.run, torch.ops.biv150_native.gate_silu_auto.default, (arg3_1, arg1_1, arg2_1, buf1, s26, 2048), {'stream': stream0}, (buf1, arg4_1, arg5_1, arg6_1, 17))
        del arg2_1
        pass
        del arg4_1
        del arg5_1
        del arg6_1
        buf3 = buf2
        assert_size_stride(buf3, (s26, 6144), (6144, 1), 'torch.ops.biv150_native.gate_silu_auto.default')
        assert_alignment(buf3, 16, 'torch.ops.biv150_native.gate_silu_auto.default')
        del buf2
        buf4 = buf1
        del buf1
        extern_kernels.mm(buf3, reinterpret_tensor(arg7_1, (6144, 2048), (1, 6144), 0), out=buf4)
        del arg7_1
        del buf3
        buf6 = empty_strided_cuda((s26, 2048), (2048, 1), torch.bfloat16)
        buf7 = empty_strided_cuda((s26, 2048), (2048, 1), torch.bfloat16)
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1.run(buf4, arg3_1, arg1_1, arg8_1, buf6, buf7, s26, 2048, stream=stream0)
        del arg1_1
        del arg3_1
        del arg8_1
    return (buf6, buf7, buf4)
