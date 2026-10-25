# v2 whole-round CPU/GPU diagnostic analysis

Identical exclusion intervals are applied to GPU events, NVTX scopes, host APIs and the elapsed-time denominator. Host waiting is not reported as CPU overhead.

## 4096 tokens / window 1

Original active envelope 234.284807s; retained 231.999898s after 4 diagnostic samples. GPU union 227.414830s (98.0237%); no recorded GPU activity 4.585068s.

| GPU event type | Retained events | Duration sum (s) | Union (s) |
|---|---:|---:|---:|
| cuda_graph | 3711 | 122.964475 | 122.964475 |
| cuda_kernel | 377137 | 104.055926 | 104.055926 |
| cuda_memcpy | 28079 | 0.394429 | 0.394429 |
| cuda_memset | 0 | 0.000000 | 0.000000 |

| Host scope | Inclusive (s) | Exclusive (s) | Exclusive with no recorded GPU activity (s) |
|---|---:|---:|---:|
| engine.step_with_batch_queue | 231.863543 | 0.501044 | 0.003955 |
| worker.execute_model | 106.180600 | 0.470454 | 0.003396 |
| engine.wait_for_output_plus_sampling | 103.245889 | 103.245889 | 0.548578 |
| gpu_model_runner: forward | 89.181687 | 89.181687 | 2.097584 |
| execute_new_1_cached_62 | 29.965951 | 0.015334 | 0.000335 |
| execute_new_0_cached_62 | 28.625339 | 0.015122 | 0.000353 |
| execute_new_0_cached_64 | 14.388575 | 0.266308 | 0.001623 |
| worker.sample_tokens | 14.103216 | 0.258872 | 0.001552 |
| gpu_model_runner: preprocess | 12.257283 | 12.257283 | 0.400137 |
| gpu_model_runner: sample | 11.031207 | 11.031207 | 0.811345 |
| execute_new_0_cached_63 | 6.944231 | 0.004198 | 0.000068 |
| scheduler.schedule | 5.679931 | 1.825346 | 0.016753 |
| execute_new_1_cached_61 | 4.243066 | 0.002096 | 0.000059 |
| gpu_model_runner: postprocess | 3.947808 | 3.947808 | 0.095548 |
| schedule: allocate_slots | 3.107684 | 3.107684 | 0.186234 |
| scheduler.update_from_output | 2.104199 | 2.104199 | 0.297614 |
| gpu_model_runner: bookkeep | 1.638839 | 1.638839 | 0.039915 |
| gpu_model_runner: AsyncGPUModelRunnerOutput | 1.117412 | 1.117412 | 0.070907 |

| Non-Graph kernel | Retained events | GPU duration (s) | Mean retained (ms) |
|---|---:|---:|---:|
| mm_kernel | 70484 | 34.031612 | 0.482827 |
| kernel_biv150_prefill_normalized | 22092 | 27.390315 | 1.239830 |
| _gqa_decode_attention | 21714 | 12.268440 | 0.565001 |
| void Gemm_tcu_mr_kernel::gemm_tcu_h<256u, 256u, 64u, 64u, 64u, 2u, (ixblasTrans_t)1, (ixblasTrans_t)1, (ixblasTCUGEMMCategory_t)0, (ixmma::MMEdgeCheckMode)0, (ixblasGEMMBoundary_t)2, (ixblasLtEpilogue_t)1, false, int, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, false, true, 4u, 4u, false, int, int, int, signed char, signed char>(int*, int*, signed char const*, signed char const*, int, int, int, int, int, int, int, int, int, int, long, long, long, long, long, long, ixblasGEMMBatchMode_t, int, void*, void*, void*, unsigned int) | 17884 | 8.293357 | 0.463731 |
| linear_kernel | 4237 | 4.783410 | 1.128961 |
| (anonymous namespace)::silu_epilogue(int const*, float const*, float const*, __ix_bfloat16*, int) | 17884 | 4.096982 | 0.229086 |
| triton_red_fused_fused_add_rms_norm_1 | 17884 | 1.488373 | 0.083224 |
| triton_poi_fused_2 | 17884 | 1.471454 | 0.082278 |
| true_div_func_kernel_rank_2 | 4237 | 1.436802 | 0.339108 |
| _weighted_hist | 8418 | 1.230452 | 0.146169 |
| triton_red_fused_fused_add_rms_norm_gate_silu_auto_0 | 17884 | 0.906442 | 0.050685 |
| true_div_func_kernel_rank_1 | 4237 | 0.782630 | 0.184713 |

## 16384 tokens / window 2

Original active envelope 558.240927s; retained 555.886648s after 4 diagnostic samples. GPU union 549.859611s (98.9158%); no recorded GPU activity 6.027037s.

| GPU event type | Retained events | Duration sum (s) | Union (s) |
|---|---:|---:|---:|
| cuda_graph | 3745 | 180.669046 | 180.669046 |
| cuda_kernel | 695318 | 368.937863 | 368.937863 |
| cuda_memcpy | 31321 | 0.252702 | 0.252702 |
| cuda_memset | 0 | 0.000000 | 0.000000 |

| Host scope | Inclusive (s) | Exclusive (s) | Exclusive with no recorded GPU activity (s) |
|---|---:|---:|---:|
| engine.step_with_batch_queue | 555.741692 | 0.545119 | 0.010219 |
| worker.execute_model | 367.123485 | 0.362354 | 0.002075 |
| gpu_model_runner: forward | 337.746136 | 337.746136 | 4.070575 |
| execute_new_0_cached_31 | 173.685250 | 0.099161 | 0.000901 |
| engine.wait_for_output_plus_sampling | 158.759080 | 158.759080 | 0.583129 |
| execute_new_0_cached_30 | 50.082011 | 0.011284 | 0.000030 |
| execute_new_1_cached_30 | 32.425555 | 0.008871 | 0.000071 |
| gpu_model_runner: preprocess | 22.674888 | 22.674888 | 0.479698 |
| worker.sample_tokens | 22.172713 | 0.285903 | 0.001745 |
| gpu_model_runner: sample | 19.496029 | 19.496029 | 0.291035 |
| execute_new_0_cached_32 | 18.926019 | 0.154144 | 0.001107 |
| execute_new_0_cached_33 | 12.478696 | 0.020294 | 0.000275 |
| gpu_model_runner: postprocess | 5.975871 | 5.975871 | 0.153540 |
| scheduler.schedule | 5.835850 | 3.466723 | 0.173466 |
| execute_new_0_cached_1 | 2.930007 | 0.029415 | 0.000005 |
| execute_new_1_cached_31 | 2.668265 | 0.000605 | 0.000012 |
| execute_new_0_cached_24 | 2.665989 | 0.001323 | 0.000017 |
| execute_new_0_cached_25 | 2.447788 | 0.001148 | 0.000000 |

| Non-Graph kernel | Retained events | GPU duration (s) | Mean retained (ms) |
|---|---:|---:|---:|
| kernel_biv150_prefill_normalized | 44814 | 209.497097 | 4.674814 |
| mm_kernel | 142978 | 68.861543 | 0.481623 |
| _gqa_decode_attention | 43806 | 42.855412 | 0.978300 |
| void Gemm_tcu_mr_kernel::gemm_tcu_h<256u, 256u, 64u, 64u, 64u, 2u, (ixblasTrans_t)1, (ixblasTrans_t)1, (ixblasTCUGEMMCategory_t)0, (ixmma::MMEdgeCheckMode)0, (ixblasGEMMBoundary_t)2, (ixblasLtEpilogue_t)1, false, int, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, (ixblasLtMatmulMatrixScale_t)1002, false, true, 4u, 4u, false, int, int, int, signed char, signed char>(int*, int*, signed char const*, signed char const*, int, int, int, int, int, int, int, int, int, int, long, long, long, long, long, long, ixblasGEMMBatchMode_t, int, void*, void*, void*, unsigned int) | 36040 | 16.711487 | 0.463693 |
| (anonymous namespace)::silu_epilogue(int const*, float const*, float const*, __ix_bfloat16*, int) | 36278 | 8.280153 | 0.228242 |
| linear_kernel | 4812 | 4.506923 | 0.936601 |
| triton_red_fused_fused_add_rms_norm_1 | 36278 | 3.010539 | 0.082985 |
| triton_poi_fused_2 | 36278 | 2.974927 | 0.082004 |
| triton_red_fused_fused_add_rms_norm_gate_silu_auto_0 | 36278 | 1.826304 | 0.050342 |
| reshape_and_cache_kernel_flash | 44814 | 1.357065 | 0.030282 |
| triton_poi_fused_mul_silu_slice_1 | 8536 | 1.170665 | 0.137144 |
| void (anonymous namespace)::quantize_cached_rows<256>(__ix_bfloat16 const*, signed char*, float*, int) | 36278 | 0.973860 | 0.026844 |

## Limits

- Host scope/API durations are elapsed host time, including GPU waiting, not CPU utilization or CPU overhead.
- Uncovered means no recorded GPU activity; it does not establish a CPU cause.
- The common mask also clips normal asynchronous work overlapping diagnostic sampling; it is not an uninstrumented counterfactual.
- Counts and token totals describe all original scheduled steps; retained host times exclude diagnostic intervals.
- Graph intervals retain complete opaque spans; missing internal kernel rows are never treated as GPU idle.
- Graph node event samples are analyzed separately and are not multiplied into whole-round kernel totals.
- All results are diagnostic; profiler and metadata overhead remain.

The raw trace remains unchanged; per-window JSON retains clipping evidence, CPU scopes, API durations, phase metadata and non-Graph kernel distributions.
