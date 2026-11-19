# BI-V150 graph event diagnostics

Analysis integrity: **PASS**. Complete valid 4K/16K pair: **True**. Diagnostic only; no official throughput claim.

All node durations below include instrumentation. The full/boundary delta is a fixed-order diagnostic comparison.

| Input | Graph | Samples | Boundary median ms | Full median ms | Paired added timing median % | DAG kernels ordered |
|---:|---|---:|---:|---:|---:|---|
| 16384 | worker-290618-window-2.json:bfe3742ebabd | 4 | 52.012573 | 54.510248 | 4.802052 | True |
| 4096 | worker-290618-window-1.json:6e328ca1f4af | 4 | 31.514686 | 33.990767 | 7.856974 | True |

## Input 16384, graph worker-290618-window-2.json:bfe3742ebabd: top driver names

Ranked by median per-sample **sum of instrumented intervals**, not native wall time.

| Rank | Driver kernel name | Timed calls | Median interval µs | Median sample interval sum ms |
|---:|---|---:|---:|---:|
| 1 | `_gqa_decode_attention` | 168 | 1049.723 | 44.102310 |
| 2 | `mm_kernel` | 536 | 31.404 | 5.541262 |
| 3 | `_ZN18Gemm_tcu_mr_kernel10gemm_tcu_hILj64ELj256ELj64ELj16ELj64ELj2EL13ixblasTrans_t1ELS1_1EL23ixblasTCUGEMMCategory_t0ELN5ixmma15MMEdgeCheckModeE0EL20ixblasGEMMBoundary_t2EL18ixblasLtEpilogue_t1ELb0EiL27ixblasLtMatmulMatrixScale_t1002ELS7_1002ELS7_1002ELb0ELb0ELj4ELj4ELb0EiiiaaEEvPT22_PT23_PKT24_PKT25_iiiiiiiiT21_SI_llllll21ixblasGEMMBatchMode_tiPvSK_SK_j` | 136 | 51.225 | 1.743987 |
| 4 | `_reduce_partials` | 168 | 8.337 | 0.350513 |
| 5 | `triton_red_fused_fused_add_rms_norm_1` | 136 | 8.025 | 0.272873 |
| 6 | `_ZN12_GLOBAL__N_113silu_epilogueEPKiPKfS3_P13__ix_bfloat16i` | 136 | 7.181 | 0.244790 |
| 7 | `reshape_and_cache_kernel_flash` | 168 | 5.635 | 0.237412 |
| 8 | `triton_poi_fused_2` | 136 | 6.815 | 0.232321 |
| 9 | `triton_red_fused_fused_add_rms_norm_gate_silu_auto_0` | 136 | 6.538 | 0.223648 |
| 10 | `_ZN12_GLOBAL__N_120quantize_cached_rowsILi256EEEvPK13__ix_bfloat16PaPfi` | 136 | 5.883 | 0.200069 |
| 11 | `triton_red_fused_fused_add_rms_norm_2` | 32 | 7.950 | 0.063073 |
| 12 | `triton_red_fused_fused_add_rms_norm_0` | 32 | 6.577 | 0.052796 |
| 13 | `triton_poi_fused_mul_silu_slice_1` | 32 | 6.281 | 0.050244 |
| 14 | `triton_poi_fused_3` | 28 | 6.838 | 0.048067 |
| 15 | `triton_red_fused__to_copy_embedding_rms_norm_0` | 4 | 10.810 | 0.010810 |
| 16 | `triton_poi_fused_1` | 4 | 6.652 | 0.006652 |

## Input 4096, graph worker-290618-window-1.json:6e328ca1f4af: top driver names

Ranked by median per-sample **sum of instrumented intervals**, not native wall time.

| Rank | Driver kernel name | Timed calls | Median interval µs | Median sample interval sum ms |
|---:|---|---:|---:|---:|
| 1 | `_gqa_decode_attention` | 168 | 540.331 | 22.691785 |
| 2 | `mm_kernel` | 536 | 37.027 | 6.109190 |
| 3 | `_ZN18Gemm_tcu_mr_kernel10gemm_tcu_hILj64ELj256ELj64ELj16ELj64ELj2EL13ixblasTrans_t1ELS1_1EL23ixblasTCUGEMMCategory_t0ELN5ixmma15MMEdgeCheckModeE0EL20ixblasGEMMBoundary_t2EL18ixblasLtEpilogue_t1ELb0EiL27ixblasLtMatmulMatrixScale_t1002ELS7_1002ELS7_1002ELb0ELb0ELj4ELj4ELb0EiiiaaEEvPT22_PT23_PKT24_PKT25_iiiiiiiiT21_SI_llllll21ixblasGEMMBatchMode_tiPvSK_SK_j` | 136 | 52.917 | 1.797973 |
| 4 | `_reduce_partials` | 168 | 10.473 | 0.439717 |
| 5 | `_ZN12_GLOBAL__N_113silu_epilogueEPKiPKfS3_P13__ix_bfloat16i` | 136 | 9.577 | 0.325990 |
| 6 | `triton_red_fused_fused_add_rms_norm_1` | 136 | 8.787 | 0.299204 |
| 7 | `triton_poi_fused_2` | 136 | 7.823 | 0.266306 |
| 8 | `reshape_and_cache_kernel_flash` | 168 | 5.723 | 0.241219 |
| 9 | `triton_red_fused_fused_add_rms_norm_gate_silu_auto_0` | 136 | 7.027 | 0.239954 |
| 10 | `_ZN12_GLOBAL__N_120quantize_cached_rowsILi256EEEvPK13__ix_bfloat16PaPfi` | 136 | 6.042 | 0.205619 |
| 11 | `triton_red_fused_fused_add_rms_norm_2` | 32 | 8.892 | 0.070046 |
| 12 | `triton_poi_fused_mul_silu_slice_1` | 32 | 7.631 | 0.061169 |
| 13 | `triton_red_fused_fused_add_rms_norm_0` | 32 | 7.050 | 0.056408 |
| 14 | `triton_poi_fused_3` | 28 | 7.821 | 0.054735 |
| 15 | `triton_red_fused__to_copy_embedding_rms_norm_0` | 4 | 10.779 | 0.010779 |
| 16 | `triton_poi_fused_1` | 4 | 7.735 | 0.007735 |

## Interpretation limits

- Node intervals include inserted timing nodes/dependencies; they are not native kernel latency.
- Full minus boundary and ratios are paired fixed-order instrumentation diagnostics, not unbiased overhead.
- Boundary also contains timing events; there is no uninstrumented graph GPU total in these JSON files.
- Interval sums and event-window unions must not be reported as native GPU wall time.
- DAG comparability describes allowed concurrency; event-window overlap does not prove kernel concurrency.
- Selected diagnostic steps repeat computation and synchronize; throughput is not a score.
- Scheduler metadata is factual. Cached/new membership never classifies decode/prefill.
- Sample GPU offsets are graph-relative; scheduler monotonic_ns is not an absolute GPU timeline anchor.
- Function handles are process-local and only matched within their own worker window.
- Call estimates count graph nodes times scheduled replay counts, excluding extra diagnostic replays.
- Rankings are per graph, not workload-wide hotspots; sampled shapes/steps exclude eager work and unsampled graphs.
