# M64 resource-led tile and complete real-weight validation

The resource probe frozen at 6e5dc2f passed all 24 configurations/cases. M64 changes from (32,64,128,4), about 0.745x the native complete gate, to (64,64,128,8), about 1.111x. Actual Corex metadata reports 64-thread warps and VGPR 61 / no spills / 16 KiB shared for the winner. These values do not establish measured occupancy or DRAM traffic.

The b05679e frozen follow-up tested this exact kernel/tile across all 34 compressed layers under worker-default FlagGems and inference mode. All 34 pass unchanged full-output native tolerances, independent CPU INT64 column checks, changed-input graph replay and three alternating complete-call timing rounds. Median speedups span 1.11218–1.11703x; the minimum of all 102 rounds is 1.11072x. Inputs remain synthetic; weight identities are recorded.

This commit integrates only exact M32 and M64 rows, retaining M32's validated tile. M33/M40/M63/M65/prefill remain original fallback. The C++/CUDA implementation and the main quantize_weight allocation order are unchanged. The dispatch check covers 32 row/weight cases plus noncontiguous/unsupported-threshold fallback and fake/fullgraph/changed-input graph for both selected shapes. Complete model accuracy and fixed-parameter 4K/16K validation are still required; no component ratio is claimed as model throughput.

The base64 gzip tar contains original JSON, logs, launch identity and compiler IR/assembly/resources; archive.json records its identity. Historical negative timings and numerical checks remain unedited.

## Worker-policy preflight diagnosis

The legacy9/40 checks pass. Initial expanded worker-policy validation exposed an empty-tensor CUDA-comparator exception and then an M8/real-BF16 original-fallback discrepancy. Both initial failures are retained. Comparisons now run on CPU at unchanged tolerances. A fresh process that does not import the fused operator reproduces large native-only repeated M8/real-BF16 differences and CPU-reference failures; the issue is not established as a new-fusion regression. Other tested rows and compressed M1/8/16 remain exact in the probe.

Do not relabel this known numerical failure as passed. The expanded check separately verifies unsupported calls forward the exact TensorImpls and min_rows to the unchanged original op; selected M32/M64 numerical/graph/fullgraph checks stay strict. If an unsupported numerical comparison differs, its original repeat must also differ, otherwise the test fails; such baseline nondeterminism remains explicitly numeric_comparison_passed=false. Full official model accuracy is still mandatory. Diagnose the original M8 BF16 issue as a separate path, not by broadening M64 selection or hiding the failed original reference.
