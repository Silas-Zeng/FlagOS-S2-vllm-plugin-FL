# M64 resource-led tile and complete real-weight validation

The resource probe frozen at 6e5dc2f passed all 24 configurations/cases. M64 changes from (32,64,128,4), about 0.745x the native complete gate, to (64,64,128,8), about 1.111x. Actual Corex metadata reports 64-thread warps and VGPR 61 / no spills / 16 KiB shared for the winner. These values do not establish measured occupancy or DRAM traffic.

The b05679e frozen follow-up tested this exact kernel/tile across all 34 compressed layers under worker-default FlagGems and inference mode. All 34 pass unchanged full-output native tolerances, independent CPU INT64 column checks, changed-input graph replay and three alternating complete-call timing rounds. Median speedups span 1.11218–1.11703x; the minimum of all 102 rounds is 1.11072x. Inputs remain synthetic; weight identities are recorded.

This commit integrates only exact M32 and M64 rows, retaining M32's validated tile. M33/M40/M63/M65/prefill remain original fallback. The C++/CUDA implementation and the main quantize_weight allocation order are unchanged. The dispatch check covers 32 row/weight cases plus noncontiguous/unsupported-threshold fallback and fake/fullgraph/changed-input graph for both selected shapes. Complete model accuracy and fixed-parameter 4K/16K validation are still required; no component ratio is claimed as model throughput.

The base64 gzip tar contains original JSON, logs, launch identity and compiler IR/assembly/resources; archive.json records its identity. Historical negative timings and numerical checks remain unedited.
