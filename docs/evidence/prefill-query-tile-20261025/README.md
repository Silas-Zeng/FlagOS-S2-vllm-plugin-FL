# Selective wider prefill Q tile — Draft PR46

Team base main0915f01. Current service candidate is frozen at aa2661f. No official upstream upgrade; the operator computation and compiler lifetime transform are developed here. No vendor implementation switch.

The existing service splits natural single-token decode prefixes in TritonAttentionImpl.forward. The new opt-in policy therefore applies only to the remaining single-sequence prefill with at least1024 query rows, using BM256/Q32 instead of BM128/Q16. Decode, small queries and unsplit multi-sequence inputs retain the existing route. The candidate has its own JIT cache and must prove Q loading outside the K loop in both TTGIR and LLVM; an unknown cold transformation falls back before launch.

## Results

| Configuration | 4K component | 16K component | Decision |
|---|---:|---:|---|
| BM64/Q8,w8 with proved hoist |0.52260x|0.50875x|Negative; registers114/shared32768 unchanged |
| BM128/Q16,w4 with proved hoist |0.90757x|0.94694x|Negative |
| BM256/Q32,w8,2048 query rows |1.04193x|1.10199x|Proceed to actual suffix |
| BM256 on unsplit mixed batch |0.75369x|0.86884x|Outside candidate scope |
| BM256 on actual1985/2017-row suffix |1.03277x|1.09161x|Proceed to service |

All ratios are median baseline/candidate durations from three alternating rounds. Same packed, shuffled paged KV, same input/output addresses; uniform-Q marker cases compare all rows with CPU FP64, random Q/K/V compares all outputs with the accepted baseline plus selected CPU FP64 rows. Both warmed compiled kernels are captured directly, retaining the verified hoist; changed-input poisoned replay matches eager. These are component results, not model throughput.

The actual serving-policy GPU probe at aa2661f passes both suffixes and exact fallback for small/multi-sequence queries. Input bytes remain unchanged. CPU checks cover selection/grid, unsupported scopes, capture fallback, rejection of unproved cold variants and propagation of GPU launch errors without retry. New feature/source identity enters compile-cache semantics.

## Formal service validation

Candidate aa2661f is running full105-question accuracy followed by three fixed warmups and one scored round for each official4K/16K workload. The next arm uses the same source with the feature off for a matched control. Fixed competition CLI and benchmark are preserved. Source defaults off. No formal precision/throughput pass or merge is claimed yet.
