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

The frozen aa2661f candidate completed: accuracy101/105 (0.9619),4K4881.40346 total tok/s with4008.68942ms mean TTFT,16K4031.54021 tok/s with124355.83761ms mean TTFT. All256/128 requests and expected input/output token totals passed. KV capacity is547632 tokens. Source defaults off; no throughput improvement or merge is claimed. The official accuracy run did not emit Q32_SELECTED: short accuracy queries do not cover the new >=1024-row branch. Separate GPU suffix/CPU-reference checks cover that branch, and benchmark warmup emitted Q32_SELECTED without fallback.

The same aa2661f source with the feature off scored5494.16802 tok/s,3715.30512ms TTFT on4K;16K control completed3953.73240 tok/s,127304.87881ms TTFT. Initial16K candidate/control is+1.968%, whereas4K is-11.15%; neither is a warmed acceptance claim. Both services completed and cleaned up. The initial candidate is11.15% slower on4K, but the comparison includes an identified warmup gap: during the candidate score,23 new MM compiler variants were written between18:37:36 and18:37:49, alongside an event-timing autotune warning and a zero-generation logging interval. The control score created no new MM variants. Normal decode intervals and short warmup timings are similar. This supports investigating cold tuning; it does not prove that all elapsed difference is compilation or establish a warmed gain. Raw outputs, cache timestamps and log excerpts are in service-initial/ with SHA256 inventory.

Three128-output,64-request warmups do not exercise all late shapes of the1024-output,256/128-request official workloads. The revised driver optionally runs each complete workload once unscored before one scored round, preserving both results. It also records added Triton metadata per score. File additions are a diagnostic, not proof that all tuning is absent. A read-only SQLite online-backup tool produces a shared initial tuning snapshot, copied into each arm's private FLAGGEMS_CACHE_DIR; it does not change the shared database. The fixed serving CLI, official scoring workload and original negative results remain intact. No production source changed after aa2661f.

Follow-up fullwarm-v2 is frozen at7f4a9a3 and has started the feature-off arm first, after original control cleanup. Both arms copy the same online-backup databases (manifest in service-initial/fullwarm-v2-tuning-snapshot.json). No production-source difference exists between aa2661f and7f4a9a3. The next arm is feature-on; both use complete-workload warmup. Formal results remain pending.
