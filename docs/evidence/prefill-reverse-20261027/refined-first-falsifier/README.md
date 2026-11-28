# Refined causal-skew route: completed first component screen

The measured source is `140cecc07ba199d199eeb620debe3082195ffacb`, with actual
parent `241220723a09061c182764f8f738e5ec01190367`. A fresh fetch before this CPU
archive found team main `56f1838192abc940b3fb9accef490022ac125000`. Its single-sequence
Q32, normalized and Q-fragment files are byte-identical to the measured parent;
the control remains B256/Q32/w16/K64/s1, grid 65x2, with actual Q-hoist selection.
No measured kernel/factory/bench or production file changed during this archive.

The original owner is PID622007/start341609077, observed at the same start in Z
in the downloaded manifest. This is existing completed work, not a new device run.
`precision_gate=false`, `official_score=false`, `service_installed=false`.

| Component | Query/KV rows | AB | BA | AB | Median baseline/candidate |
| --- | --- | --- | --- | --- | --- |
| chunk4k | 2048/4096 | 1.032543384256113 | 1.0345287859985568 | 1.0360690303866011 | **1.0345287859985568** |
| chunk16k | 2048/16384 | 0.9907340053569528 | 0.9925539958554824 | 0.9938249606510651 | **0.9925539958554824** |

All three speed ratios and their medians are recomputed from original milliseconds
in [summary.json](summary.json). These are component observations: approximately
3.45% positive at short context and 0.75% negative at long context. The old
unconditional long-context median was 0.9671327473679263. The refined route reduces
that loss but does **not** meet the desired zero-loss long-context condition.
Three ordered rounds are not a service throughput or statistical-stability result.
The positive short-context direction remains open.

Both timed cases report exact direct baseline/candidate identity, changed-input
graph replay, side-stream identity, unchanged inputs/output canaries and passed
CPU-double reference guards. Pattern0 checks all4,194,304 elements against the
reference tolerance; pattern1 uses16,384 reference elements. Direct comparisons
still cover4,194,304 elements for both patterns. These records have no inherited
baseline oracle exceptions. They do not establish exact agreement with real-number
arithmetic or full-model precision, and this CPU audit does not recompute tensors.

All13 fixed-pointer metadata states completed with exact recorded device route
IDs on both heads, direct/graph identity, unchanged inputs and inactive-output
canaries. The two graphs and the exact-source route probe were captured once each;
all12 transitions report changed metadata, including the final return to short
context. CPU recomputation of each decision agrees with the original records:

| State | cu_q start | Actual query | KV | Prefix | Order |
| --- | --- | --- | --- | --- | --- |
| short_prefix | 0 | 2048 | 4096 | 2048 | reverse |
| long_prefix | 0 | 2048 | 16384 | 14336 | forward |
| prefix6144 | 0 | 2048 | 8192 | 6144 | reverse |
| threshold_below | 0 | 2048 | 10239 | 8191 | reverse |
| threshold_equal | 0 | 2048 | 10240 | 8192 | reverse |
| threshold_above | 0 | 2048 | 10241 | 8193 | forward |
| prefix10240 | 0 | 2048 | 12288 | 10240 | forward |
| actual1024 | 0 | 1024 | 5120 | 4096 | reverse |
| actual1023_padded | 0 | 1023 | 5115 | 4092 | forward |
| actual1025_partial | 0 | 1025 | 5125 | 4100 | reverse |
| nonzero_start1 | 1 | 1024 | 5120 | 4096 | forward |
| nonzero_start31 | 31 | 1024 | 5120 | 4096 | forward |
| return_to_short | 0 | 2048 | 4096 | 2048 | reverse |

This is two timed cases plus13 metadata states, not the full ordinary numerical
matrix. Original `full_matrix=false` and
`refined_route_numerical_matrix_complete=false` remain unchanged. Compiler reuse
of metadata loads is unproven: `metadata_cse_device_verified=false`.

Both exact compiled kernels have actual **postlaunch** SDK counts113 registers,
0 spills and compiler metadata w16/s1/shared65536/use_sme114691/warp_size64.
The original compile snapshots have null register/spill fields; the later
Manager `loaded_resources` and final case snapshots establish the observed
counts after matching binary hashes and every saved ASM hash. The archive tool
recomputes the strict phase comparison. It also executes the unchanged Manager's
pure-text Q-hoist verifier against the saved actual TTGIR/LLVM: Q load outside
the loop,0 LLVM Q-memory reads in loops,32 total, for both binaries. Typed ABI,
compiler options and target records match. Both actual private Managers report
selected compiles and no fallback/error. No occupancy, driver local/static-shared
attributes, complete resource equivalence or scalar-load CSE is claimed.

The [archive manifest](archive-manifest.json) preserves all26 raw files/1,633,508B
plus the original [download manifest](download-manifest.json), whose SHA256 is
`563a8efb2f1e019d97f42dc7b646987a500f2cada2c0776822a35b7450c12e10`.
Every original SHA and byte count was reverified. The deterministic
[raw archive](causal-skew-140-raw.tar.gz) is200,917B, SHA256
`09e951664336231b36a354bdb891fc28558869f2a55ac827042db0c0a99d62b0`.
It contains the result, owner/log, normalized/conditional source, both Manager
override TTGIR files, route-probe source and each actual compiled kernel's
source/TTIR/TTGIR/LLVM/**ASM/cubin**. Override IR and actual compiler-output IR
are separate stages; this audit preserves both without claiming byte equality.

Reproduce the CPU checks/archive, also from an extracted copy of this archive:

```bash
python3 tools/archive_biv150_prefill_causal_skew_cpu.py \
  --snapshot /path/to/extracted-or-original-snapshot \
  --out /tmp/causal-skew-archive-recheck --self-test
```

Eight injected direct/reference/changed-input/resource/timing/route/probe/
changed-metadata faults must be rejected. This tool imports no Torch or Triton,
initializes no driver, performs no SSH and launches no device work.
