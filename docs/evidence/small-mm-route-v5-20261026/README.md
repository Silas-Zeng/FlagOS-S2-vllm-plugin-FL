# PR53 v5: global MM binding is closed for measured regression

Runtime commit: `97d5d479bdf03567859085a787a408e7bd16eeea`. Main integration baseline: `34942f7e28da5c954ab05878dec75807da9d7a93`.

Both arms use this same runtime source, native on, the same pinned native library and tuning snapshot, fixed competition service/benchmark commands, three short warmups and one full workload warmup before one scored round. Only the stable-small-row binding changes: candidate `mm`, control `linear`.

| Workload | MM total tok/s | Linear total tok/s | MM change | MM TTFT ms | Linear TTFT ms |
|---|---:|---:|---:|---:|---:|
| 4K | 5253.616573 | 5626.933347 | -6.6345% | 3782.942804 | 3457.978113 |
| 16K | 4051.084796 | 4103.124533 | -1.2683% | 124190.503045 | 122497.280848 |

All 256/128 requests succeeded, zero failed, concurrency 64 and exact requested input/output token counts. Both drivers completed and cleaned up their owned services. Candidate math: **102/105 (0.9714285714)**, 105 unique predictions, 105 scored, zero failed. The control skips a repeat 105 evaluation; its precision is **not** reported as a new independently measured pass.

The v5 component records include both functional and out-buffer paths at M=1/2/4/8, default and side streams, changed-input poisoned CUDA Graph replays, CPU FP64 comparisons, and 3,072 direct operator checks. Mixed request semantic/count fixtures passed. This establishes correctness of this implementation, not a throughput gain.

**Decision:** close this global transparent MM/linear interception implementation; do not merge it. It regresses both matched workloads. The existing main BF16 correctness kernel remains on main; this closure does not discard the accepted historical GQA/native work. Historical PR28 replay is tracked separately in PR56 and receives the next exclusive GPU slot. Do not equate these scores with the historical-best pair or treat the matched linear control as a fresh formal 105 qualification.

The archive contains full scored benchmark JSON, manifests, warmups, metadata changes, mixed fixture records, candidate accuracy report and 105 prediction rows, component results, and the exact queue/state. Remote originals remain under `/workspace/experiments/minicpm-s2/small-mm-route-{mm,linear}-model-v5-20261026`.

Archive SHA256: `4bf98dd580e864cd0220c5fe60537f6d6d2624bd5e860c4c81901aec85b11ce2` (735378 bytes).
