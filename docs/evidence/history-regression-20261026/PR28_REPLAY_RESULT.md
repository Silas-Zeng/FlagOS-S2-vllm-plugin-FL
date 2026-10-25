# Historical PR28 replay: 4K 5739.28 tok/s, additional mixed fixture fails

Historical accepted runtime `84c1be5fa5ae3d1f0b0485c5d430798913eb78f5` (PR28) remains an ancestor of main. Original accepted results: **100/105**, 4K **5765.810177399745**, 16K **4048.1358315864945** total tok/s. They are retained as the historical reference.

On 2026-10-26 Shanghai, replayed that exact clean source using the current pinned native library `5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe`, private current tuning snapshot, and frozen PR56 tool `5dc2b1a95d934ae1df360db42aba779c21b4972f`. Fixed service arguments and official benchmark parameters; three short warmups, one complete 256-request 4K workload warmup, then one scored 4K round.

Result: **5739.277280046471 total tok/s**, TTFT **3697.1882001398626 ms**; 256 completed, zero failed; exactly 1048576 input and 262144 output tokens. Driver and owned service cleanup completed. This is only 0.46% below the old 4K observation: the 5700-class implementation is reproducible and must not be discarded.

This replay does not rerun 105 accuracy or 16K. The added HTTP-overlap mixed fixture, executed after startup graph capture but before workload warmup, fails the long arithmetic request: 1024 generated token IDs contain `[8]` followed by 1023 zeros, with no answer. The concurrent counting request is correct. This is an actual observed regression case; it does not prove model-step overlap, locate the failing operator, prove failure after workload warmup, or invalidate the historical 100/105 evaluation. These extra diagnostic gates are not a claim that the old official result was disqualified.

The driver explicitly records `acceptance_passed=false`, `diagnostic_scores_only=true`, `qualifies_historical_baseline=false`. These fields describe this run under the added guard, not the old contest result. Its raw throughput helps isolate cost; it is not reported as a newly qualified full baseline.

A recent same-source PR53 linear control measured 5626.933346681921 / 4103.124533375926, but its source is `97d5d479bdf03567859085a787a408e7bd16eeea`, not exact main, and it does not rerun 105 accuracy. PR51's own measured 5520.225676096365 / 4103.509891285446 uses a different historical comparison window. Do not turn these into a precise isolated cost for the BF16 fix. In particular, PR51's +5.09%/+8.66% was against native-off control, not against PR28. Its 4K result is 4.26% below historical PR28 and should have triggered a historical regression review before merge.

PR53 global MM binding is now closed with complete evidence: correct 102/105 but -6.6345% 4K and -1.2683% 16K against its matched linear control. It was not merged. PR55 integer PV remains deferred while the historical baseline is recovered.

Next: isolate the broad small-row BF16 projection wrapper and cache-semantics changes from old code, reduce the fix to projections that require it, and verify mixed behavior after workload warmup. Retain historical GQA/native optimizations; do not reset main or deploy a version with an unresolved observed fault merely on raw TPS.

Archives:

- `history-pr28-currentconditions-4k-final-20261026.tar.gz`: 54370 bytes, SHA256 `aba0257ac34d58e886e77de61c04930cb422be6c7a30ae8dc7905a1c51ffbe16`.
- `history-pr28-startup-mixed-fixture-20261026.tar.gz`: 8527 bytes, SHA256 `50c306cb2d2bfe0424d3514c6ff8e6a3b4833f15709ecfa65738afe9e219cc9c`.

Raw remote output: `/workspace/experiments/minicpm-s2/historical-pr28-currentconditions-4k-v1-20261026`. The source worktree is unchanged. Current main was rechecked as `34942f7e28da5c954ab05878dec75807da9d7a93`.
