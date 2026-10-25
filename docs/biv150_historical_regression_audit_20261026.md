# Historical-best regression audit

The accepted PR28 runtime84c1be5fa5ae3d1f0b0485c5d430798913eb78f5 scored4K5765.810177/16K4048.135832, precision100/105 and all request/token/TTFT gates. Its merge6a013bd is an ancestor of current main34942f7. Later reference5754.210990/4049.306052 also exists; this is not a single isolated5700point. The production optimizations were not discarded.

Current PR51 package5520.225676/4103.509891 is4K-4.2593%/16K+1.3679% relative to those separate old PR28 measurements. Its stated+5.09%/+8.66% is only against a native-off safe control. It is not an improvement over the historical best. The failure to require a historical matched regression before merging PR51 is explicitly corrected in its body. PR53 remains an unqualified experiment; it does not replace the accepted reference.

Read-only evidence so far:

- Historical exact source, original SO(eacac719), full manifest/105report/scores/warmups/Inductor artifacts remain in code/gqa-wave-coverage-v3-20261024 and gqa-wave-main-cache-formal-v2-20261024/full. No old artifacts are modified.
- Historical actual native installation equals current: min_rows17, skip_edge_layers4, fused_silu, residual rows1–16, cached quantization17–2048/thread256 and gate_silu_auto. Torch2.10.0+corex4.5.0, vLLM0.24.0, FlagGems5.3.5 and EvalScope1.11.1 are recorded historically.
- Existing current global MM/linear tuning rows equal the earliest f104 private snapshot. Fullwarm2c5 adds four MM keys, changes zero existing MM keys and changes no linear keys. This is not a timestamped copy of the Oct24 database, so it does not prove that entire historical tuning state is recovered.
- Original and current SO(5cbb8853) host .text match exactly. Their embedded device ELF sections also all match, including allocated .note/.rodata/.text/.data. Remaining fatbin bytes differ outside this ELF. Whole-SO hash changes alone do not establish a native instruction regression. Replay holds one exact5cbb SO constant anyway.
- Initializing configs retain BF16, model length131072, TP1, no model/KV quantization flags, FULL_DECODE_ONLY, prefix caching/chunked prefill. Full raw logs remain authoritative; do not infer every unrecorded inherited env value from old manifest's limited env list.

## Controlled replay

First allow current PR53v5 exact queue owner419113/start329923968 and its child to complete/clean up. Do not run parallel GPU work. Replay old exact source84c with the same current private2c5 snapshot,5cbb SO and fixed CLI/benchmark helper as the adjacent current control. Use three short plus one full unscored4K warmup then one official4K-shaped score, not four scored repeats. This isolates code under current conditions; it is not a claim to reproduce the unrecorded Oct24 cache/hardware state exactly.

The new mixed fixture runs and raw outputs are retained. An explicit diagnostic-only option can retain a performance measurement after semantic failure to locate the regression, but always records acceptance_passed=false/qualifies_historical_baseline=false. Such a score cannot justify production use or erase the failure. A passed old105result does not cover this added fixture, and a failure on a different pre-fix main cannot retrospectively invalidate PR28 without replaying it.

The new tool is external to the source under test and uses one current benchmark helper. It never edits frozen runtime code, global packages or global caches; it verifies commit/clean source/library/snapshot, uses private caches, asserts fixed requests/tokens, records newly compiled metadata, and preserves the qualified harness's owned-process cleanup AST exactly. Legacy missing-small-hook handling is allowed only for a source that actually lacks that file and feature0. Partial4K diagnostics do not claim complete4K+16K qualification. Based on this replay, add a pre-PR51 source arm or a precisely isolated policy correction; do not launch unrelated GPU experiments before the baseline question is resolved.
