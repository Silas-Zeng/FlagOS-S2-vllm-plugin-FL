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


## Current matched control and further exclusions

The actual original and current full compilation_config/kernel_config strings compare exactly equal, including compile endpoint2048, FULL_DECODE_ONLY, capture sizes and Inductor combination flags. Official4K benchmark argv also compares equal after normalizing only output directory. The SO embedded device ELF's .text/.rodata/.data/.note and all its other sections compare byte-identical; allocated device code .text SHA e32191f3601644b67d5f0d7670dab0c0f9761955f0a192f5a9fac955845efc58. Neither command changes nor whole-SO hash is an evidenced explanation.

Current PR53v5 candidate completes102/105 and4K5253.616573/TTFT3782.942804ms,16K4051.084796/124190.503045ms. Its adjacent same-source native-on linear control now scores4K5626.933347/3457.978113ms; control16K remains running. Thus transparent MM binding loses about6.6%4K and is not eligible to merge. Current5626 point is still a separate run below historical5765 and is not treated as a recovered baseline.

CPU artifact inspection shows some Inductor best-config choices differ(e.g. matching pointwise cache IDs select512/4warps versus1024/8warps), while the reduction and combination artifacts require correlation. Original emitted modules contain extern_kernels.mm calls, retained linear control contains opaque small-linear calls. File counts/choice differences are clues only, not runtime frequency/time or causal proof.

The actual CPU waiter is now live PID/SID441671/start330432963, exports/queue_history_regression_v1.py; state history-regression-queue-v1-state.json. It waits only for the real PR53v5 owner419113/start329923968 to exit, then checks terminal state/port/GPU idle and exact old source/tool commits before launch. Frozen tool source5dc2b1a is separate from original source84c and source files were verified clean. It records all current private cache/library identities and never overwrites old artifacts. New INT8-PV GPU work is deferred. Team local clean main was fast-forwarded from0bf9d2e to actual shared34942f7; no running source/package was changed. Additional evidence archive6229bytes,SHA d91c1d07be5eed0515a1f4afad002c982374e9d6272b34072965d750c71def58.


## Reproducible generated-code and device audit

Shared CPU tools analyze_biv150_codegen_artifacts.py and compare_biv150_native_elf.py were actually run against the original/current libraries and all three real compiler-cache roots. Each root has21emitted Triton definitions/13unique normalized math bodies; all13match after stripping function name/decorators. Original and MM-mode each contain11generated extern MM call sites; linear-mode has11opaque-linear call sites. This is generated-code structure, not actual invocation count. It does not prove equal launch metadata, input dtype/layout, machine code or cost.

Recorded hardware metadata in all three roots matches: cc71,16SM,warp64,262144registers/SM,8192threads/SM and4096threads/block. Do not apply a64K-register NVIDIA occupancy model to these records. Some saved pointwise launch choices differ but the counts of five XBLOCK2/R0_BLOCK2048/16warp reductions match. Any performance attribution still requires actual selected launches and matched measurement.

The legacy source's package resolution was checked with the same interpreter/PYTHONPATH/cwd, without importing its package or GPU runtime; it resolves to the exact84c source and has no new small-GEMM hook file. Actual server marker/provenance still must be checked when replay starts. Additional AST/device archive2204bytes,SHA580113fa91090dfcab378ae8c220dcbc1c9cd6ed3dc9367ea408e79267e16f00. At04:04the exact owners419113/434166 remain live and the control benchmark is score-16384; historical waiter441671 remains live waiting, with no concurrent GPU test.

## 历史最好为何没有直接恢复为当前配置

用户记得的已验收成绩确实存在：PR28/runtime84c1be5，100/105，4K5765.810177、16K4048.135832。合并提交6a013bd仍是当前main34942f7的祖先，原日志、源码和成绩保留。2026-10-26相同旧源码在当前条件下重放4K5739.277280，仅低0.46%；本次重放没有重跑105或16K，不能冒充新的完整资格。

后来PR51修复小行数BF16投影的不稳定输出，合入时测得5520.225676/4103.509891。相对历史的两次独立观测，4K低4.2593%、16K高1.3679%；该PR报告的+5.09%/+8.66%是相对native-off安全对照，不能作为超过历史最佳的证据。合并前没有做足历史性能回归审查，已经明确纠正这一判断。不同时间/source/cache的分数差异不能全部归因于某一个算子。

旧源码重放时，新增并发混合请求在完成启动图捕获、完整workload预热前出现长问输出[8]+1023个0，计数请求正常。这是需要处理的真实异常，但不是比赛判定旧成绩不合法，也没有证明旧源码充分workload预热后仍错。历史100/105和正式两档成绩继续作为有效参考。

PR58保留所有tiny-row保护，并恢复编译器输出缓冲规划：新候选102/105，4K5760.266264、16K4049.275300，前后混合请求、请求/token及TTFT均通过，已接近历史速度。同源码关闭该优化的完整控制是103/105、5624.274341/4110.860080；因此候选实际成对变化为4K+2.41795%、16K-1.49810%，暂不合并，保留4K增益继续修16K。原始控制archive867035B SHA507765fd0e7995c194367826b6ee8077771b78ccaa3a7b31ce2720c276478096，已共享PR58。

PR59首轮全权重工作集组件进一步测得：32/64行MM更快，2048行provided-output linear更快（依投影1.07–1.29x）；2880数值/双流/变化输入Graph检查通过。下一步只按算子几何和行数寻找通用分流，保留既有小行保护。组件收益不能等同于服务吞吐收益，尚未选择或合入生产策略。
