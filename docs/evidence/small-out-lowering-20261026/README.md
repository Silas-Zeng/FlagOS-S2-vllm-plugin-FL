# Inductor-owned output buffers for the BF16 correction

Base main: `34942f7e28da5c954ab05878dec75807da9d7a93` (refetched before publication).

Historical PR28 reproduced 5739.28 tok/s at4K (PR56), while the stable BF16 correction wraps QKV/O/down/edge-gate in a pure custom op. PR57 gate-only actually restores nine generated MM sites, but its initial mixed fixture still fails. Existing numerical records also show stock O/QKV/down out and gate-functional instabilities at some small rows. Removing protection indiscriminately is therefore not established as safe.

This alternative retains all existing small-row protection and the exact CUDA math. Optional `VLLM_FL_BIV150_SMALL_OUT_LOWERING=1` lowers only the plugin's own functional op to an Inductor ExternKernelOut choice. The compiler owns the contiguous output allocation/reuse; the callback writes the same masked BF16 kernel into that buffer for M1-8 and uses original torch.mm(out=...) above8. Eager behavior and unsupported compiler inputs keep the previous functional implementation. Default disabled, no main deployment.

The installed PyTorch2.10 source was read to verify ExternKernelChoice.bind/output_node, ExternKernelOut and fallback_handler interfaces. Canonical active state and helper source enter compile-cache identity. No SDK/global kernel mutation, no official-upstream upgrade, no request/benchmark detection, and no CLI changes.

CPU passed:40 output callback cases including M0/1/2/4/8/16/32/64/2048 and strided fallback, four symbolic-M output-layout cases, four unsupported compiler fallback cases, buffer identity, unchanged existing CUDA-math AST, default-disabled equivalence, and separate active cache identity. These mocks do not prove the real compiler ABI or GPU correctness.

Pending: actual Torch compile and generated out-buffer call verification; real-weight CPU FP64 comparison; default/side streams and changed-input poisoned CUDA Graphs; then model105, original mixed request before/after warmup, and matched sufficiently warmed4K/16K scores. No GPU job is queued while PR57 owns the personal instance. No measured gain or merge readiness is claimed.

A separately considered compiler-hint hypothesis was not accepted: fixed-snapshot rows1/2/4/8 generally use BLOCK_M=M, so the simple wrapped-row contiguity counterexample does not explain the observed failures at those shapes. This PR is a buffer/planning experiment, not a claimed fix of that unproved cause.

## Real compiler component passed

Frozen1d00e0d ran36 real-weight projection/row cases and576 FP64/default+side/changed-input poisoned Graph checks, all passed with no errors;8 actual generated out modules were found. This qualifies the component, not a model speedup. Archive9007bytes SHA25616188416f7f04fe93bbcb3cb58498ae460feaf504b22a14b820ff5d170036a32.

The initial CPU waiter failed before starting any child because its upstream path incorrectly said gate-only-v2 instead of v1. No GPU task ran from that queue. After confirming PR57 terminal cleanup, noGPU processes and closed9031, the component was directly launched under PID462835/start331269046 and finished successfully. Both records are preserved.

Next source adds opt-in selection recording and an owned model controller. CUDA arithmetic remains unchanged. The controller verifies actual M2 selection in all four projections and generated out calls, runs original105 and sufficiently warmed single scored4K/16K, and repeats mixed after warmup. Its finally cleanup AST matches main. Re-run the component with recording enabled before model testing. Default behavior and CLI remain unchanged. No performance gain or merge readiness is claimed.

## Actual model first semantic gate passed

Frozen61d198e: recording-enabled component repeat again passes36 cases. The real model installs the lowering, reports all4 M2 projection geometries using the compiler output, and generated model code has11 actual extern out sites (static sites, not runtime counts/times). The initial mixed long arithmetic gives85 and concurrent1..128 counting succeeds with HTTP overlap. At06:12 Shanghai, complete105 is active with43 prediction rows. This is not a full accuracy, warm-after-score semantic, or throughput qualification. Runtime stays frozen61d; this evidence-only commit does not hot-edit it.

## Actual model precision and first4K score (06:42 Shanghai)

Frozen61d198e300f0889a109d43cc9eec2e7b87455984 completed102/105 (0.9714),105 unique predictions/no failed predictions. Archive924150bytes SHA25678dfae958f9f67f0d4eb9dbdfcca8d0fd2397a0f4961c289e268280d89c16ac8 preserves formal results. Actual M2 selected all4 projection geometries;11 generated model out sites. Initial mixed fixture passes.

After3short+1full4K warmup, one official4K round:5760.266264365806 total tok/s; mean TTFT3735.5155516306695ms;256completed/0failed;1048576input/262144outputtokens. Historical PR28 scored5765.810177399745/4048.1358315864945 and100/105; its optimizations remain in main. These separate rounds do not establish a causal gain.16K, post-warm mixed checks and exact-source output-lowering-disabled control are still pending. No merge readiness claimed. partial-4k-scores.json and partial-4k-state.json are snapshots while the controller remains in throughput.

## Sequential matched control prepared

queue_biv150_small_out_control.py waits for exact live upstream queue/driver identities, requires candidate cleanup, both workload gates, formal precision and pre/post-warm mixed semantics, then dispatches the identical recorded acceptance command with only output directory and out-lowering1→0 changed. Full105, native-on, all small-row protections, snapshot, warmups, serving and benchmark arguments remain identical. A CPU command-difference contract passed. It is CPU-only until both owners exit and aborts without touching any occupied GPU/port. This does not yet constitute an executed or qualified control.

## Completed candidate (07:10 Shanghai), exact-source control running

Frozen61d completed102/105, both working request/throughput/TTFT gates, initial and post-score mixed semantics, generated out proof and owned cleanup. Scores4K5760.266264365806/TTFT3735.5155516306695ms;16K4049.275299845021/124241.82034386467ms. Exact256/128 successes, zero failures and full token counts. Both scored rounds follow3short+1complete-workload warmup;16K fullwarm4045.366566.4K remains just below the original5771.232(+20%) target; do not mark the unbounded goal complete.

Full performance/commands/warmups/semantics/controller/cleanup archive60839bytes SHA256d13a2c3662b0b8dc0d43950a5f63826b1abcc085a78878a7f0a75fc9c68ba3ef. Formal105 raw data are in the separately hashed924150byte archive above. Candidate scored4K adds4index/0MM metadata;16K metadata counts remain to be inspected. A mid16K metrics snapshot has11 cumulative preemptions, but includes previous workloads; it is not a per-score count or proof of the capacity effect.

The exact-source full105 control started only after all candidate owners exited/cleanup passed. Queue474606/start331601392, driver475424/start331734914, server475428/start331734936, Engine475576 (starttime rechecked at observation). Outputsmall-out-lowering-control-v1-20261026, currently accuracy. Default remains disabled, no deployment/merge or matched speedup is claimed.
