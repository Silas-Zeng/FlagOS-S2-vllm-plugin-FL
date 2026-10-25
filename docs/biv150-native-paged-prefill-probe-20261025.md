# Native paged prefill capability probe

This draft adds an explicit standalone component experiment based on accepted
team main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`. It changes no serving hook,
dependency, cache allocator, official command line, or benchmark protocol.
Before committing, the branch was fast-forwarded to current team main
`0915f01d5efc2b31af65182ddb87e2d2f62bab0c`, which adds the completed profiling
tools/evidence; its production `vllm_fl` and `csrc` trees remain identical.

The completed baseline 16K trace attributes 209.497097 seconds / 44,814 calls to
`kernel_biv150_prefill_normalized`. Its accepted implementation includes zero
initial softmax mass and the Q-fragment lifetime optimization. The probe compares
against that exact installed stack and fails if Q-fragment dispatch is absent.

## Installed interface and previous coverage

Read-only inspection of the personal instance found `ixformer
0.7.0+corex.4.5.0`. In its installed
`contrib/vllm_flash_attn/flash_attn_interface.py`, `flash_attn_varlen_func` at line
69 supports GQA, bottom-right causal masks, `block_table`, and `out=`. Its paged
branch at line 197 calls `ops.infer.ixinfer_flash_attn_unpad_with_block_tables`.
The corresponding C++ symbol is exported by `libixformer.so`; that library
references `cuinferFmhaFwdFuncV3`, declared in `/usr/local/corex/include/ixinfer.h`
at line 4293. These are ABI/source observations, not runtime support claims.

The native reference indexes cache as `[blocks, heads, page, dim]`. Actual serving
uses packed `[blocks, 2, 16, 2, 128]` storage. K/V NHD views have strides
`[8192, 256, 128, 1]`; their HND transpose views have strides
`[8192, 128, 256, 1]`, and are not contiguous. This experiment asks whether that
exact view is accepted. It never silently gathers or changes the allocator.

The earlier throughput repository contains `bench_ixformer_prefill_upper_bound.py`,
`bench_ixformer_gather_prefill.py`, and `test_ixformer_prefill_semantics.py`. A
search of the local archives and personal-instance analysis/probes/sweeps/baseline
files found their sources, but no reproducible completed ixformer timing or
correctness outputs. The old dense probe also generated independent dense and
paged K/V, so its comparison could not establish numerical parity. The recorded
materialized-attention experiments use explicit BMM and FP32 score/softmax
buffers; they are a different implementation, not native FlashAttention results.

## Probe and numerical gates

Run after the GPU owner has scheduled an idle window, with the worktree on
`PYTHONPATH`, the pinned SDK Python environment, and an isolated Triton cache:

```sh
python tools/bench_biv150_native_paged_prefill.py --out /path/to/new/result.json
```

The default cases are Q2048/K4096, Q2048/K16384, ragged Q `[1,17,33]` with K
`[65,130,257]`, and fresh Q2048/K2048. Each uses noncontiguous shuffled physical
pages, a real-width 8192-entry block table, finite unused/tail canaries, BF16
Q16/KV2/D128, and one preallocated output shared by the two implementations.

Two input patterns are tested per case. Zero Q plus a sequence/head/channel/position
V marker permits a full-output CPU FP64 uniform-softmax reference. Changed random
Q/K/V use the same graph addresses, full-output comparison against the accepted
Triton path, and CPU FP64 attention on eight fixed query rows (all rows for the
small ragged case). `--reference-all-rows` enables the expensive full CPU check
for random inputs too. The default component tolerance is `atol=rtol=0.005`;
raw maxima, mismatch counts, and tolerance failures are retained. This is not
the model precision gate.

Every eager validation and Graph replay first poisons the output with NaN.
The native `out=` buffer identity is checked. Changed-input Graph replays must
be finite and numerically equal to native eager. The nondefault stream and all
strides are recorded; input bytes, including canaries, must remain unchanged.
Any failure retains a partial JSON report and exits nonzero, without timing a
failed candidate or falling back to an unreported implementation.

## Timing and current validation status

After the numerical gates, warm eager calls alternate order on the same
nondefault stream. CUDA-event samples include dispatch gaps and any internal
native copying. The accepted Q-fragment wrapper deliberately avoids Graph
capture, so a captured generic baseline would silently select another kernel;
the probe does not make that comparison. Native Graph replay is a separate
capability/numerical check.

The result is a hot component comparison, not end-to-end service throughput.
`service_acceptance_passed` stays false. Direct-view failure must be reviewed
before introducing a gather/prepack alternative and measuring its complete cost.

Only CPU static compilation, `--help`, `--describe`, and diff checks have been
run for this draft. No GPU capability, correctness, timing, or service result is
claimed. The GPU owner will run the probe separately.

The baseline call also preserves the service binder types: `q_descale=None`,
FP32 GPU scalar K/V scales, and integer `softcap=0`. Per-case compilation records
retain the normal hash, signature/constants/attributes, options, transformed IR
content hash, mechanism proof, and actual candidate hash. Candidate hashes alone
are not a portable identity gate: Corex includes `ir_override`'s temporary
absolute filename in its options hash.
