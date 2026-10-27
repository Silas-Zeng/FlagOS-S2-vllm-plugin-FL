# First Q32 eager numerical feedback

This private opt-in entry defaults to CPU and preserves the frozen reader/d6/bb
tools. It has no timer, service hook, queue, Graph or side-stream path. The first
scope is2048 fresh BF16 query rows,4096 preceding context tokens, hence6144 total
KV tokens, one sequence,16Q/2KV/D128/PAGE16/Q32. Formal batchTokens is unchanged.
16K, Graph and side stream are explicitly untested. This is deliberately an
eager feedback stage, not full validation.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_numeric_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_q32_numeric.py
# Root alone, after choosing its serial device slot:
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_q32_numeric.py --run-gpu --numeric-diagnostic \
  --commit EXACT_NEW_NUMERIC_COMMIT --out FRESH_OUT \
  --source-root /workspace/experiments/minicpm-s2/cjj-konly-prefill-q32-compile-v1-20261027/sources
```

The supplied source-root must be the retained original d6 source/debug root.
The corrected per-role binding uses d6 for baseline/unhoisted, and the original
bb source root for hoisted (derived from each archived TTIR). Every file's bytes
are compared to its original archive. The fixed
override is read from its original compiled path
`/workspace/experiments/minicpm-s2/cjj-konly-q32-hoist-compile-v1-20261027/fixed-q32-i8-operand-hoist.ttgir`.
It must already exist and match ef347cd; neither original path is modified or
impersonated by the new validator checkout. New output records both roots and
the current exact clean commit. The SDK54/helper59/JIT source hashes are checked.

Before CUDA the whole-job native PID query must be empty, full ixsmi idle, and
9031 closed. One disclosed tiny CUDA initialization establishes the actual
queried host PID. Every phase requires the same singleton, same container
PID/start, same root binding and fresh token. A foreign/missing PID, bad query,
stale token or active service stops; no other process is stopped. This tiny
initialization is GPU work outside numerical checking, not zero-launch compile.

Actual2048-row Tensor descriptors are allocated and all three roles warm up
before any encoder/reader launch. They must match reviewed kernel hashes, typed
ABI/options, source and TTIR/TTGIR/LLVM/cubin bytes. All warmups first export
all actual files and record, then saves the full expected record and a separate
expected/actual difference JSON. Unknown changed specialization stops with those
actual bytes retained; nothing is replaced to match the proof. Printed ASM is
also retained and compared. A bounded ASM-text difference requires exact cubin,
exact preceding IR/ABI/options, the same instruction/control sequence and each
role's strict actual audit; see the separate9f binding-refusal report. The
hoisted actual positive auditor also reruns. Handle
initialization records known positive register/warp fields and known zero spills.
No resource decline is interpreted as performance.

Both existing fixture patterns and seed20261025 are used. The ae00 codec is
vendored from the pinned source archive into the fresh private output. Its full
16KiB/page encoded allocation, code, scale, unchanged BF16 V and untouched gap
bytes are compared to an independent CPU original-BF16 bit oracle. Source inputs
must survive encoder and consumers unchanged. Each separate output has prefix
and suffix canaries and NaN-poisoned live rows. All output bytes must match
between hoisted and unhoisted I8, including signed zero. Q32 has no exposed
scratch/max/sum ABI; this entry claims output equality only.

Selected rows0/1/15/16/31/32/1024/2047 have independent original-BF16 closed FP64
attention and quantized INT64-QK/FP64-scale/global-softmax references. The latter
does not model BF16 P rounding or online softmax rounding; it is not an exact
instruction-level oracle. The original `.005 + .005*abs(reference)` pointwise
gate and `max(.005,1.2*baseline_relative_RMSE)` original-BF16 relative gate remain.
Baseline pointwise failure, nonfinite values, codec, buffer, input, actual-IR or
bitwise failures are never bypassed. `--numeric-diagnostic` permits recording a
finite candidate consumer precision failure; its true/false gate is retained.
There is no timing under either mode.

Each pattern saves original CPU BF16 inputs/metadata, independently computed
encoded bytes/codes/scales, actual encoded bytes, full three outputs and both
selected-row references as a losslessly gzip-compressed PT artifact with SHA.
No input, output or reference is truncated to fit a collector limit. These are future Root-run
artifacts; CPU preparation does not produce or load device tensors. The BF16
control is canonical unhoisted SDK code, not accepted service main. All service,
performance, full-validation and competition precision gates remain false.
