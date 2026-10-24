# Residual small-M input cache component probe

## Current: complete component pass, production-route service candidate

Frozen `2dad25e` completed **36/36 records and all 180 input patterns**, including
the previously skipped outlier. Original/new raw INT8/scales and whole gate are
bit-exact; corrected independent CPU reference at unchanged .005/.005 and
changed-input Graphs pass. Three alternating warmed complete-gate rounds for
the 12 thread256 configurations observe median speedups **1.03523–1.05299x**,
geometric mean1.04474x. The 64-thread regressions are not selected. Raw build
state, original9 checks, result and summary are in
`evidence/residual-cache-20261024/reference-fixed-full/`. This is a component
result, not new model accuracy or throughput.

The new candidate syncs merged main `6a013bd` (PR28) and explicitly stacks
PR31 `0248416` (runtime e972997). Only the existing compressed small-M auto
route now calls the validated residual implementation with256 threads. The
original residual entry point stays unchanged as a comparison. M17..2048 input
cache, BF16-weight fallbacks, all math, layouts, INT8 GEMM and SiLU are retained.
Installation metadata and acceptance explicitly require the new cached residual
range/thread count, so the service cannot silently validate the old route.

Before loading a real model, the new actual-route check covers all M1..16,
pitch2048/2080, five patterns (160), old/new raw and complete outputs, independent
CPU reference, zero padded rows and changed-input Graphs. Real BF16 fallback
and M0/17/32/40/64/2048/2049 are also checked; original9 and40 cached-dispatch
checks remain, plus PR31's28784 CPU policy tuples. Actual private compilation
and these new device checks are pending. Next full service still requires105
questions and warmed one-round fixed4K/16K, full tokens/requests and TTFT.

The candidate depends on PR31 and will be separately frozen. Its queue must
wait for PR32's diagnostic to exit, additionally require PR31 e972997's formal
state.completed, and refuse a busy GPU/port. No running frozen source is edited;
unvalidated production changes stay on this Draft PR. Historical sections below
retain earlier snapshots, superseded by this current state.

The earlier quant-row cache probe improved M17..2048 full gate by1.5–4.9% with
256 threads. That path is being evaluated separately in a real model. This
probe checks M1..16 residual activation quantization, which uses a different
kernel and has not inherited that optimization.

The new kernel retains BF16 input values in registers between amax and both
main/residual quantization passes;64/128/256 threads are compared. It keeps the
existing32-row padded GEMM,16 CTAs, quantization scales, rounding, residual math,
GEMM and epilogue. Production gate_silu_residual and gate_silu_auto continue to
select the old kernel. New entry points are standalone probes only.

The36 cases cover M1/2/4/8/15/16, input pitch2048/2080 and3 thread widths.
Every case tests zero/tiny/normal/large/outlier data, requires bit-exact raw INT8
values and scales including padded rows, full-gate bit identity, sampled CPU
INT64 dot reference and changed-input Graphs. Three alternating warmed timing
rounds measure the complete gate, plus isolated quantization timing. CPU reference
retains the previously validated tolerance; it is not a model precision score.

The probe will run only after the owned quant-cache service driver exits and
the GPU and port are free. Source and native SO are private. Local lint/syntax
checks precede sharing; native compilation, numeric and performance gates are
pending. No service gain is claimed.


## Initial result:35/36 passed, CPU reference boundary needs diagnosis

Frozen6b794ad built successfully and passed existing9 gate checks.35/36 cases
passed; only M15/pitch2080/256threads failed on the sampled CPU reference:
1/90 values, max abs0.0625 and relative0.008667 vs0.005/0.005 limits. The new vs
old raw INT8/scales and complete-gate bit identity checks precede that reference
check in the script and passed at the failing input.90 values are the15×6 CPU
sample, whereas the full gate has15×6144 values. Therefore this log does not
show a candidate-vs-old kernel regression: the unchanged original has the same
reference discrepancy on that input. Do not mark all36 numeric gates passed.

256-thread successful cases observe complete-gate gains around3–5%;64 threads
regress and will not be selected. Raw initial result/check evidence is in
`evidence/residual-cache-20261024/initial/`. No model installed or score credited.
Keep draft PR26 open for reference/rounding diagnosis, preserving the original
failed result. Do not silently relax its tolerance. Next recheck must identify
the input pattern and preserve failing inputs, compare original and candidate
against the same CPU reference explicitly, and audit activation-residual and
FP32 epilogue rounding. PR25 is now merged; sync main before further changes.


## Contracted arithmetic reference audit prepared

An offline compile of the original frozen source, using its original build
flags, emits ml_fma_f32 for value - first * main_scale and for the dequantized
main + residual sum (residual product rounded first). The old oracle instead
uses eager split GPU operations for quantization, then FP64 dequantization.
Assembly excerpts and SHA identities are in
`evidence/residual-cache-20261024/reference-code/`. This proves a reference
semantic difference, but does not yet prove which difference caused the
recorded failure.

The new diagnostic replays the original RNG sequence, including its early exit
and changed-input draws, and explicitly compares old and new against both
unchanged legacy and independent CPU FP32/FMA references. It records the failing
pattern and saves the complete small CPU fixture (input, selected INT8 weights,
scales, raw quantization and both outputs) with a hash. No tolerance is changed.
It checks CPU quantization against actual original raw quantization before using
that oracle. No performance or model acceptance is inferred from this audit.
Its private queue will wait for the owned PR28 coverage process to exit before
building or using the GPU; it refuses any other busy GPU/port.


## Reproduced reference failure; corrected probe oracle

Frozen d4df0cb completed36 records/179 patterns, with zero candidate-vs-original
raw/full-gate differences, zero independent CPU raw-quantization mismatches and
zero corrected-reference tolerance violations. Original M15/pitch2080/t256
failure reproduced on the `large` pattern: both original and candidate have the
same one legacy-reference violation (abs0.0625). The original failed-result
control flow skips its next pattern, so this audit covers179 rather than180.
Do not call that a completed180-pattern performance recheck.

The saved fixture SHA is
3c4a41ba509736eb864d6a016593a1240d2f817843217b4844baa93bb15104ed.
Its portable bytes are archived as `reference-audit/fixture-base64.json`:
base64-decode data, verify SHA, then torch.load(..., map_location='cpu',
weights_only=True). Raw audit and original9 checks are adjacent. The legacy
oracle performs its quantization on GPU, despite its old CPU-reference name;
CPU-only emulation of main_scale = peak * FP32(1/127) reproduces the saved legacy
output exactly, with2 main and5 residual INT8 differences from the original.
Direct CPU division matches original raw quantization. This strongly identifies
the legacy GPU scalar-division strategy as the triggering reference discrepancy;
direct GPU legacy intermediate capture has not yet been retained. FMA differences
also exist in source/assembly, but are not by themselves the cause on this fixture.

This probe now explicitly uses the independently validated CPU FP32/FMA oracle
and compares both old and new against it at unchanged0.005/0.005. The old legacy
helper and historical failure evidence stay unchanged. Next run must complete
all36×5 patterns and changed-input Graph/performance checks, including the one
skipped outlier case. Production dispatch remains unchanged until that recheck
and then actual model acceptance pass. No model result is inferred from179.
