# Residual small-M input cache component probe

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
