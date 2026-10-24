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
