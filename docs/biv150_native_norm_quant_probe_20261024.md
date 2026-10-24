# Native IR RMSNorm + activation quantization fusion probe

This corrects the unrun01281a4 FlagGems-reference prototype. Its waiting queue
was cancelled before GPU execution. This new branch uses a separate checkout,
frozen source and queue; current model acceptance is untouched.

## Authoritative service path

The actual engine log has custom_ops=['none'] and priorities
rms_norm=['native'], fused_add_rms_norm=['native']. Installed
CustomOp.dispatch_forward returns forward_native before the OOT branch when
disabled. Installed RMSNorm.forward_native calls native IR fused_add_rms_norm.

Pinned source SHA256 values:

* vllm.ir.ops.layernorm:d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910
* vllm.ir.op:fd0355b398c0d96235bb6aba0c9c08f73c199429bd8ec2aed227178c297c1eb2
* vllm.model_executor.custom_op:4b4ba01623005d6a93698eaec007a934dd512bf9ec7241090927de819621de2e
* vllm.model_executor.layers.layernorm:696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c

The probe uses fused_add_rms_norm.impls['native'].impl_fn directly, compiled
with Inductor/fullgraph. This is the installed reference implementation, not a
handwritten eager approximation or the OOT FlagGems path. It remains a component
reference: it does not prove identical whole-model compiler partitioning.

Native IR normalizes the unrounded FP32 residual sum, casts the unweighted
normalized value to BF16, multiplies by BF16 norm weight, then rounds to BF16
again before quantization. The first cast differs materially from FlagGems.
The candidate preserves both casts and residual BF16 writeback. It fuses this
with main quantization for M17..2048 or original residual correction for M1..16.
Vendor int_mm, padding and epilogue remain the existing ones. Model parameters,
attention, framework scheduling and service configuration are unchanged.

## Predeclared numerical and timing gates

18 cases: M1/8/16/17/32/40/64/256/2048 × unit/random norm weights; each tests
zero/tiny/normal/large/cancellation/outlier inputs. Before any GPU result, the
native variant declares atol0.005/rtol0.005 for norm and full-gate comparisons
with the compiled native reference and independent CPU FP64 norm reference.
Parallel reduction/rsqrt may differ by rounding; bit identity is recorded, not
claimed as guaranteed. Residual BF16 outputs remain bit-exact. Every INT8 value
and scale must match the original quantizer applied to the candidate's own
BF16 normalized output, including padding, bit-exactly. Debug/write-normalized
and performance/no-normalized-store variants must emit identical INT8/scales.
Changed-input Graphs check full gate and residual using those same gates.

This is a new pre-execution contract, not relaxing a failed GPU result: the old
FlagGems prototype never ran. Any actual service change still requires full105
precision, fixed warmed4K/16K request/token/TTFT gates and verified import paths.

Timing retains identical two input-reset copies on both sides, includes three
alternating warmed Graph rounds, and records norm+quant subpath cost and actual
performance-kernel IR digests/resources. No component speedup is a service
throughput score. Only contiguous BF16 geometry is covered. Failure stores IR
for diagnosis and stops the queue from claiming numeric success.

All new operators are probes; no model/dispatcher install is added. The private
queue waits for residual-cache probe84926/start_ticks318602347, then refuses
busy GPU/9031, builds a private SO and checks existing gate routes before probing.
Local lint/syntax/diff checks pass; GPU build, numerical and performance results
are pending.


## Final first-variant result: numerical gate failed

Frozen8ab178c built successfully and passed the existing9 gate checks. Only7/18
fusion cases passed the predeclared numerical/CPU/Graph gates;11 failed. All9
random-weight cases failed on normal inputs, with BF16 norm differences up to
0.03125 and roughly0.2% of elements exceeding the existing0.005/0.005 bound.
Two large ones-weight cases also failed. INT8/scale self-consistency passed up
to the recorded failure points, but that does not fix normalized-input mismatch.
Full evidence is in `evidence/native-norm-quant-20261024/final/`.

No tolerance was relaxed, no model service installed, no accuracy/throughput
credited. Close draft PR27 for this failed variant. The fusion direction is
preserved for a redesigned experiment: inspect the actual compiled native
reference and candidate IR first. Source-level BF16 casts alone do not prove
how the fused compiler materializes intermediate rounding. Current ones/random
contrast suggests that boundary is worth inspecting; it is not yet a proven
cause. Any replacement needs a new frozen variant and draft with its own gates.


## Generated-code finding after failure

Read-only collection found four existing Inductor native-norm wrapper/kernel
artifacts containing the2048-column rsqrt reduction; originals, paths and SHA256
are archived in `evidence/native-norm-quant-20261024/compiled-reference/`.
The generated triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 computes
`tmp19=tmp13*libdevice.rsqrt(...)`, then `tmp20=tmp19.to(tl.float32)` and
`tmp22=tmp20*tmp21`, finally storing to BF16 output. It does not materialize the
source-level pre-weight BF16 cast. The failed candidate explicitly did.
This supplies a concrete compiled-code mismatch consistent with random-weight
failures. It does not prove every failure's cause: large ones-weight failures
also require auditing the reduction layout and libdevice rsqrt implementation.

The next variant should derive its arithmetic and launch layout from selected
compiled kernels, record their actual selection, and add quantization without
inventing another intermediate rounding boundary. Do not loosen the failed
variant's gate or install it. Closing PR27 preserves this direction and its
7 passing cases for a redesigned draft, rather than discarding fusion wholesale.
