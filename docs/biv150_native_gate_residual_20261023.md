# Small-batch activation residual prototype

Base: gate-only experimental branch5b88eac (full precision failed99/105).
Latest verified team main remains3283672; this branch is not integrated.
Implementation:ed79689. The model installer still calls the old single-pass
operator: this new operator is microbenchmark-only until numerical and speed
checks pass. Do not mistake the inherited model behavior for an enabled fix.

For1<=M<=16, the existing native INT8 GEMM already pads input to32 rows.
New `gate_silu_residual` uses main quantized activations in rows0..15 and residual
quantization in rows16..31, still calling one32-row GEMM with the same INT8
weight. Main scale is max(abs(x),1e-6)/127, residual scale is main_scale/254;
with symmetric nearest rounding, residual covers half the main step. The
FP32 epilogue combines both integer accumulators before the existing BF16
linear rounding and FP32 SwiGLU. This reduces activation quantization error;
weight quantization error remains. Extra quantization/epilogue work can still
cost time; identical GEMM dimensions do not prove zero overall overhead.

The prototype has device/dtype/geometry guards for BF16 M1..16,K2048,N12288.
Untouched original operators provide the within-library baseline. No scheduling,
serve/benchmark argument, KV dtype, model graph or installed package is changed.

Prepared24 numerical cases cover M1/2/4/8/15/16 and zero/normal/4x/outlier inputs.
The reference uses independent CPU INT64 sampled-column matrix products;
full-output RMSE is compared against FP32 unquantized activations with the same
quantized weight. Three alternating GPU graph timing rounds retain all results.
Additional checks mutate graph inputs across zero/nonzero, test fullgraph/fake
Tensor support, and reject unsupported M17. CPU syntax/lint passes only: actual
extension compilation and GPU correctness/performance are not yet established.

Remote source:/workspace/experiments/minicpm-s2/code/native-gate-residual-20261023.
Queue117144 waits for historical diagnostic103728 and verifies service/GPU idle,
then builds and runs the operator probe. State: native-gate-residual-probe-state.json;
logs:native-gate-residual-{queue,build,probe}.log under the experiment root.
No whole-model pass or retained4K gain is claimed.
