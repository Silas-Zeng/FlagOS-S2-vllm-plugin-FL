# Residual RMSNorm + activation quantization fusion probe

This independent operator probe is based on6762da8, the candidate containing
team main1ad9a708 and cached quantization for compressed gate M17..2048. Its
model precision/throughput acceptance is running separately. The fusion is not
installed in any model or dispatcher. Production operators are unchanged from
the parent; the native SO adds only reference/probe entry points.

## Actual implementation boundary

Read-only copies of installed FlagGems sources match the pinned5.3.5 checkout:

* modules/normalization.py SHA256 d00f40f876532204b496634a03687eab4b0c364d775ee6375f28f47257d356f4
* fused/fused_add_rms_norm.py SHA256192cbddc2283a564a9eaf90bfae1641a8a052398c0d7dd14f83f6a67f76ec3b9

Installed llama.py confirms post_attention_layernorm feeds only the MLP at this
boundary. The actual MiniCPM config has hidden_size2048 and rms_norm_eps1e-6.
Iluvatar dispatch prefers FlagGems; the probe calls gems_rms_forward itself,
records use_c_extension, and refuses source-hash drift. Successful service
integration would still need to prove the actual selected backend and alias
scope; configuration preference alone is not actual service selection evidence.

Both inputs are BF16, but normalization uses their unrounded FP32 sum. The
residual output stores that sum as BF16. The FP32 weighted normalized value then
rounds to BF16 before amax and quantization, preserving the existing separate
operation boundary. Direct quantization of the unrounded normalized FP32 value
would change the integer input. M1..16 retains the same residual correction and
32-row GEMM; M17..2048 uses main quantization. All padding is initialized.

The Triton kernel keeps the audited RMSNorm math and4 warps, adds INT8/scale
production in registers and optionally omits the normalized BF16 write. The new
native gate_quantized_probe calls the same vendor int_mm and production fused
epilogue. A separate native reference exposes all original quantized inputs and
scales, including padding, so checks do not rely on matching a few output rows.

## Gates and timing limits

18 cases: M1/8/16/17/32/40/64/256/2048 × unit/random norm weights. Each tests
zero/tiny/normal/large/cancellation/outlier inputs, exact BF16 norm/residual,
exact INT8/scales and complete gate, CPU FP64 norm/residual reference and
changed-input CUDA Graph. Norm reference uses the established0.005/0.005
tolerance; every candidate-vs-actual-path comparison requires bit identity.
Only contiguous BF16 tensors are eligible. This does not validate arbitrary
strides, dtypes, graph compilation or a decoder-layer rewrite.

Three alternating warmed Graph rounds compare full norm+gate. Both sides copy
the same two immutable source inputs to private preallocated working buffers
before each call, preventing repeated in-place normalization from drifting the
timed data. These equal reset copies are included in timing and recorded. They
make this a controlled component comparison, not a direct service latency
estimate. Norm+quant subpath timing and compiled register/shared metadata are
also recorded. No standalone speedup is added to service throughput.

The owned queue waits for residual-cache probe84926/start_ticks318602347, then
refuses a busy GPU/port. It builds a private SO and passes the existing9 gate
checks before the new probe. Model/service parameters and current running
checkouts are untouched. GPU compilation, numeric and timing results are pending.
