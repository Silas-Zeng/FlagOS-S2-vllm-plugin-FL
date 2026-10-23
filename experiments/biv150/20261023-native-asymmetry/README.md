# Native W8A8 4K / 16K performance asymmetry: historical log audit

The prior candidate beb3518 scored4K5318.130579 total tokens/s and
16K3094.243113; same-session BF16 baseline9a43eb1 scored4835.561243 /
3366.817953. The 16K regression remains unresolved; attention-dominated
work alone explains dilution, not the negative overall result.

Audit reads original server logs, score JSON and manifests, without rerunning
GPU work or changing framework arguments. Raw records, logger samples and
SHA256 identities are in log-audit.json. The formal-window approximation is
result.date minus duration through result.date (completion); periodic logger
samples can straddle phase boundaries and are not per-step tracing.

- Baseline16K: KV capacity522960 tokens;66 formal-window logger samples;
  running30 in40 samples,31 in9,29 in5.
- Native16K: KV capacity545616 tokens (the earlier4K service had545712);
  71 samples; running31 in36,32 in14,30 in9,33 in2.
- Native4K:25 samples; running64 in13,63 in3,62 in6; peak whole-log
  KV utilization58.8%, while both16K logs reached100%.
- None of these three complete service logs contains a preemption/recomputation
  warning. There is no observed evidence supporting a recomputation explanation.
  Absence of logged warnings is not per-step proof of scheduler behavior.

The observed active-batch distribution changed when compressed weights freed
KV memory. A shape/path change remains a causal hypothesis, not an established
explanation of the full8% regression. Captured graph padding can also differ
from logger running counts, so one must not infer kernel width directly from
these histograms. Next diagnostic should compare actual captured shapes and
per-layer/kernel timings around the30–33-active-request regime, with all official
performance results retained. No failed native candidate is merged on this basis.

## Follow-up: preserve the fast 4K path without changing scheduler parameters

The old compressed-weight C++ path bypasses `min_rows` fallback whenever the
public BF16 weight is a zero-stride marker. Raising MIN_ROWS therefore cannot
select original BF16 weights: those weights no longer exist. The relevant C++
source is identical between beb3518 and the shared fd98718 delivery.

Prepared probe `tools/bench_biv150_weight_only_decode.py` keeps the SAME INT8
weight layout/scales as W8A8, but uses BF16 activations and exact INT8-to-BF16
weight-tile conversion for BF16 matrix multiplication; channel scales apply
in the FP32 epilogue. Gate/up additionally fuses BF16 rounding and SwiGLU.
It avoids activation quantization, minimum32-row padding and a materialized
INT32 GEMM result. This is a new W8A16 operator, not an original-BF16 fallback.
Its numerical reference is FP32 using the same quantized weights. RMSE against
original BF16 weights is reported separately; model accuracy remains unproven.

It compares W8A16 with native W8A8 and actual FlagGems BF16 at M1/8/16/32/40/64/2048
for QKV/O/gate-up. Candidate route selection must follow measured operator
shapes and layer type, remain safe under Graph replay, and retain all slow cases.
Do not choose routes from benchmark names or 4K/16K labels.

Official serve, max_num_batched_tokens, max_num_seqs, capture configuration and
benchmark concurrency remain at the baseline configuration, including implicit
defaults. Probe row counts are independent inputs, not scheduler configuration
changes. With the same weight representation, memory-capacity confounding may
be reduced but is not assumed eliminated: any service candidate must log actual
KV capacity, actual captured shapes, full precision and both official scores.

Queue93278 waits for the existing probe queue90478, which in turn waits for
full acceptance84302. Probe codecaefaf3 is shared; no W8A16 GPU result exists yet.

### Candidate selection after the probe

A second axis is layer selection: the prior configuration quantized QKV, O and
gate/up across the interior layers. QKV perturbations feed attention and stored
K/V, while gate/up has the largest selected projection and the measured fused
GEMM/activation benefit. If all-projection routing remains slow or precision is
marginal, evaluate a gate/up-first candidate while keeping QKV/O BF16. This is a
hypothesis about the benefit/precision tradeoff; it does not establish that QKV
quantization caused the old throughput regression. Native versus W8A16 routing
must still be chosen from measured shape costs. Each reduced layer selection
changes weight memory and therefore requires recording actual KV capacity again.
Do not silently count a partial-layer result as retaining the old full10.6% gain.

## Completed probes: reject the current W8A16 kernel

W8A16 completed126/126 numerical checks against independent FP32 with the same
quantized weights. Every tested configuration is slower than native W8A8.
Best per-shape speedups range about0.074–0.398x; no W8A16 serving hook is added.
No spills were observed in the best rows. The gate/up BF16 timing uses explicit
PyTorch activation operations, which can cost more than the service's compiled
single activation kernel: it is not an exact full-graph service reference.
The native W8A8 comparison and the negative conclusion remain valid.

The same-weight BF16 layout probe passed all20 cases bit exactly versus the
baseline. M32 O/gate-up improved about4.3%/6.2%, but M64 regressed about5.9%/3.9%
and prefill showed no consistent benefit. It is not a universal layout change
and storing duplicate full weights would alter memory usage. Do not install it.
Raw outcomes are retained in evidence/, including all negative configurations.

The observed native M32 QKV/O GEMMs are modestly slower than BF16, while native
gate/up remains faster. This motivates a gate-only service experiment; it does
not prove that these small differences explain the full prior16K regression.

## Prepared execution-shape diagnostics (not installed in formal acceptance)

`tools/biv150_execution_shape_trace.py` records host scheduled tokens/requests,
maximum tokens per request, actual graph mode and padded token/request counts.
It wraps the original model-runner helper and returns the exact original result;
no dispatch result, scheduler limit or GPU tensor is changed/read. It ignores
helper calls outside execute_model (dummy/profile/capture), fingerprints the
helper source, and records effective scheduler limits from the runner object.
Periodic snapshots can omit final calls after forced termination, so counts must
not be treated as complete per-step tracing without a clean final flush.

CPU self-test covers33->40 padding, ignored dummy calls, result-object identity,
original exception propagation, uninstall restoration, and simulated disk failure
without interrupting inference. The self-test's2048/256 limits are synthetic;
they do not measure the current service. No GPU/model trace has been collected.

A useful subsequent separate diagnostic compares the old BF16/native pair around
input16000 versus16384 at32 requests, with the official serve configuration kept
unchanged. At16000,32 inputs fit the recorded baseline token capacity; at16384,
32*16384=524288 exceeds baseline522960 but fits the old native capacity. This can
expose the cache boundary and actual graph padding rather than inferring them
from logger running counts. These are diagnostic workloads, not replacement
formal benchmark scores, and require isolated sources plus verified GPU idleness.
