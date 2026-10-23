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
