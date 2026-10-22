# BI-V150 operator experiments, 2026-10-23

Integration baseline: team `main` commit `9a43eb1e883a1ba986a4ae8052f1d00d8fd7aa5f`.
This experiment does not merge a newer official framework or modify scheduling.
Device: one BI-V150 32 GiB; COREX 4.5.0, PyTorch 2.10.0+corex.4.5.0.

## Address arithmetic and segment reduction: rejected

`tools/bench_biv150_fastpaths.py` compares the shipping GQA kernel with constant
strides, 32-bit indices and smaller reduction launches. The same seeded tensors
and shuffled physical pages are used in every arm. A warmed CUDA graph is replayed
8 times per sample, with 15 samples reported as a median. Mixed sequence-length
tests cover zero-length graph padding, page boundaries and partial pages; short
sequences also use an independent dense FP32 attention reference.

The attached CSV is operator latency, not end-to-end serving throughput. Address
variants are checked bit-for-bit. Warp-count variants use 0.005 absolute/relative
tolerance because their FP32 reduction tree changes; actual errors are recorded.

The best primary-shape change is below 1%, while some short shapes regress.
32-bit indices alone regress on primary shapes. None of these variants is enabled
in production: the measured result does not justify another specialized path.

Remote raw logs: `/workspace/experiments/minicpm-s2/operator-fastpaths-20261023/`.

## Follow-up attention probes

`softmax-results.json` covers unchanged-maximum branch elision, bounded delayed
renormalization with integral log2 anchors, internal FP16 dot products, and `.cg`
KV loads. All shapes include shuffled pages and mixed/empty lengths. Two long
rows per uniform case are also compared with an independent dense FP32 reference.
The first three algorithms regress, so they are not enabled. `.cg` loads on full
pages are bit-identical and improve primary shapes by approximately 1%; the tail
load remains unchanged. Its end-to-end benefit has not yet been measured.

`compiler-results.json` records register limits and SME masks. Lower limits spill
registers and regress; other choices stay within measurement noise. No compiler
option override is enabled.

## Native W8A8: opt-in experiment, model acceptance pending

The native operator uses a CUDA row quantizer, vendor integer GEMM and a CUDA
BF16 epilogue. A 2D epilogue avoids per-element integer division; the quantizer uses
64-lane warp shuffles. Small compressed-only batches are zero-padded to 32 rows.
The compressed-only path preserves the public BF16 weight shape using a scalar
broadcast marker and retains INT8 weight data instead of duplicating both copies.

`native-compressed.log` contains seeded QDQ comparisons, warmed CUDA-graph timing,
and dynamic fullgraph checks. Gate/up speeds up approximately 1.6–1.9x at small M
and 1.4x at prefill M. This is not an end-to-end throughput or accuracy result.
The runtime hook is off by default; do not ship it as accepted without complete
MATH-500 Level 3 and serving benchmark evidence.

## Full-layer quantization: rejected on accuracy

`native-all-rejection.json` records six wrong answers after 77/105 scored
Level-3 examples. Even if every remaining answer were correct, final accuracy
would be only 99/105 = 0.942857, below 0.95. The evaluation was stopped for that
mathematically conclusive reason; it is not a complete 105-example score and no
throughput result was accepted for this arm. The next arm preserves the first
and last four transformer layers in BF16 while retaining compressed-only middle
layers. It requires its own complete acceptance.

## Fused scale GEMM capability probe: rejected

The COREX Lt API exposes outer-vector scaling attributes, but no valid INT8/BF16
heuristic is supported for the tested row/column layouts and compute types.
PyTorch's scaled-MM accepts FP8 but runs much slower: 1.62 vs 0.117 ms at
M64/N12288/K2048 and 8.58 vs 1.16 ms at M2048. INT8 scaled-MM is rejected by the
PyTorch API. Transposed INT8 weight packing offers no consistent improvement.
See `scaled-gemm-probe.log`; the current native INT8 operator remains unchanged.

## Edge-layer preservation: rejected on complete accuracy

`native-edge4-accuracy.json` records the full 105-example evaluation, 105 unique
questions, no failed requests, and accuracy **0.9429**. Keeping the first/last
four layers in BF16 is insufficient. No throughput score is accepted for it.

## Native gate/up + SwiGLU fusion

The fused epilogue reads integer GEMM accumulators, reproduces BF16 gate/up
rounding, and applies the existing FP32 SiLU/multiply formula before BF16 output.
It avoids materializing the full gate/up activation and the following read pass.
`native-fused-service-ref.log` uses the `gems_silu_and_mul` dispatch API reference,
with bit-identical tested outputs and dynamic fullgraph checks. Versus that BF16
linear+API activation chain, measured speedup is approximately 2.0x for small batches
and 2.36–2.39x at prefill sizes. **Correction:** inspection of the actual baseline
Inductor artifact found a generated `triton_poi_fused_mul_silu_slice_1` using FP32
sigmoid and two multiplies, rather than this FlagGems API call. Therefore these
numbers do not measure speedup against the compiled service path. A separate
compiled-kernel reference is required, and end-to-end scores remain authoritative.
Integration stays opt-in, TP=1, matching MiniCPM geometry and the selected native
weights; other MLPs retain their original forward and down projection dispatch.
The next complete-model arm retains down projection in BF16 to reduce quantization
error and tests QKV/O/gate-up on middle layers with this fused epilogue.

That arm (`beb3518`, QKV/O/gate-up, skip first/last four layers, fused SwiGLU,
compressed-only weights, BF16 down) completed all 105 Level-3 questions with
accuracy **0.9524 (100/105)**, 105 unique questions and no failed requests.
See `native-fused-no-down-accuracy.json`. This passes 0.95 narrowly. It still
requires measured serving throughput and TTFT before any default enablement.
