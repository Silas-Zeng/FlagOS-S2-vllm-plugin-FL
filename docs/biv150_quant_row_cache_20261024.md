# Cached activation row quantization probe

Base: team main1ad9a708 (merged validated gate PR24). No model hook change and
no service start. Existing native operations retain the former code path by
default; only new probe entry points select a candidate. Separate private SO.

The current row quantization loads each BF16 activation twice: once for amax,
again after scale reduction. The probe retains these values in registers and
compares64/128/256-thread row blocks. The64-thread case uses one actual64-lane
warp and needs no cross-warp shared max/barrier. The128/256 cases keep proper
cross-warp barriers. Max, division, half-away rounding, clamp and INT8 result
remain the same. The full gate GEMM and BF16-rounded SwiGLU epilogue are unchanged.
This may increase registers or lose parallelism; no gain is assumed.

The public strict probe accepts K2048/N12288 and M>=17 with compressed markers.
Small-M residual is unchanged.36 full-gate cases cover M17/32/40/64/256/2048,
contiguous/2080-pitch activations, and the three block widths. Each checks raw
quantization+scale bit equality, independent CPU quantization, gate bit equality
at input scales0/1e-9/1/100, changed-input Graph,3 alternating warmed full-gate
timing rounds, and isolated quantization timing. Existing9 gate checks run first.
Raw quantization timings include its public entry allocation/contiguous copy,
while Graph replays record the kernels. Full gate timing decides usefulness.

Private build/check/probe runs only after verifying clean source and idle GPU/
port, with no package installation. No candidate is installed in a real model;
full accuracy, service throughput/TTFT, memory, and a chosen source revision
would require separate acceptance. Local validation: Ruff, syntax, diff checks;
C++/CUDA compilation and GPU results pending.


## Completed standalone probe

Runtime66705f85636bfb1f06bd1da5063c32693e8bb3c9: private build and9 unchanged
gate route checks passed; all36 new numeric/Graph cases passed bit-exact against
the old quantizer, independent CPU full quantization reference and full gate.
Raw evidence: `evidence/quant-row-cache-20261024/result.json`.

For256 threads, complete gate median speedups for contiguous/pitched inputs:

|M|pitch2048|pitch2080|
|---|---:|---:|
|17|1.04834|1.03459|
|32|1.04768|1.03222|
|40|1.04910|1.02937|
|64|1.04260|1.02939|
|256|1.02812|1.02264|
|2048|1.01526|1.01525|

64-thread cases regress for M17..64.128 threads improve these cases but less than
256.256 threads improve every tested full-gate case and are the proposed bounded
service choice for compressed gate M17..2048. No activation scale, quantized
value, GEMM or epilogue differs bitwise on the tested cases. This evidence is
operator-only and does not establish any end-to-end gain. Small-M residual and
uncompressed BF16 fallback remain independent and unchanged.
