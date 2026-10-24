# Cached activation quantization on current team main

Parent abf760a contains the standalone probe and team main1ad9a708 (mergedPR24,
including PR22/PR23 and the compressed residual gate). Full-gate36-case results
support256 threads: minimum speedup1.01525, maximum1.04910; all bit-exact to old
quantization/CPU references/full gate/changed-input Graphs.

The service dispatch selects this quantizer only for compressed gate geometry
K2048,N12288,M17..2048. Small-M residual, M0, larger M and original BF16-weight
fallback retain their previous behavior. Weights, activation scales, integer
GEMM, epilogue, attention, model runner and scheduling remain unchanged.
No B40 segmentation experiment is included, so the service comparison remains
against the merged main implementation.

The owned queue builds a private SO, runs the existing9-case route checker,
then40 full-gate boundary/stride cases (including M2049 unchanged fallback),
independent CPU quantization and changed-input Graphs. Only after these pass
does it run fresh105-item accuracy and fixed warmed4K/16K formal benchmarks.
The existing service runner adds strict installation checks for the cached-row
range and256-thread selection. All competition arguments and benchmark loads
remain byte-identical. No model/service improvement is claimed before results.
