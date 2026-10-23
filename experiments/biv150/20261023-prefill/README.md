# Paired prefill diagnosis, 2026-10-23

Team main verified through GitHub API: 9a43eb1e883a1ba986a4ae8052f1d00d8fd7aa5f.
Independent branch: codex/20261023-prefill-diagnostics. No production operator
changes, scheduler changes, dependency installs or official-upstream merges.
Owner instance only; services ran serially and cleaned their own process groups.

Profiler-only service options: torch profiler, first 8 iterations, record shapes,
FULL_DECODE_ONLY. Warmup: 8 requests of 4096/128; diagnostic: 16 requests of
16384/1. These are NOT official benchmark scores. Trace files remain at
/workspace/experiments/minicpm-s2/prefill-diagnostics-20261023/{baseline,native}/trace.
Native source is the exact previous beb3518 configuration, not the failed
final-default variant. Baseline uses the unchanged 9a43eb1 production sources.

The traces contain 336 attention launches in both arms and 1344 BF16 GEMM
launches in the baseline. Baseline cumulative GPU kernel durations: 2975.75 ms,
including attention 1932.44 ms (64.94%) and GEMM 898.98 ms (30.21%). Native:
2718.06 ms total, attention 1932.75 ms. Native reduces this sampled kernel sum
by 8.66%; this is not wall-clock speedup or full-model attribution. It does not
explain the earlier full 16K regression by a slower sampled prefill stage.
Decode/request progression and kernel internals still require investigation.

The launch-only sweep uses the pinned actual unified-attention source, SHA256
54c495dcea404582a1526a96ae7e472965cbd03c8fa0931c71ef72223bf04e1f.
It clones only the Python launch wrapper into an independent namespace and
reuses the original JIT kernel. Installed vLLM files are not edited. Q length
2048; KV lengths 2048/4096/8192/16384; shuffled physical pages, causal BF16
attention with 16/2 heads and head size128. 13 launch configurations per length
(52 total) all passed baseline numerical comparison at atol/rtol .005.
Timing uses warmup and five CUDA-graph event measurements, three calls each.

No alternative beats the existing BM128/TILE64/8-warps/1-stage configuration
consistently. The 2-stage equivalent is bit-exact but 1–2% slower. Wider TILE128
changes rounding (max observed difference .00024414) and is slower; BM256 is
also slower. No candidate is installed or merged and no model accuracy score
is inferred from these operator checks. Short/edge/mixed-shape independent
reference checks remain necessary for any future actual candidate.

Liveness/cache/address probe: 5 variants × 4 lengths (20 checks), all bit-exact.
Original kernel: 102 registers, zero spills, 65536 bytes shared. Delayed V:
100 registers, zero spills, 49152 bytes shared, but approximately 2% slower at
16K. `.cg` and 32-bit physical-page index variants are within timing noise.
No variant is promoted. Generated modules copy only the hash-verified pinned
source into this experiment's directory; installed source files remain untouched.

Occupancy-combination probe: 19 variants × 4 lengths (76 checks), all within
.005 tolerance. BI-V150 reports 16 multiprocessors, 128 KiB shared memory and
262144 registers per multiprocessor, warp size64. Late-V + TILE32 reduces shared
to40960 bytes and registers to91, crossing the shared-memory limit from two
to three resident blocks (before other resource constraints), yet is about
6% slower at16K. All other tested combinations are slower too. These results
do not support shared-memory occupancy alone as the cause of the prefill cost.
No variant is promoted and no full benchmark or model accuracy is inferred.

Matrix-lowering probe: use_sme=1, one/two pipeline stages and late V (16
checks across four lengths). No stable benefit; SME output is bit-exact and
resource counts match the default one-stage lowering. Evidence is matrix-results.json.

Profiler scope clarification: vLLM bench sends an initial test request. With
first-eight-step prefill capture, the recorded 336 attention launches are the
first single 16K prompt's 8×2048-token prefill chunks, not concurrent full-load
prefill attribution. A forthcoming decode-only diagnostic uses direct parallel
HTTP requests without an initial test prompt and graph mode NONE, solely to
make individual kernels visible. Its results must never be reported as official
FULL_DECODE_ONLY benchmark throughput. Actual captured batch shapes must be checked.
