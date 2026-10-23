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

Decode diagnostic capacity correction: the initial graph-NONE service still
reported 522960 KV tokens. The 16K/32-request diagnostic was stopped by the
530432-token guard before profiling and requests, with owned service cleanup.
The revised diagnostic uses 15360 input tokens, 512 output tokens, 32 direct
parallel requests and requires 507904 KV slots. This fits the entire intended
workload and avoids preemption in the diagnostic; it is NOT the official 16K
benchmark and does not count toward acceptance. Profiles are delayed to step300,
8 steps recorded, actual sequence shapes must still be verified from the trace.
Output root: /workspace/experiments/minicpm-s2/decode-visible-15k-20261023.

Materialized-score algorithm probe: BF16 Q/K/V, vendor batched GEMM producing
FP32 scores (verified supported by this COREX torch build), custom causal
FP32 softmax to BF16 P, then vendor PV GEMM. It packs shuffled paged caches
and query head groups, materializes score chunks of64/128/256/512 query tokens,
and includes all packing/copy costs in the event timing. It is standalone and
not registered into serving; only a single-sequence prototype is implemented.
At16K fastest12.943ms versus pinned kernel10.705ms (0.827x), all compared at
qlen2048. Some8K comparisons fail .005 tolerance (max observed .0087891),
so these initial comparisons required an independent audit before attributing
the discrepancy. The short-case dense reference was not reached in this initial
sweep because baseline comparison failed first. No tolerance was weakened.
Subsequent component checks showed that the materialized short-case path agrees
with dense FP32 while the pinned prefill denominator has extra initial mass.
The algorithm still loses on16K performance and is not adopted. No model score
is inferred; initial mismatches alone are not proof the new algorithm is wrong.

Normalization investigation: private instrumentation reproduces original output
exactly, matches reference masks/scores/maxima, but finds denominator excess
up to1.0000114 in q16/KV128. Starting cumulative mass at0 reduces denominator
error to1.14e-5. Eight dense FP32 reference checks pass for a normalized in-memory
clone. The standalone first softmax update behaves normally; exact compiler
cause remains unresolved. The production candidate and registered-stack checks
are on codex/biv150-prefill-normalization-20261023, not installed by this
experimental branch. Initial instrumentation with extra kernel parameters
failed compiler option conversion; the private diagnostic reuses unused
pointers with the associated features disabled and succeeds.


## Prefix mask experiment (27 records)

Conditional skip and split-loop variants skip causal masking only for complete
KV tiles strictly before every valid query. Of27 records,24 passed sampled
independent FP32 windows; the normalized reference and both candidate variants
all failed the same single BF16 element at q2048/kv2048 (absolute difference
0.015625, relative about0.00775, against0.005 tolerances). This is not silently
counted as a pass or attributed only to the candidates. Long conditional-mask
variants showed roughly1.7–2% operator improvements; small shapes regressed.
No serving candidate was installed. Full raw errors and timings are retained
in prefix-mask-raw.json.

Query-head grouping completed36 records.32 pass independent FP32 windows; all
four normalized/candidate variants fail the same single BF16 element for
q2048/kv2048 at the existing strict tolerance. Long grouped candidates run
about0.33–0.50x as fast as baseline; they are rejected. Raw headgroups-raw.json
retains both performance and tolerance failures. No service hook was added.
