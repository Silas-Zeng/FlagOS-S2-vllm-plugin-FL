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


## Started full model acceptance

Frozen source6762da85b0a64bdb0b60ad2b4f73335aed2ea80b passed private native
compilation, existing9 operator-route checks and all40 added row-boundary/stride
cases, including CPU quantization, bit-exact full gate and changed-input Graphs.
Evidence: `evidence/quant-cache-service-20261024/cached-dispatch-check.json`.

18:36:58 personal-instance GPU was idle with no running compute processes.
Owned queue79227/start_ticks318577084 started in its private checkout and run
folder quant-cache-service-20261024. Acceptance driver79671 started18:38:09;
API79674 and engine79831 became ready. The runner verified the installation
metadata for cached M17..2048 with256 threads before full105-item evaluation.
No new full precision/throughput result is available yet. Current main remains
1ad9a708 and no unvalidated runtime has been merged.


## Complete precision result

The frozen6762da8 runtime completed fresh105-question Math500 Level3 at
19:06:07 on the personal BI-V150:102/105 (0.9714),105 unique questions,
0 failed predictions. Full report, verdict and runtime/native-SO manifest are
archived under `evidence/quant-cache-service-20261024/accuracy/`.

Draft PR25 tracks this direction. The fixed sufficiently warmed4K/16K formal
throughput and TTFT acceptance continues on the same frozen service. No new
service throughput is credited yet. Subsequent shared commits only add docs
and evidence; running source and service arguments remain unchanged.


## Complete formal4K observation,16K still running

The same frozen6762da8 runtime completed256/256 requests,0 failed,1048576 input
and262144 output tokens in227.416809s:5763.514163 total tok/s, mean TTFT
3683.410328ms. Gain vs original4809.36 is19.8395%; this remains below the20%
threshold5771.232. The PR24 observation5750.700805 was0.2228% lower, from a
separate single round; this is not a controlled isolated causal estimate.
Warmup TPOT113.722974/101.133267/101.279981ms met the stability gate.
Raw result and warmup snapshot are in `evidence/quant-cache-service-20261024/score-4096/`.

16K warmup285.653800/279.706251ms stabilized and the fixed128-request formal
round is running. No repeat sampling is scheduled to cross a goal threshold.
