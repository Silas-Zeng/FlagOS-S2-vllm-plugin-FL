# Cached activation quantization: full accepted service result

The row quantizer previously reloaded BF16 activations after computing amax.
For compressed MiniCPM gate K2048/N12288/M17..2048, a256-thread kernel now keeps
values in registers between these passes. Small-M residual, M0, larger M and
original BF16-weight fallback retain the previous implementation. Quantized
weights/scales, vendor int_mm, epilogue, attention, request ordering, scheduler,
model and fixed competition service/benchmark arguments are unchanged.

## Provenance and operator evidence

Frozen runtime6762da85b0a64bdb0b60ad2b4f73335aed2ea80b incorporates team
main1ad9a708 (PR22/23/24). Subsequent branch commits only archive evidence;
csrc, vllm_fl and tools remain byte-identical to the measured runtime.
No B40 segmentation experiment is included. Private native SO SHA256:
b014522cc8b05592bdd3de15c1e2d53bebfa284b7b0428c3ace14f7f6424b2de.

The standalone36 cases passed bit-exact old-quantizer/CPU INT8 and scale,
complete gate and changed-input Graph checks.256 threads improved every tested
complete-gate case, ratios1.01525–1.04910.64-thread small-M regressions were not
selected. Private build, existing9 route checks and40 additional row/stride/
CPU/changed-input Graph checks passed, including M2049 unchanged fallback.
The original component evidence is in biv150_quant_row_cache_20261024.md.

## Full model acceptance on personal BI-V150

Fresh Math500 Level3:102/105 (0.9714),105 unique questions,0 failed predictions.
After sufficient warmup, one fixed formal round per length:

|Load|Total tok/s|Vs original reference|Mean TTFT ms|Requests|
|---|---:|---:|---:|---|
|4K|5763.514163|+19.8395%|3683.410328|256/256,0 failed|
|16K|4010.102823|+19.2053%|125077.489141|128/128,0 failed|

Input/output counts are exact:4K1048576/262144 and16K2097152/131072.
Durations227.416809s and555.652585s. Working precision/request/throughput/TTFT
acceptance passes. Actual compute-order and Q-fragment serving paths were
selected; no recorded scope rejection or Q-fragment fallback. Queue completed
normally. Shutdown cleanup warnings are not evidence of an inference failure.

Warmup TPOT(ms):4K113.722974/101.133267/101.279981;
16K285.653800/279.706251. Both stabilized before the formal round.
Compared with PR24's separate single-round observations, totals are0.2228% and
0.1949% higher; these small differences are not isolated causal estimates.
The controlled component timings support the operator improvement.

Neither length reaches the overall20% goal5771.232/4036.836. This does not
prevent retaining a verified incremental operator improvement. Work continues
past20% or30%; no repeated sampling was used to cross a score threshold.

All full acceptance/report/manifest/raw-score/warmup evidence is in
`evidence/quant-cache-service-20261024/final/`. Earlier standalone and boundary
checks remain alongside it. Draft PR25 can now be marked ready and merged after
fresh main/head verification; dependent PR26/27 must then sync to integrated main.

## Reproduction

Build the SO in the desired private checkout before serving:
`MAX_JOBS=2 python tools/build_biv150_native_w8a8.py`.
Use that same source path via PYTHONPATH. The owned build/check/acceptance queue
is tools/run_biv150_gate_team_queue.py. Formal service and benchmark arguments
are preserved in the final manifest and raw score files; the research protocol
uses one formal round after the recorded sufficient warmup.
