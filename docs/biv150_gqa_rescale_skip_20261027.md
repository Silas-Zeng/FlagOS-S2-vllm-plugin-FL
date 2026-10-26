# Conditional online softmax correction on BI-V150

Final decision (2026-10-27): close this threshold8 PAGE16 GQA implementation
without merging. The unchanged numerical guard failed and both measured
components were slightly slower. No model105, service throughput or TTFT run
was started. This is a bounded negative result for this implementation, not a
rejection of every conditional-rescale design or a future prefill variant.

Baseline: team main2412207. Valid Graph profiles establish unchanged GQA as a
decode hotspot. PR69 page pipeline and PR70 stride folding are closed with raw
negative evidence. No repeated whole profiling.

KDA's KernelWiki led to the original conditional online rescaling derivation:
https://tridao.me/blog/2026/flash4/ . Adapt the mathematical mechanism to BI-V150,
without assuming Blackwell TMEM or NVIDIA warp32.

A uniform eight-head CTA decision retains previous normalization anchors until
any proposed maximum exceeds its anchor by8 log2 units. Thus probabilities are
<=256, and at model length131072 the FP32 denominator is bounded by33554432.
On correction, scale sum and accumulator together and update per-row anchors.
Without correction, accumulate in the old anchor units. Save those same anchors
in partial maxima: the existing final segment reduction remains conjugate.
Threshold0 is a bounded unchanged-max control. No floating bitwise equivalence
claim: FP32 fusion and BF16 probability rounding can change. Do not compose
threshold8 with a software exp implementation limited to nonpositive inputs.

The source factory changes only two full/tail recurrence blocks, with exact
source inverse. Preserve original page16 while/Q lifetime/chooser/strides,
BF16 KV allocation and reducer. First falsifier uses packed4K/B64 and16K/B32,
8192 table pitch, narrowed scratch, actual baseline641478, independent FP64
oracles, zero/tails/ragged, changed-input Graph/eager equality and side stream.
Save actual compiler artifacts/resources and three alternating warmed complete
producer+reducer pairs. The preparation above is preserved; completed GPU
evidence and its limits follow below. There is no model gain claim.

If promising, expand batch/layout/stream coverage and integrate the actual
service, then complete105 >=.95 and fixed4K/16K with adequate warmup and TTFT.
Production vllm_fl/csrc stay unchanged in this component Draft.

## Completed component evidence

Frozen tested source: `7f6bbdd9e89a32e791a82c5588e348e2941202c1`, against team
main `241220723a09061c182764f8f738e5ec01190367`. Both source and the original
`rtol=.01, atol=2e-4` all-output guard remain unchanged. No numerical or
performance code was edited to close this experiment.

| Case | Segments | Baseline median ms | Candidate median ms | Median paired speedup | Passed |
| --- | ---: | ---: | ---: | ---: | --- |
| B64 / KV4096 | 2 | 0.525586009 | 0.529435575 | 0.9928053674x | false |
| B32 / KV16384 | 4 | 1.029431224 | 1.035959125 | 0.9936323725x | false |

Each median speedup comes from three alternating warmed producer+reducer pairs;
it is component timing, not service throughput. Candidate and baseline each
load 93 registers, zero spills and 8192 shared bytes, with one warp, one stage,
`use_sme=14`, warp size64. Baseline hashes match the current actual 4K/16K GQA
kernels `641478e6...` and `e29a0c84...`. Candidate hashes are `3706dda6...` and
`5adf6b1a...`; all actual compiler source, TTIR, TTGIR, LLVM IR, ASM and cubin
files are retained.

The full-output baseline allclose check passed 0/2 steady/ragged checks at 4K
and 1/2 at 16K. The two changed-input Graph checks in each case were finite and
equal to eager output, while their additional baseline-allclose guard passed
0/2 at 4K and 1/2 at 16K. All six sampled FP64 checks per case passed the original
sample criterion, and the side-stream outputs were finite. Those sampled
checks do not override the failed full-output guard; they establish neither
whole-output error nor a model precision score. No failing-element tensor dump
was recorded, so a precise numerical root cause is not established.

The completed owner was read again during CPU archiving: PID601797,
start_ticks340415344, stateZ, wait_status0. The script returns normally after
recording failed rows; **process exit0 does not mean passed=true**. Both original
records retain `passed=false`. Final summaries explicitly retain
`precision_gate=false`, `official_score=false`, `model_accuracy_tested=false`.
No model105 was run because this implementation has no component performance
upside to trade for accuracy recovery.

## Mathematical review and limits

In exact arithmetic, retaining an old anchor preserves that row's existing
normalization units. If any of the eight heads exceeds its anchor by more than
eight log2 units, the uniform CTA correction scales both the running sum and
accumulator into the new per-row anchor units. The first page necessarily
corrects the initial negative-infinite anchor. The emitted partial maximum is
that same anchor converted to natural-log units, so the existing reducer remains
algebraically conjugate. The ideal probability bound is256 and the denominator
bound at131072 tokens is33554432; these are real-arithmetic range bounds, not
floating-point error proofs.

Rounding BF16 probabilities in different anchor units and changing FP32
expression/control structure need not reproduce the original floating result.
The source inverse proves a narrow transformation, not bitwise equivalence.
Actual conditional-branch frequency was not recorded, and unchanged resource
counts do not establish why the measured version is slower. Threshold0 and
prefill were not run. No tolerance was enlarged and no failed guard was removed.

## Archived originals and verification

Evidence folder: `docs/evidence/gqa-rescale-skip-20261027/`. The tar includes
34 regular-file members: all 24 actual compiler artifacts, generated source,
original result/proof, owner/log, and four frozen runner/helper/baseline sources.
`__pycache__` is excluded. CPU-only archiving streamed the archive over SSH;
no remote files, services, dependencies or GPU work were changed. Every original
file SHA256 was rechecked after download, and the 24 compiler files were also
checked against the hashes embedded in the original result.

- Archive `gqa-rescale-t8-v1-20261027-evidence.tar.gz` (253044 bytes):
  `18972458191dd8125ee2a3cc427703f98efeffb7cd0c61c095f914ef895a36d5`.
- Original `result.json`:
  `d22c4801340eee3ad8dbddfd2cfd8a4750f86bc90e73f19154024248248667d7`.
- Manifest:
  `2b7f94243f50c625acd58f389cdceb424ced116b0be7ab7363575160f8d81a62`.
- Generated source:
  `d354032c663c922c1b0032f84599c0ec2769b82abc63b5d3e6c8a3db97d08ab8`.

`archive-verification.json` records terminal ownership and hash checks;
`final-summary.json` records the bounded close decision without altering the
original result. Local CPU source regeneration reproduced the archived generated
source hash and exact inverse proof without importing Torch or running GPU.
