# Conditional online softmax correction on BI-V150

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
producer+reducer pairs. GPU result pending; no model gain claim.

If promising, expand batch/layout/stream coverage and integrate the actual
service, then complete105 >=.95 and fixed4K/16K with adequate warmup and TTFT.
Production vllm_fl/csrc stay unchanged in this component Draft.
