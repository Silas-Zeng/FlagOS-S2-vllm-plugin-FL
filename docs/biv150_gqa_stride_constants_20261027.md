# GQA decode signed64 constant strides

Component-only experiment on current team main2412207, independent CJJ branch.
The original GQA still accepts seven runtime int64 strides. Specialize their
actual tensor values while retaining each expression as signed64 through
explicit tl.full((),stride,tl.int64) aliases. Preserve original while loop,
BF16 Q/K/V, floating arithmetic, page16, scratch/reducer, physical allocation,
chooser and launch options. Do not import latest upstream framework.

The narrow source factory has an exact byte inverse, rejects unexpected
declarations/reassignment/alias collisions, and requires no Torch for its
source check. Save actual source constants/signature and every compiler artifact;
require expected stride values in the real candidate specialization. First
falsifier is packed4K/B64 and16K/B32,8192 table pitch/narrowed scratch, baseline
actual model hash641478, independent FP64 samples, uniform/random/ragged zero
and partial pages, changed-input poisoned Graph, exact baseline output and
nondefault-stream comparison. Three warmed alternating whole producer+reducer
pairs. No complete profiling prerequisite and no model gain claim.

Production vllm_fl/csrc remain unchanged. Only after meaningful component
potential, broaden batching/layout/stream coverage, integrate into the real
service and pass complete105 plus fixed4K/16K with adequate warmup. A passing
component is not model acceptance. No GPU result yet.
