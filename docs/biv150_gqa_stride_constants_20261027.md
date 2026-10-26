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

## Final device result: close without merging

Frozen e4553be completes both cases. All outputs are exactly equal to baseline
for uniform/random/ragged patterns, changed-input poisoned Graph replay and
nondefault stream, and independent FP64 checks pass. Actual compilation includes
all seven intended stride constants; LLVM i64 multiply instructions fall from29
to0 in both cases. This verifies the mechanism, not a throughput improvement.

Warmed producer+reducer medians are0.99484365x4K/B64 and0.99543348x16K/B32.
All three alternating pairs per case are below1. Loaded resources remain93
registers,0 spills,8192B shared,one warp/one stage. No net gain. No model105 or
long fixed-service qualification was run and production remains unchanged.
Close this specific specialization without declaring all address/data-layout
work exhausted. Baseline remains the actual model641478/e29a0c kernels.

All28 originals are in docs/evidence/gqa-stride-constants-20261027/
gqa-stride-constants-final-evidence-20261027.tar.gz, SHA256
8a5ac39857ac7df2529ebbba06263ca361d004eb2408b85157885a4a99be8a03.
Manifest hashes were replayed locally. PID601555/start340300039 is terminalZ;
GPUidle after completion. No service installation or merge.
