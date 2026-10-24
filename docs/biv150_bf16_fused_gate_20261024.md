# BF16 paired-column fusion: corrected reference, negative performance

Personal BI-V150, 2026-10-24. This probe is not installed in the model.
Source a3a2a32 preserves the old paired-column kernel and changes only the
independent reference: BF16 is copied to CPU before conversion to FP64.

A smoke check of GPU BF16 [0, 1, -2, 0.125] returned four zeros through
`x.double().cpu()`, but correct values through `x.cpu().double()`. Therefore the
previous 517510c reference was invalid. Its failed checks do not establish
incorrect candidate computation. The raw old results are retained and labeled.

With the corrected reference and unchanged tolerances, 53/56 configurations pass.
All passing configurations are slower than actual FlagGems mm plus one fused
activation kernel. Best median baseline/candidate speed ratios by M:

| M | Ratio |
| --- | ---: |
| 1 | 0.7822 |
| 8 | 0.8266 |
| 16 | 0.9034 |
| 32 | 0.7313 |
| 40 | 0.5850 |
| 64 | 0.7536 |
| 2048 | 0.3848 |

Three split-K configurations still fail the original numerical tolerance.
The probe includes BF16 projection rounding, sampled independent CPU FP64
references, changed-input graph replay for the selected M8 configuration, and
three alternating timing rounds. It uses original weights and pitched inputs.
These results reject this implementation, not all BF16 GEMM/activation fusion.
A future attempt must change load/layout or compute structure rather than repeat
the same eight parameter configurations. No model-level accuracy or throughput
claim is made for this probe.

Raw evidence: [corrected reference](evidence/biv150-bf16-fused-gate-20261024/corrected-reference-probe.json),
[invalid old reference](evidence/biv150-bf16-fused-gate-20261024/invalid-gpu-fp64-reference-probe.json).
Remote result: /workspace/experiments/minicpm-s2/bf16-fused-gate-cpu-reference-20261024/probe.json.
