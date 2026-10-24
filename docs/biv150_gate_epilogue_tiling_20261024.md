# Gate SiLU epilogue tiling component probe

Base is validated team main5eda0b0. Production gate dispatch, quantization,
vendor INT8 GEMM, BF16 projection rounding and FP32 SiLU formula stay unchanged.
Probe kernels vary only64/128/256 threads and1/2/4/8 outputs per thread,
retaining coalesced column accesses. Fewer thread blocks and more independent
work per thread may reduce scheduling overhead; register pressure may regress.
Both ordinary and small-M residual epilogues use their existing expressions.
No fast-math flag, approximate replacement or tolerance change is introduced.

The108 cases cover M1/8/16/17/32/40/64/256/2048. Each requires isolated native
bit identity, independently sampled CPU FP32/FMA epilogue reference, five
activation patterns with complete gate bit identity, and changed-input Graph
replays. Three alternating warmed rounds measure isolated epilogue and complete
gate separately. Save failures privately with checksums. No model score or
production path change is claimed. Only measured wins may progress to a bounded
shape policy and then full precision/4K/16K/TTFT service acceptance. Draft PR is
created before any GPU work. Queue after owned compiled-norm probe and refuse
busy GPU/port; shared Python packages and competition parameters remain untouched.


## Component result:108/108 numeric pass; no global dispatch

Frozen814c916 built and passed all108 full-gate/isolated CPU/changed-input Graph
configurations. For the major M32/64/2048 shapes, the best observations mostly
retain256threads/1item, which is the same workload as the original kernel;
small sub-percent differences there are not evidence for new math or a useful
new policy. Larger per-thread output counts often regress and are not selected.

M256/256threads/2items is a bounded candidate: complete-gate ratios in three
alternating rounds1.02019/1.01610/1.02157, median1.02019, isolated epilogue1.17469.
Raw108-case evidence is under `evidence/gate-epilogue-tiling-20261024/final/`.
This does not prove actual M256 inference coverage or a service gain; no
production dispatch changed and no full-model score is claimed. Keep draft for
actual shape coverage or a later bounded M256 integration. Do not apply2/4/8
items globally or extrapolate the isolated17.5% improvement to model throughput.
