# Cached activation row quantization probe

Base: team main1ad9a708 (merged validated gate PR24). No model hook change and
no service start. Existing native operations retain the former code path by
default; only new probe entry points select a candidate. Separate private SO.

The current row quantization loads each BF16 activation twice: once for amax,
again after scale reduction. The probe retains these values in registers and
compares64/128/256-thread row blocks. The64-thread case uses one actual64-lane
warp and needs no cross-warp shared max/barrier. The128/256 cases keep proper
cross-warp barriers. Max, division, half-away rounding, clamp and INT8 result
remain the same. The full gate GEMM and BF16-rounded SwiGLU epilogue are unchanged.
This may increase registers or lose parallelism; no gain is assumed.

The public strict probe accepts K2048/N12288 and M>=17 with compressed markers.
Small-M residual is unchanged.36 full-gate cases cover M17/32/40/64/256/2048,
contiguous/2080-pitch activations, and the three block widths. Each checks raw
quantization+scale bit equality, independent CPU quantization, gate bit equality
at input scales0/1e-9/1/100, changed-input Graph,3 alternating warmed full-gate
timing rounds, and isolated quantization timing. Existing9 gate checks run first.
Raw quantization timings include its public entry allocation/contiguous copy,
while Graph replays record the kernels. Full gate timing decides usefulness.

Private build/check/probe runs only after verifying clean source and idle GPU/
port, with no package installation. No candidate is installed in a real model;
full accuracy, service throughput/TTFT, memory, and a chosen source revision
would require separate acceptance. Local validation: Ruff, syntax, diff checks;
C++/CUDA compilation and GPU results pending.
