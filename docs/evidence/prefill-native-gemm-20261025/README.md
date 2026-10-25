# M2048 prefill BF16 native BLAS experiment

Base: team main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`. Latest fetch was attempted before creating this branch; network timeout was followed by GitHub API verification of that exact main commit. No upstream framework upgrade is involved.

PR43's complete 4K diagnostic found 33.988 seconds in 70,484 non-Graph BF16 mm kernels. CPU enqueue correlation associates these launches with 526 model steps; 522 scheduled 2048 tokens. The earlier PR42 negative set covered M32/M64 and does not test these large-row shapes. All measurements here are components, not service results.

This experiment calls the installed Corex `ixblasGemmEx` directly using BF16 inputs/output and FP32 accumulation. It bypasses the registered ATen mm/linear dispatch; calling `at::mm` again would select FlagGems. Installed `CUDABlas.h` declares BF16 gemm but the dynamic libraries do not export that specialization. The current BLAS handle getter is exported and its disassembly proves that it sets the current Torch stream. The installed legacy GemmEx ABI takes `ixDataType`, so FP32 compute is `IX_R_32F=0`; the distinct `IXBLAS_COMPUTE_32F=68` enum must not be substituted. Pointer mode is saved/restored; math mode is unchanged.

The C++ probe is not registered into any model path. Build and benchmark tools refuse a busy GPU or a service on port9031. They use an isolated shared library, real MiniCPM weights, the current worker's FlagGems `mm.out` baseline, and an independent CPU FP64 oracle for every output element. Finite numerical failures remain visible and may receive diagnostic timings. Non-default stream and changing-input Graph tests precede three alternating timing rounds. Timings reuse a single weight; cold weight-ring and service validation remain pending.

Initial planned probe: M2048, layer20 QKV/O/down and existing BF16 edge layer0 gate/up. No layer is selected using accuracy outcomes. Expansion to layers4/20/37 and other existing BF16 edge layers follows only if this native route warrants further evaluation. Actual service integration, first 105-question accuracy, warmed fixed 4K/16K throughput/TTFT and a matched control are still required before merge. M32/M64 production routing is unchanged because this branch currently has no production hook.

Current status: source and benchmark prepared; Python syntax checked; C++ compile, GPU numerical tests and all performance measurements pending. The GPU is reserved for PR43's baseline profiling first.

## 首轮真实权重组件结果（fe3abc9，尚无服务验收）

独立out为prefill-native-gemm-v2-20261025，原v1队列因profiling导出超时而退出，没有运行GPU。v2在collector真正退出、Report saved、SQLite quick_check=ok、两窗validation通过后开始；原失败记录保持原样。编译/链接/加载通过，SO SHA4fdf4fa03d1514ab63e8cabc581149913a014b942f8ec6e70ab4db0205f0cc3b。

4个几何各3轮交替计时：QKV median1.059618x/min1.058734x，O1.243085x/min1.238557x，down1.338432x/min1.334176x，BF16 edge gate/up1.214994x/min1.213961x。仅热权重组件计时，尚无服务收益。

严格CPU FP64容差四组均未通过，原结果保留。但4几何×5种输入的20个直接比较全部numeric_differences=0、max_abs=0，候选与原FlagGems基线逐元素数值相同；两者对CPU参考的统计也完全相同。12个变化输入Graph与随后eager数值相同。timing输入只记录二者对CPU的指标一致，没有直接比较，不能额外声称逐元素一致；不称byte-bit-exact或未经审计的一ULP。

下一阶段将扩展功能式custom-op、真实权重覆盖和冷权重链，再做完整105题精度与充分预热的固定4K/16K服务验收。原始数值和计时见v1-component-result.json，保留所有full_cpu_passed=false。
