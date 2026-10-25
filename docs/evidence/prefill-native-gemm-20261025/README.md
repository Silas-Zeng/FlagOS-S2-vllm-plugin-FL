# M2048 prefill BF16 native BLAS experiment

Base: team main `0bf9d2efeaa10144dd7f0ff4597298df029f9328`. Latest fetch was attempted before creating this branch; network timeout was followed by GitHub API verification of that exact main commit. No upstream framework upgrade is involved.

PR43's complete 4K diagnostic found 33.988 seconds in 70,484 non-Graph BF16 mm kernels. CPU enqueue correlation associates these launches with 526 model steps; 522 scheduled 2048 tokens. The earlier PR42 negative set covered M32/M64 and does not test these large-row shapes. All measurements here are components, not service results.

This experiment calls the installed Corex `ixblasGemmEx` directly using BF16 inputs/output and FP32 accumulation. It bypasses the registered ATen mm/linear dispatch; calling `at::mm` again would select FlagGems. Installed `CUDABlas.h` declares BF16 gemm but the dynamic libraries do not export that specialization. The current BLAS handle getter is exported and its disassembly proves that it sets the current Torch stream. The installed legacy GemmEx ABI takes `ixDataType`, so FP32 compute is `IX_R_32F=0`; the distinct `IXBLAS_COMPUTE_32F=68` enum must not be substituted. Pointer mode is saved/restored; math mode is unchanged.

The C++ probe is not registered into any model path. Build and benchmark tools refuse a busy GPU or a service on port9031. They use an isolated shared library, real MiniCPM weights, the current worker's FlagGems `mm.out` baseline, and an independent CPU FP64 oracle for every output element. Finite numerical failures remain visible and may receive diagnostic timings. Non-default stream and changing-input Graph tests precede three alternating timing rounds. Timings reuse a single weight; cold weight-ring and service validation remain pending.

Initial planned probe: M2048, layer20 QKV/O/down and existing BF16 edge layer0 gate/up. No layer is selected using accuracy outcomes. Expansion to layers4/20/37 and other existing BF16 edge layers follows only if this native route warrants further evaluation. Actual service integration, first 105-question accuracy, warmed fixed 4K/16K throughput/TTFT and a matched control are still required before merge. M32/M64 production routing is unchanged because this branch currently has no production hook.

Current status: source and benchmark prepared; Python syntax checked; C++ compile, GPU numerical tests and all performance measurements pending. The GPU is reserved for PR43's baseline profiling first.
