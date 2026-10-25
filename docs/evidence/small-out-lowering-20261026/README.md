# Inductor-owned output buffers for the BF16 correction

Base main: `34942f7e28da5c954ab05878dec75807da9d7a93` (refetched before publication).

Historical PR28 reproduced 5739.28 tok/s at4K (PR56), while the stable BF16 correction wraps QKV/O/down/edge-gate in a pure custom op. PR57 gate-only actually restores nine generated MM sites, but its initial mixed fixture still fails. Existing numerical records also show stock O/QKV/down out and gate-functional instabilities at some small rows. Removing protection indiscriminately is therefore not established as safe.

This alternative retains all existing small-row protection and the exact CUDA math. Optional `VLLM_FL_BIV150_SMALL_OUT_LOWERING=1` lowers only the plugin's own functional op to an Inductor ExternKernelOut choice. The compiler owns the contiguous output allocation/reuse; the callback writes the same masked BF16 kernel into that buffer for M1-8 and uses original torch.mm(out=...) above8. Eager behavior and unsupported compiler inputs keep the previous functional implementation. Default disabled, no main deployment.

The installed PyTorch2.10 source was read to verify ExternKernelChoice.bind/output_node, ExternKernelOut and fallback_handler interfaces. Canonical active state and helper source enter compile-cache identity. No SDK/global kernel mutation, no official-upstream upgrade, no request/benchmark detection, and no CLI changes.

CPU passed:40 output callback cases including M0/1/2/4/8/16/32/64/2048 and strided fallback, four symbolic-M output-layout cases, four unsupported compiler fallback cases, buffer identity, unchanged existing CUDA-math AST, default-disabled equivalence, and separate active cache identity. These mocks do not prove the real compiler ABI or GPU correctness.

Pending: actual Torch compile and generated out-buffer call verification; real-weight CPU FP64 comparison; default/side streams and changed-input poisoned CUDA Graphs; then model105, original mixed request before/after warmup, and matched sufficiently warmed4K/16K scores. No GPU job is queued while PR57 owns the personal instance. No measured gain or merge readiness is claimed.

A separately considered compiler-hint hypothesis was not accepted: fixed-snapshot rows1/2/4/8 generally use BLOCK_M=M, so the simple wrapped-row contiguity counterexample does not explain the observed failures at those shapes. This PR is a buffer/planning experiment, not a claimed fix of that unproved cause.
