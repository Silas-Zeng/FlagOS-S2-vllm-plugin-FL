# M2048 native INT8 GEMM + original BF16-boundary SiLU

Base: team main `899ce77470b0b539aebc108fd122f668c2d5894d`.
Status: local CPU-only prototype; not compiled or numerically/performance qualified.
Default OFF means no production import, registration loader, dispatcher or service
configuration change. All implementation files are under `tools/`.

The current production `csrc/iluvatar/native_w8a8.cpp:68` materializes a full
INT32 `[2048,12288]` matrix before `native_w8a8.cu:50` applies SiLU. This private
component accepts the **same prequantized AQ/WQ and row/column scales**, then writes
the caller's BF16 `[2048,6144]` output directly. It does not implement, replace or
reconfigure the quantizer. No persistent packed weight or global INT32 intermediate
is allocated. Both gate and up projections independently round to BF16 before FP32
activation, preserving `float(acc)*(row_scale*col_scale)`, `expf`, division and final
BF16 round. No fast-math flag is added.

One configuration only: CTA output 64x64, 256 threads/four 64-lane warps, K64
single-stage shared storage. Each warp owns 16x64 output. Four A16x64 panels and
four each of gate/up B64x16 panels occupy 12 KiB. Ordinary packed-word stores build
the SDK's swizzled shared images; there is no invented SME intrinsic. Two K32 I8
MMAs per K64 step use `FragmentA/B/C` and `mma_sync_tcu`, with CTA barriers before
consumption and before reuse. This first version does not overlap global loads
with MMA. Actual registers, spills, shared allocation and instruction lowering
must be read from the resulting SO before numeric execution.

The saved CoreX headers, originally `/usr/local/corex/include/IX/crt-ixsdk/`, are
bound in the compile tool: `iluvatar_mma.h` SHA256
`36a0b4f3e5d518780f4549f7a42c9938c982b493c26f0696daa22091c635e17d` and
`iluvatar_mma.hpp` SHA256
`ddb5cd4a1d53aa2cacad501a1b2d138715c045bbf837a162d6c20a144ca45094`.
The implementation's `CoordToOffset<8,row/col>` supplies the layouts; header
`load_matrix_sync_tcu` supplies the fragment read positions. The signed-I8 path
uses `__ivcorex_matrix_mad_i32x4_i8x8`; C component j maps to
`row=lane/16+4*j,col=lane%16`. CPU enumeration covers all bytes, words and outputs.
Header declarations and CPU checks alone do not prove device behavior.

PR37 (`1f06b7e`, `tools/biv150_int8_gemm_silu.py`) already tried four Triton
dual-accumulator configurations, including M2048, which was slower. This prototype
changes to an explicitly constructed native shared/MMA dataflow; it does not claim
that eliminating INT32 or changing implementation guarantees a win. Only a full
same-input, same-output-buffer component comparison can qualify it. The old
non-Graph trace's combined GEMM/epilogue cost is not a current main899 gain estimate.

CPU-only checks and default plan (stdlib, no Torch import):

```sh
python3 -B -W error tools/check_biv150_native_i8_gate2048_cpu.py
python3 -B -W error tools/compile_biv150_native_i8_gate2048.py
```

Root's later explicit compile-only invocation, on the designated SDK and clean
frozen checkout, using a new output directory outside the checkout:

```sh
python3 -B tools/compile_biv150_native_i8_gate2048.py --compile-only \
  --root-owned-exclusive --out /workspace/experiments/minicpm-s2/UNUSED_GATE2048_COMPILE_DIR
```

The tool records compiler/header/source/SO identities, standalone IR/assembly and
**actual final SO** IR/SASS, but calls no operator and requires CUDA to remain
uninitialized. Its compile success grants no numeric or timing qualification.
After separate qualification, an explicitly loaded library exposes only
`torch.ops.cjj_native_i8_gate2048_probe.gate_out(aq,wq,row_scales,col_scales,out)`.
It rejects other shapes/dtypes/layouts/devices and any input/output storage alias,
and uses the caller's current CUDA stream without allocating or synchronizing.
Real weights, full-output numeric comparison, changed-input Graph and side-stream
checks remain pending; this change deliberately supplies no new service harness.
