# Private normalized2D128 reader: compile-only component

Independent branch based on actual team mainDAC1b8f, depending on PR79/80/82/86.
The source is the existing80 `normalized2d128` reader, not a new kernel/factory.
Ten borrowed files/data are exact copies of frozen80d2ceb3b with individual
SHA/size provenance in `borrowed80.json`. Original79/80/83/86 files and
production vllm_fl/csrc are untouched. The frozen factory retains the24c source,
SDK54c495/helper59d9d3/normalizer957720/codec409ca0 checks and exact inverse.

Geometry is copied from80's Q32 numerical plan:2048 fresh queries after4096
prefix tokens, total KV6144,773 physical pages, one sequence and one segment.
Only the explicit component profile changes to BM128/Q16/K64/w8/s1, yielding
the full grid`(129,2)`. This is a forced compile profile. For single-sequence
Q2048 the actual main policy normally selects Q32, so the BM128 BF16 compile
control is not an actual-parent performance baseline. No timing/throughput or
dispatch comparison is made. Numerical short-query/tail validation will follow
actual artifacts later, not inferred from this geometry.

The old80 factory/reference/allocate/kwargs/descriptor helpers are reused
unchanged. Its old CLI/plan supported-profile list remains unchanged. The new
adapter supplies this already-existing profile/geometry, verifies output does
not alias either control cache, and does not call the old main56-based compiler
controller. Source candidate SHA is
`205b6a902b906b6a80af8db6246fc9a3fea5e73f6125a16dc809042dc1527740`,
identical to80's independently replayed source proof for this profile.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_2d128_compile_cpu.py --out NEW_CPU_JSON
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/bench_biv150_konly_2d128_compile.py --out NEW_CPU_DIRECTORY
```

Both default paths use only stdlib/source syntax/metadata models. No Torch,
Triton, JIT, CUDA, RNG, copy/fill, kernel or service is executed. Fake empty
allocation descriptors account for42139668 bytes without allocating that data.
Fourteen refusals cover descriptor units/offsets/extents/aliases, forbidden
launch subscript/direct call, missing or wrong typed K/scale/V schema. These
synthetic ABI checks are not actual compiled pointer proof.

Python3.12.15/3.14 checker outputs are byte-identical, SHA
`465f9cce7dc00e30f2fdda65bee5b1efb001460ddc8fdceda6d60754f06025dc`.
The default entry plan SHA is
`1cf04f04a6555f2d1332b04383f24bf5dafddc93dda4b54abaac0440b463025c`.
The files are retained under `docs/evidence/konly-prefill-2d128-20261027`.

After root's105 finishes and native GPU/9031 are idle, root may explicitly run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/bench_biv150_konly_2d128_compile.py --compile-only \
  --commit FROZEN_HEAD --out FRESH_COMPILE_DIRECTORY
```

The entry checks clean exact source/productionDAC, actual installed and loaded
SDK/helper SHA, actual JIT SHA, corex71/warp64 and idle native query/closed9031.
It uses an isolated compiler cache, optional module version attributes rather
than mandatory Triton distmetadata, and no unrelated native W8 library.
Only `empty/view/unbind/as_strided` describe uninitialized buffers. Both BF16
control and K8 candidate use the same fixed BM128/w8 geometry. A `WarmupOnly`
boundary exposes warmup and rejects launch subscript/direct call. There are two
planned JIT warmups and zero reader/encoder/reducer launches; no data values are
initialized or read for numerical results.

Actual emitted source/TTIR/TTGIR/LLVM/ASM/cubin must all be nonempty and are saved
with hashes, full typed signature/constants/attributes/kwargs, actual descriptors
and metadata. Candidate K is required`*i8`, private scale`*fp32`, V`*bf16`;
baseline K/V remain`*bf16`. Actual w8/s1/warp64 options must match. Compilation
failure/unknown ABI preserves the original artifacts/errors and exits nonzero.
Prelaunch register/spill/shared observations are preserved but never treated as
loaded resource qualification. Even populated counts are prelaunch-only.

The factory source quantizes Q before its KV loop. **Actual BM128 operand reuse
is unknown/not audited.** Q32's accepted actual hoist/SSA/LLVM proof is neither
copied nor applied; no IR postprocessing occurs. `compile_passed` means only
source/ABI/options/artifact compilation, not a mechanism, resource, numerical,
Graph/side, lifecycle, all-reader or model gate. Data are uninitialized and no
kernel runs, so accuracy, performance, precision_gate and service readiness stay
false. No SSH/GPU work, service install or new queue was performed by this CPU
preparation; root alone decides the later compile job.

Actual mainDAC was API-confirmed. The initial fetch command did not execute
because automatic approval review timed out; the single allowed retry then
completed a normal fetch successfully. No unsafe action was inferred from the
review timeout, and no official upstream branch was merged.
