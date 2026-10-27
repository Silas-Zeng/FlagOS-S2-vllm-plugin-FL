# Fixed Q32 I8-hoist override compile frontend

This is a separate opt-in frontend. The original d6 driver, descriptor allocator,
reader24c, codec, CPU hoist pass54608c3 and all original evidence remain unchanged.
It admits only the witnessed Q32 case and fixed override SHA
`ef347cd98b1603f9deec35b9ac3e0407efde022e399972b14ca5a60a4226891e`.
There is no caller-supplied IR, new case, service hook, numeric/timing runner or
queue. This preparation ran only CPU fixtures; Root owns future device work.

```bash
# CPU only: source/plan/override preflight; imports no Torch/Triton:
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_q32_hoist_compile.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_q32_hoist_frontend_cpu.py --out NEW_CPU_JSON
# Future Root serial opt-in only; not run during this preparation:
python3 tools/bench_biv150_konly_q32_hoist_compile.py --compile-only --commit EXACT_NEW_HEAD --out FRESH_OVERRIDE_OUT
```

Preflight verifies the original d6 driver/allocator file hashes, archived target
proof, current source modules and full Q32 plan. The candidate/baseline generated
source bytes match the original d6 compile exactly. Only candidate `.warmup`
receives one extra backend option: `ir_override=FRESH_OVERRIDE_OUT/fixed-…ttgir`.
Every Tensor, scalar, grid, warp/stage option and baseline call is forwarded from
the unchanged d6 code. No direct kernel-launch interface is exposed by the
candidate adapter. Unknown options, an existing override, wrong grid/scalars,
repeat warmup and non-Q32 profiles refuse.

The adapter changes only its privately imported Python driver's host loader and
export callbacks inside a restoring context. It does not patch the installed
SDK, JIT runtime or serving process. d6's clean/exact commit, unchanged production,
pre-CUDA native-PID/ixsmi/9031 checks, actual SDK/helper/JIT pin checks, real empty
Tensor descriptors and `.warmup`-only behavior remain in force. There is no extra
CUDA initialization, Tensor fill, encoder/writer, attention call or kernel launch.
An opt-in job still creates a CUDA context and allocates real memory as disclosed
by the d6 compile frontier; this is not zero device interaction.

Fresh output ownership is recognized only when d6 writes its initial compile
summary after successful exclusive directory creation. Existing or unowned
output directories cannot receive the new sidecar. The fixed override is created
exclusively and rehashed before/after candidate warmup. Host callbacks restore on
success and failure. CPU fixtures verify baseline forwarding, candidate's sole
option difference, archive metadata export, original evidence protection and
adapter restoration without importing a framework.

All original d6 actual source/TTGIR/LLVM/ASM/cubin/options/typed ABI/descriptors
are retained. The candidate requested-options record includes the actual override
path. Actual JIT source and typed ABI/constants/attrs must match the original
record; mismatch refuses after available exports have been saved. Baseline
never gets an override and remains canonical **unhoisted** BF16 source, not an
accepted service timing baseline. No performance comparison is licensed here.

The generated JIT modules live in the **new** output directory; their source
bytes equal the d6 modules. The fixed TTGIR deliberately retains **original d6
debug source locations**. `override-frontend.json` keeps both provenance identities,
the fixed hash, actual output hash and per-stage status. A new checkout is not
presented as the old compile's source root. The parent driver commit field denotes
the current checkout freeze; the sidecar separately pins the unchanged driver
source to d6 and reader source to24c.

`actual_override_compile_completed` means a warmup requesting the fixed option
returned a complete candidate archive. It does not establish that LLVM/ASM kept
the physical motion. Exact exported-TTGIR hash equality is recorded separately;
normalization or unknown lowering is retained for review. Override application,
actual I8 Q hoist, ASM positive grammar, runtime binding, precision, performance
and service remain unverified/false. Nothing waits for an invented audit positive
before collecting the first real feedback. Root must inspect the new actual
IR/ASM/synchronization/resources before any numerical or timing decision.
