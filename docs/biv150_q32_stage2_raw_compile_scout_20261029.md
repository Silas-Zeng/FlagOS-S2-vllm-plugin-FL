# Private raw Q32/stage2 compile scout

Current baseline after a normal merge: main `899ce77470b0b539aebc108fd122f668c2d5894d`. This experiment preserves
the complete existing Q32 GPU source, codec/math, 16KiB page layout and descriptor
fixture. It fixes N1, total Q2048, BM256/BQ32/K64, grid `(65,2)` and 16 warps; the
single compiler-option change is `num_stages=2`. Production files and PR123's frozen
worktree are unchanged. There is no production hook; default invocation is CPU-only.

This is deliberately **raw stage2**, not a hoisted-stage2 implementation. The installed
compiler runs its pipeline in the TTGIR stage and then applies `ir_override`; supplying
the existing stage1 override would replace the pipeline result. This frontend therefore
does not construct or pass any override, rejects an enabled external override manager,
and requires the actual compiled metadata to explicitly contain `ir_override=None`.

## Reuse and exact boundary

The source factory reads the frozen PR116 driver and ABI mapper at immutable commit
`844a69f10539c551500e29e2304061aef7d8671b`, checking both full-file SHA-256 values.
That dependency object must explicitly accompany a remote source bundle; it is not
assumed to exist and there is no fetch fallback. The driver is adapted by 24 exact-count,
fully reversible edits. Its allocation fixture, identity/ownership/idle checks, artifact
preservation, handle loading and resource checks remain; the second warmup and all hoist
proposal/override/native-proof work are removed. Only one geometry and one warmup remain.

The existing snapshot helper is loaded as a private copy with exactly one metadata
recorder scope edit `(16,1,64)` to `(16,2,64)`. Its production source stays unchanged;
only `kernel_snapshot`, static source-pin validation and artifact preservation are used.
No `Adapter` is installed or constructed, no service gate changes, and no stage1 receipt
or Q64 result grants this scout any qualification.
The main proposal file is read only for its pinned SDK/helper/JIT/codec identity literal;
its proposal functions are neither imported nor invoked.

## CPU and planned device commands

```sh
python3 -B -S -W error tools/probe_biv150_konly_q32_s2_compile.py
python3 -B -S -W error tools/check_biv150_konly_q32_s2_compile_cpu.py
```

Root alone may run the following after PR123 and a fresh exclusive/idle check:

```sh
python3 -B tools/probe_biv150_konly_q32_s2_compile.py \
  --compile-only --root-owned-exclusive --commit EXACT_FROZEN_HEAD \
  --out /tmp/cjj-q32-stage2-raw-compile-v1-20261029
```

It allocates uninitialized compile descriptors and performs one actual JIT warmup plus
handle initialization. Attention/writer/encoder launches and initialized tensor elements
remain zero. Context/allocation APIs count as GPU API work, not numerical execution.
Original TTIR/TTGIR/LLVM/ASM/final cubin, typed ABI, metadata, source identities and handle
resources are saved. Unknown/spilled runtime resources stop with their actual records
preserved. ELF scalar/vector/private resource fields require later independent review;
runtime `n_spills=0` is not a claim that all ELF spill counts are zero.

The only question is whether actual stage2 creates a useful, reviewable load pipeline
while leaving room for a separately proven Q operand lifetime. No pipeline/hoist is
asserted by a successful compile. If the new CFG, alias ownership or loop spill cost is
unpromising, stop here and report **throughput unmeasured**. Do not introduce an async
proof framework, early-Q relocation, Q64/K128/warp/stage sweep, numeric test or timing
entry in this change. All device/resource/numeric/Graph/performance/service gates remain
pending; CPU fixtures are explicitly synthetic metadata and do not run `warmup`.

Historical first-freeze (`585d187`) CPU verification: **31 checks passed** with Python 3.12.14 and
`-B -S -W error`. The generated driver has 164 lines versus the reused parent's 170;
the allocation fixture and `save`/`git`/`main` functions are AST-identical, and the
24-edit inverse restores the complete parent bytes. Snapshot scope inversion, one
warmup/no-override/no-proposal structure, fixed geometry, wrong ownership/source,
stage1/stage3 recorder rejection and disabled/unknown compiler-override guards are
covered. Synthetic snapshots do not establish actual SDK warmup or device semantics.

CPU report SHA-256: `2cb4a357429750abcaa809fe65037a2b6c2074b2907c47d1a6b6a0709d282beb`.
Default plan SHA-256: `138865e0aed873dd11928ed1c19ff5925c48b630a5dd69a6c1b02b1cbaf04f4d`.
That historical CPU check did not invoke the target compiler or GPU. It preceded the
failed target attempt recorded below.


## First actual attempt and narrow cold-JIT repair

The original `585d187` attempt, owner179458/start362684972, stopped at
`fn.compile` with `AttributeError: JITFunction object has no attribute compile`.
The owner closed and all seven original files (67317B) remain local. Summary SHA256
`44707e9bf656d999bf1b923c2db6f42baa181bd3bc1b0d701c73883d65179030`;
full archive SHA256 `d4cc55a367e763c5e44bcb19e07ae940205266e369215bd872007f3d8d5a2a7e`.
There were zero warmup attempts and zero attention/writer/encoder launches, while
CUDA import/context and descriptor allocations had occurred. This is an API access
failure before stage2 compilation, **not a stage2 performance result**.

The saved actual JIT source (`66a168ae…`, lines700–711) assigns `self.compile`
only inside lazy `create_binder`; its constructor builds a defaultdict using that
binder. A fresh JITFunction therefore has no `compile` attribute before warmup.
The repair resolves `..compiler` using `create_binder.__globals__['__package__']`,
exactly matching that method's relative import, then reads the exported compile
function's own `__globals__['knobs']`. It does not call the binder or compiler and
does not assume the JIT-side knobs object is the compiler-side object. The actual
saved CoreX compiler (`a97f0670…`, line287) reads `knobs.compilation.override`.
The saved generic compiler (`f9646f23…`) has the same relevant access; the repair
uses the runtime relative import rather than guessing either module path.

All four prohibitions remain: environment override disabled, no `ir_override`
launch argument, actual compiler manager strictly `is False`, and the eventual
compiled metadata explicitly `ir_override is None`. Unknown or missing manager
still rejects. The only generated-driver changes relative to the first attempt
are the production-base diagnostic and this prewarmup lookup expression. GPU Q32
source, allocation/ABI fixture, one warmup, stage2 option and resource rules are
unchanged. Production bytes equal main899; the old585 frozen worktree is untouched.

The corrected CPU suite passes **34 checks**, including a cold object without a
compile attribute, exact relative-import selection, deliberately different JIT-side
knobs, and rejection of true/unknown/missing compiler manager or compile export.
Report SHA256 `b4ad969e99334fb7b3d59524b036f307fe3f0fcec6b9b573d6e07cf661e2be14`;
default plan SHA256 `f05751f1bf1fe2997f449e8c86039d2de3b65bf71a7cda091d495466c73a3381`.
A separate local source/failure replay verifies all seven original hashes, actual
SDK AST lifecycle, the exact narrow reverse-diff and unchanged GPU bytes; report
SHA256 `0f7bb98a5c84454749716f67b24ae1d396a4895f2f1f0ea19206684e233fca4e`.
These CPU checks have not performed the corrected device compile or granted any
pipeline, numerical or performance qualification. The next output must be a fresh
v2 tag; never overwrite the original failed output.
