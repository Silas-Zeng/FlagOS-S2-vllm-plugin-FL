# Fresh service Q32 I8 operand-hoist CPU proposal

This independent own branch starts at freshly fetched/API-confirmed team main
DAC1b8fb48e7983e48283032efeee9b592c5fbe5. It changes only tools/docs. It depends
on PR80's private source/codec work and PR89's default-off encoded-cache service
facade; PR87 provides independent other-reader validation, not this Q32 proof.
No framework import, JIT/SSH/GPU execution, installer, service hook or queue is
included. This is preparation for a fresh actual specialization, not recovery
of a measured service gain.

`tools/biv150_konly_service_q32_hoist.py` exposes
`propose(actual_ttgir, actual_meta) -> (override_or_None, proof)`.
Missing/false `enabled` returns no override. Non-Boolean enable or unknown IR/
metadata raises `Unsupported`. Callers may handle that exception only before
launch to retain the exact correct **private unhoisted encoded reader**, original
args/grid/kwargs. It must never send encoded bytes to a stock BF16 reader, and
must not catch postlaunch errors as fallback. There is no automatic activation.

The narrow scope is B256/BQ32/T64,w16/s1,one sequence,16 query/2 KV heads,D128,
private KV_QUANT_MODE0 and at least1024 CPU-visible query rows, complete original
grid(qrows//32+1,2). No4K/16K/benchmark label selector is present. Metadata must
bind actual compiler hash, source digest, input TTGIR hash, function symbol,
selected constants/grid/options/target and pinned SDK54/helper59/JIT66/codec
bytes. These values must be extracted from the actual compiled object and the
actual source/format receipt. CPU schema checking does not authenticate caller
claims; proof explicitly retains caller/runtime qualification as incomplete.

The proposer independently discovers the input's typed first5 pointer arguments
(outBF16,QBF16,K_i8,scaleF32,VBF16), target module attributes, one KV scf.for,
Q global BF16 tile, RN32 codec and256x128 I8 result. Region ancestry and basic
block identity prove Q load/code/initialized allocation precede the loop in the
same block; CFG dominance proves the original zero-I32 QK accumulator. The Q
pointer addptr/splat/broadcast chain must originate at actual Q parameter1.
The fresh QCode memdesc may be used only by its allocation and a unique local
load: alias/subslice/escape/store/deallocate/additional use rejects. The operand
must directly feed the I8/I32 QK and no other use, with original BF16 PV geometry.
Layout names, SSA names and debug filenames are discovered rather than matched
to an old example. Structured parent/child nesting is checked; unknown branch/
parallel/async or shape/layout grammar refuses instead of guessing.

Only that one immutable local_load is moved immediately before its KV loop in
the same parent block. All other operation text, math, masks, stores and barriers
are unchanged. A strict inverse reconstructs the original input byte-for-byte.
This is a CPU SSA/memory certificate for a proposed compiler input; it is not a
claim that the backend actually retains Q operands in registers. A zero-trip
loop still only speculatively reads the already initialized local allocation.
Physical shared-bank reuse, publication/reuse barriers, native operand roots,
register liveness/occupancy and resource spill behavior remain **fresh native
compile** gates, independent of this source-level move.

The historical original Q32 TTGIR fixture has SHA
5e87a8fddac1cf254628a409212f5a6fed2be616179bdefb894e101c40b78264.
It is extracted unchanged from PR80's original-compile archive and is used only
for regression tests. A second positive changes debug paths, SSA identifiers
and shared/MMA layout names; scope metadata cases1024/2048/4096/16384 exercise
parametric geometry, not runtime attestation. Nineteen negative cases cover
Q memory escape/store/alias, operand extra use/layout, wrong Q source pointer,
extra loop, BM128, nonzero C, PV dtype, target/warps, sibling-block dominance,
numseq/BQ/query/grid/bool/stale TTGIR and SDK identity. CPU3.12/3.14 outputs match.
No new actual service TTGIR has been supplied yet, so actual service/fresh
compile/hoist/native synchronization/numeric/Graph/performance gates all remain
false. Old ef347 overlays, raw hashes and old actual native proofs are never
used as admission predicates for this tool.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_service_q32_hoist_cpu.py --out /tmp/NEW-cpu.json
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/propose_biv150_konly_service_q32_hoist.py
```

The first command runs only the historical/synthetic CPU contract; the second
prints disabled/no override. For future actual input, write fresh metadata with
schema `cjj-konly-service-q32-hoist-v1`, profile `q32`, format
`cjj-konly-rn32-rne-v2`, source/compiled hashes, function name, `pins`, `target`,
`launch_options`, `constants`, `query_rows` and `grid` (see the checker `meta()`
for schema only; its synthetic hashes are **not real runtime evidence**).

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/propose_biv150_konly_service_q32_hoist.py --enable \
  --ttgir ACTUAL_NEW_SERVICE.ttgir --metadata ACTUAL_NEW_SERVICE.json \
  --out FRESH_PROPOSAL_DIRECTORY
```

A service integration must first obtain the actual private Q32 specialization
from the actual selected tensors/kwargs (including real fused-QKV row pitches,
nonzero offsets and pointer divisibility). It must preserve those same objects,
args/grid/options when fresh-compiling with the generated override, and include
source/input/override hashes plus enable semantics in compile-cache identity.
It must not replace those inputs with the contiguous old microbenchmark fixture.
Cache reuse needs the same authenticated actual specialization/source and the
same CPU-visible admitted policy scope. Expensive source/Git/file/IR audits belong
to installation/first compilation, not every hot call; runtime scope remains a
separate cheap checked contract. Unknown specialization retains unhoisted.

Only after Root's isolated fresh compile may a new actual certificate be issued:
actual TTGIR must preserve the exact moved operation (allowing independently
checked scoped SSA pretty-printer renaming), actual LLVM/ASM Q operand roots must
dominate KV and remain unmodified through it, native I8-QK/BF16-PV and shared-bank
barriers must be checked, resources known and spilling rejected. Numerical/Graph/
side, all-service lifecycle/model105 and official full-cost throughput/TTFT remain
subsequent gates. A CPU proposal does not authorize the service to claim hoist,
precision acceptance, capacity reclamation or performance recovery.
