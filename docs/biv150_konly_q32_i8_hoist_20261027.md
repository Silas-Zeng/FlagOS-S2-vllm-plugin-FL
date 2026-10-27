# Q32 I8 Q operand: actual IR review and one-operation CPU pass

The d6 compile-only job626170/start343488382 completed in the same-start Z state.
Its40 original files/3,047,276 bytes plus the original download manifest are
preserved without edits in `q32-i8-hoist-v1/original-compile.tar.gz` (411,921B,
SHA f3e4f8e1f4414fc1181818746d568c2e7213211f3924db223a8a1e9d549fe183).
The archive checker rechecks every original SHA. The original summary still says
completed=true/errors=[]/0 launches, **actual hoist=false**. This CPU pass does
not modify d6 driver/source/ABI, reader24c, codec, original reports or production.

The witnessed candidate hash is
`c0a523d9d0d4d5dde57b241df9d22c9c38fa71e480878d18e690b4eb002e24f7`.
TTGIR/LLVM/ASM SHA values are respectively5e87a8fd…/250c1e69…/c5c004c0…;
full values are pinned in the new module. Its shared-memory metadata is57,344B;
registers/spills remain **unknown** because handles were not explicitly loaded.
The BF16 control678ec5b4… has98,304B shared and is raw/unhoisted canonical source,
not the accepted service Q32 baseline. Neither shared size is a speed result.

## Actual SSA and physical path

Line numbers below reference the unchanged candidate `actual.*` archive members.

| Layer | Evidence |
| --- | --- |
| TTGIR Q global read |277: masked BF16256×128 `%Q_156`; pointer derives from original query_ptr. |
| TTGIR codec |278–311: BF16→FP32, max-abs, two precise divisions, RNE, clamp and I8 `%4`, all outside KV loop. |
| TTGIR QCode storage |312: `%5 = ttg.local_alloc %4`, a complete256×128 I8 allocation. `%5` has only its definition and one read; no alias, write, deallocation or escape. |
| TTGIR repeated operand read |352 starts the sole KV `scf.for`;421 loads `%10` from `%5` into **opIdx0 / kWidth4 / I8** dot layout inside that loop.423 directly consumes `%10` in I8×I8→I32 QK. |
| TTGIR original PV |464 remains BF16256×64 × BF16 64×128→FP32, with original scaled C; source softmax, masks/scales/output are unchanged. |
| LLVM |16 BF16 global Q reads at265–400 derive from pointer argument1, outside natural KV loop header1796.48 RN32 codec divisions dominate that loop.2215–2237 contain16 two-byte Q shared reads in loop block1975. Their SSA shuffles directly feed I8 operand A; debug inline ancestry resolves to QCode codec call source line238. |
| Native matrix lowering |LLVM2299–2326 has16 signed-I8 matrix calls;2812–2899 has32 BF16 PV calls. ASM3751 onward contains16 `ml_matrix_mad_i32_i8_m16n16k32`;4517 onward has32 BF16 matrix instructions. |
| ASM physical reload |The BB0_39/BB0_38 loop contains32 Q byte SLB reads beginning3721, with byte permutes before the I8 matrix calls. K operand reads use a different packed/shared path. These are static instruction counts, not elapsed-time fractions. |

The code/scale ABI is the actual packed K-only format: K I8 page stride16384,
V BF16 page stride8192, flat FP32 scale view with offset0 and embedded scale
offset1024 FP32 elements. The old mode-NONE k_scale_cache_ptr is an unused dummy
KCode pointer; the new private key_code_scale_ptr is the real FP32 view. Actual
Tensor metadata is preserved and checked against the d6 CPU descriptor contract;
this is not an end-to-end runtime-memory or precision qualification.

## Narrow transform and safety preconditions

The new **I8-specific** grammar does not reuse the BF16 pass. It requires this
exact actual TTGIR SHA and normalized-Q32 symbol/case; M2/raw3D/other hashes are
not licensed. It independently checks a unique KV loop, same-parent dominance
of original Q global read/codec/cast/allocation, complete I8 allocation, exactly
one immutable allocation use, one direct I8 QK consumer, original zero I32 C,
I8 opIdx0/kWidth4 layout, and the original BF16 PV site.

The transform moves **only** line421's `ttg.local_load` immediately before the
KV loop, removing two indentation spaces. Q global read, quantization, local
allocation/store and all math/masks/barriers/other operations remain byte-identical.
An exact inverse restores the original file. The prepared override SHA is
`ef347cd98b1603f9deec35b9ac3e0407efde022e399972b14ca5a60a4226891e`.

This local read is speculatable even for a zero-trip KV loop: the complete QCode
allocation was already initialized, including zero-filled masked query rows, in
the same CTA/parent region. The move crosses no conditional domain and reads no
new global data. No fake dependence, arithmetic, cache write, synchronization
deletion, layout rewrite or source math change is introduced. Compiler memory
planning and inserted synchronization **must be checked after recompilation**.
The long-lived I8 operand may change registers or shared allocation; no amount
of shared-memory saving or speedup is assumed.

## CPU replay; no numerical/timing license

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_q32_i8_hoist_cpu.py --out NEW_CPU_JSON
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_q32_i8_hoist_cpu.py --emit-override NEW_OVERRIDE.ttgir
# Optional original-byte extraction for direct comparison:
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_i8_hoist_cpu.py --extract-original NEW_RAW_DIRECTORY
```

The optional sidecar captures original typed ABI, arguments/descriptors, target,
requested16-warp/1-stage options and source hashes so Root can bind a future
separate override compile to the witnessed d6 case. This commit adds **no** JIT,
GPU, SSH, override-compile runner, numeric/timing runner or queue.

The CPU faults reject changed dtype/layout, extra uses/aliases/writes, changed
QK/PV/codec/loop domain, unknown profile/hash, LLVM native-call changes, swapped
matrix operands, broken debug provenance and CFG, and missing codec divisions.
A critical negative combines the moved TTGIR with the original LLVM/ASM: it
still reports Q reads inside KV and **actual hoist=false**. A source/TTGIR move
alone cannot pass the physical mechanism gate.

The LLVM auditor is deliberately limited to the observed16 two-byte Q reads and
their SSA/debug/CFG chain. New packed/vectorized or direct-register lowering is
unknown, not automatically a mathematical/performance failure. There is no
observed post-hoist ASM yet, so its positive auditor remains explicitly
unsupported/false. Root must compile the override separately, preserve complete
new artifacts/options/ABI and inspect whether all Q operand reloads left the KV
loop, with QK/PV/math and required synchronization intact. Any unknown layout
needs a new evidence-based grammar; the original false proof remains immutable.

Only after that mechanism review may Root consider bitwise raw-I8 vs hoisted-I8
producer fields, original BF16/FP64 precision gates, codec/slot/unmapped/finite/
Graph/side-stream checks and complete accepted-main cost. None is licensed or
claimed here: actual hoist/binding/precision/performance/service remain false,
capacity is unchanged, and no device work was performed in this CPU preparation.
