# Q32 backend variation: actual-ASM numerical admission

Root's6439 eager-v2 job628141/start344328946 ended same-start Z with62 raw
files/5043823B. All three roles were warmed and exported before the old baseline
exact-cubin gate refused. No encoder or reader launched; the admission's tiny
CUDA initialization did run. This is not a precision result. All62 files plus
manifest are preserved unchanged in a667281B archive; the original error remains.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_backend_cpu.py --extract FRESH_EXTRACT
```

All three actual source bodies, typed ABIs, options, TTIR, TTGIR and LLVM are
byte-identical to their correct per-role references. Hashes remain678e/c0a/68f.
All three printed ASM and cubin SHA values differ from their prior compile.
ASM differences are constant temporary-register/selection-operand variants;
full instruction opcode sequences stay1901/4029/4029 instructions and unchanged
control/synchronization text. Full ASM diffs are1067/974/1513 bytes. The actual
hoisted ASM still passes the Q-register, loop, native MMA and barrier audit.
The per-role source-root repair worked. Cubin changes are a backend compilation
observation; this preparation does not claim binary equivalence or correctness.

The explicit v3 entry uses source/typed ABI/options/TTIR/TTGIR/LLVM exactness plus
the current actual ASM checks to admit **numerical validation**. Baseline keeps
actual LLVM exactness and64 native BF16 matrix instructions; unhoisted keeps the
original I8 loop grammar; hoisted runs the positive actual SSA/loop/register/
barrier/native I8-QK/BF16-PV audit. Instruction opcode sequence and control text
must match. Actual cubin must be fully exported, recognized as target ELF64 and
match its own recorded SHA; equality to a previous compilation is not required.
Both ASM texts, full bounded diff, both cubin SHA values and the explicit false
cross-run equality flag remain in the sidecar. The historical exact-machine
default/checker is unchanged, so9f and6439 evidence never becomes retroactive pass.

This does not weaken any numerical gate. Only the subsequent real execution can
establish output correctness: all-output hoisted/unhoisted bitwise, original
BF16/quantized FP64 gates, independent full codec, finite, baseline-reference,
canary and read-only-input guards remain unchanged. Unknown source/IR/ABI/options,
missing actual cubin, changed opcode/control or failed hoist audit stop before
reader execution. Small CPU counterexamples cover those boundaries. Reader,
codec, mathematical expressions, d6/bb and production bytes are unchanged.

The first scope remains2048 queries+4096 prior context, two patterns, eager only,
no timing/Graph/side/16K/service. The existing --run-gpu CLI is unchanged except
for Root supplying the new exact commit and fresh output. No SSH/JIT/GPU or queue
was used during CPU preparation; precision_gate remains false pending real data.
