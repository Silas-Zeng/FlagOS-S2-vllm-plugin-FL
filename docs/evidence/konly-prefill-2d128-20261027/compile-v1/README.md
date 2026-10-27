# Actual normalized2D128 zero-launch compile

Root ran unchanged78ebd23bc0ae1517293dbbd07b4aef895dff7755, owner636649/
start345102304, collector same-start terminal Z. Complete38 raw files/3,077,682B
plus the unchanged collector manifest are in deterministic `raw-v1.tar.gz`
(407415B, SHA2566ec6083034bc046940c7eb9f64274189210d012030e371194796472860fd991e).
All original sources, logs/owner, metadata/signatures, compiler cache files,
actual TTIR/TTGIR/LLVM/ASM/cubin and Python sources remain byte-identical.
`archive-manifest.json` lists every member SHA/size and summary SHA.

Original completed/compile_passed=true,errors=[],2 warmups and0 attention
launches. Q2048,prefix4096/KV6144,pages773,grid(129,2),BM128/Q16/K64/w8/s1.
This is a forced component profile, not the current main parent policy choice
for2048 rows; no downgraded main performance control is asserted.

Actual BF16 control hash780806697389702032978aa79c9e12ec47a54b6b84153e20b3578e57c010b479,
shared65536. Actual candidate hash
aacfe384e0d1562b51a8f9eb27e174a9bb66caf4a06e660ff4cbc94fb4230fcd,
shared40960. Registers/spills are **None prelaunch**, and loaded-resource/occupancy
proof remains false. The private mode remains KV_QUANT_MODE=0, with actual
K*i8 / key_code_scale*fp32 / V*bf16 in signature and TTIR/TTGIR. SDK/helper,
generated source and factory's exact AST/byte inverse match frozen pins.

The new audit establishes a precise **unhoisted** actual mechanism, without
changing the original unknown/false fields. TTGIR Q BF16 global load277,
RN32/RNE I8 cast311 and Q shared allocation312 precede KV loop352; Q operand
local_load421 remains inside the loop. I8/I32 QK423 and BF16 PV464 are native
matrix paths. LLVM CFG dominance/natural-loop and backward operand SSA tracing
find16 two-byte shared Q operand reads inside the KV loop, while16 original
BF16 Q global reads and48 RN32 divisions dominate it. ASM independently has32
byte Q operand reads,16 I8 QK and32 BF16 PV native matrix instructions in the
KV loop body BB0_37 (headerBB0_38/backedge present). This does **not** inherit
Q32 hoist proof or claim a working BM128 operand hoist.

The actual I64 scale address chain is full F32 page0 +physical_block*4096 +1024
+slot*2 +kv_head, masked by tile_mask with other1.0. K I8 strides remain
(16384,256,128,1). V is the already-offset BF16 view at4096 elements; its
per-page8192 element stride adds no second V-field origin. The source transform
leaves BF16 V/PV and softmax unchanged. Native code inspection establishes this
one specialization's intended mechanism; numerical behavior, actual loaded
resources, Graph/side, model, performance, all-reader lifecycle and service
activation remain unverified. The archive is not permission to use an encoded
cache with a BF16 generic fallback.

The CFG/SSA observer is copied unchanged from PR80's observer and independently
applied to these BM128 original bytes; the actual hashes/geometry/source and
all observations are BM128's own. Seven semantic faults reject Q-load placement,
scale offset/mask, native I8/PV, Q byte read and LLVM operand-origin changes.
No source transformation or override is introduced.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_2d128_actual_cpu.py --out /tmp/NEW-actual-cpu.json
```

This default is stdlib only: no framework import, JIT or device work. The full
compile summary itself can be recomputed by `biv150_konly_2d128_actual.original()`.
Historical78e source and evidence were not edited. Fresh main API is DAC1b8f;
the new normal HTTP/1.1 fetch timed out at its15-second low-speed limit and is
not claimed successful. Production remains identical to DAC, with no installer.
