# Separate after-QK diagnostic hypothesis

CPU manual review of the owner's actual3f1 compile artifacts confirms a partial
mechanism: all full/tail V packed loads, expansion, conversion and scale loads
follow the last QK tensor MMA. They still interleave with score scaling/max and
softmax, so the original after-all-softmax strict contract remains false. The
new flag is a separately named hypothesis, never a promotion of the strict proof.

The owner compiled source `3f1a2586cecc71c3badefc165a9bf9b2321a6db3`, candidate
kernel SHA256 `b562d2cdbabf0dba9b7e495fd9b6a48b7dc70f7ca64220fe441efcbdd3cb66c8`,
dependent on PR73 source4a. The candidate kernel/old4a benchmark/validation and
all production bytes are unchanged by this CPU contract update.

## Original evidence and manual findings

[compile-v1/archive-manifest.json](evidence/v8-late-read-20261027/compile-v1/archive-manifest.json)
preserves all21 original files/1411427B plus the original download manifest in
one190568B tar.gz, verified per-file SHA and complete byte-for-byte roundtrip.
The original [compile-proof.json](evidence/v8-late-read-20261027/compile-v1/compile-proof.json)
has SHA256 `ed669a9454e3c7efa40f405490e3593869c8f18894d2e2a0483143c4b969fb46`:
completed=true, compile_only=true, kernel_launches=0, mechanism_passed=false,
and its original strict-failure error is retained. PID621841/start341524944 was
same-start terminal Z in the downloaded owner record. No additional remote read,
JIT or GPU was performed for this CPU review.

| Actual ragged-late_v8 location | Full path | Tail path |
| --- | --- | --- |
| ASM last QK MMA |555|1429|
| ASM scale load |557|1732|
| ASM8 packed V loads |572–839|1576–1715|
| ASM32 byte expands |590–932|1738–1828|
| ASM64 i8→i16→BF16 conversions |608–965|1833–1896|
| ASM last softmax exp |748|1771|
| ASM first PV MMA |1040|1979|
| LLVM last QK MMA |468|1250|
| LLVM8 packed V loads |576–611|1367–1451|
| LLVM32 byte extracts |577–615|1372–1459|
| LLVM32 i8→BF16 conversions |656–687|1510–1541|
| LLVM scale load |618|1468|
| LLVM last exp2 |514|1302|

All identified operations are after QK MMA in actual ASM and LLVM, not merely in
source order. LLVM is also after-exp in both paths; the ASM scheduler interleaves
some expands/conversions with softmax. In particular590<748 and1738<1771 explain
the preserved strict failure. Before this change, the old V8 packed-word loads
and byte expansion preceded QK. This does not prove elapsed-time improvement or
that all V-related address registers have short live ranges.

Actual compiled hashes: frozen V8
`d2481b9f7db900083e1fdbc307add5f2f66b5a34fe3b1ef6c577db69e9c9e760`, late V8
`428d26d5a579b3946f88d348bc065ea9a5c9cbfcba5bd7a59b07550d5d2a2d65`.
Observed late resources137regs/0spill/4096B shared versus143/0/4096, same
use_sme79, w1/s1/warp64, Triton3.6.0. The complete runtime descriptor plan equals
the independently derived real packed/narrowed ragged plan. BF16 MMA32,
V b16 shared stores64, other b32x2 shared stores4 and barriers9 remain; store
immediates/modes and V blocked/shared layout definitions also agree. These are
staging checks, not a formal proof of every physical-register address expression.
All original numeric and data-integrity gates are still necessary.

The legacy proof did not serialize every compiler option. This update requires
a **fresh explicit compile proof for the new commit**, records fusion/FTZ/input
precision/maxnreg/backend/arch/sanitize/debug options, and checks option equality
against frozen V8 and again against the actual runtime specialization. The old
strict-failed artifact cannot directly license device work.

## Explicit contract and future owner-only CLI

Default `--compile-only` and default device mode keep the original strict gate.
`--after-qk-diagnostic` selects `after_qk_diagnostic_v1` separately:

- Both full/tail LLVM and ASM must have exactly the audited V operations, all
  strictly after the last QK tensor MMA and before first PV; scale load included.
- Every original type/MMA/V layout/shared/store/barrier schema check remains.
  Shared-store immediate/mode/barrier signatures must agree with frozen V8.
- Packed/narrowed descriptor plan, w1/s1/warp64, use_sme, compiler version/options,
 4096B shared and zero-spill staging remain linked to actual compiled hashes.
- `mechanism_passed` continues to mean the strict result. Only the separate
 `diagnostic_mechanism_passed` can admit this explicit partial mechanism.
- Device mode also requires explicit `--numeric-diagnostic`. Original baseline
 CPU numerics, codec, slots, unmapped bytes, Graph/eager and finite guards remain
 hard; candidate precision failures stay false under the original thresholds.

If root elects one small future4K component falsifier, first compile the actual
4K specialization (ragged's79 mask does not license steady's78 mask):

```sh
python tools/bench_biv150_v8_late_read.py --compile-only --after-qk-diagnostic --case steady4k --out NEW_COMPILE_OUT
python tools/bench_biv150_v8_late_read.py --run-gpu --after-qk-diagnostic --numeric-diagnostic --case steady4k --ir-proof NEW_COMPILE_OUT/compile-proof.json --out NEW_DIAGNOSTIC_OUT
```

Use the same clean source checkout/private runtime/cache for compilation and
the guarded run. Before the first actual launch, the real tensor arguments must
produce the exact proven compiled hash and TTGIR/LLVM/ASM SHA; partial mode also
checks all recorded options/metadata/register/spill values. A mismatch stops
before launch. Source/flag changes require a new proof. The separate
mechanism-contract.json records the selected hypothesis alongside the untouched
original numeric result format. Full writer+producer+reducer timing remains
separate from CPU error attribution. No scheduling, JIT or device queue is
started by this CPU review.

[cpu-after-qk.json](evidence/v8-late-read-20261027/cpu-after-qk.json) verifies the
actual legacy partial ordering and preserved strict failure, old pre-QK negative,
early load/scale/expand/cast mutants, staging mutants, rejection of partial proof
in default strict mode, runtime hash/IR/options/metadata/resource faults, and the
unchanged4a numeric/writer/codec/Graph policy. Historical CPU records are preserved.
There is no new throughput or accuracy result for this entry point. PR73's
0.7226/0.6822 steady component ratios remain negative baseline evidence;137regs
does not imply that this scheduling hypothesis recovers those costs.
