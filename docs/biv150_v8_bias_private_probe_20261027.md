# Private A/bias consumer: source readiness, not a device result

This update wires a separate opt-in private driver on PR78. It does not edit
PR73/4a, PR75/ac6 or production source. The A generator and generated source
SHA256 `dba93c46e32f1761c15df73494185122e8784ecaa769ff2c5b7e078b3dd24fbd`
are unchanged. No SSH, JIT, GPU, service or queue was used for this implementation.

## Frozen reuse and boundaries

The complete PR73/4a probe is an inert text fixture pinned to
59546f4657d48e05e3e15d5d5c807f4568126f073dd9683d371e3766cba530ad.
Its scalar reference and validation modules are copied byte-for-byte, pinned
and checked for their actual import paths. The descriptor functions are exact
function slices from75/ac6, including true packed rank/shape/offset/gaps and
scratch/max/sum narrow strides. Main56f's BF16 GQA and benchmark utility hashes
are pinned. Runtime output, token, slot and cache tensors use real descriptors;
no1D pointer stubs or manual use_sme override is used.

Ten host AST adaptations surround the frozen probe. Reversing every adaptation
restores the entire original module AST. Original independent BF16/FP64 numeric
oracles, thresholds, two changed-input patterns, writer/codec/invalidslot,
unmapped bytes/padding, Graph/eager output, finite and baseline hard guards are
therefore retained. The additional four-field Graph/eager byte gate and A/old
gate are hard failures, including in numeric-diagnostic mode.

These files are private tools, not a serving hook. Standalone factory proof
continues to describe only the generator; the new driver is source-wired but
has never been validated on the device.

## Mandatory compile-only phase

The driver defaults to CPU preview. Explicit `--compile-only` first writes the
exact original/a7fc and generated A source into a new external output directory.
It uses uninitialized real tensors and warmup only, with kernel_launches=0.
Source BF16 and encoded caches have separate allocations, matching the full
encoder/writer harness; the ac6 constructor is reused twice. Per-case allocation
upper bounds are roughly6.84MiB ragged,532.75MiB4K and1034.38MiB16K, plus one small
BF16 output/token/slot set. Allocator/module caching may retain earlier blocks.
No tensor contents are initialized or consumed by a kernel in this phase.

Each case freezes source SHA, exact call ABI (tensor dtype/rank/shape/stride/
offset/contiguity, scalars and grid), options, metadata, registers/spills and all
exported IR/ASM/cubin for ten roles: current-main BF16 GQA, old V8 GQA, A GQA,
the original reducer, and both variants' encoder and ENCODE=false/true writer.
Partial artifacts/metadata survive compilation errors. The known current-main
BF16 specialization hash must match before accepting the proof.

The conservative w1 full/tail mechanism gate requires:

- Old V8's actual LLVM numeric I8 conversions and64+64 total template ASM CVTs
  must be recognized. A must contain no numeric integer-to-float conversion in
  TTGIR/LLVM or the old I8→I16→BF16 / I8→F32 ASM path.
- Each path must show32 actual scalar FP32 bitcast/bias operations and eight
  packed V loads. Unknown/vectorized/strength-reduced grammar stops for review.
- Four typed BF16 QK/PV dots and32 BF16 MMA template instructions must remain.
  Shared layout, store/barrier counts and store immediate/mode signatures must
  match old V8; register count is recorded without a speed inference.
- Old/A ABI, options, warp/configuration and compiler version must agree.
  Registers, shared bytes and use_sme are separately recorded for both sides;
  a difference in these known resources is allowed. Every one of the ten roles
  requires a positive integer register count, integer zero spills (None/bool
  are rejected), and known shared/warp/stage/use_sme/version metadata.
  A different source/hash alone is never mechanism proof.

This preserves the two original falsifiers: did numeric V conversion actually
disappear, and are the32 b16 V stores per path still present? It makes no LDG
reduction claim: old V8 already has packed-word loads. Any failed or unknown
mechanism stops before numeric execution and timing.
An unrecognized staging or lowering grammar is an unknown result requiring
review; it is not evidence of incorrect mathematics or negative performance.

## Numeric and full-cost ordering

For each case, the driver runs frozen V8 first with the complete original
numeric framework but **no timing**. A then runs the same seeded framework.
Full original CPU K/V, table, lengths, slots and per-pattern Q/new tokens are
compared, followed by logical narrowed scratch/max/sum/output bytes for both
patterns. NaN payloads and positive/negative zero are compared as raw uint8
bytes. Original FP64/consumption checks are still recorded independently; an
A/old match cannot turn the known ragged failure into passed precision.

Only after both patterns pass every original hard gate and A/old equivalence
can full-cost timing run. Each round measures current-main BF16, frozen V8 and A
via the original token writer+producer+reducer call, with the appropriate
variant's writer. The primary ratio is main_BF16/A, with old_V8/A reported only
as a secondary component diagnostic. Initial cache encoding, CPU references,
byte comparison and persistence are outside timing. Attention-only timing is
disabled. Numeric-diagnostic permits only the original finite candidate
precision failure; baseline/codec/Graph/finite/bitwise failures still stop.

Before the first actual launch of any of the ten roles, actual warmup must
match a proven ABI/hash/options/metadata/resources and exact TTGIR/LLVM/ASM SHA.
Same clean checkout and source paths are required across the two explicit
phases. Proof files cannot license an uncompiled case or another source commit.

CPU replay from this worktree root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_decode_factory_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_v8_bias_probe_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_v8_bias.py --case ragged
```

Future owner-controlled commands, **not executed or queued here**, after access
and exclusive device use are established; both output paths must be new and
outside the checkout, with a private runtime/cache:

```sh
python tools/bench_biv150_v8_bias.py --compile-only --case ragged --case steady4k --case steady16k --out NEW_COMPILE_OUT
python tools/bench_biv150_v8_bias.py --run-gpu --numeric-diagnostic --case ragged --case steady4k --case steady16k --ir-proof NEW_COMPILE_OUT/compile-proof.json --out NEW_RUN_OUT
```

[consumer-source-cpu.json](evidence/v8-bias-decode-20261027/consumer-source-cpu.json)
records exact AST reversal, real old IR rejection, numeric/hard-gate faults,
hash/ABI/IR/options/resource faults before a fake launch, raw-byte comparison,
and timing-before-equivalence rejection. The only positive IR fixture is
explicitly non-executable parser text; it is not new actual IR. Original old IR
bytes and SHA are retained in the small old-v8-ir archive.

No device, throughput, precision or service acceptance is asserted.16KiB/page
and codes/scales are unchanged. PR73's0.72/0.68 baseline ratios remain negative
evidence; beating slow V8 alone cannot establish a useful current-main gain.
