# Private N48 K pair-read candidate

This CPU-only candidate depends on shared PR112 source
`e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8`. It changes only the loop in
`issue_k_i8`, plus private symbol names. It has no service installation or
activation. The namespace is `cjj_native_k8_compact_n48_kpair_probe`.
Actual compile, ISA, resource, device numerical, Graph/side, performance and
model qualification are pending; no candidate SO hash is invented.

The original actual library is
`2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1`.
Its CUDA source SHA is
`1aca8c4be8d8d8e8d818cad529c0716d16b1e30a871ee3c8ec48b12d62338ab7`.
Its S4 SO-extracted ISA has eight scalar K global loads per complete page at
`0x16090..0x16118`, eight dependent shared stores and waits through`0x16148`,
followed by the unchanged barrier. Actual SASS SHA:
`81023af333c6c9481627e37d1bd782126ec6642d2dc28ccf0efe32c451b5a370`.
Static resources are100 VGPR/66 SGPR/8192B shared/private0/zero metadata spills.
These are reference observations, not candidate resources or occupancy proof.

## Exact change and preserved scope

An8B-aligned base selects256 `uint2` loads distributed as four iterations per
lane, instead of512 scalar uint32 loads distributed as eight iterations.
Each pair remains within one KV slot: `word=pair*2`, `slot=word/32`,
`dim=(word%32)*4`. The two original shared words receive `.x` and `.y`.
They are64 bytes apart in shared, so this does not pretend the writes are
contiguous or remove the eight per-lane shared stores.

Both reads stay behind `slot<valid`; invalid slots write zeros without loading
K. The uint2 ABI is guarded by source static assertions for size8, alignment8,
x offset0 and y offset4. The future actual compiler must establish those
assertions; the CPU model does not establish vendor vector lowering.

The host currently guarantees only4B alignment. A base whose low three bits
equal4 executes the **byte-exact original scalar loop**. The dispatch predicate
is uniform within the unchanged one-warp block, although its compiled placement
and cost remain unknown. No host input is newly rejected.

Unchanged items include:

- The full page/head base calculation, page12544, K slot stride256, head128,
  scales and V offsets; N48/S4 host scope and Q pitches2048/2560.
- Shared8192B and its K/V ranges, all barriers, K-before-V issue and delayed
  V wait, Q quantization, QK/softmax/PV, outputs, RN32/RNE codec and flags.
- The existing dc5 writer, reducer, layouts and production dispatch.

Reversing the one loop replacement and private names recovers every original
CUDA/C++/codec byte. The compile frontend reverses exactly to the original112
frontend after three identity/description substitutions. There is no PR113
shared relocation or PR103 Q hoist.

This loop replacement is independently portable: its page-base line is outside
the edit. A later separate experiment could retain the accepted8b86 path's
16384B base calculation and use the same transport loop, provided its K mapping
and ABI are independently checked. This branch contains only the compact N48
implementation and does not grant such a future port device qualification.

## CPU evidence and limits

`check_biv150_native_k8_compact_n48_kpair_cpu.py` enumerates all512 shared words,
valid lengths0…16, both KV heads, page IDs0/1/32767/2147483646, base residues
0/4/8/12 modulo16, and three byte payload families including all256 byte values.
It checks complete read byte multisets, shared write address/bit multisets,
zero fill, unique writes, untouched middle/V/guard bytes, and the exact scalar
fallback read/write sequence. No float reference or device framework is used.

The1,632 image pairs pass:816 vector-branch cases and816 scalar fallback cases,
835,584 word comparisons and3,342,336 byte comparisons. The modeled requested
read transaction counts are417,792 original and313,344 candidate across this
mixed corpus; this is **not an actual instruction or memory transaction count**.
Five altered-source negative cases fail the exact source contract. The CPU
report and default compile plan are preserved under
`docs/evidence/native-k8-n48-kpair-cpu-20261029/`.

The4B-but-not8B CPU result applies only to the K load/store function. Original
2ad's host check does not establish that its full cache/V SME access supports
that placement. Original2ad and the future candidate must both test this
whole-cache boundary; if the original fails, preserve it as an unsupported or
unresolved boundary rather than claiming that scalar fallback qualifies it.

## Minimal later device sequence

1. Root schedules the existing compile-only flow in an isolated output folder.
   Inspect the **actual final SO** and retain raw LLVM/SASS: uint2 ABI, eight
   scalar versus four vector loads if achieved, tail masks, scalar fallback,
   K/V ordering, unchanged math and codec, static/dynamic shared, registers,
   private and both spill fields. Unknown resources remain unknown. A backend
   that splits uint2 back into scalar loads may erase the intended benefit.
2. Use original2ad and the candidate on identical N48 inputs. Reuse the existing
   PR112 four-case fixture (live41/42×Q pitch2048/2560), full saved arrays, FP64
   gates, first48 checks, Graph-before-eager, side stream, guards and whole-cache
   checks without changing tolerances. Include aligned bases at0/8 modulo16
   and separately the4/12 boundary for **both** libraries. These new boundaries
   have not already been device-tested. Keep the prior ideal diagnostics.
3. Only after actual numerical review, reuse the44769-page/stride2560 fullstep
   shapes and dc5 writer+producer+reducer costs, warmup and ABBA to compare
   original2ad versus this candidate. Both arms use the original K8 golden.
   Existing TL48 measurements remain a separate historical reference; twoN32
   is not the performance denominator. Do not mix PR113 into the comparison.
4. Model precision and official throughput are separate later service gates.
   This candidate is not a reason to delay or overwrite the frozen111 service.

Added alignment control, address temporaries, vector register lifetimes,
instruction scheduling, code size and memory coalescing can worsen cost.
K still reads2048 useful bytes per head/page and writes2048 shared bytes. No
HBM reduction, register improvement, occupancy increase or model gain is claimed.
If the actual ISA/resources give no useful change, stop before an expensive
model run. An occupancy query for either native kernel must use64 threads and
dynamic shared0; static shared must not be counted again.

## Commands

CPU only:

```sh
python3 -B -W error tools/check_biv150_native_k8_compact_n48_kpair_cpu.py --out /private/tmp/kpair-cpu-NEW.json
python3 -B -W error tools/compile_biv150_native_k8_compact_n48_kpair.py
```

The following is a future Root-owned command, not an automatic action:

```sh
"$PY" -B -W error tools/compile_biv150_native_k8_compact_n48_kpair.py \
  --compile-only --root-owned-exclusive \
  --out /workspace/experiments/minicpm-s2/cjj-native-k8-n48-kpair-compile-v1-20261029
```

The entry preserves the original zero-kernel compile flow and before/after
Torch CUDA initialization observations. It does not run attention, codec,
writer, reducer or input initialization, and does not approve the resulting SO.
