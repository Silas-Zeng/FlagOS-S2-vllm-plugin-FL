# Q64 compile-only screen on the accepted 16 KiB encoded cache

Base: team main `a455a3de364051fbea80226870c239de3a48e716`. This private
experiment has no production import, environment switch, dispatch or installer.

Hypothesis: a real query tile of 64 may amortize the existing K/V tile loads
across more queries while retaining the accepted I8 QCode operand hoist. This is
not a predicted gain: the larger FP32 accumulator and live Q fragment may cost
more registers, shared memory, instructions or residency. The old BF16 Q64 test
used 201 registers / 131072 B shared and was slower than its BF16 Q32 candidate;
that is a warning, not evidence for this I8 implementation.

Only geometry changes: `BLOCK_M=256/BLOCK_Q=32` becomes `512/64`, with
`grid=(query_rows//64+1, 2)` and a private function name. K64, w16/s1, 16 KiB
page layout, RN32/RNE codec, QK/PV math, causal masks and all output stores stay
byte-identical. The existing CPU hoist transform retains its single movable
`ttg.local_load` and exact inverse; only its shape/metadata checks are adapted.
No Q128, compact layout or tuning sweep is included.

Shortest falsification path:

1. CPU: reverse both source transforms, check complete query/head coverage,
   masked tail rows, 16 KiB address intervals and metadata rejection paths.
2. Root alone: fresh compile of Q32 and Q64, each unhoisted then hoisted, using
   identical uninitialized tensor descriptors. No attention, encoder or writer
   launch is available in this frontend. Save the original TTIR/TTGIR/LLVM/ASM/
   cubin, typed ABI and actual handle resources. Unknown shape grammar stops.
3. Inspect actual Q64 operand dominance, shared-bank synchronization, final
   machine resources/spills and physical layout. A source-level move is not
   proof of a backend hoist. Stop if it is lost, fails compilation or has an
   unacceptable resource cost; do not manufacture an old Q32 native receipt.
4. Only after that review: reuse the original full-output/codec/INT64-QK/FP64,
   poisoned-output, readonly/cache, Graph and side-stream gates. Retain the
   original BF16 proxy diagnostic failures and numerical tolerances. Then use
   same-input warmed Q32/Q64 ABBA, including full and 1985/2017-row tails. A
   production change needs separate model precision and official throughput.

The compile screen alone grants no numerical, Graph, performance or service
qualification. Logical QCode bytes per CTA double from 32768 to 65536 and the
logical FP32 accumulator from 131072 to 262144 bytes; these are tensor sizes,
not physical register/shared allocations or an occupancy estimate.

CPU check (no framework import):

```sh
python3 -B -W error tools/check_biv150_konly_q64_compile_cpu.py
python3 -B -W error tools/probe_biv150_konly_q64_compile.py
```

The optional `--historical-ttgir PATH` checker argument accepts only the old
Q32 fixture with SHA `5e87a8fddac1cf254628a409212f5a6fed2be616179bdefb894e101c40b78264`.
It expands dimensions synthetically to exercise the same parser/inverse and
negative paths; the resulting text is explicitly **not actual Q64 IR**.
No historical or new compiler artifact is added to this change.

Root-only compile command, after freezing the exact clean commit and arranging
exclusive ownership (the frontend does not launch automatically):

```sh
python3 -B -W error tools/probe_biv150_konly_q64_compile.py \
  --compile-only --root-owned-exclusive --commit EXACT_FROZEN_COMMIT \
  --out /workspace/experiments/minicpm-s2/NEW_Q64_COMPILE_DIRECTORY
```

The output contains four fresh original artifact groups (`q32/unhoisted`,
`q32/hoisted`, `q64/unhoisted`, `q64/hoisted`), typed ABI, compiler metadata,
actual handle register/spill/shared observations, and the two one-operation
hoist proposals. The compile descriptor is the existing contiguous Q2048,
773-page fixture, pitch2048, table8192; no tensor is initialized or read by an
attention kernel. CUDA context creation, allocation and handle loading are
explicitly recorded separately from zero attention/writer/encoder launches.
There is no `--gpu`, benchmark or production activation option.

`compile_passed` means compilation, unchanged ABI, exact TTGIR modulo scoped SSA
names and known zero-runtime-spill handles. It does not certify physical QCode
hoisting, shared-bank safety or acceptable resource cost. The old Q32 native
observer hardcodes instruction counts and is **not** reused as Q64 qualification.
The final ELF/ASM/LLVM must be reviewed afresh; scalar and vector register
spills, dynamic versus static shared memory and runtime fields stay distinct.
Unexpected backend output stops before any reader launch, with artifacts kept.

CPU result: 134 Q32/Q64 geometry cases, all 64 tail lengths, 2355264 query/head
rows, 3166208 K/V dimension positions; 23 rejected geometry/metadata/IR faults
when using the optional historical parser regression. Full source byte inverse
and complete AST equality after the two geometry constants/private symbol are
verified. These counts are address/transform checks, not numerical validation.
The full local report SHA is
`eaa6e042659b99736608c1782c4a35d2d7d3e2a37894450b5d2c07e977b6de8c`.
