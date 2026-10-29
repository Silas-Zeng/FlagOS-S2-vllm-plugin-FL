# PR118: extend the pending mixed-prefix scope from measured host windows

This independent child starts at PR118 `da73e99ff07f9741dcf031dd1f3a73a7070d5eeb`.
Team main remains `a455a3de364051fbea80226870c239de3a48e716`, per Root's fresh
fetch/PR receipt. The original worktrees and31/63 reports are unchanged. Nothing
was fetched, pushed, compiled by CoreX, loaded on a GPU or installed in a service
while preparing this child. Root will publish it to the same PR118.

The sealed119 A windows motivate this exact extension:

| Actual prefix rows | Original TL segments (performance A) | Pending Native segments | Context | Same-S accepted8b reference | Reference-only zero-length rows |
|---|---:|---:|---:|---:|---:|
|29|8|4|16384|N32/S4|3|
|30|8|4|16384|N32/S4|2|
|31|8|4|16384|N32/S4|1|
|60|4|2|4096|N64/S2|4|
|61|4|2|4096|N64/S2|3|
|63|4|2|4096|N64/S2|1|

4K score had21546 noncapture TL decode layer/kernel host calls: N61/S4=7980
and N60/S4=6090, together65.30%. The complete16K score had42168: N30/S8=23142
and N29/S8=8022, together73.90%. These are scoped host-call counts, not GPU time,
Graph replay counts, model-forward counts or a promised gain. They must not be
divided by42 to invent a request/step count. The measured Q pitch was2048,
writer K/V pitches256/2560, width16 scratch and34276 physical16KiB pages.
[Selected local evidence identities](evidence/native-k8-mixed-coverage-20261029/scope-evidence.json)
retain both actual-review SHA references.

## Minimal source changes

Only the host whitelist, the mixed adapter's PAIRS/qualification/C3 literal,
component contract and their CPU checks expand. All original `.cu`/codec/header
bytes remain identical to accepted main8b. The accepted CUDA launchers already
use the supplied batch only as `dim3(batch,2,S)`; each CTA owns one sequence,
KV head and segment. No valid_rows device parameter, candidate padding, grid
substitution, Q/K quantizer, page, softmax, PV, wait/barrier or reducer change is
introduced. The existing seven symbol-renaming macros remain identical for host
and device compilation, retaining the independent mixed namespace. New C++ host
bytes therefore require a fresh actual SO and binding review; old118/8b receipts
do not qualify the new row scope.

N32/N64 still select accepted8b through the original adapter. N28/N40/N48/N59/N62
retain their original metadata and TL routing;29/30 are no longer fallback cases.
The original natural-prefix eligibility and Graph NONE boundary, current stream,
writer-owned16KiB epoch, typed roles, nonalias bounds, one-call ticket, stock
Native fence, decode-before-prefill ordering and no-postlaunch-retry rules remain.
Segments are changed together with width16 *views*, with the original physical
partial/scalar strides32768/2048/128/1 and256/16/1 retained. Candidate Q/length/
table rows remain exactly N, and the prefill suffix is still `[N:2048)`.

## CPU checks and pending actual work

[Routing report](evidence/native-k8-mixed-coverage-20261029/cpu.json) exercises all
six N at Q pitch2048/2560 (12 cases) through the actual original producer/reducer
launch AST with opaque CPU tensors. It checks shared Native/reducer Tensor
identities, all valid row/head/segment address projections, unchanged suffix and
metadata reuse. Readonly new K/V fixture pitches are256/2560. The fixture uses
small opaque cache metadata, not real34276-page contents or device allocation.
Nine prelaunch mutations run on newN61 and three no-retry mutations on newN60;
old32/64 routes and explicit59/62 (plus28/40/48) TL fallbacks remain checked.
The original standalone helper94db fingerprint regression and pending-ON refusal
remain in place. No production SHA pin is populated by a CPU fixture.

The component contract now requires a real N32 or N64 reference, rather than
`N+1`, and explicitly counts all zero-length reference rows. Candidate metadata
and queries are not padded. Total Q stays2048; decode contexts are4096/16384,
paired prefill tables must be valid, and actual writer K/V pitches256/2560 and
width16 scratch must be exercised. Padded reference work remains outside candidate
performance cost. The shipping TL with S4/S8 is A; same-S TL is diagnostic B.
Same-S native-reference O/M/L/final must be bitwise equal for every valid row.
Different-S A/C retain independent original FP64/pagewise-BF16-P numerical gates
and all BF16 proxy results; different partitions are not forced bitwise equal.

The build CPU check retains original clean/dirty/ref, symbol/preprocessor, old8b
refusal and compiler-failure fixtures. Its failure-path review template now reads
scope from the already captured plan, leaving it unknown for a deliberately
missing-plan fixture; it must not rerun source verification after that fixture
has moved the factory root. All new actual-ABI review flags remain false.

```sh
python3 -B -W error tools/check_biv150_native_k8_mixed_cpu.py
python3 -B -W error tools/check_biv150_native_k8_mixed_build_cpu.py
python3 -B -W error tools/compile_biv150_native_k8_mixed.py
python3 -B -W error tools/probe_biv150_native_k8_mixed_component.py
```

There is still no numeric GPU runner in this initial PR118 frontend. Default
execution is a stdlib CPU plan. Root's future first device step is the existing
compile-only entry, with the exact new frozen commit and a new output directory:

```sh
python3 tools/compile_biv150_native_k8_mixed.py \
  --compile-only --root-owned-exclusive \
  --expected-source-commit <full-new-frozen-commit> --out <fresh-absolute-output>
```

Actual new SO symbols/stub ownership, S2/S4 final-SO IR/SASS/math/resources and
dual-library coexistence precede numeric admission. All new29/30 and60/61 review
flags, plus the original31/63 flag, stay false until checked. Complete component
numeric/Graph/side/readonly/guards, writer+decode+reducer+unchanged-prefill cost,
service request coverage, Math105 and warmed same-condition current-main4K/16K
remain pending. Default OFF and qualification pins remain `None`, so enabling
mixed Native still refuses. No1175898, compact12544,120, Q64, Kpair or6KiB variant
is included, and no service, accuracy or performance qualification is claimed.
