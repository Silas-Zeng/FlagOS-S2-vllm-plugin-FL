# Dynamic gather: isolated compile-only screen

This child preserves all2773 production bytes (including accepted117/main56f).
It adds only a compile frontend and a small stdlib check. No Torch/Triton/NumPy
is imported by the default path. No numerical/performance/service eligibility is
claimed. The frozen kernel SHA remains c5613dc8645e862fee10be013f2238d436aa5760d3f5501be560c257b8d3ece5.

```sh
python3 -B -S tools/compile_biv150_dynamic_hidden_gather.py
python3 -B -S -W error tools/check_biv150_dynamic_hidden_gather_compile_cpu.py
```

Root-only after its previous job is terminal, with the production gather flag
still OFF/unset:

```sh
python3 -B tools/compile_biv150_dynamic_hidden_gather.py \
  --compile-only --root-owned-exclusive \
  --expected-source-commit FULL_NEW_FROZEN_HEAD \
  --out /absolute/fresh-compile-out
```

The actual installed CoreX JIT source is saved and checked against the previously
used66a168ae… warmup implementation. Wrong versions/target are refused with
available diagnostics retained. This does not upgrade or install anything.
Explicit Root serial coordination is the exclusion authority; the9031 check is
only an extra fence, with no namespace PID-table idle inference.

M=(1,63,64,65,1782,2018,2048) and
N=(1,7,8,9,31,41,42,43,59,60,61,62,63,64). All84 legal N<=M combinations use two
warmups each. Hidden[2048,2048]BF16, indices[64]I32, output[64,2048]BF16 are allocated
once with torch.empty. Per-case zero-offset narrow views carry the actual ranks,
shapes and strides while keeping all three pointers/alignment identical. These
CUDA allocations initialize the context; this is not zero GPU/context work.
Deterministic-algorithm mode is refused before empty allocation, without changing
its policy, to avoid automatic uninitialized-memory filling.
No tensor contents are initialized, copied or read. Uninitialized indices are
safe only because the frontend calls frozen JIT warmup, never the kernel.

`kernel_launches=0`,168 attempted/completed JIT warmups, explicit handle-load
counts, actual Python object identities and compiler hashes are separate fields.
Warmup counts are neither launches nor compiler-invocation counts. Strong refs
retain all returned objects so IDs cannot be recycled. Each unique returned
object exports every available actual artifact and source, full TTIR/TTGIR/LLVM/
ASM/cubin, exact signature/constants/attrs including typed dictionary keys,
requested options and actual metadata/resources. `_init_handles()` loads only;
known/unknown n_regs/n_spills and metadata are recorded explicitly. No absence or
None is interpreted as zero spill. Disk cache contents are included in the final
artifact manifest. No SDK internal compile hook is invented/instrumented.

The ABI audit requires both row extents to remain actual i32 arguments and
absent from constants, with only8/128 tile constants. `do_not_specialize` is saved
as an observation, not a pass. The straight-line TTIR parser follows actual
index-load/payload-load/store SSA: selected-row tail, signed loaded-index >=0 and
<input_rows, column<2048, and matching payload/store masks. It also checks integer
16 payload copying. Its scope is explicitly TTIR dependencies; affine addressing,
final machine masks and real payload equivalence still need manual/device review.
Unsupported grammar remains verified=false with the raw files retained. Small
synthetic SSA CPU checks exercise this parser, not a real compiler or worker.

`completed=true` means the whole matrix and mandatory artifact collection
finished. It does not require a single hash, known resources or a supported mask
grammar, and grants no numerical permission. Those observations are independent
booleans. If shapes produce multiple real keys/objects, that is recorded rather
than normalized away; descriptor/use_sme/alignment variants remain possible.
The source is loaded directly from its frozen checkout path; the saved file is
an evidence copy, not a new debug/source root pretending to be the old source.

This screen does not exercise real `_cold`, CPU provenance/receipt/full-call
costs, runtime branch hits, payload/Graph/side correctness, service or timing.
Those remain subsequent Root-controlled steps. No new numerical or performance
framework, production hook, queue, GPU runner or service activation is added.
