# Private K-only reader compile frontier

This adds an opt-in **compile-only** entry on top of PR80 reader freeze
24c741cc152fc7a53f0b8ce27c1e988898fc59ec and its dependency PR79 ae00. The
factory, codec, generated reader hashes and original CPU evidence are unchanged.
There is no service hook, numerical/timing runner, queue, SSH action or dependency
installation. This preparation was exercised only with stdlib CPU fixtures.

Default use imports no Torch/Triton, checks pinned source inputs and prints a CPU
plan. Only `--compile-only --commit EXACT_DRIVER_HEAD --out FRESH_DIRECTORY`
can import the target framework, allocate CUDA memory or call JIT warmup. Root
owns the future serial invocation. This document is not permission to run it in
parallel with another GPU job.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_prefill_compile.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_prefill_compile_cpu.py --out NEW_CPU_JSON
# FUTURE ROOT OPT-IN ONLY; not executed during this CPU preparation:
python3 tools/bench_biv150_konly_prefill_compile.py --compile-only --profile q32 --commit EXACT_DRIVER_HEAD --out NEW_Q32_OUT
# Separate jobs, only after Root reviews the earlier result:
python3 tools/bench_biv150_konly_prefill_compile.py --compile-only --profile raw3d --commit EXACT_DRIVER_HEAD --out NEW_RAW3D_OUT
python3 tools/bench_biv150_konly_prefill_compile.py --compile-only --profile multi2 --commit EXACT_DRIVER_HEAD --out NEW_M2_OUT
```

The default profile is normalized Q32. Raw3D and M2 are independent explicit
profiles; there is no multi-profile automatic queue. Normalized BM128 remains
source-prepared by the original factory but is outside this minimal entry.

## Tensor descriptors and real allocation work

The compile geometry follows main56f's existing prefill component fixture:
`pages=2*sum(ceil(klen/16))+5`, a **full** `[N,8192]` table, `q_descale=None`, and
two 0D FP32 k/v descale tensors. This differs from PR79 decode's narrowed table;
the distinction is explicit. It describes this private component specialization,
not a claim that all live service descriptor variants are identical.

| Profile | Query/KV lengths | Source/launch configuration |
| --- | --- | --- |
| Q32 | Q4096, KV4096, one sequence | normalized L0; BM256/BQ32/TILE64/w16/s1; grid `(129,2)` |
| M2 | Q2048+2048, KV4096+4096 | same normalized source; grid `(130,2)`; never raw BM128 fallback |
| raw3D |64 sequences, Q1/KV4096 each | independent raw L1; BM16/BQ2/TILE16/S16; grid `(96,2,16)`; SDK warp/stage defaults left unspecified and recorded from compiled metadata |

Only `torch.empty`, dtype views, `unbind` and `as_strided` construct buffers.
Nothing initializes lengths/table/Q/cache/scale/scratch values. There is no
`zeros`, RNG, fill, copy, encoder, writer, synchronization kernel or attention
launch. CUDA context creation and real memory allocation **are GPU API work**;
they are not hidden as zero device interaction. The tool's explicit kernel
launch count remains0 because only JIT `.warmup` is invoked.

Original BF16 caches use a real5D packed allocation and rank4 `unbind` views,
strides `(8192,256,128,1)` in BF16 elements. The candidate uses a separate raw
allocation, I8 K rank4 stride `(16384,256,128,1)`, original BF16 V at offset4096
BF16 elements, and a full flat FP32 view at offset0 for the Kscale ABI. All views
preserve their real rank/shape/stride/contiguity rather than1D pointer stubs.
Q/out have `(T,16,128)` BF16 layout. Raw3D scratch/max/sum use **contiguous logical
S16** layouts `(T,16,16,128)`/`(T,16,16)`; no GQA narrowed-gap reinterpretation.

Q32 needs about49MiB for the declared buffers, M2 about65MiB, and raw3D about
1.01GiB (the component fixture retains unused canary pages and both cache
allocations). The plan records exact byte counts. This memory is uninitialized
and temporary;16KiB physical pages remain unchanged and no capacity is reclaimed.

## Actual-source and archive boundary

Before framework import/CUDA, opt-in requires a clean exact driver commit,
unchanged production source, actual installed SDK54c495/helper59d9/JIT66a168
file hashes, an empty native compute-PID query, full ixsmi's explicit idle marker
and closed9031. Busy or unknown state refuses immediately without a wait or a
process stop. This one-shot compile admission does not implement or license the
whole-job native-PID transition contract of a future numerical runner.

Generated candidate source must still match the recorded24c CPU hash. The tool
saves baseline/candidate modules and frozen SDK/helper sources before compiling,
then checks imported actual SDK files again. Baseline is the **canonical BF16
source without the accepted service's BF16 IR-hoist transform**; it is a compile
control, not an accepted service timing baseline. No old BF16 pass is borrowed.

Each producer warmup exports available TTIR/TTGIR/LLVM/ASM/cubin, actual JIT
source, SHA/length manifests, grid, requested options, compiled metadata/options,
signature/constants/attrs, JIT version, actual Tensor descriptors and resources.
Raw3D additionally compiles the unchanged SDK reducer. Both roles retain separate
resources/use_sme; equality is not required. Handles are not explicitly loaded.
Unknown registers/spills stay null, never become a zero-spill claim. Missing
required actual source/TTGIR/LLVM/ASM/cubin causes an incomplete result while
preserving available exports, source files, cache and failure diagnostics.

`completed=true` means the requested compile archives are complete **only**.
There is no audited I8 operand-hoist SSA grammar yet: `auditor_supported=false`,
`verified=false` stay in every record even when compilation succeeds. Root can
review actual raw IR to decide whether and how to add a narrowly evidenced
mechanism contract. Source order, a native opcode, register counts or complete
archives cannot license numerical/timing execution. Actual runtime binding,
codec/all-reader correctness, precision, performance and service remain unproven.

The CPU checker uses explicitly synthetic empty Tensor metadata and synthetic IR
in a temporary directory to falsify rank/stride/view-origin/alias/stub mistakes,
unknown resources and missing LLVM. It also invokes every default CPU profile
in a child process and confirms no Torch/Triton import;31 rejection/unknown-resource
fixtures pass. Synthetic records are
never kept as real target evidence. Root's later79 near-precision diagnostic
speedups do not change this reader's original numerical gates or qualification.
