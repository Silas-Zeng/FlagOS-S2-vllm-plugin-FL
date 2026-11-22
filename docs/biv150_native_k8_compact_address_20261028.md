# Compact NativeK8 address-only component draft

CPU preparation only, dependent on PR93 `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46`
(original377 sources / original22e library) and PR91 format reference
`dc5fd2e5b86a3d57aceaad26e8556d169ec7462a`. Fresh team-main fetch and API both
reported `4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b`. No official framework upgrade.
No source or frozen tree from PR91/93/96/98 is modified. Only new tools/kernels/docs.

The private `cjj_native_k8_compact_probe` library preserves the original K-then-V
issue, wait8/barriers, tail masks, Q codec/hoist, I8 QK, FP32 online softmax,
pagewise BF16 P, BF16 PV, scratch stores and reducer. It does **not** combine the
unvalidated V-first variant. The source factory reverses exactly three CUDA
global-address edits and five C++ typed-layout/span edits (plus isolated symbol
names and three error messages) to original377. The codec body is unchanged.

| Role | Pointer | Page step / elements | Last page-local byte |
| --- | --- | --- | --- |
| K I8 `[P,16,2,128]` | layer fullbase | 12544 I8 | 4095 |
| scale F32 `[P*3136]` | same fullbase; internal +1024 F32 once | 3136 F32 | 4223 |
| reserved | untouched | bytes4224..4351 | 4351 |
| V BF16 `[P,16,2,128]` | fullbase+4352B once | 6272 BF16 | 12543 |

The V SME descriptor keeps its 512B slot pitch. Its four 16-row/64B column reads
per head cover exactly the original logical V, staying inside the 8192B V region.
Shared K `[0,2048)` and V `[4096,8192)` remain unchanged; do not replace shared
4096 with global V4352. Page bytes, F32 scale step and BF16 V step are different
units. CPU checks exhaust all page bytes, both SME head footprints, large page IDs
and a two-page nonzero-base serialization/remap. This proves address formulas,
not GPU instruction lowering or numerical behavior. No Torch/Triton/NumPy is used.

```sh
python3 -B tools/check_biv150_native_k8_compact_cpu.py
python3 -B tools/probe_biv150_native_k8_compact.py
```

Both immutable dependency commits must exist in the local Git object database;
if a shallow checkout lacks dc5, fetch the shared PR91 history rather than replace
it with another checkout. Default mode cannot compile or launch. Root alone may
run the following in its clean, frozen worktree after coordinating the previous
job's terminal state. The controlled compiler/library load and binary inspection
perform zero attention/encoder/codec/reducer launches and no input initialization.

```sh
python3 -B tools/probe_biv150_native_k8_compact.py \
  --compile-only --root-owned-exclusive \
  --compiler /usr/local/corex/bin/clang++ \
  --objdump /usr/local/corex/bin/ixobjdump --out /absolute/new/output
```

The original compile body is adapted only for namespace; sources and header,
compiler/version/options/Ninja/object/SO, standalone LLVM/ASM and logs are frozen.
`compile-summary.json` is left unchanged after creation. A separate
`actual-so-inspection.json` binds its SHA and SO SHA to final-SO `--sass`/`--IR`
outputs, preserving export failures. Exports alone do not set actual-IR/SASS audit
true. Standalone flags differ from CppExtension's flags; source/standalone .s
cannot qualify the final library. Root must inspect final SO dataflow/addresses,
math/waits/tail, code bodies and known static resources; any unknown stays unknown.

This first commit deliberately has no numeric/fullstep/service launcher. Its
`admit_fixture` is a CPU contract for exact typed roles and a current positive
epoch, with explicit compact format, real layer base/storage identity and checked
layout. It is **not** a live storage registration or content certificate, and
must not be fed arbitrary JSON as device admission. The later numeric frontend
must create its own live guarded compact allocation, retain it strongly, verify
codec/writer contents, and enforce that receipt at dispatch. Calling the raw
diagnostic Torch op bypasses that planned frontend and is not licensed by this PR.
Unknown/old format and encoded nominal-BF16 aliases cannot fall through to stock
native; the current compact runtime's ownership-before-native boundary remains a
separate prerequisite for any service bridge. Old 16KiB Q32-hoist proof is invalid
for the new format. There is no new service hook/default activation here.

Next numeric stage: serialize the **same** CPU logical K-code/scale/BF16-V cache
into original16KiB and compact12544 buffers; use identical Q, physical block table,
lengths and width16 narrowed scratch at N64/S2 and N32/S4. Require every active
O/M/L and final bit identical against actual original22e, plus inactive poison,
empty/zero, partial-page, nonzero-offset, whole allocation/padding/guard and alias
checks. Then verify both append writers on identical new KV/slots, Graph replay
before fresh eager, side stream and post-timing same-arm bits. Keep the existing
FP64 and original BF16-proxy gates unchanged; the known Q8 proxy failure is not
compaction precision success. No timing until the implementation gates pass.

N40 is excluded. Main4229 `_pick_segments(40,2,16)` explicitly returns S16; the
CUDA template admits S2/S4 only and host currently fixes N64/N32. A later private
N40/S4 host/grid instance could reuse the S4 CUDA math but needs independent
format-equivalence and numerical evidence. It neither covers N40/S16 nor changes
the shipping policy; a real policy/performance comparison must include accepted
main S16 and cannot compare partial tensors across different segment partitions.

Actual compile/mechanism/resources/numerics/bitwise/performance/model/service and
capacity qualification are all pending. PR91's one-layer/two-page allocation
does not establish 42-layer registration/allocator/scheduler behavior, capacity
recovery or service throughput. Memory occupancy and waiting requests motivate
testing; they do not prove a benefit or an additional qualified shape.
