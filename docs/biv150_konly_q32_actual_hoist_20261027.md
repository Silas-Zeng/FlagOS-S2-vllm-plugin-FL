# Actual Q32 I8 operand hoist: CPU review

This adds a separate positive audit for Root's compile-only job626927/start343979040,
same-start Z. The original bb843 driver/source, original `hoist=false` and
`ASMunsupported` reports are unchanged. There were zero kernel launches.

The archive contains all42 original files (3151371 bytes) plus the download
manifest, including logs, source, TTIR, TTGIR, LLVM, ASM, cubin and cache metadata.
The compressed archive is424316 bytes and is independently rehashed before use.
The original40-file compile remains separately archived in `q32-i8-hoist-v1`.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_actual_hoist_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_actual_hoist_cpu.py --extract /tmp/q32-hoist-actual-fresh
```

The actual candidate hash is
`68f376d4394d07e21221008dd44099e2178ffd8b18a4e11a610bd175088f5ddc`.
The actual metadata selects the fixed `ef347cd9...` override, while generated
source, typed ABI and original baseline remain unchanged. The accepted grammar
is this normalized-Q32 specialization only. It does not license raw3D/M2 or
arbitrary layouts. The actual TTGIR differs from the fixed input only in scoped
SSA spelling: normalization uses definition positions and lexical regions, so
the reused while/for `%arg26` names cannot conflate separate values.

Actual candidate file locations (one-based lines):

| Layer | Evidence |
| --- | --- |
| TTGIR | Q BF16 read277; RN32 divisions288/293; QCode allocation312; operand read352 before KV for353; I8 QK423; BF16 PV464 |
| LLVM | Q32 byte stores1612–1705; publish barrier1739; Q operand reads1777–1814 in `.lr.ph108`, dominating KV header1832 |
| LLVM | Shared bank reuse barrier2128 before K stores2129/2130; acquire2251 before32 K operand reads2252–2283 |
| ASM | Publish barrier3446;32 Q byte reads3471–3624 and24 packing operations3554–3657 before loop |
| ASM | KV body3753/header4707/exit5119; Q fixed in v32:v39 and never written in any of the full loop's masked subpaths |
| ASM | I8 QK16 instructions3849–3921 read the four invariant register pairs; BF16 PV32 instructions4311–4667 |
| ASM | Reuse barrier4883 before K shared stores4884/4887, including the first iteration entered at header4707 |

The new auditor checks actual SSA roots, CFG dominance, barriers, exact scoped
TTGIR equivalence and physical register construction/invariance. CPU faults
remove publish/reuse/acquire barriers, mutate the source operand or dot type,
reinsert a Q reload, overwrite a live Q register, and replay old unhoisted IR.
Pinned actual hashes additionally limit the positive claim to the observed IR.

Shared allocation changes57344→33792 bytes. Registers/spills are **unknown**:
this compile did not initialize handles. Shared size and static instruction
counts do not establish runtime, accuracy or performance. The BF16 control is
the canonical raw unhoisted SDK control (98304 shared), not the accepted service
main kernel. Binding, bitwise execution, precision, performance and service
validation remain false. No capacity was reclaimed.
