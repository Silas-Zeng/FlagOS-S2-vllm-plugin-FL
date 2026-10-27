# Q32 eager9f binding refusal and per-role repair

Root's job627487/start344209834 ended same-start Z after warmup of baseline.
The original error is preserved: `baseline: actual ttgir differs from the
reviewed compile`. Reader launches=0, encoder launches=0; the disclosed admission
tiny CUDA initialization did run. This was a binding refusal, not precision data.
All20 raw files/1050358B plus manifest are archived unchanged (150910B tar.gz).

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_role_binding_cpu.py --extract FRESH_EXTRACT
```

The incorrect9f comparison imported baseline.py from original d6's source root,
but selected the later bb override compile's baseline record. Both use source
SHA b6703eef... and kernel hash678ec5...; the source/debug paths differ. Actual
hash, typed ABI, signature/constants/attrs, all options and warmup resources match
d6 exactly. Query rows4096→2048, grid129→65, physical pages517→773 and the output
canary offset2048 BF16 elements were recorded actual geometry changes; they did
not change the observed Triton ABI, SME flags or specialization.

The actual source/TTIR/TTGIR/LLVM/cubin bytes match d6 exactly. Compared with the
incorrect bb reference, replacement of only its literal source directory
explains all97 TTIR,96 TTGIR and1 LLVM path occurrences. No operation, pointer
hint, instruction flags or mathematics changed in these IR comparisons.

Actual ASM differs from the correct d6 printed ASM in three small constant-temp
selection regions (actual lines719–730,2186,2201–2202); both full texts are
preserved and the checker prints their1500-byte unified diff. For example, the
temporary values0x5410/0x7632 swap v2/v3, and the csel source operands swap with
them. The complete actual cubin is byte-identical to d6, independent of printed
ASM. This does not assert the bb cubin is identical: it is a separate artifact
with different debug paths and a different .text variant. No time is inferred.

The repaired numerical binding explicitly derives each source root from that
role's archived TTIR, never from the override TTGIR:

| Role | Source and reviewed record |
| --- | --- |
| BF16 semantic control | Original d6 source/debug root and original d6 baseline |
| Unhoisted I8 control | Original d6 source/debug root and original d6 candidate |
| Hoisted I8 | bb source/debug root and bb candidate; fixed override still retains d6 TTGIR locations |

All three warmups export every actual file, descriptor, full expected record and
difference JSON before any binding refusal. No encoder/reader runs during this
stage. Source/TTIR/TTGIR/LLVM/cubin, kernel hash, typed ABI and options must match
the exact reviewed role. An ASM text discrepancy is retained as a full sidecar,
and is accepted only with that exact machine/IR binding plus the actual strict
audit: same full instruction-opcode sequence and control/synchronization text,
original BF16 matrix grammar for baseline, actual negative I8-loop grammar for
unhoisted, and positive I8 SSA/loop/Q-register/barrier audit for hoisted. Unknown
IR, changed cubin, ABI/options, instruction/control grammar or missing files stop.
This is limited to these three pinned Q32 roles, not a general ASM bypass.

The repair changes only test binding/provenance and warmup export order. Reader,
d6/bb driver, original override, mathematics, codecs and all numerical/finite/
canary/bitwise/read-only/native-PID gates are unchanged. The128MiB collector is
not a numerical limit. Full artifacts remain losslessly compressed. The scope
stays2048 queries+4096 prior context, eager only, no timing; Graph/side/16K and
all formal precision/performance/service gates remain false. A new device run
requires Root's serial opt-in; the original9f result is never rewritten.
