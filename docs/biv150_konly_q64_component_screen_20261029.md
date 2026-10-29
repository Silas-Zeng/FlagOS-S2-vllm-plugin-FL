# One actual hoisted Q32/Q64 component screen

Child of `844a69f10539c551500e29e2304061aef7d8671b` (Draft116), on accepted
main `a455a3de364051fbea80226870c239de3a48e716`. Production, the Q64 source
transform and compile frontend remain unchanged. This adds one private
frontend for Q2048, prefix4096/totalKV6144, 773 existing16KiB pages, Qpitch2048,
table8192, K64/w16/s1 and original patterns0/1. No long context or tail sweep.

The independently replayed actual compile review is SHA
`256c95d945b44bfdd3dc0acd1f6a83da791f1a72ab8b0ba3d5fc6f052606caa0`.
Q32 binds122 VGPR/33792B dynamic shared/runtime spill0/ELF scalar spill96;
Q64 binds237/67584/runtime0/ELF scalar485. This explicitly replaces the old
Q32-only `shared<=65536` admission for this private pair. The exact full
source/typed ABI/kernel hash/TTIR/TTGIR/LLVM/cubin snapshot must match the
reviewed compile before the first reader launch. Supplemental ASM is separately
retained as diagnostic evidence; its text SHA may differ only while every
execution identity and the complete final cubin remain exactly equal. The original absolute source
and override paths from the actual compile owner are reused to avoid changing
path-dependent compile identity. Unknown or changed identities stop.

The original `e4dd64e453fbe1bdeba4d8b2a1b551167e9446c4` numeric contract and
its two scalar-codec dependencies are loaded from SHA-pinned original source
files. The original `policy` function is extracted unchanged from its frontend.
Only the reference row list adds63/64; its exact inverse recovers the original
source. Pattern0's full2048-row uniform oracle is AST-identical to the original.
RN32/RNE, INT64 QK, FP64 global softmax and all original tolerances are retained.
BF16 quality failures require explicit `--numeric-diagnostic`; quantized-reference,
finite, codec/cache, full-output bitwise, guards and readonly failures always
stop. The legacy `hoisted_unhoisted_bitwise` field is retained only to call the
unchanged policy; the report explicitly identifies both actual arms as hoisted
Q32 and hoisted Q64. Model `precision_gate` remains false.

The unchanged accepted16KiB writer is also reused, with real K/V source stride
2560 and2048 append slots corresponding to the final2048 tokens. Both arms use
the same writer. Before every numerical append+reader phase, only those exact
K/scale/V bytes are poisoned; every byte of all773 pages must then equal the
independent CPU codec, including unused pages and padding. Full writer input
backings, slots and other reader inputs must remain unchanged.

PatternA eager and Graph outputs are checked first. Both actual hoisted Graphs
read changed patternB **before any fresh eagerB**, for reader and append+reader
modes. Each result is checked against the original reference and its subsequent
fresh eagerB result; changed outputs are required. Every eager/capture/replay
uses the recorded nondefault stream. There is no normalized/fallback Graph.
Full before/after output buffers, full cache results, CPU originals, independent
references and input fingerprints are saved locally before assertions can stop.

Only after these gates, each mode runs direct Q32/Q64/Q64/Q32 ABBA on identical
patternB. Each graph contains4 complete steps; each role is warmed by8 graph
replays and each timed block uses16 graph replays (64 complete steps). Event
and synchronized wall time are recorded separately, with full result/cache/input
checks after every timed block. `reader` includes Q quantization, shared staging,
QK/softmax/PV and epilogue. `append-reader` also includes the unchanged writer.
No partial-loop timing is called full cost, and neither mode is a service score.

CPU commands (no frameworks/GPU imports):

```sh
python3 -B -W error tools/probe_biv150_konly_q64_screen.py
python3 -B -W error tools/check_biv150_konly_q64_screen_cpu.py \
  --raw ACTUAL_COMPILE_RAW --review ACTUAL_256C_REVIEW --references ORIGINAL_REFERENCE_FOLDER \
  --execution-sources ACTUAL_SDK_SOURCE_FOLDER --failed-raw ORIGINAL_FIRST_STOP_RAW
python3 -B -W error tools/probe_biv150_konly_q64_screen.py --cpu-preflight \
  --raw ACTUAL_COMPILE_RAW --review ACTUAL_256C_REVIEW --references ORIGINAL_REFERENCE_FOLDER
```

`ORIGINAL_REFERENCE_FOLDER` needs only the following four source files from
`e4dd64e453fbe1bdeba4d8b2a1b551167e9446c4:tools/`, each pinned in the contract:

- `biv150_konly_i8_qk_reference.py`
- `biv150_konly_decode_quant_reference.py`
- `biv150_konly_q32_numeric_contract.py`
- `bench_biv150_konly_service_q32_bitwise.py` (only its unchanged policy is loaded)

Root can stage these existing source files separately; this patch neither
copies a numerical framework nor needs historical Git objects on the device.
The actual compile raw and approved review are private inputs; do not publish
compiler binaries/logs/cache/arrays with this source change.

Explicit Root-only device template, after clean freeze and exclusive scheduling:

```sh
python3 -B -W error tools/probe_biv150_konly_q64_screen.py \
  --gpu --root-owned-exclusive --numeric-diagnostic --commit EXACT_FRONTEND_HEAD \
  --raw PRIVATE_ACTUAL_COMPILE_RAW --review PRIVATE_ACTUAL_256C_REVIEW \
  --references ORIGINAL_REFERENCE_FOLDER --out NEW_PRIVATE_OUTPUT
```

CPU checks exercise48 negative resource/identity/policy cases, reference exact
inverse, original uniform mathematics, Graph-B ordering and no-framework import.
GPU compilation, reader/writer numerics, Graph, side stream and ABBA for this new
frontend are **pending**. Long-context and1985/2017-row work is conditional on a
positive result here; no production qualification is granted.

## First execution stopped before readers; execution/diagnostic correction

The original `898076dd85788f678c664db07dbb0ba74a0baced` run stopped with
`fresh compiled identity differs for q32`, reader/writer/Graph counters all0.
Its22 original files (1895627 bytes) remain unchanged at the private raw path;
manifest `c05a64453e02915ce67b4062c3778a64b8c9fc7c94450c3414cbaed9037eff45`,
summary `9e88abb22aac73d2ac7f3ae8bc4815ec44b423d529bc8445303c15ad36a592e4`.
All fields and artifacts were equal except the supplemental ASM text:13 lines
changed constant/register/select allocation. It was not just a filename change.
The complete final cubin remained byte-identical (`d1017266…`), as did all LLVM,
TTIR/TTGIR, source, kernel hash and typed ABI. No reader numerics, timing, or new
runtime resource observation was reached. This is not a measured regression.

Read-only installed source establishes the execution boundary:

- `spec/iluvatar/triton/compiler/compiler.py`, SHA `a97f0670…`, lines361–393
  saves each stage but392–393 deliberately do not advance `module` for `asm`.
- `backends/iluvatar/compiler.py`, SHA `a0d2bbda…`, lines460–475 separately
  translate the same LLVM into ASM and cubin; line192 sets `binary_ext=cubin`.
- The same spec `CompiledKernel`, lines474–480, reads that binary into
  `self.kernel`; lines517–518 pass it to `load_binary`. The generic compiler is
  a different implementation and is not silently substituted for this proof.

The frontend now checks those actual installed file SHAs before framework
imports and records/checks the actual `CompiledKernel` implementation path.
It removes only `artifacts.asm` from execution-snapshot equality, requires that
its actual SHA still be present, and records both ASM identities. It verifies
that the actual loader argument `kernel.kernel` hashes to the exact reviewed
cubin, then retains every original resource check. There is no per-run ASM hash
allowlist. Changed LLVM/cubin/ABI/resources or missing diagnostic ASM still stop.
The real first-stop snapshot is replayed by the CPU checker; new negative cases
confirm that executable changes cannot use this diagnostic exception.

The first-stop audit remains preserved with its original pending-source-proof
status; the selected source evidence here supplements it rather than rewriting
history. The cause of the two independent ASM translations choosing different
register allocation was not separately reproduced. No device retry or new
numeric/performance claim is included in this correction.
