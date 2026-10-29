# Private multi2 Q operand hoist

Baseline: team main `56f88737fcf3b940760b9ca3027a925cd5a7d77d`.
The production encoded reader already quantizes Q outside its KV loop. This candidate
only moves the initialized QCode shared-to-dot-operand load to that loop's preheader.
It preserves BQ32/BM256, K64, 16 warps, one stage, two sequences and `total_Q // 32 + 2`
grid; the GPU Python source, codec, QK/softmax/PV arithmetic and writer remain exact.
The proposal and loader use private modules. There is no production hook or environment
flag activation, and the existing single-sequence Q32 qualification is unchanged.

The single-variable hypothesis is that actual multi2 lowering can retain Q's dot operand
across KV tiles, avoiding repeated local reads without an offsetting register/spill cost.
Actual multi2 CFG must satisfy the existing structural proof without weakening it.
Backend resource or loop-code regression can stop the experiment at compilation; this
would be a compilation-stage decision, not a measured throughput result.

## Reused implementation and dependency

`tools/biv150_konly_multi2_hoist_source.py` applies four reversible metadata edits to the
main Q32 proposal: separate schema/profile, `num_seqs=2`, and grid `+2`. Its movement,
dominance, pointer origin, memory alias, zero-trip and math checks remain unchanged.
The private metadata creator also binds the unchanged multi2 symbol/JIT-source digest.
Three diagnostic strings are renamed to describe the private two-sequence contract.

The compile frontend reuses the exact driver and ABI mapper from shared PR116 commit
`844a69f10539c551500e29e2304061aef7d8671b`, read by immutable Git object and checked
against full-file SHA-256. This object must be available in the device repository
(include it in the source bundle if absent); there is no network fetch fallback.
The driver inverse lists every edit. Changes replace the old two-geometry screen with
one multi2 unhoisted/hoisted pair, including table/length/cu-Q descriptors for two sequences.
No Q64 geometry, numeric implementation, resource qualification or result is inherited.

## Compilation and subsequent stop points

Default CPU plan: `python3 -B tools/probe_biv150_konly_multi2_hoist_compile.py`.
After Root verifies exclusivity and freezes the exact commit, the device invocation is:

```sh
python3 -B tools/probe_biv150_konly_multi2_hoist_compile.py \
  --compile-only --root-owned-exclusive --commit EXACT_FROZEN_HEAD \
  --out /tmp/cjj-multi2-operand-hoist-compile-v1-20261029
```

The existing frontend rejects a dirty/wrong source, production changes, occupied GPU,
open service port and enabled encoded service; it uses a new private compiler cache.
Compile descriptors are allocated but uninitialized. There are zero reader, writer or
encoder launches. CUDA context/allocation and exact handle loading are device API work,
so `GPU_executed` becomes true during this phase without implying numerical execution.
All original TTIR/TTGIR/LLVM/ASM/cubin, typed ABI, metadata, proposed IR and handle
resources are preserved. No actual SO/cubin/resource hash is invented in advance.

Root must inspect the final multi2 LLVM/ASM/ELF: Q operand residency in its actual KV
loop, memory ownership/synchronization, unchanged codec/math/ABI, register/shared costs
and both runtime spill and ELF scalar/vector/private fields (they are different metrics).
Do not reuse the single-sequence Q32 instruction counts or its receipt. A large resource
regression or ineffective transformation is a reason to stop before numeric screening.

If the backend is promising, reuse the existing encoded multi2 full-array reference and
fixture from the allreader numeric tools (original I64/FP64 contract; BF16 proxy remains
a diagnostic), then test changed-input Graph-B before fresh eager-B and side stream.
Only after that: direct same-input unhoisted/hoisted multi2 timing, including reader Q
quantization/staging/epilogue and separately append+reader if the existing writer is reused.
No source-stage success grants numeric, Graph, performance, model or service qualification.
Stage-2 pipelining is deferred and is not implemented here.

## CPU verification at this freeze

Python 3.12.14, stdlib only, with `-B -S -W error`:

```sh
python3 -B -S -W error tools/check_biv150_konly_multi2_hoist_cpu.py
python3 -B -S -W error tools/check_biv150_konly_multi2_hoist_cpu.py \
  --actual-ttgir PATH_TO_ARCHIVED_B/kernel_cjj_konly_multi2.ttgir \
  --report /private/tmp/cjj-agent-multi2-operand-hoist-cpu-20261029.json
```

The optional regression input is only accepted at SHA-256
`bd9c7c5178d0db8a41f6f45c39f3d17a71a375b36645173ee7796e6343a51fe5`, from the
existing B service archive (`5ca2afaa91bc9133d4170a22410daef4e22f5e00`, production
prefill source identical to main56). Raw compiler artifacts are not included here.
The CPU-selected 2048-row metadata tests the intended compile contract; it does not
claim those were the archived request lengths or establish a new actual compiled object.

The actual multi2 IR satisfies the unchanged proof: Q global read 277, quantized cast
311, initialized shared allocation 312, old KV loop 352–466, operand load 421,
QK 423 and PV 464. The proposal moves only the operand load to line 352; the loop
header becomes 353. Full IR byte reversal succeeds. The proposed IR SHA is
`e74096adb7a98af2a8240f023acb2bf5b6d0789874e463f11adfa9778af62484`.
This is a CPU proposal, not the output of the new device compiler.

Default contract checks: 51 passed. With the archived actual multi2 IR: 58 passed,
including default-OFF/no-SDK import, original proof functions, private scope/ABI,
wrong-freeze/ownership rejection, and actual-IR repeat-hoist/alias/operand rejection.
CPU report SHA-256: `fb26b14c98a8df7877d6f149902b796cdcf79c92173c2b2fac493e01a0e18d08`.
Plan SHA-256: `99afedbd46aff04a21ac3212f384561a8c231760e93b24cd5b4370c791b7b6cb`.
Actual compilation, resource review, numeric/Graph/side-stream checks, component timing,
Math105 and formal model throughput remain unperformed for this candidate.

Suggested Draft title: `BI-V150: screen private multi2 Q operand residency`.
Dependency: immutable compile helpers in PR116/`844a`; no dependency on its Q64 result
or any service change. The proposed experiment remains private and default OFF.
