# Private M2/raw3D first numerical feeds

This separate CPU-prepared frontend keeps the24c/d6/e256 source, kernels,
encoder, original reducer and numerical thresholds unchanged. It has no service
hook, Graph, side stream or timing. Root alone may explicitly invoke device work.

M2 uses total Q2048 with two heterogeneous sequences: Q=[2047,1], KV=[4096,33]
and Q=[2046,2], KV=[4095,17]. The second sequence crosses the query prefix and
checks qlen1/2 mapping. This is a private component geometry, not a change to
formal batchTokens. The old actual compile used Q4096 and is never called a
service performance baseline.

raw3D uses64 sequences, all qlen1,16 segments and ragged KV lengths including
1,2,3,15,16,17,31,32,33,63,64,65. It preserves token-indexed scratch/max/sum
strides and launches the original SDK reducer. Each role has independent
workspace with suffix guards; active segments must be finite with positive
exp sums, while inactive segments remain NaN. This first scope does not license
raw qlen2: the frozen CPU descriptor contract excludes it. A later private
metadata/address scope and causal-empty segment check are required for qlen2.

The exact40 M2 and55 raw3D actual compile files, plus their manifests, are
preserved in two SHA-bound archives below2MiB each. Original d6 flags, including
false I8-hoist/binding/precision/performance, stay untouched. Runtime imports the
original remote tag/sources directory, not a new checkout pretending to be it.
All actual roles' source/TTIR/TTGIR/LLVM/ASM/cubin, metadata, typed ABI and
differences are saved before binding checks. Source/ABI/options/TTIR/TTGIR/LLVM
must match those reviewed originals; actual ASM opcode/control/synchronization
and native I8-QK/BF16-PV counts must match. Actual cubins are retained and hashed,
without asserting cross-compile binary equivalence. Unknown specialization
stops for review. No Q32-only hoist is borrowed by M2 or raw3D.

Two input patterns run per case. Independent CPU BF16 inputs generate FP32
RN32/RNE codes/scales/full packed bytes; the real encoder must match all bytes.
The uniform-Q pattern checks every output row with closed FP64 prefix averages.
Random Q uses selected long-sequence rows and all short rows, independent INT64
QK/FP64 global softmax and BF16 V. Closed references do not model BF16 P or
online-softmax rounding. All full outputs must be finite; canaries, read-only
inputs, codec, actual IR, baseline FP64 and quantized-reference pointwise gates
are mandatory. The unchanged original-BF16 proxy may remain false only with
explicit --numeric-diagnostic; this never bypasses an implementation failure.
CPU originals, expected/actual encoded cache, codes/scales, selected references,
full outputs and raw3D workspace are saved losslessly before numerical gating.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_allreader_numeric_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_allreader_numeric.py --profile multi2
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_allreader_numeric.py --profile raw3d
# Root only, after its exclusive slot is idle:
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_allreader_numeric.py \
  --profile multi2 --run-gpu --numeric-diagnostic --commit EXACT_FRONTEND_COMMIT \
  --out FRESH_OUT --source-root /workspace/experiments/minicpm-s2/cjj-konly-multi2-compile-v1-20261027/sources
# raw3d uses --profile raw3d and the corresponding original directory:
# /workspace/experiments/minicpm-s2/cjj-konly-raw3d-compile-v1-20261027/sources
```

Whole-job empty→owned native-host-PID admission precedes CUDA initialization.
One tiny initialization is recorded outside numerics; each phase requires the
same native singleton/container PID/start/token/roots and closed9031. Nothing
is queued, stopped or deployed. Loaded resources are observed after actual
launches without forcing zero spills. writer_ms/fullcost_ms=null, timing=false,
precision_gate=false, capacity_reclaimed=false. No local SSH/JIT/GPU was used.
