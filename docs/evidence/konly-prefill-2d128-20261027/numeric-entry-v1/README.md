# Bounded eager BM128 consumer entry: CPU-prepared, not executed

This follows the independently reviewed actual78e zero-launch compile archived
in `../compile-v1`. The old factory/kernel/compile entry/source/actual originals
are unchanged. No TTGIR override or operand hoist is added: actual BM128 Q shared
operand reads are known to remain in the KV loop. Production remains DAC bytes.

The new entry is default stdlib CPU. Explicit Root-only `--run-gpu` prepares
exactly the existing two input patterns, Q2048 after4096 prefix/KV6144,773 pages,
BM128/Q16/K64/w8/s1/grid(129,2). It compares two same-BM128 private functions:
original BF16 control and unhoisted I8-QK candidate with unchanged BF16 V/PV.
The control is **not actual-main Q32 performance policy** for2048 rows. There
is no timing code, shape scan, service hook, public quantization flag or installer.

The device source directory must be the original78e compile directory inferred
from its actual TTIR debug locations, preserving both source bytes and file
paths. Clean exact frontend HEAD, DAC production equality, SDK/helper/JIT disk
pins, actual loaded SDK/helper source identity and corex71/warp64 are checked.
Both warm compilations preserve all actual artifacts before any reader launch.
Actual hash/typed ABI/options/JIT-version and source/TTIR/TTGIR/LLVM must match
the reviewed records. Actual candidate mechanism and baseline native BF16 paths
are reaudited, and ASM control/synchronization must match. ASM/cubin originals
are saved; printer/native-binary byte equality is reported rather than asserted
as backend equivalence. The same numerical hard gates remain required.
Handles must initialize with known positive registers and zero spills for both
roles; the original prelaunch None records are not substituted for this gate.

Each pattern constructs an independent original BF16 CPU cache and query,
then uses the unchanged RN32/RNE CPU codec to form a full physical encoded
cache. It uploads that cache and verifies full readback bytes. There are
**0 GPU encoder launches and4 private reader launches**. The native-PID
admission protocol performs one separate `torch.zeros(1,cuda)+synchronize`
initialization, verifies a fresh sole native host PID, then guards exact source,
owner/start, closed9031 and that same sole native PID before every phase. Thus
this eager entry is not zero-device-work, unlike the historical compile entry.
Its two-role phase sequence is the sole change from the copied PR80 protocol.

Every output starts as NaN with a live BF16 token of value71 on both sides.
All full outputs must be finite; guards and readonly input/cache/metadata hashes
must remain unchanged. Pattern0 additionally checks all2048 query rows/head/
channels against the existing independent uniform-causal cumulative reference.
Random pattern1 and both patterns' selected rows0,1,15,16,31,32,1024,2047 use
two distinct closed FP64 references: original BF16 Q/K/V attention, and CPU
RN32/RNE Q/K integer-code dot (INT64), scales, global FP64 softmax and BF16 V.
The latter does not model online-softmax or BF16 probability rounding; original
thresholds remain unchanged. Raw original/quantized errors are separate.

The same pointwise0.005+0.005*abs(reference) implementation threshold is retained.
Original BF16 relative RMSE must be<=max(0.005,1.2*baseline relative RMSE), as in
frozen80. Nonfinite, original baseline/reference failure, CPU cache mismatch,
implementation/quantized-reference failure, actual IR/resource, readonly or
canary failure is hard and cannot be bypassed. Only the original BF16 quality
miss can continue under explicit `--numeric-diagnostic`, always retaining its
false numeric result. A diagnostic completion never sets precision_gate=true.

Lossless per-pattern artifacts preserve original CPU inputs, full actual encoded
cache readback, both references, both full outputs, actual output-canary rows
and before/after readonly hash observations. Readonly comparisons are reported
as hash observations, not independent saved post-read input images. An error
preserves available artifact/result/compile records and exits nonzero.

CPU3.12.15 and3.14 default-plan/checker outputs match byte-for-byte. The checker
uses real saved records for strict binding, verifies separate quality versus
implementation gates, and rejects21 ABI/options/JIT/hash, hard-gate, diagnostic,
foreign native PID, port, stale owner, token and phase faults. It imports no
Torch/Triton/NumPy and makes no device call. CPU is no proof of actual typed
binding, numerical accuracy, loaded resources or fixture GPU execution.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_2d128_numeric_cpu.py --out /tmp/NEW-numeric-cpu.json
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/bench_biv150_konly_2d128_numeric.py
```

Root may later run from a clean frozen checkout only after its prior workload
and controller are terminal/cleaned and GPU/service idle. Use a fresh output:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/bench_biv150_konly_2d128_numeric.py --run-gpu --numeric-diagnostic \
  --commit EXACT_NEW_FRONTEND_HEAD \
  --source-root /workspace/experiments/minicpm-s2/cjj-konly-prefill-2d128-compile-v1-20261027/sources \
  --out FRESH_OUTSIDE_CHECKOUT
```

This tests only the two requested eager geometries/patterns. Short-query/qlen1/
tails/multiseq,16K, Graph/side, all service readers, codec writer integration,
fused RoPE, cache lifecycle/freshness, model105 and throughput/TTFT remain future
gates. full_numerical_matrix_verified=false, precision_gate=false and model/
service/performance=false. Source preparation does not deploy or accept K-only.
