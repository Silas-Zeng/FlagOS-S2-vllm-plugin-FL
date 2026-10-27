# Same-main native / Triton GQA operator diagnostic

Fresh team mainb3b87d2, independent own branch. Root-only future device execution;
author performs no SSH/JIT/GPU/build. Draft prepares one bounded pair, not model
profiling. Main4K fullwarm6449.65 versus score5655.19 is an observation, not a
proven28-second JIT explanation: only two new prefill compiler file spans total
about2.8s, selected SQLite rows also changed and timestamps do not isolate cost.

Two arms use the actual same b3 producer math and original reducer: actual native
S2/S4 SO content SHAaace693… versus direct exact b3 Triton GQA (no native dispatch).
Both include the same previously tested ENCODE=False BF16 append. K8 is excluded
from the pair; its old measurements cannot be mixed into this new native control.

One fixed34276-page physical16KiB arena matches the548416-token layer budget.
The deterministic no-overlap permutation seed20261028 maps active4K/B64 pages
16384 and16K/B32 pages32768 across that arena; table row stride8192. Each step
appends from length-1 to post-append length, including page/tail boundaries.
Resident inputs/cache, actual ABI, full outputs/guards/readonly and independent
CPU FP64 gates precede one warmed ABBA. Same-arm repeat/Graph outputs must be
bit exact; cross-arm bit equality is recorded, while both must independently
pass original.005+.005|reference| pointwise and.005RRMSE gates. Initial BF16 pack
copy cost is separate; no model/TTFT/service gain is inferred.

The explicit device runner is now prepared. Default execution and the CPU checker
import only stdlib. Device entry requires exact clean commit, fixed aace library
SHA, unchanged b3 production source,9031 closed and low GPU memory; the namespace
process table alone never establishes exclusivity. Root still owns scheduling.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_tl_pair_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_native_tl_pair.py
python3 tools/bench_biv150_native_tl_pair.py --run-device --commit FROZEN_HEAD \
  --library /workspace/experiments/minicpm-s2/cjj-main-b3-native-gqa-build-v1-20261028/corex_native_gqa_s2_probe.so \
  --out NEW_ABSOLUTE_OUT
```

Without --case it executes both exact cases, two numeric controls each (dense and
ragged/empty/page tails), then one dense ABBA per case. It preserves full output
and partial bits, FP64 references, q/newKV/table/slots in compressed NPZ, actual
Triton source/IR/ASM/cubin/typed ABI and native source/SO/descriptor identities.
Half-GB caches are represented by deterministic CPU seed, complete SHA and append
oracle instead of archived full copies. The CPU input generator is explicitly
Torch CPU BF16 with seed20261023+B+length and its actual version recorded; this
fixed larger arena replaces old small active-only fixtures, so old timings are
not paired observations. Source/destination initialization is separate.

Timing captures the complete same BF16 append+producer+original reducer4 into
resident CUDAGraphs (unroll8), checks full Graph/eager bits, warms each128replays,
then records a single Native/TL/TL/Native sequence of64replays each. Both GPU event
cost and Graph host-launch/synchronization wall cost are reported. Python normal
producer dispatch is outside this captured operator timing. This cannot explain
all model/CPU/SQLite/JIT/cache effects. Initial GPU pack-copy is separate; host
random generation/H2D/reference/validation are setup, not a free service encoder.
No new timing or numeric result exists until Root executes this frozen source.

First28d748 device attempt (Root owner688424/start347203865) passed the complete
append-cache SHA check, then failed before any attention/ABBA at int64 slots
serialization: bits() lacked element_size8 and raised KeyError8. Original failure
is retained; it is not a numerical or performance result. The next tools-only
revision adds torch.int64 byte reinterpretation and a stdlib recorder regression
for signed sentinels/extremes. Inputs/layout/reference/timing gates are unchanged.

## Existing Root v2 result (CPU audit only)

Source36c5b0b, owner688719/start347244779 is terminalZ. All94 original files /
15,217,427 bytes independently match the download manifest. The actual local
native SO payload matches aace693b…; source-function SHA and complete actual
Triton typed ABI/artifact hashes match this exact b3 control. No device rerun.

| Resident case | Native complete step | Triton complete step | TL/Native |
|---|---:|---:|---:|
| 4K/B64/S2 |0.4163701534ms|0.5278745294ms|1.26780108x|
| 16K/B32/S4 |0.8058110178ms|1.0312613845ms|1.27978070x|

These include identical BF16 append, producer and original reducer4. Each row is
one ABBA with two observations per arm, after128warmreplays (unroll8). Within-arm
range/mean is0.02363%/0.02717% (Native/TL) at4K and0.00857%/0.02826% at16K.
Graph launch/sync wall means reproduce1.26773573x/1.27975935x. Initial same535.56MiB
arena GPU pack-copy is separately1.92941/1.92833ms. Normal Python producer dispatch,
model layers/prefill/scheduling/SQL and initialization are outside this captured
operator timing. No confidence interval, whole-model gain or explanation of the
14% warm/score difference is claimed; K8 is not an arm.

All four NPZ controls independently show exact Native/TL bits for full output,
selected partial scratch/maxima/sums. All8 saved-full-FP64 metric comparisons
recompute with zero pointwise violations, maximum RRMSE0.0022802426 (old gates
unchanged). Additionally, CPU FP64 segment composition is regenerated from every
saved partial, and length0/1 closed attention is regenerated from saved newV.
Native producer resource attributes are unknown; actual loaded Triton producer /
reducer / append are93/11/6 registers respectively, all0spill.

Evidence boundaries are explicit. Local Python environments lack Torch2.10;
therefore the535MiB seeded cold arena and original closed-attention FP64 array
were **not independently regenerated from seed**. Metrics against the original
saved FP64 array are independently recomputed. Outer guard buffers, same-arm
second-run and Graph-vs-eager separate snapshots and full readonly/cache payloads
were not stored; their successful source assertions/runtime flags are retained,
not presented as an independent CPU replay of absent arrays. Selected partial
finite/NaN masks, complete cross-arm arrays, table/permutation/slot geometry and
actual source/SO/ABI are independently replayed. Original raw files remain in
Root's archive; this commit shares small unchanged result/manifest/receipt plus
CPU summary, avoiding another large cache/artifact package.

The original v1 KeyError8 failure result is preserved separately and all its
available files were SHA-checked; it contains no attention or timing result.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_tl_pair_evidence.py --arrays
```

This reader accepts --raw-root and --library for the original immutable records.
Default paths point to Root's existing local raw/build archives. --arrays uses
NumPy CPU only; no Torch/Triton/service/JIT/GPU is imported or started. Summary:
`docs/evidence/native-tl-gqa-pair-20261028/result-v2/independent-cpu-replay.json`.
