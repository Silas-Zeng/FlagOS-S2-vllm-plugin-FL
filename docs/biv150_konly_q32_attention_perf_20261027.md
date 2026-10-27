# CPU-prepared attention-only comparison against actual DAC1 main

This independent entry does not edit the frozen2d/d2 readers, codecs or numerical
gates. Its accepted-main BASE is DAC1
`dac1b8fb48e7983e48283032efeee9b592c5fbe5`, requiring a separate clean checkout and
actual imported plugin path. It checks four relevant attention source files are
unchanged from the56f evidence baseline; it does not call56f the newest main.
DAC1's worker/compile-semantics/compute-order/LM-head changes are not claimed to
be service-equivalent or exercised by this attention component.

The materializer follows PR74's actual parent-control method: install accepted
baseline/M2/Q32 in the private process, call the real high-level SDK attention
with BF16 K/V, and capture the Q32 parent-selected arguments and CompiledKernel.
The hash must exist in the actual main Manager.optimized table with selected
mode and BF16 lowered_proof. The actual normalized source body must equal its
audited source; the actual source/IR/cubin/typed ABI/options and Manager proof
are exported. No canonical B256 raw arguments are sent through the SDK B128
entry. The d6 raw96KB kernel remains only a semantic output comparison.

Only after per-length two-pattern checks may timing run: independent codec/full
cache bytes, full I8 hoisted/unhoisted bitwise, actual-main/d6-BF16 full bitwise,
finite, canary/read-only guards, main FP64 baseline and quantized FP64 pointwise
implementation must pass. An original-BF16 proxy failure remains false and
requires explicit --numeric-diagnostic. Implementation/baseline failures cannot
use that diagnostic permission. Candidate actual IR proof also reruns for this
length; old4K evidence does not license16K. Unknown specialization stops with
actual exports retained. Original component numerical thresholds are unchanged.

One explicit KV length runs per invocation:4096/6144/16384, always2048 queries.
Those mean2048/4096/14336 preceding context tokens. Default6144 reproduces the
existing4096-prefix capacity. Query batchTokens are unchanged. Each invocation
freshly validates its actual length before any event measurement.16K is not
claimed tested by CPU preparation or earlier6144 data.

The timed controls are only the materialized actual main BF16 Q-hoisted binary
and the proved I8 Q-hoisted binary. Direct warmed JIT capture preserves the main
selected binary, avoiding the SDK wrapper's intentional generic capture fallback.
Each Graph must first match fresh eager output and preserve canaries/inputs.
One fixed warmup (four paired replays) precedes one AB/BA event diagnostic
round (64 replays per sample). No cache flush, cache setting or parameter sweep
is added. SDK host-dispatch, startup, compilation, whole-cache encoder and writer
costs are excluded: writer_ms=null, fullcost_ms=null, service_performance=false.
The encoder is all-page cold setup, never labeled a production incremental writer.

Records include actual cache allocations, logical KV payload once, query bytes,
strides/table pitch, optional device L2 property and unflushed fixed-fixture
replay scope. Logical payload is not measured traffic. A resident repeated
fixture cannot establish HBM or service speed. Resources are read from actual
postlaunch objects without forcing a spill count. A whole-job empty→owned native
PID transition and exact phase checks reject occupied GPU/9031, changed roots or
foreign/stale/missing PID; nothing is queued or stopped.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_attention_perf_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_q32_attention_perf.py --kv-length 16384
# Root after105 releases its serial GPU slot: materialization/numerics/capture only
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_q32_attention_perf.py --run-gpu --prepare-only --numeric-diagnostic \
  --kv-length 6144 --commit EXACT_NEW_FRONTEND_COMMIT --out FRESH_OUT \
  --main-root CLEAN_DAC1_CHECKOUT \
  --source-root /workspace/experiments/minicpm-s2/cjj-konly-prefill-q32-compile-v1-20261027/sources
# Or replace --prepare-only by --attention-only-diagnostic for the one bounded
# timing round, only after the same invocation's actual implementation gates.
```

Default is CPU only, no Torch/Triton import. There has been no local SSH/JIT/GPU,
queue, main materialization or timing. CPU tests cover incorrect parent/fallback
proof, wrong main root, per-length geometry, baseline/implementation/finite hard
failure, explicit diagnostic permission and native namespace/token rejection.
No model105, formal precision, full-cost or service pass is claimed.
