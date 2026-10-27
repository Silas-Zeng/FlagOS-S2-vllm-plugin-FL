# Private Q32 changed-input Graph and side-stream frontend

This new entry defaults to CPU and does not edit2d79's source, codec, numerical
policy or production. It reuses the verified three-role Q32 reader bindings,
2048 query-row allocation and4096-prefix capacity (6144 total-KV capacity).
There is no timer, service hook, queue,16K path or local device invocation.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_graph_side_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_q32_graph_side.py
# Root alone after choosing its serial GPU slot:
PYTHONDONTWRITEBYTECODE=1 python3 tools/bench_biv150_konly_q32_graph_side.py --run-gpu --numeric-diagnostic \
  --commit EXACT_NEW_GRAPH_COMMIT --out FRESH_OUT \
  --source-root /workspace/experiments/minicpm-s2/cjj-konly-prefill-q32-compile-v1-20261027/sources
```

Pattern0 first runs real eager attention, then captures one Graph per role using
the same Tensor objects and already-validated JIT specialization. Pattern1 changes
the complete Q payload, rotates the384 valid physical page-table entries, and
changes runtime length6144→6143 before any first replay. The allocations/grid and
formal batchTokens stay fixed; the mutable-length boundary has4095 actual prior
context, within the original4096 capacity. Table padding remains unchanged.

Each first Graph replay sees the new data before any fresh eager on that pattern.
The live output was already poisoned by the parent. Fresh eager and side stream
are separately NaN-poisoned, and all three full outputs must match bitwise for
each role. Stream dependencies explicitly order input writes and result reads.
The parent's independent codec, baseline/quantized FP64, finite, full I8-hoist
output equality, canary and read-only-input checks still run. Original proxy
precision failures retain their false labels under explicit numeric diagnostic.
The CPU FP64 reference only substitutes runtime length/table coverage for its
old geometry constants, with a byte-inverse proof; mathematical expressions and
numeric gate bytes stay fixed.

The frontend clones only the pinned private host runner and removes exactly its
prelaunch resource-count rejection, with exact byte inverse. Unknown/nonzero
spill counts are observations, not a numerical hard gate. Every real eager,
Graph-replay and side-stream attention then reads n_regs/n_spills/metadata directly
from the original returned/captured CompiledKernel object, recording whether it
is the same object as warmup. No resource count is adjusted or fabricated, and
no extra run is scheduled to obtain a preferred count. Capture API calls and
actual execution modes have separate counters; capture does not imply execution.

All per-mode full outputs, changed-control digests, postlaunch resource fields
and extra native-PID checks are saved in fresh private outputs. Original fixed
source/debug roots/override are preserved. The original whole-job empty→owned
native-PID transition remains; every additional mode also requires the same host
PID/container identity/root binding/token and closed9031. Existing output paths
are refused without touching their evidence. Host adapters restore on exit and
never install a service or edit installed SDK modules.

CPU checks cover source/reference inverses, metadata rotation/tail boundary,
existing-output refusal, foreign/namespace/stale-token rejection, adapter restore
and observation of unknown/nonzero resources. Actual Graph/side/numeric results
are untested until Root explicitly runs this entry. Full validation, formal
precision, performance and service remain false. Any later performance work
must include the accepted main BF16 Q-hoisted control; the d6 raw98304B control
used here is a semantic comparison and cannot stand for service performance.
