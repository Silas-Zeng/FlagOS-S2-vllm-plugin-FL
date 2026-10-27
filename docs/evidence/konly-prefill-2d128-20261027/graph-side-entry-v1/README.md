# CPU-prepared changed-input BM128 Graph/side entry

This adds only new tools over frozenf04. The original78e reader/factory, original
ef4 failure,4d indexing probe and f04 host/numerical/resource gates are unchanged.
It does not introduce any hoist/kernel transform, timer, installer or service hook.

Root f04 eager v2 completed with4 reader/0encoder launches and both patterns'
7 hard gates true. Independent CPU read of all40 raw original hashes and all
four full8,396,800B pre/post guarded-buffer pairs found: both complete71 guards
unchanged, pre interior entirelyNaN, post interior entirelyfinite and byte-equal
to saved full outputs, saved canaries equal to the actual full-buffer boundaries,
and all8 readonly hashes unchanged. This establishes only that new f04 run's
boundaries. It cannot prove the oldef4 actual suffix was unmodified.
Pattern1 quantized-implementation relative RMSE is0.0022699815155583425; original
BF16 relative RMSE is0.009644218032913768, so original quality remains false.
Root may provide this actual f04 summary as the mandatory eager precondition;
no Graph/model/service/performance qualification follows from it.

Default execution is stdlib CPU. Explicit device mode validates the actual f04
completed/errors/count/binding/resources and both exact7-hard-gate records,
allowing a separately recorded original-BF16 diagnostic miss. It refuses the old
ef4 failure, missing gates or unearned precision. It pins the f04 host source,
resource gate, numerical contract, admission protocol and shared fixture bytes.
No original host source is rewritten. Scoped Python loader/reference/input
adapters restore all hooks even on exception.

Both original BF16 control and unhoisted I8-QK candidate retain fixed
Q2048/prefix-capacity4096/KV-capacity6144,773 physical pages,
BM128/Q16/K64/w8/s1/grid(129,2). Pattern0 runs actual eager, then captures each
role's warmed kernel and stable Tensor addresses. Pattern1 changes Q/cache data,
rotates the384 unique physical pages in the padded table and changes actual
GPU runtime KV length6144→6143 (prior context4096→4095, partial last slot).
Query cu_q stays[0,2048]; no changed query metadata is invented. Before replay,
actual GPU Q/table/length full-backing hashes must equal their new CPU sources
and differ from each role's capture data. Descriptors, I8 K/F32 full-base scale/
already-offset BF16 V pointers and all original source/IR/resource guards remain.

For changed input each role must execute in exact order **Graph replay first,
then fresh eager, then explicit side stream**. Full outputs must be byte-identical
across these three modes. Every mode immediately checks finite full output,
full CPU-buffer71 guards and readonly full-backing hashes before another mode
can overwrite anything. Whole guarded post-mode CPU images, full outputs,
canaries and hash observations are saved. GPU advanced negative indexing is
never used to read guards. Genuine0D FP32 descales are fingerprinted through a
metadata-only full-storage typed vector, avoiding scalar byte-view failure or
pitched reshape copies. Side stream explicitly waits for current work and
records Tensor use; current stream waits before observation.

The actual returned/captured kernel hash must match warmup; actual resource
fields must be known/zero-spill after real attention. Unlike the older Q32
prototype, **no resource guard is removed**. The same quantized INT64-QK/FP64
reference and original BF16 FP64 quality gates apply with runtime-aware6143
geometry, and pattern0 retains the full2048-row uniform causal reference.
Changed Graph/eager/side equality propagates those numerical checks to all
three outputs. Nonfinite/implementation/canary/readonly/IR/resource/source/
PID/metadata failures remain hard; only original BF16 quality can remain a
false diagnostic under explicit `--numeric-diagnostic`.

Operation counts are reported by mode:4 eager calls,2 capture calls,2 replay
calls and2 side calls. Capture calls are not claimed as actual attention
executions. The inner f04 legacy frontend has4 wrapper entries; it is not a
claim that this Graph job has only4 attention operations. There is still0 GPU
encoder work, plus the unchanged separately accounted tiny Torch/native-PID
initialization. No timing is performed and actual BM128 hoist remains false.

CPU3.12/3.14 contract/plan outputs match. The contract checks22 malformed
metadata/eager-gate/mode-order/foreign-PID/control-hash/pointer faults, exact hook
restoration,0D-scalar full-storage addressing and readonly backing deduplication.
It runs no framework/JIT/device operation and is no actual Graph proof.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_2d128_graph_side_cpu.py --out /tmp/NEW-graph-cpu.json
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/bench_biv150_konly_2d128_graph_side.py
```

Root only, from clean exact freeze with a fresh external out and idle GPU/service:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/bench_biv150_konly_2d128_graph_side.py --run-gpu --numeric-diagnostic \
  --commit EXACT_GRAPH_FRONTEND_HEAD \
  --source-root /workspace/experiments/minicpm-s2/cjj-konly-prefill-2d128-compile-v1-20261027/sources \
  --eager-summary /workspace/experiments/minicpm-s2/cjj-konly-prefill-2d128-numeric-v2-20261027/numeric-summary.json \
  --out FRESH_OUTSIDE_CHECKOUT
```

No device execution has been done by this preparation. This remains a bounded
one-sequence two-pattern component. Full numerical matrix, arbitrary service
shape/format lifecycle/Graph epoch, all-reader/writer integration, model105,
throughput/TTFT and merge eligibility remain unverified; precision_gate=false.
