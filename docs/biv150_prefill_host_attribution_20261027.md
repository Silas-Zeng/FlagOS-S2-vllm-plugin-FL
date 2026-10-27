# Bounded actual-host attribution diagnostic

The5967 component retained a4K event median1.039341 but wall ratios
.980065/1.148599/.869846, with roughly.2ms discrete wall gaps. Its two requested
cases and actual certificate passed; full numerical matrix/model/service gates
remain unverified. All75 originals are preserved separately at evidence
commit3941971, archive SHA3fd8d1ba36054181c8cd7887f7f53f1c097c07188fe6af554b32146c50f28672.
This entry collects missing CPU evidence without sweeping additional score rounds.

`tools/diagnose_biv150_prefill_host_overhead.py` defaults to CPU preflight/plan.
It imports no Torch/Triton and executes no actual host profile by default.
`--self-test` explicitly tests only the stdlib observer's success/error cleanup.
The initial593 observer-only3.12.15 and3.14 tests passed, with artifacts in
`evidence/prefill-reverse-20261027/host-attribution-v1/`: default-plan SHA
1465a9090a67eb742c5375a39a12a4bbd15cf24e034a011f10fd81276e777530,
self-test-plan SHA77637a4ac944c512461da541c41e11b17c084653556bbf86246558783ed8b841.
The corrected resource-schema tests and current plan are recorded below. The
existing CPU report remains30dc23a4….

Explicit `--run-gpu` is solely for root's later serialized diagnostic. It runs
one baseline/candidate block, each3 warm calls then2 profiled calls by default
(bounded1..4). The input defaults to query2048/KV4096; there are no timing-round
sweeps or speedup calculations. The original benchmark is untouched. Its startup
statements are copied literally and checked for AST equality, retaining source,
package/native identity, GPU-idle/port, M2/Q32 chain and scope qualification guards.
The same original SDK function, parent policy and LocalSdkBinding are invoked.
Warmed typed ABI/options/target, Q-hoist, observed resource equality, actual route,
output exactness/canaries, ownership and native identity are checked. This does
not rerun the complete Graph/side/numerical matrix or grant component acceptance.

CPU preflight:

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python tools/diagnose_biv150_prefill_host_overhead.py \
  --source "$PWD" \
  --sdk-source /usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py \
  --helpers-source /usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_attention_helpers.py \
  --target-sources docs/evidence/prefill-reverse-20261027/host-binding-v1/actual-target-sources.tar.gz \
  --self-test \
  --out /tmp/NEW_CPU_PLAN
```

For a later actual diagnostic, root can use the same command with a fresh output
directory and add `--run-gpu --commit FROZEN_HEAD --native-library ORIGINAL_5C_SO
--calls 2`. This work does not launch or queue it. Initial device behavior remains
unverified until root runs the frozen driver.

## Measurements and limits

- cProfile records actual original frames. Exclusive self-time buckets separate
  parent policy, certificate, descriptor, route/one-shot, host scope, Manager,
  SDK signature binding/normalization, Triton binder/launcher and native sync.
  All function records and per-arm cProfile files are retained.
- A narrow line observer measures the postlaunch Boundary audit span without
  wrapping/replacing methods, preserving the actual-parent-frame witness.
  Its span overlaps Boundary self time; it is not an additional additive bucket.
  The observer also counts live descriptor calls by block_m.
- Explicit timestamps retain the benchmark's charged placement: before begin
  record, SDK calls, end record, end synchronize. Initial stream synchronization
  remains outside that window. Calls are fewer because this is a diagnostic.
- gc callbacks preserve enabled state, thresholds and existing callbacks. No
  disable, forced collection or threshold change occurs. Pause generation/thread
  and timestamps are recorded. Observer callbacks and profiling still perturb
  allocation/GC; they are removed on normal and exceptional exits.
- Process and optional thread rusage capture CPU time, faults and voluntary/
  involuntary context switches. Bookends include observer setup/cleanup and are
  broader than the explicit wall window; maxrss is a high-water observation.
  These counters do not directly identify a scheduler or syscall cause.

Instrumentation perturbs host execution, GC and GPU enqueue gaps. Short warm
prehistory also differs from the full correctness harness. These durations are
not performance scores and cannot retrospectively prove the cause of an older
round. No ratio is produced, no mandatory check is deferred, and no service hook
or optimization is added by this diagnostic.

## Audit semantics and lifecycle before any further optimization

| Work | Current role/lifetime | Treatment here |
|---|---|---|
| Original parent/Manager descriptor, capture, async/hook guards | Per-call runtime qualification of mutable state | Retained |
| Certificate source/full-module compilation | Installation qualification; invalidates on code/default/scope change | Retained |
| Fresh CpuContract/Frame/Call/lease and late-execution checks | Current one-shot runtime protocol, not just reporting | Retained |
| Boundary.last and Boundary.calls | Postlaunch component evidence; last holds Tensor references until replacement; list bounded512 and consumed by acceptance checks | Every record remains generated in charged calls |
| Verified Manager compile/Q-hoist/ABI and first loaded resources | Opaque compile/launch acceptance and provenance | Retained |

Boundary.last/calls are not read to choose the kernel: they are written after
selected.run returns. Their cost may be diagnostic-only for a future service
integration, but this component currently consumes them for acceptance. Turning
them off only for timed calls would change the experiment and is not done.
A future service design must specify evidence emission and retention consistently
for both correctness and timing before changing that lifecycle. Repeated runtime
guards cannot be removed merely because a profile identifies their cost.

The goal remains retaining the kernel benefit through a valid real-service
integration. A successful diagnostic is not model105, official throughput/TTFT,
full-matrix, or new host-performance acceptance.

## Initial593 diagnostic resource-schema refusal

Root's first diagnostic owner627340/start344164972 ended same-start Z. It warmed
the baseline, then safely rejected the resource check before any profiled arm:
raw Manager.loaded_resources dictionaries lack the explicit
post_launch_resource_evidence consumed by compare_resource_records. Actual counts
113/0 alone must not imply an observed phase. The original diagnostic retains
exactly its AssertionError, empty arms, parent_restored=true and
diagnostic_checks_passed=false; it contains no profile/score result.

All26 originals1272541B plus download manifest are preserved unchanged in
`host-attribution-resource-v2/failed-v1-raw.tar.gz`, 178239B, SHA256
`e6179bd686d1c2f463bbad67fc493d9bebe06031dce6df5cd462e5103f439d68`.
Every file SHA and unpacked archive byte was independently verified. Source593
and its old plans remain in history; the source on the target is not overwritten.

The correction follows the original bench: resource_evidence(snapshot,
manager_record) verifies pre/post/candidate hash, metadata and ASM identity and
attaches its result as post_launch_resource_evidence. It uses the actual
Manager-loaded phase, without inventing already_launched or weakening unknowns.
The existing strict comparator is unchanged. CPU self-test now replays the exact
preserved5967 parent/reverse records, then rejects unknown counts and a changed
manager binary hash. The real archive is read only as JSON; no SO/cubin is loaded.

Corrected3.12.15/3.14 self-test output matches byte-for-byte in
`host-attribution-resource-v2/`: default-plan SHA
`c2e4a7e42217527cb9baa59f178e4d67d2e0d971ee9b48b2e51f1ada8535fd5a`,
self-test-plan SHA `2c4b6a5db9bf670c79a247e0ce80abdb13546191fd7fd959f763263d0c599210`.
The self-test requires the committed host-binding-completed-v4/v4-raw.tar.gz.
Bootstrap, timing placement, observer, scope/runtime guards and original benchmark
are unchanged. Actual diagnostic execution remains for root; no device task is
started or queued here.
