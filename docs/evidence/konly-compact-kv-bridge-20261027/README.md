# Compact K-only KV bridge: CPU source candidate

**Current source stage:** default-off service wiring now follows native-GQA main
`b3b87d2` and immutable PR89 `d8ef055`, with ownership routing before stock native;
see [service-wiring-b3-native.md](service-wiring-b3-native.md).
The complete installer passes explicit CPU metadata mocks. Actual Tensor/IR/
numeric/model/capacity/throughput remain unverified. The DAC1/2487 report and
phase-one `cpu.json` below are preserved historical stages, not new-main evidence.

## Historical phase-one study (69ac65d)

Base: fresh fetched main `dac1b8fb48e7983e48283032efeee9b592c5fbe5`.
Depends on PR88 contract `65db624d39f554f2330fef578930c4048a109f06` and PR89
encoded service `5b228e2c6e6cf0c4365cf0595c340f83b827e470`. This branch does not
change either worktree, production source, service, GPU runner, or device queue.
It does not use PR89's subsequent uncommitted fresh-Q32 overlay.

The new code provides executable host operations and exact source adapters for
a **future** private 12544B allocation. It has no installer or automatic hook.
Importing its factory/checker uses only stdlib. All device, binding, capacity,
accuracy and performance claims remain false; `activation_ready=false`.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_compact_bridge_cpu.py
```

Inputs are eight existing target-source archives and immutable Git objects from
the PR88/89 commits, not PR89's working files. Source paths are listed in the
checker's `ARCHIVED` mapping: `cjj-target-kv-cache-service-sources-20261027`,
`cjj-target-konly-cache-lifecycle-sources-v1-20261027`, and
`cjj-target-kv-zeroer-platform-sources-v1-20261027`, plus the actual registry
file `/private/tmp/cjj-target-kv-spec-registry-20261028.py`.
An optional complete source tree can be passed with `--vllm-root`; change the
Git object source with `--parent-repo`. `cpu.json` records exact source hashes.
The interface `7e3c2209…` and backend `1219c4d1…` also match existing C3 pins.
Every installed hash must be checked again before runtime binding. No upstream
upgrade or package replacement is proposed.

The earlier `/private/tmp/vllm-0.24.0` tree became empty during this work; the
seven archived files reproduce the same hashes without any remote recovery.
Root then supplied the actual7619B registry source, SHA256
`16b3f6458538910f72cdac48554c20432c7ac0e5122e91c8a6219a1ec25f44fe`.
The checker now replays its real registration code, with only its lazy built-in
registration import replaced by a recorded CPU callback. Built-ins-first then
compact registration resolves the private manager/uniform base correctly;
private-first demonstrably skips built-in initialization. This is metadata-only
replay, not actual scheduler/worker registration or activation.

## Actual allocation and view path

1. `Attention.get_kv_cache_spec` currently returns stock BF16 `FullAttentionSpec`
   (`vllm/model_executor/layers/attention/attention.py:581–632`).
   `ModelRunnerFL.get_kv_cache_spec:7606–7643` also records backend indexing.
   `convert_specs` supplies a narrow post-hook: all42 exact stock specs must
   convert together. Its generated frozen `CompactFullAttentionSpec` preserves
   semantic head128/BF16, modeNONE, page16 and causal full attention.
   It overrides `real_page_size_bytes=12544`, rejects padding/hybrid/unknown
   format, and never merges with stock specs.
2. `kv_cache_utils.get_kv_cache_config_from_groups:1319–1401` computes block
   count and `KVCacheTensor.size` from `spec.page_size_bytes`. With42 equal
   private specs this stays one homogeneous group. It therefore produces
   **12544*num_blocks per layer**, rather than shrinking a view over16KiB pages.
   The CPU checker executes the unchanged DAC1 allocator method
   `ModelRunnerFL._allocate_kv_cache_tensors:7170–7212` with a metadata-only
   Torch substitute:42 allocations×2 pages request1053696 bytes.
3. Stock `_reshape_attention_kv_cache` (`gpu/attn_utils.py:194–252`) cannot
   reshape that storage into a contiguous BF16 KV tensor. Setting stock
   `page_size_padded=12544` also fails its16384B lower bound and is not a shortcut.
   `reshape_compact` instead checks exact private type, fresh independent int8
   storage size, NHD order, page count, alignment and no packed allocation, then
   calls the real API `torch.as_strided(raw.view(torch.bfloat16), …)` when supplied
   a runtime. This code has not been executed with an actual Tensor/device.

| View | Shape | Element strides | Byte offset |
| --- | --- | --- | ---: |
| Nominal BF16 carrier | `(N,2,16,2,128)` | `(6272,2176,256,128,1)` |0|
| INT8 K code | `(N,16,2,128)` | `(12544,256,128,1)` |0|
| Full FP32 scale view | `(N*3136,)` | `(1,)` |0|
| BF16 V | `(N,16,2,128)` | `(6272,256,128,1)` |4352|

The carrier's K plane **overlaps** the V region and is only an ABI carrier.
Its BF16 values are not keys. Stock Triton backend shape/identity and
`forward`/`do_kv_cache_update` `unbind(1)` can stay unchanged; they yield nominal
BF16 K/V views with stride6272 and a4352B pointer gap. Every owned read/write
must reach the compact encoded facade. A stock fallback would be incorrect.
Generic copying, serialization, or `numel()*element_size()` accounting of the
carrier cannot substitute for its physical storage extent.

## New required host boundaries

- C3 `_cold` rejects `type(spec) is not FullAttentionSpec` explicitly.
  `c3_source` changes exactly that rejection to an injected
  `compact_spec_is_owned` predicate and proves an AST inverse. That predicate
  must certify the exact registered compact class/format and active owned epoch;
  it must not be a blanket `isinstance` relaxation. Existing source, device,
  dispatcher, builder, batch-order and backend identity checks remain.
- Registry registration must associate the compact spec with a **private
  FullAttentionManager subclass** and its own uniform base. Call the registry's
  `_ensure_registered(config)` before adding private entries: its implementation
  otherwise skips built-in registration when any entry already exists.
  Export the generated classes under one importable package path before
  scheduler/worker RPC; a function-local class is insufficient. The factory
  supplies importable spec source, but RPC export/install is not yet tested.
- `SingleTypeKVCacheManager.allocate_new_blocks:303` and
  `allocate_external_computed_blocks:270` use exact-type lists for
  `new_block_ids`. Merely inheriting FullAttentionManager loses these zeroing
  notifications. `manager_source` adds only `CompactFullAttentionSpec` to those
  two lists; inverse AST restores the original algorithms. The CPU checker
  executes both generated methods and checks the new IDs. External KV transfer
  remains outside the private service scope despite preserving this method.
- Original `KVBlockZeroer` (`worker/utils.py:80–225`) accepts the subclass and
  derives physical bytes from `kv.stride(block_dim)*element_size`. Its original
  constructor is replayed for42 distinct carrier bases: page3136 i32 values,
  tile64,49 tiles/page. It covers all12544 bytes, including scale/padding.
  **Cost caveat:** old16KiB pages used tile1024 and4 programs/page; the unchanged
  compact zeroer launches49, or12.25× as many programs per page. Fewer bytes do
  not prove a cheaper zeroer. A masked ceil-tile zeroer would be a separate change.
- No connector means `use_uniform_kv_cache` returns false
  (`kv_connector_model_runner_mixin.py:115–158`), so the independent raw path
  applies. Reject connectors/offload, cross-layer packing/sharing, speculative
  caches and live reinitialization; inherited PR89 ownership/transfer fences
  must remain active. Prefix sharing may only reuse pages within the same cold
  compact epoch. On/off format changes require a fresh worker/prefix/Graph epoch.

## Writer/reader bridge prepared

`parent_host_source` adapts PR89's real typed broker and runtime registry from
the pinned5b source. It changes the explicit format ID, physical extents,
typed strides and V pointer gap. Its byte inverse preserves all original
storage identity, bounds, readonly-alias, epoch, capture and dispatch guards.
The CPU checker executes the adapted typed broker's complete role dictionaries
and binding validator against the independent compact recipes.

`kernel_source` accepts only the six pinned active PR89 source hashes. It changes
ten address literals across writer, decode, normalized2D128, Q32, M2 and raw3D:
code page16384→12544, scale page4096→3136 F32 elements, full V page8192→6272
BF16 elements, and full V offset4096→2176. The scale offset1024 F32 elements
and slot/head/dim terms stay unchanged. Each output reverses byte for byte to
its parent, preserving codec, QK, softmax, P, BF16 V/PV, masks, reducers and
ordinary bounds guards.110592 CPU address comparisons use emitted writer and
scale expressions plus the PR88 oracle, including page/slot/head mappings.

All these generated kernels need fresh compile/ABI/actual-IR checks. The old
Q32 override, hoist proof, hashes and resource/performance observations do not
license a changed page/scale stride. The current private89 service still uses
16KiB and gains no capacity from this branch.

## Remaining activation work

There is intentionally no callable service activation in this commit. Before a
future opt-in service candidate, materialize/export spec and manager modules;
verify remaining installed hashes and registry/RPC behavior; bind the exact
get-spec and reshape hooks; extend C3's owned-layout receipt; bind the generated
broker/runtime/kernels with a new implementation digest and format-aware compile
fingerprint; and prove every backend route rejects unknown/stock interpretation
of an owned compact allocation. Existing16KiB service/Graph proofs are not reused.

Then test actual storage sizes/page counts, zero/reuse/prefix lifecycle, writer
slots/unmapped bytes, all reader numerics, Graph/side stream, model precision
and fixed-parameter service throughput. Keep the original precision gates.
The theoretical1.30612× pages per unchanged byte budget is not measured service
capacity, latency or throughput; neither a component gain nor this arithmetic
establishes an end-to-end benefit.

## New independent device-entry preparation (still CPU-only)

The `device-v1/README.md` and `tools/bench_biv150_konly_compact_device.py` now provide an explicit Root-only Tensor/compile/numeric/Graph/side entry for the unchanged8bab writer and split GQA grids64/S2 and32/S4. This revision normally merges main4229, changes no production bytes, verifies embedded89 source against099f, and grants no old16KiB-page/hoist/model qualification. Default remains stdlib CPU; device results are pending.
