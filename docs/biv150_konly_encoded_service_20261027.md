# Default-off private 16KiB encoded-cache service candidate

Base: authenticated API main **dac1b8fb48e7983e48283032efeee9b592c5fbe5**. The first normal fetch attempt timed out at20s low-speed; API and the already present Git object matched DAC. Parent subsequently fetched DAC successfully. No upstream SDK/dependency upgrade. This own branch is a CPU-ready **experimental service binding**, not a service/model/performance pass.

`VLLM_FL_BIV150_KONLY_ENCODED_SERVICE` defaults0. Off executes no hook import/binding/device work. On installs once in a cold worker before KV allocation/capture. The official serve CLI stays unchanged. Author performed no SSH/JIT/GPU/service work.

## Actual seams and format

The physical cache remains BF16-shaped NHD `[P,2,16,2,128]`, strides `(8192,4096,256,128,1)` in BF16 elements,16KiB/page. Kcode is I8 at0..4095, Kscale FP32 at4096..4223, reserved4224..8191 remains, V BF16 stays8192..16383. No capacity recovery and no PR88 compact layout. Tensor dtype cannot authenticate payload; a cold, explicit per-allocation registry owns the format.

```mermaid
flowchart TD
 I[ModelRunnerFL.initialize_kv_cache] --> A[Original allocation plus private epoch checks]
 S[Original scheduler slot_mapping / padding -1] --> W[Original unified_kv_cache_update custom-op]
 W --> E[Private stride-aware do_kv_cache_update writer]
 E --> D[Original ordering dummy dependency]
 D --> F[Original TritonImpl.forward and C3 natural-prefix splitter]
 F --> U[Original normalized / GQA unified function identities]
 U --> G[GQA producer facade: K8 plus original reducer]
 U --> N[Normalized holder facade: Q32 / M2 / BM128 K8]
 U --> R[Raw SDK kernel facade: K8 plus original SDK reducer]
```

The writer is called at the real `TritonAttentionImpl.do_kv_cache_update` seam (target triton_attn.py724–767), not again in forward. Original attention.py `unified_kv_cache_update` retains the dummy dependency consumed by attention; profiling with no layer slot mapping skips the writer as before. Source K is post-RoPE; private fused-RoPE/cache writing is disabled before compilation and any attempted unqualified fused writer raises.

The83 writer function is copied byte-for-byte, with the same RN32→RNE codec, actual source token strides and int64 slot/index arithmetic. Negative/padded slots short-circuit source loads. No per-token `.cpu()`/`.item()` or newly invented freshness check: source/slot ordering, valid unique physical slots and padding-1 come from the original scheduler/block table/custom-op/stream contract. Main runner4069–4141/4360 and actual block_table326–380 are the source anchors. These are inherited contracts, not newly proven arbitrary concurrent mutation safety.

The86 broker constructs zero-copy typed aliases, independently verifies storage identity/pointer/offset/span, and separates K byte strides `(16384,256,128,1)` from V BF16 `(8192,256,128,1)`. FP32 scale points at full **layer-page0**, not at the scale field; kernels add1024 FP32 elements themselves. Nonzero aligned layer offsets are represented explicitly. Source K/V cannot alias writable cache. Strong base ownership prevents address reuse; the same impl changing allocation/view is rejected. A fresh worker/prefix/capture lifetime is required for reallocation. Prefix lookup/reuse and whole-page zeroing remain original; format stays fixed for the allocation lifetime. KV connector/offload/speculative ownership and encoded copy/swap/layout conversions are refused pending separate qualification.

## Reader dispatch and original main stack

Top-level `attention.unified_attention`, its normalization/GQA closures and C3 builder/forward identities remain unchanged. Three **inner** kernel bindings get format-aware facades: main GQA producer, raw SDK kernel, and the existing normalized-holder entry. No encoded launch falls through to ordinary BF16 data interpretation. Untagged reads after the encoded epoch is armed, unknown aliases, mode/features/geometry/ABI mismatches and foreign IR refuse.

Existing main natural-prefix splitting remains installed, with original metadata/stream preparation. Since shape/dtype metadata stays BF16 and outer functions are unchanged, it can still issue prefix GQA and suffix prefill calls. **Actual service trace is pending**; source preservation alone is not a claim that every mixed batch splits. Facade events record profile/query rows/sequence count/hash so a63/31 decode prefix can be distinguished from all-sequence2D. Non-leading/interleaved mixed calls currently use one encoded all-sequence2D reader, including its decode rows: no second85 launch and no duplicate writes. Before formal throughput, actual trace must confirm leading-prefix GQA remains and assess restoring85 plus a disjoint prefill-skip-decode path for non-leading mixtures.

Reader sources come from frozen79/80/87 factories. Profile selection preserves Q32 BM256/BQ32/T64/w16/s1 for one large sequence, M2 same tile for two, normalized BM128/BQ16/T64/w8/s1 otherwise, and raw3D original SDK grid/segments/options/reducer. General grids are `qrows//BQ+num_seqs`, not fixed benchmark geometries. Q32 is explicitly **unhoisted I8 fragment**: no old exact-hash overlay is borrowed. Fresh service IR/ABI/resources must be collected before claiming a new hoist or component speedup. Source-generation proofs and component eager/Graph evidence do not qualify new service specializations.

The fixed source dependencies already have component implementation evidence:79 native-QK/codec/decode,83 strided writer,80 Q32,87 BM128 eager/changed-input Graph+side,80 M2 and raw q1/q2 exact cases,85 scattered mixed. Original BF16 proxy failures remain; none is represented as new service/model precision acceptance. The separate81 result102/105 is only old56 stock-cache pure-decode quantization. It cannot certify this encoded-prefill candidate.

## Identity and cache safety

Private GPU functions have independent JIT objects/caches. Installed SDK/helper/backend and main Q32/M2 chain identities are pinned before any mutation. New binding construction is atomic with rollback before any encoded write. The source bundle is generated by stdlib-only materialization, with exact factory inverse proofs; generated GPU modules are never imported by CPU checkers.

`compute_order` has an explicit enabled-only **dispatch receipt**. It verifies registered facade objects/underlying parents, writer/lifetime/transfer binding objects, private kernel identities, actual dispatch code identity, allocation tags and implementation digest. Only then are parent interfaces inspected by the original metadata-stack proof. Parent `.fn` reflection is **not** treated as proof of new KV math. Default-off proof takes the unchanged parent route. New numerical certification remains false.

`compile_semantics` includes the default-false feature bit and an enabled-only `konly_encoded` record with format and implementation SHA. That digest covers every private Python implementation plus the source manifest, agrees with install/selected logs and changes the compiled graph identity. Old BF16/old-format captures are not valid on-state evidence. It adds no keys to the original Torch ABI dictionary. The main worker tail uses explicit os import, preserves CRLF bytes and updates the sole runner pin. A complete exact reverse patch proves the three original files return to DAC bytes.

## Log protocol / next root action

- `BIV150_KONLY_ENCODED_INSTALL`: installed binding, format scope, implementation SHA, unhoisted/service-unverified status.
- `BIV150_COMPILE_SEMANTICS`: enabled `konly_encoded` format and matching implementation SHA.
- `BIV150_KONLY_ENCODED_SELECTED`: actual returned compiled hash (or honest null), source digest, profile, query rows and sequence count. These are host/capture events, never fabricated Graph replay counts.
- `biv150_konly_encoded_service.stats()`: writer/selected/fallback host counts, all observed hashes, claimed allocations and explicit precision/service/hoist limits.

A root-owned fresh fixed-CLI short-request smoke must confirm real startup, actual format writer and selected readers, compute-order selection, source/on fingerprint and cleanup. Capture fresh sources/typed ABI/IR/resources; unknown resource values remain unknown. Startup/ABI failures are not numerical negatives. No new full105/throughput should be claimed without root's independent run. No author device entry is executed here.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/materialize_biv150_konly_encoded.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_encoded_cpu.py
```

The second command executes the actual new worker tail off/on without an os global, reverses the exact production patch, checks source-digest on/off separation, uses explicitly synthetic metadata/kernel recorders for routing and rejects aliases/reallocation/capture creation/transfer/feature mismatches. It does not simulate GPU math. All shape-wide service, lifecycle/Graph, model105, official throughput and TTFT gates remain open. Keep Draft.

## Final statistics export (separate instrumentation revision)

The new runtime wraps the actual `ModelRunnerFL.shutdown` and also registers an atexit fallback. It exports once, preserving the original shutdown return/error. With the existing absolute private `TRITON_CACHE_DIR`, the default output is its parent `konly-encoded-final-{pid}.json`; an explicit absolute `VLLM_FL_BIV150_KONLY_ENCODED_STATS_PATH` may contain the literal `{pid}`. No file is written per token. Export uses exclusive temporary creation plus atomic replace, refuses an existing final file, and reports failures without hiding model/shutdown results. Missing export is an evidence gap. Abrupt process termination can prevent Python shutdown/atexit; absence cannot be interpreted as zero activity.

The file records process PID/start tick, source implementation digest, all host counts and observed kernel hashes, per-shape grid/dtype/stride/offset/selected scalar/capture counts (bounded4096, explicit overflow), and actual returned CompiledKernel metadata/typed signature/constants/attrs/source digest/artifact SHA/resources when available. Unknown metadata/resources remain null or explicitly unrepresented; no Tensor repr/data read occurs. New-hash SELECTED events replace the former first-role-only event, so later compiled variants are visible. The complete shape table distinguishes differently sized launches sharing a compiled hash. These are host/capture observations and still **do not count kernel executions inside Graph replay**. No service numerical or hoist certificate is invented.

CPU tests exercise one-shot shutdown/atexit export, original return preservation, atomic files, shape records, missing-resource unknown status and no-repr serialization. Kernel/writer/codec source and the three original production files remain byte-identical to359 in this instrumentation revision. Its implementation/compile fingerprint changes because runtime source is included. The original359 smoke source/output are untouched.
