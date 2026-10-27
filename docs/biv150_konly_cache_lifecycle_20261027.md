# K-only cache service lifecycle: source map and CPU byte contract

This independent CPU-only branch depends on PR79's stored-K8 decoder/codec and
PR80's private prefill readers. It implements no kernel, production hook,
installer or GPU entry. Production remains byte-identical to team main
`56f1838192abc940b3fb9accef490022ac125000`. The existing M2/Q32 chain is preserved.
The ordinary BF16 chain cannot consume encoded K. Device numerical, Graph,
side-stream, model105, official throughput and TTFT gates remain unverified here.

Source provenance includes the original seven actual target files from PR79's
IO review, the two actual attention/context files from PR74, and fourteen newly
downloaded lifecycle/writer files in the first freeze:23 files953321B. Follow-up
RoPE fusion/zeroer/platform evidence adds18 path copies. The current inventory
is41 downloaded copies1577908B,31 canonical paths/unique content hashes. Eight
`vllm/vllm` nested copies and two overlaps with the original set are byte-identical;
they are not extra active implementations. Five small archives preserve every
original manifest/file SHA, with no framework import. Other-vendor
FlagGems files are inventory evidence, never assertions of BI-V150 execution.
The CPU checker also pins six current-main source files. It parses source ASTs
without executing modules and records exact symbol line spans in `cpu.json`.
The initial normal fetch timed out at15s low speed; authenticated API rechecked
main56f and the matching local commit before development. This is not a claim
that the fetch succeeded or an upstream SDK upgrade.
Before the second freeze, a normal `git fetch --no-tags origin main` with
HTTP/1.1 succeeded; FETCH_HEAD/origin/main and fresh authenticated API all remain
the same56f. The branch already contains that actual main; production stays equal.

## Physical format and the dangerous writer seam

The supported hypothetical layout is contiguous NHD
`[P,2,16,2,128]` BF16, strides `(8192,4096,256,128,1)` in BF16 elements,
zero storage offset, quant modeNONE. HND, page/layer packing, nonzero offsets,
different block/head sizes and unknown ownership require rejection until
separately qualified. These are explicit contract values, not observed live
Tensor strides. Real base alignment, storage extent/aliasing and active symbols
still need binding verification before any mutation.

| Region, half-open page byte offsets | Bytes | Private interpretation |
|---|---:|---|
|0..4096|4096|I8 Kcode, strides `(16384,256,128,1)` bytes|
|4096..4224|128|FP32 Kscale,32 words; base1024 and strides `(4096,2,1)` FP32 elements|
|4224..8192|3968|Reserved, preserve on append|
|8192..16384|8192|Original BF16 V, base4096 BF16 elements, strides `(8192,256,128,1)`|

Active payload12416B does not reclaim capacity: each physical page remains16KiB.
Each valid append logically stores256B code+8B scale+512B V=776B versus the old
1024B K/V stores. This excludes source loads, codec reduction/division/rounding,
transaction granularity, launch and any scratch cost; it is not a measured
bandwidth or latency reduction. Whole-page transfer still moves16KiB.
Relative Kcode row/head offsets align to128B and FP32 scales to4B; actual base
alignment remains a real-handle gate. Legal positive int32 page index
`2**31-1` times16384 plus16383 fits in signed I64 (`2**45-1`). Perform the widen
before multiplication, not after an overflowing int32 product.

Actual `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:33–139` loads
slot asI64, returns on negative slots before reading source K/V, then stores K/V
at the same **element** coordinates. NONE applies no K quantization. For the
original BF16 geometry, K slot occupies512B. Private code slot occupies256B and
its two scale words are elsewhere. The ordinary writer must never first write
the live encoded K field and then be patched by the private encoder.

The CPU counterexample initializes live prefix scales to1, lets the old writer
write slot8 (`4096..4608`), then correctly writes private slot8 code/scale/V.
Fifteen other slots' scale pairs remain corrupted, including all eight earlier
live prefix tokens. For slot0–7, old writes can also hit other slots' code
regions; for slot9–15 they overwrite reserved bytes. The V field has the same
original coordinates and can be retained independently. Encoding an overlapping
whole live BF16 K page is likewise unsafe unless all needed source bytes are
first materialized in separate storage under correct ordering.

Current79/80 scale helpers use a full-allocation FP32 pointer and add1024
explicitly. A shaped scale view whose data pointer already starts at4096B would
need the offset removed in its consumer; supplying it to the current helpers
would add the offset twice. Preserve the exact pointer convention in every
encoder/reader binding. Kcode uses byte strides while V uses BF16 strides, so
the old writer's one shared K/V element-stride argument cannot stand for both.

The checker models writing **already encoded** code/scale plus original V;
it does not reimplement PR79's RN32→RNE codec or RoPE. It covers all48 physical
slots across three pages, exact touched spans, untouched neighbouring tokens,
scales/reserved/pages, padded negative slots without reading their payloads,
and preflight rejection before mutation. This is a byte contract, not a writer
ABI or device implementation proof.

The actual ordinary writer wrapper:344–441 passes `key.stride(0)` and
`value.stride(0)`, and its launch token dimension is `slot_mapping.shape[0]`.
The existing private79 append control hardcodes source `token*256+head*128`;
that control is not the production writer ABI. `Attention.forward` only reshapes
key/value; a QKV split can retain a larger row stride. The CPU source-byte
counterexample compares row pitches256 and2560 across three rows and shows why
the latter cannot use the contiguous control. No actual service stride is
asserted. Future binding must honor/prove input row extent, row pitch, contiguous
head/dim layout or reject it; padded slots must short-circuit source loads.

## Real call and lifetime map

| Actual or main source anchor | What is established and what must be preserved |
|---|---|
|Actual attention.py:498–545|Both direct and custom-op paths call cache update before attention when the selected backend excludes writes from forward. Shared-KV layers skip it. Do not duplicate the writer inside forward.|
|Actual attention.py:670–736|`get_attention_context` obtains layer cache and per-layer slot mapping from ForwardContext, including speculative list metadata. `unified_kv_cache_update` calls the actual impl writer and returns a dummy dependency. Preserve that ordering token/custom-op schema rather than adding an unordered hidden side effect.|
|Actual triton_attn.py:724–767|Ordinary modeNONE calls the ordinary writer; built-in per-token quant mode writes both K/V and is a different format/codec. It is not a shortcut to private K-only.|
|Actual triton_attn.py:769–808|The declared fused RoPE path is enabled through ROCm AITER and calls its own cache writer. The actual fusion/API and its source-only disabled condition on pinned PlatformFL are detailed below; actual binding/pass execution remains a device/service gate.|
|Main FlagGems attention.py:471–497|Alternate backend unbinds axis0 and calls FlagGems. Actual generic FlagGems flash writer:32–134 stores equal element coordinates; `use_c_extension` delegates to another op. Neither its native branch nor other vendors are inferred active.|
|Actual attn_utils.py:194–247; main ModelRunnerFL:7200–7309|View construction handles packing, padding and stride permutations. Our fixed NHD contract rejects those alternatives. Main runner is a separate implementation, not a subclass that automatically inherits every target GPUModelRunner method.|
|Main ModelRunnerFL:4069–4141,4360|Slot buffers are per-layer/group, padded slots become-1; ubatches slice these mappings. Actual BlockTable kernel:326–380 computes physical page/slot, widens page toI64 and handles CP/locality. Python request row swap is metadata movement, not a payload swap.|
|Actual cache_manager.py:202–242 → coordinator → FullAttentionManager:542–588|Full attention prefix lookup reuses existing pages; it does not encode/copy payload. `add_local_computed_blocks`:182–232 touches references. Format must be immutable for the relevant cache lifetime, or invalidated/keyed explicitly before mixed reuse.|
|Actual block_pool.py:542–635|Allocation evicts old hash metadata and changes refcounts; free/touch manage ownership, not payload conversion or zeroing. Freed bits may survive. Reuse must not preserve a stale validity/format generation.|
|Actual kv_cache_utils.py:539–604|Hash inputs are previous hash, token IDs and extra keys such as LoRA/MM/salt/embeds; there is no private format argument. Do not assume cross-format prefix hashes isolate representations.|
|Actual block_pool.py:358–457|Partial-prefix registration exists and adds a lookup key to the same existing block without copying; it is not evidence that default FullAttention16 activates this feature. If admitted later, partial ownership and append isolation need their own proof.|
|Main ModelRunnerFL:1218–1219,1351,1434–1445|New-block zeroing and resumed preemption update metadata through separate seams. Actual KVBlockZeroer:41–221 enumerates absolute INT32 segments and clears the full physical span; for fixed block-dim0 NHD it is4096 words/four1024-word chunks per16KiB page. This can remain unchanged under that layout. Zero/uninitialized scale bytes cannot be read as valid tokens before the private writer completes. Rebuild its absolute-address metadata if storage changes.|
|Actual connector mixin:78–112|Bind metadata → start asynchronous load → model → wait_for_save/get_finished/errors → clear metadata. Deferred speculative finalization uses64–72. Submission is not readiness; retained references, format agreement and completion events remain required.|
|Main GraphWrapper:145–253|Capture stores the runnable's graph by BatchDescriptor; later calls replay it without rerunning Python reader/writer selection. Install a consistent format before capture and recapture on incompatible binary/storage lifetime changes; changing a later Python flag cannot repair an already captured BF16 writer.|

The ordinary SDK/normalized/Q32/M2/GQA/generic readers and mixed decode/prefill
all share cache payloads. Codes81,7f form BF16 bits0x7f81, a NaN; codes7f,00
form0x007f, not reconstructed K=127. Dtype/shape/NONE checks cannot identify the
format. Once encoded, ordinary BF16 fallback is invalid. Before first encoding,
unsupported configurations may keep their entire cache stock. Afterwards every
reachable reader/writer must understand the format, or a separately verified and
charged conversion must precede any stock consumer. No such conversion or
service activation is implemented here.

## Actual RoPE fusion contract and bounded options

The new canonical `vllm/compilation/passes/fusion/rope_kvcache_fusion.py`
SHA`faed3153db71f91bddcccf99cfab47dca7881933769d262dec52e22b5bfad163`
establishes the following source chain. Its nested-install alias is identical.
The fused impl:46–77 gets the same cache/slot context and calls
`do_rope_and_kv_cache_update`; its return is only a zero-length ordering dummy.
Registration:92–97 explicitly declares `mutates_args=["query","key"]`.
The unquantized functionalized replacements:339–354,369–384 return the dummy,
updated Q, updated K and original V. Thus the declared API is not cache-only.
The original pattern runs RoPE before `unified_kv_cache_update`, so a preserved
unfused path supplies post-RoPE K to the private writer.

Actual `_aiter_ops.py::triton_rope_and_cache`:2450–2489 explicitly passes
`q_out=query,k_out=key` to external
`aiter.ops.triton.fused_kv_cache.fused_qk_rope_reshape_and_cache`. This is an
in-place output binding proof, not a proof of the absent AITER device body or
of an executed BI fusion. It still writes key/value caches. Calling it against
the live encoded K and then encoding returned K has the same corruption hazard.

The fusion pass registers patterns only when the layer's capability is truthy
(:440–451), and applies only within its configured compile range(:461–465).
Pinned main `PlatformFL._enum=OOT` inherits actual
`Platform.is_rocm(): return self._enum == PlatformEnum.ROCM`. Therefore the
pinned FL/Triton source chain fails the AITER ROCm prerequisite; the unsupported
decorator returns`None` (falsy), not literal`False`. This is a conditional source
conclusion for those exact classes. Active platform/impl method identity,
patched symbols, pass matches and compiled graphs are not observed by this
CPU task and remain admission checks.

The simplest future stored-format route is to keep fusion unsupported for the
private backend before compilation/capture, retaining original separate RoPE
and a format-aware writer. It changes no official serve parameter, but still
requires review/numerical/all-reader/service gates and is not implemented here.
If a fused path is later retained, use a verified direct private writer or a
separate BF16 scratch target plus the codec. Scratch must never alias live
encoded K; matching page strides/slots and returned rotated K need explicit
proof. A compact dummy cache cannot reuse arbitrary physical page IDs without
remapping. Scratch allocation, extra writes/reads/launches and Graph lifetimes
belong in complete-step cost. Neither alternative is declared free or fast.

## Copy/swap: preserve the operations that really preserve bytes

| Operation | Bounded conclusion |
|---|---|
|Whole16KiB raw page, or complete K8192B plus V8192B fields|Can preserve code, scales, padding and V without quantization work when physical offsets/page stride/format agree. Existing main raw-page/pitched helpers are candidates for reuse, not shown active service paths.|
|Same-width opaque BF16 word load/store, no numerical conversion|Can preserve payload bits. The CPU model round-trips all65536 BF16 patterns, including signalling/quiet NaNs, and16 pages containing every pattern in V. Actual device copies still require independent bitset tests and stream/alias/lifetime proof.|
|Actual `_custom_ops.swap_blocks`:2659–2685|ABI explicitly uses contiguous byte-block size and calls native `_C_cache_ops`. This is compatible in principle with16KiB raw transfer, but the native implementation/binding is not in this source set. Do not call this verified device memcpy.|
|Actual copy_kv_blocks:184–218 → main PlatformFL:405–426|Helper delegates to actual platform. Main implementation indexes `[:,ids]`, assuming page dimension1. Current Triton semantic shape is[P,2,...], page dimension0. A three-page/id2 CPU structural example rejects. Do not silently apply that alternate-layout helper to Triton; verify the real override or keep transfer rejected.|
|Actual receive postprocessing utils:221–296|BF16-token/head reshape/permute/index_copy can preserve each word yet still rearrange code/scales as if they were old K. The CPU HND→NHD word-permutation example preserves every16-bit word but changes the private scale region. These layout/blocksize conversions require private-format transformation or rejection.|
|Logical per-token K copy/gather/dequantization|Must carry the corresponding code256B, scale8B and V512B with correct indices; old BF16 K512B token geometry is insufficient. Native gather/dequant wrappers are obligations, not transparent copies.|
|Arithmetic/converting copy|May alter payload: CPU FP arithmetic on BF16 NaN0x7f81 produces0x7fc1. A simple F32 roundtrip on this Mac happened to preserve it, so the test uses explicit add-zero and does not falsely equate dtype change alone with corruption.|

Current main C3 already excludes KV transfer, cross-layer/shared-cache and
unsupported scopes. Preserve these exclusions until coverage is proven; this is
not a current-main service regression or a proposal to rewrite all copies.
For any future raw transfer, source/destination must agree on codec version,
physical page size/layout, allocation generation and ready/retired events.
No capacity reduction, new transfer activation or vLLM parameter change follows.

## Remaining admission gates and replay

Missing/unknown format ownership, active writer identity, tensor byte offset,
alignment/extent, unsupported shape/layout, alternate/fused writer, uncovered
prefill/generic/short-query/multi-sequence fallback, stale generation/capture,
unsettled streams, unsupported copy/dequant or incomplete initial-format
transition must reject **before mutating live cache**. A Boolean claim that
coverage is complete is not a binding proof. This branch provides no installer.

The newly collected Python fusion/zeroer/platform sources close their source
definition gaps. Remaining optional external bodies are the called AITER kernel
and selected native copy/FlagGems implementations **only if** those branches are
admitted. A conservative private FL scope can reject them instead. Actual loaded
model rotation/caller identities, live Tensor/storage strides/extent/alignment,
all-writer/all-reader bindings, prefix/reallocation/Graph lifecycle and stream
proofs remain required before activation. Keeping an immutable format per
layer/allocation lifetime can avoid cross-format prefix mixing; selecting format
afresh from each query shape cannot. There is no installer or completed service
closure, and no reason to fetch unrelated backend kernels merely because their
source files exist.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_cache_lifecycle_cpu.py --out NEW_CPU_JSON
```

Python3.12.15/3.14 outputs match. Source/hash/AST and CPU bytes are the only
current gates. Keep Draft; no installed service, numerical/model acceptance,
device-copy correctness, performance claim or GPU queue is implied.

Initial`cpu.json` (23-source6ab freeze) remains unchanged, SHA
`5776cfa3c9ed7ee2a376f465158fceecfb24f0bbda94f873d46fb18469ea7cc5`.
Current`cpu-source-closure-v2.json` includes41-source copies, explicit mutation/
output-binding/Platform enum AST checks, pitched-source counterexamples and
full-page zero address coverage. Its SHA is
`f14525be9e47519f42ffab0be47fbdc260c5cf821b1e9b8631413b9c973eb1f3`.
The41-copy record includes both duplicate trees, not41 distinct active paths.
