# K-only cache service lifecycle: source map and CPU byte contract

This independent CPU-only branch depends on PR79's stored-K8 decoder/codec and
PR80's private prefill readers. It implements no kernel, production hook,
installer or GPU entry. Production remains byte-identical to team main
`56f1838192abc940b3fb9accef490022ac125000`. The existing M2/Q32 chain is preserved.
The ordinary BF16 chain cannot consume encoded K. Device numerical, Graph,
side-stream, model105, official throughput and TTFT gates remain unverified here.

Source provenance includes the original seven actual target files from PR79's
IO review, the two actual attention/context files from PR74, and fourteen newly
downloaded lifecycle/writer files:23 files953321B. Three small archives preserve
every original manifest/file SHA, with no framework import. Other-vendor
FlagGems files are inventory evidence, never assertions of BI-V150 execution.
The CPU checker also pins six current-main source files. It parses source ASTs
without executing modules and records exact symbol line spans in `cpu.json`.
The initial normal fetch timed out at15s low speed; authenticated API rechecked
main56f and the matching local commit before development. This is not a claim
that the fetch succeeded or an upstream SDK upgrade.

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

The checker models writing **already encoded** code/scale plus original V;
it does not reimplement PR79's RN32→RNE codec or RoPE. It covers all48 physical
slots across three pages, exact touched spans, untouched neighbouring tokens,
scales/reserved/pages, padded negative slots without reading their payloads,
and preflight rejection before mutation. This is a byte contract, not a writer
ABI or device implementation proof.

## Real call and lifetime map

| Actual or main source anchor | What is established and what must be preserved |
|---|---|
|Actual attention.py:498–545|Both direct and custom-op paths call cache update before attention when the selected backend excludes writes from forward. Shared-KV layers skip it. Do not duplicate the writer inside forward.|
|Actual attention.py:670–736|`get_attention_context` obtains layer cache and per-layer slot mapping from ForwardContext, including speculative list metadata. `unified_kv_cache_update` calls the actual impl writer and returns a dummy dependency. Preserve that ordering token/custom-op schema rather than adding an unordered hidden side effect.|
|Actual triton_attn.py:724–767|Ordinary modeNONE calls the ordinary writer; built-in per-token quant mode writes both K/V and is a different format/codec. It is not a shortcut to private K-only.|
|Actual triton_attn.py:769–808|The declared fused RoPE path is enabled through ROCm AITER and calls its own cache writer. Whether a compiler fusion/override reaches it, rotates input K in place or only writes rotated K into cache needs the actual fusion body. No claim of closure from the backend condition alone.|
|Main FlagGems attention.py:471–497|Alternate backend unbinds axis0 and calls FlagGems. Actual generic FlagGems flash writer:32–134 stores equal element coordinates; `use_c_extension` delegates to another op. Neither its native branch nor other vendors are inferred active.|
|Actual attn_utils.py:194–247; main ModelRunnerFL:7200–7309|View construction handles packing, padding and stride permutations. Our fixed NHD contract rejects those alternatives. Main runner is a separate implementation, not a subclass that automatically inherits every target GPUModelRunner method.|
|Main ModelRunnerFL:4069–4141,4360|Slot buffers are per-layer/group, padded slots become-1; ubatches slice these mappings. Actual BlockTable kernel:326–380 computes physical page/slot, widens page toI64 and handles CP/locality. Python request row swap is metadata movement, not a payload swap.|
|Actual cache_manager.py:202–242 → coordinator → FullAttentionManager:542–588|Full attention prefix lookup reuses existing pages; it does not encode/copy payload. `add_local_computed_blocks`:182–232 touches references. Format must be immutable for the relevant cache lifetime, or invalidated/keyed explicitly before mixed reuse.|
|Actual block_pool.py:542–635|Allocation evicts old hash metadata and changes refcounts; free/touch manage ownership, not payload conversion or zeroing. Freed bits may survive. Reuse must not preserve a stale validity/format generation.|
|Actual kv_cache_utils.py:539–604|Hash inputs are previous hash, token IDs and extra keys such as LoRA/MM/salt/embeds; there is no private format argument. Do not assume cross-format prefix hashes isolate representations.|
|Actual block_pool.py:358–457|Partial-prefix registration exists and adds a lookup key to the same existing block without copying; it is not evidence that default FullAttention16 activates this feature. If admitted later, partial ownership and append isolation need their own proof.|
|Main ModelRunnerFL:1218–1219,1351,1434–1445|New-block zeroing and resumed preemption update metadata through separate seams. `_zero_block_ids` delegates to target KVBlockZeroer; its body is not yet in this source set. Zero/uninitialized scale bytes cannot be read as valid tokens before the private writer completes.|
|Actual connector mixin:78–112|Bind metadata → start asynchronous load → model → wait_for_save/get_finished/errors → clear metadata. Deferred speculative finalization uses64–72. Submission is not readiness; retained references, format agreement and completion events remain required.|
|Main GraphWrapper:153–251|Capture stores the runnable's graph by BatchDescriptor; later calls replay it without rerunning Python reader/writer selection. Install a consistent format before capture and recapture on incompatible binary/storage lifetime changes; changing a later Python flag cannot repair an already captured BF16 writer.|

The ordinary SDK/normalized/Q32/M2/GQA/generic readers and mixed decode/prefill
all share cache payloads. Codes81,7f form BF16 bits0x7f81, a NaN; codes7f,00
form0x007f, not reconstructed K=127. Dtype/shape/NONE checks cannot identify the
format. Once encoded, ordinary BF16 fallback is invalid. Before first encoding,
unsupported configurations may keep their entire cache stock. Afterwards every
reachable reader/writer must understand the format, or a separately verified and
charged conversion must precede any stock consumer. No such conversion or
service activation is implemented here.

## Copy/swap: preserve the operations that really preserve bytes

| Operation | Bounded conclusion |
|---|---|
|Whole16KiB raw page, or complete K8192B plus V8192B fields|Can preserve code, scales, padding and V without quantization work when physical offsets/page stride/format agree. Existing main raw-page/pitched helpers are candidates for reuse, not shown active service paths.|
|Same-width opaque BF16 word load/store, no numerical conversion|Can preserve payload bits. The CPU model round-trips all65536 BF16 patterns, including signalling/quiet NaNs, and16 pages containing every pattern in V. Actual device copies still require independent bitset tests and stream/alias/lifetime proof.|
|Actual `_custom_ops.swap_blocks`:2659–2685|ABI explicitly uses contiguous byte-block size and calls native `_C_cache_ops`. This is compatible in principle with16KiB raw transfer, but the native implementation/binding is not in this source set. Do not call this verified device memcpy.|
|Actual copy_kv_blocks:184–218 → main PlatformFL:405–426|Helper delegates to actual platform. Main implementation indexes `[:,ids]`, assuming page dimension1. Current Triton semantic shape is[P,2,...], page dimension0. A three-page/id2 CPU structural example rejects. Do not silently apply that alternate-layout helper to Triton; verify the real override or keep transfer rejected.|
|Actual receive postprocessing utils:221–296|BF16-token/head reshape/permute/index_copy can preserve each word yet still rearrange code/scales as if they were old K. These layout/blocksize conversions require private-format transformation or rejection.|
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

Priority remaining actual sources: compiler RoPE/cache fusion and its called
implementation (being collected separately), target
`vllm/v1/worker/utils.py::KVBlockZeroer`, and the selected native implementations
only if those native copy/FlagGems writer branches are actually admitted. There
is no need to declare every unrelated backend active. Model-specific rotation
and fusion output ordering must be resolved before choosing split RoPE+writer
or a separate scratch-cache materialization. An old writer must never use the
live encoded K as its dummy target; any scratch/extra launch/read/write cost is
part of the eventual complete-step timing.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_cache_lifecycle_cpu.py --out NEW_CPU_JSON
```

Python3.12.15/3.14 outputs match. Source/hash/AST and CPU bytes are the only
current gates. Keep Draft; no installed service, numerical/model acceptance,
device-copy correctness, performance claim or GPU queue is implied.
