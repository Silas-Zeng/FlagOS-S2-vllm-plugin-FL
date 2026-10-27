# K-only layer-relative cache view broker: CPU only

This small independent branch is based on actual team main
`dac1b8fb48e7983e48283032efeee9b592c5fbe5` (PR84), fetched normally and confirmed
by authenticated API. It depends on PR79/80/82/83 but changes none of their
frozen files or production vllm_fl/csrc. There is no kernel, GPU entry, Tensor
creation, backend/service installation or encoded-cache registration.

`parameters(layer, role, format_id=...)` returns only cache-pointer/stride kwargs
as integer-address `View` recipes. The caller must explicitly request the private
codec interpretation. That request does not verify cache contents or grant
activation. Unknown role/format/layout rejects; there is no BF16 fallback.
It supplies neither query/output arguments nor a full writer slot/source
admission. No dispatch, grid, M2/Q32 chain or official parameter is changed.

## Frozen pointer conventions

| Role | Cache pointer arguments | Pointer relative to this layer's page0 |
|---|---|---|
|`decode79` (inline and preQ cache subset)|`key_cache_ptr`, `key_scale_ptr`, `value_cache_ptr`|I8 K0; FP32 scale0; BF16 V8192B|
|`prefill80` (common2D/3D cache subset)|`key_cache_ptr`, `key_code_scale_ptr`, `value_cache_ptr`|I8 K0; FP32 scale0; BF16 V8192B|
|`writer83`|`code_full`, `scale_full`, `value_full`|All three pointers at page0; kernel adds V4096 BF16 elements|

K structured view is `[P,16,2,128]` with I8 strides
`(16384,256,128,1)`. Reader V uses BF16 strides`(8192,256,128,1)` and
already-offset pointer. Full scale is a flatFP32 view of`P*4096` elements;
79/80/83 add`page*4096+1024+slot*2+head` themselves. Passing a scale pointer
already at4096B adds the bias twice. Writer full BF16 is flat`P*8192` elements;
passing the reader's V pointer likewise adds8192B twice.

Decode gets its exact six K/V stride scalars. Prefill gets eight exact K/V
stride scalars, mode0, original unused mode0 dummy K/V scale pointers and zero
dummy scale strides. The actual per-head scale uses the additional private
`key_code_scale_ptr`. `USE_PER_TOKEN_HEAD_SCALES` is internal constexpr source,
not a raw parameter; injecting it is rejected. The broker never enables built-in
mode2 or substitutes it for private K-only data.

ABI evidence is extracted from exact commits:79`bf94ba4`,80`d2ceb3b`,
83`58bace0`, main`dac1b8f`. Seven complete source snapshots and manifest are in
`frozen-abi-sources-v1.tar.gz`,29715B, SHA
`8a68f9211c9c09fc17fdc124aef12e50fd2aba0c8a8f8b8d28054f78ad56230b`.
The80 SDK source is pinned54c495 from its frozen input archive. CPU parsing
checks exact raw parameter-name subsets, scale offsets, actual79/80 offset-V
construction and83 full-V formula. These are source schemas, not compiled typed
signatures or proof that the producers/readers are installed.

## Layer base and metadata checks

The layer observation is BF16 NHD`[P,2,16,2,128]`, actual element strides
`(8192,4096,256,128,1)`, explicit device`cuda:0`, storage identity/base/bytes,
absolute storage offset in BF16 elements and independently supplied data pointer.
Every field is strictly typed and internally checked. P is positive, each
physical page remains16KiB and complete layer span must fit storage. F32 base
alignment and signed-I64 pointer/span bounds are required. Padded/HND/interleaved
page layouts, unknown IDs and inconsistent data pointer/offset reject.

The layer tensor's storage offset may be nonzero. Full-base means **this layer's
page0**, not the backing allocation's first byte. For example, a layer beginning
at allocation byte16384 has full scale storage offset4096 FP32 elements, reader
V storage offset12288 BF16 elements, and writer full-V offset8192 BF16 elements.
Every derived view retains the same storage identity, allocation base/size and
device; typed absolute offsets produce the same layer-relative pointer.

`validate_binding` recomputes the complete expected cache subset and rejects
extra/missing fields, foreign storage, wrong dtype/shape/strides/pointer/offset,
pre-offset scale/full-V confusion and byte-vs-element stride errors. It does not
trust a mutable plan dictionary as a certificate. The optional readonly-source
observations admit BF16 K/V[rows,2,128] with head128/dim1 and row pitch>=256.
K/V may share QKV backing with separate offsets; either source sharing or
overlapping live-cache backing rejects conservatively. Caller still needs83's
row/slot count, finite valid K and trusted unique/fresh slot contract.

The old80 fixture descriptor checks use absolute storage offsets0/4096. They
remain unchanged. This independent broker validates the raw kernels' layer-local
pointer arithmetic for nonzero offsets; it does not claim that the frozen80
fixture validator suddenly admits them, nor silently relax that validator.

## Replay and limits

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error \
  tools/check_biv150_konly_view_broker_cpu.py --out NEW_CPU_JSON
```

Python3.12.15/3.14 outputs match `cpu-v1.json`, SHA
`8a5047a03d106c8ecf71009e93745604d65653f2cb7b6d82a408443e6e78ed95`.
Fifteen serialized role/layer parameter outputs cover1/3/17 pages and storage
offsets0/4/256/16384/65536 bytes.98304 address comparisons cover every slot,
head and dimension on first/last pages, plus a virtual large last-page I64 case.
All three roles reach identical code/scale/V bytes.29 faults reject, including
unknown roles, wrong source/typed storage, alignment/bounds/layout, pre-offset
views, extra internal constexpr and mixed stride units. Readonly QKV sharing is
explicitly accepted while cache alias is rejected. No framework is imported.

This is a stateless address planner. It has no payload registry, opaque Tensor
creation, data-pointer cache or lifetime token. Actual device views/compiled ABI,
all-reader coverage (generic/short/mixed/warmup), source/slot provenance,
freshness/stream ownership, Graph epoch/capture and storage reallocation remain
activation prerequisites. Address equality after reuse cannot prove generation.
Holding a stale observation or printing `format_id` cannot grant encoded status.
There is no GPU/runtime installer, performance measurement, model105 acceptance,
throughput/TTFT result or new test queue. Keep Draft.
