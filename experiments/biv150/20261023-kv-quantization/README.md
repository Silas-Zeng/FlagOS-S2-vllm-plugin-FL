# BI-V150 quantized KV feasibility: negative operator results

No quantized-KV serving hooks or model quantization configuration are installed.
All tests are isolated, serialized after the warmed formal serving run exits.
Official service parameters and benchmark are unchanged. This branch includes the
validated team-main normalization fix3283672; prototype conflicts use exactly
main's already checked implementation. Old prototypes remain in Git history.

Existing W8A8 linear and FP8_scaled_mm experiments were separate from KV storage.
Correction: the local2026-10-22 progress report already records an INT8 KV
operator probe:BF160.5227ms versus INT80.8246ms, a negative result. This turn
adds FP8, storage layouts, launch searches and direct/dual INT8 dots; it is not
the first historical INT8-KV test. No full-model quantized-KV acceptance is known.

## Scope and accounting

Initial:30 configurations/shapes; launch/layout:70; integer-QK:5;
dual integer-QK/PV:5. All sampled independent numerical-reference checks pass.
These are synthetic operator cases, not Level3 model-accuracy evaluations.

Per-token/head symmetric scales are FP32. INT8 and E4M3FN payloads have128 bytes
plus4-byte scale per head, versus256-byte BF16. Offline cache quantization is
excluded from attention timing. Optional append tests include a new fused Triton
quantization/cache-write kernel; negative attention-only results already rule out
performance acceptance for the implementations tested here.

Compact allocation measures actual reduced storage; inline and page-packed
layouts reuse the original BF16-sized allocation and do not claim reclaimed
capacity. Page-packed scales are contiguous after each page's payload. The append
checks cover padded/empty rows, tail pages, shuffled pages and both storage types.

## Results

| Approach |4K B64 BF16 / candidate ms|16K B32 BF16 / candidate ms|
|---|---:|---:|
|INT8 payload, vector dequant|0.5266 /0.8322|1.0331 /1.6458|
|FP8 payload, score/probability scales|0.5265 /0.6825|1.0320 /1.3959|
|INT8 QK dot, BF16 PV|0.5271 /0.6754|1.0318 /1.3250|
|INT8 QK and PV dots,32-token tiles|0.5269 /0.6522|1.0312 /1.2900|

Warp1/2/4, segment2/4/8/16 and fixed-allocation page layout searches did not beat
the unchanged BF16 reference policy. The fastest listed dual-INT8 chain is still
about24-25% slower. Baseline measurements before and after each candidate remain
close; no competing serving/GPU workload ran. These are operator times and cannot
be converted into model throughput claims.

Integer-QK quantizes the query per head. Dual-INT8 additionally quantizes
probability times the per-token value scale independently in32-token blocks.
Its independent FP32 reference explicitly models this probability approximation;
passing it does not establish equivalence to unquantized attention or model IQ.
No complete Level3 or full-serving run of a quantized-KV implementation was done,
because none of these operators has shown an advantage. No quantized-KV patch is
merged into main. Raw JSON records include source hashes and actual launch args.

Separately,64 BF16 head-subgroup cases passed numeric checks but long-shape best
speedups were0.576x /0.540x. Their raw results are in the decode evidence folder.

A normal-only E4M3 byte payload was also tested: subnormal payloads are flushed
to zero, and normal values are decoded by integer bit-field reconstruction into
BF16. All five sampled independent references pass, but the operator remains
slower; this custom representation is not installed in serving.

FP16 KV plus FP16 dots (BF16 queries converted at load, BF16 outputs) passed five
independent-reference cases but long-shape speedups were1.0003x /0.9995x, within
noise, before cache-write conversion costs. No serving integration is justified.

The deferred BF16 softmax experiment tested70 cases:66 passed numeric checks,
4 failed compiler PassManager (recorded verbatim). Best long-shape speedups were
0.582x /0.671x; no version was installed. Compiler failures were not counted as
correctness passes. Raw data is in the decode evidence folder.
