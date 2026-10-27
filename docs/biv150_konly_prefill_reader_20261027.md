# K-only native-I8 QK reader: private CPU preparation

This Draft depends on **PR79, frozen ae00f58629ad85850fec1b8b8d2da28757af5f8b**.
It starts from freshly fetched team main56f1838192abc940b3fb9accef490022ac125000.
It changes no production code and never edits the active79 or74 worktrees.
The full frozen SDK/normalizer/codec inputs are preserved in the21,018-byte
`evidence/konly-prefill-reader-20261027/source-inputs.tar.gz`; its manifest pins
every original byte. Two stdlib CPU references are copied **byte-identically**
from ae00, including its distinct v1 history; this reader uses only codec v2.

The scope is the current BF16 MiniCPM geometry:16 query heads,2 KV heads,
GQA8, dimension128, PAGE16, causal, no optional bias/sink/window/TD/FP8 modes.
No general dtype or arbitrary-layout compatibility is claimed. QCode is a new
variable: original BF16 `Q` remains the dtype source for the unchanged V cast.
Only Q/K operands of QK become I8; P, BF16 V/PV and FP32 online softmax remain.
There is no INT8 P or V, P8 PV, two-page QK concatenation or KV capacity claim.

## Complete K-reader map and explicit profile limits

Line numbers below refer to the archived SDK54c495… and helpers59d9d3… originals.

| Route | K reads and scaling | Profile boundary |
| --- | --- | --- |
| SDK generic2D and3D | Both use the same `kernel_unified_attention`: pointer K load464, cast474, QK518/520. Address457–460 uses a vector of independently mapped physical pages. | One Kscale helper covers the pointer path;2D/3D masks and epilogues remain original. |
| TD K reader | Descriptor call421–433, helper111–143; K and V share descriptor machinery. | Explicitly rejected. No claim that pointer-byte strides can license TD. |
| main normalized mixed/prefill2D | Main `biv150_prefill_normalized.py:59–66` changes initial L to0 and specializes11 stride parameters with I64 aliases. | `normalized2d128`: BM128/BQ16/TILE64/w8/s1; includes qlen1 requests in a batch that also has prefill, with no batch permutation. |
| Parent Q32 | `biv150_prefill_q32.py:44–59` selects BM256/BQ32/w16 for one sequence and totalQ≥1024. Kernel source is the same normalized clone. | `q32`: separate guard/profile; grid `T//32+1,2`. Never feed BM256 arguments to SDK raw BM128. |
| Parent M2 | `biv150_prefill_multi2.py:50–67` selects the same BM256 source for two sequences and totalQ≥1024. | `multi2`: grid `T//32+num_sequences,2`.513+513 and1025+1 require34 programs, not33. |
| SDK raw3D fallback | SDK981–995 chooses3D only for no-prefill/threshold/buffer conditions; initial L remains1. BF16 host defaults762–777/864–866 give BM16/BQ2/TILE16. | `raw3d`: independent canonical source, segment shapes1/2/4/8/16. Original SDK leaves warp/stage options unspecified; this prototype does not invent them. |
| SDK raw2D with pure-decode/high-batch or unknown features | Same pointer K read, but SDK raw L/host profile differs from the accepted normalized prefill profiles. | Mapped, **not admitted by this first factory**. Unknown profile stops; it cannot consume encoded K as BF16. |
| Specialized main GQA3D | `biv150_gqa_decode.py:342–424` intercepts eligible pure decode; its full/tail K reads are already the separate PR79 work. | Not modified or duplicated here. Its source uses exp2/log2 units; SDK helpers396–418 use natural exp. This factory retains the SDK's natural softmax scale. |
| Reducers | SDK663–748 and main GQA reducer read only partial output/max/sum, never K. | Byte-identical; no reducer rewrite or shared-format inference. |

Main mixed splitting (`biv150_mixed_decode.py:192–225`) creates decode-prefix and
prefill-suffix views with adjusted cu_q, then calls the existing forward twice.
The source reader itself also handles noncontiguous qlen1 requests using original
cu_q indexing. Historical C3 outside counts are before local ordering and do not
prove an unoptimized decode suffix. Benchmark/profile event frequency is not a
time share. This preparation does not repeat a batch-ordering experiment.

## One packed format; three different pointer units

PR79 codec v2 is `cjj-konly-rn32-rne-v2`: scale zero→1, otherwise
`max(RN32(maxabs/127),2^-126)`; code is
`clamp(RNE(RN32(BF16_value/scale)),-127,127)`. The exact ae00 codec helper is
included in each generated source. Original BF16 caches must be retained as
separate encoder inputs; in-place compression is not licensed here.

| View | Pointer origin and address unit |
| --- | --- |
| Kcode I8 | Whole packed allocation, offset0; strides `(16384,256,128,1)` in bytes. |
| Kscale FP32 | Flat whole-page view, offset0; `page*4096+1024+slot*2+head` in FP32 elements. Passing a view already offset1024 would be wrong and is rejected. |
| V BF16 | Same allocation, offset4096 BF16 elements (=8192B); strides `(8192,256,128,1)` in BF16 elements. |

Each page still allocates16KiB: Kcodes0–4095B, Kscales4096–4223B, reserved
4224–8191B, original BF16 V8192–16383B. The address oracle covers every dimension
of both heads/all16 slots at page0/1/2^16/2^31−1, including offsets beyond2^43.
Masked final-tile lanes can read padded table metadata (the original SDK table
load is unmasked); K/V/scale data loads remain masked. The CPU oracle rejects an
insufficiently padded table and does not dereference poisoned invalid pages.

Built-in SDK `KV_QUANT_MODE=2` is **not** this format: it also reads Vscale and
changes P/PV at559–560. This factory requires mode0 plus a new private Kscale
argument. Its descriptor validator checks explicit format, dtype, ranks, units,
view origins, storage bounds and aliases. It rejects encoded→BF16 fallback.
This is a CPU metadata contract, not a claim of validated runtime Tensor binding.

Raw SDK3D flattens scratch addresses using the logical NUM_SEGMENTS value
(SDK596–600); it has no GQA scratch-stride arguments. A narrowed S4 view retaining
physical S16 strides therefore fails this profile's contract. It must receive
the corresponding contiguous logical `(T,16,S,128)`/`(T,16,S)` layout, or remain
unavailable. The GQA/PR79 narrowed descriptor cannot be copied blindly.

## Source proof and numerical boundary

The CPU factory derives normalized kernels using the exact pinned main stdlib
stride transformer, or independently extracts SDK raw3D. Removing the Q codec,
Kscale ABI/helper, K-cast change, I32 QK expression and private guards returns
each canonical kernel **byte for byte**. All V, P/PV, masks, FP32 softmax,
segment/reducer and output expressions are preserved. Mutants changing those
expressions, scale mask, QK output dtype or source pins are rejected.

Q quantization occurs once per active CTA after its original Q load and before
the KV loop; raw3D repeats it per active segment, which is a real future cost.
**Actual I8 operand hoist is unverified.** The accepted BF16 Q-fragment transform
matches a BF16 local_load and cannot certify the new I8 operand. Source order
does not establish physical placement, registers, staging, synchronization or
speed. A future actual-IR gate must prove the I8 operand conversion/local read
is outside the KV loop; an unknown grammar stops before numerical/timing work.

I32 raw dot is exact for given codes: magnitude≤128*127^2<2^24. Intended FP32
score order is `(I32*sQ)*sK*natural_softmax_scale`, each multiplication RN32;
actual target lowering and FTZ behavior are not proven by Python syntax.
Independent CPU references separately compute original-BF16 global FP64
softmax×V and quantized-operand global FP64 softmax×V from the original rows.
Neither emulates BF16 P, tensorcore PV accumulation or device FP32 softmax.
Their toy-fixture differences are observations, not a component precision pass.

## CPU replay and next mandatory evidence

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_prefill_cpu.py --out NEW_CPU_JSON
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_prefill_cpu.py --emit-dir NEW_SOURCE_DIRECTORY
```

The checker imports no Torch/Triton and executes no JIT or kernels. It covers
four source profiles,8 complete causal grid/mask cases,16,384 address vectors,
zero/tiny/half-boundary v2 codec fixtures and malformed-source/format/stride/
alias/mask/scratch counterexamples. The preserved CPU JSON records the counts.
`--compile-only` and device/timing options deliberately do not exist.

Wait for PR79 device codec qualification, then root may arrange actual compile
and numerical work. The next gates must bind actual source/options/descriptors,
prove native I8 QK/I32 output and hoisted Q operand, record all known resources,
and retain original BF16 PV/softmax/reducer behavior. End-to-end memory, codec,
baseline-independent precision, finite, changed-input Graph and side-stream
checks remain mandatory. Full cost must include Q quantization, writer and all
producer/reducer work against current main BF16/Q32/M2; producer-only timing
cannot qualify a service improvement. No deployment hook, complete runner,
actual IR/resource result, numerical pass, throughput/TTFT/model105 result or
capacity recovery is present in this Draft.
