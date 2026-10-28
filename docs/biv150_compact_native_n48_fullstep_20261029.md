# N48 same-shape compact fullstep: CPU-prepared component

This tools-only child starts at `1c0de57574f66f5401c0d1443ccbc57bb1d44a67`.
Frozen N48 CUDA, codec, producer C++ and production are unchanged. No GPU was
executed while preparing this component. It has no service/model/performance
qualification until Root runs and reviews its separate actual result.

The performance denominator is the **real production compact TL48** reader
`fc25c86a13fc273bb7368b83af4ca264524b4e12:vllm_fl/patches/biv150_konly_compact_decode.py`
(`ea955c009c7b3810ac6e6332ca4ac3d01ef535281f4332bcc708f016aea5e8b7`).
The complete file is materialized unchanged. Its original RN32/RNE helper is
loaded normally and checked by object identity and exact source SHA
`16afc46585f950f4b9c06c831b0e0dbf5515ee941a476a9ed300c492ebf947f9`.
The call uses the shipping GQA ABI: grid `(48,2,4)`, ROWS/queries-per-KV8,
HEAD128/PAGE16, one warp and one stage. Scale is FULL page0 F32 with
`page*3136+1024+slot*2+head`, and V is already offset4352 bytes.

The other producer is actual library
`2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1`,
namespace `cjj_native_k8_compact_n48_s4_probe`. Its actual loaded-library set and
operator schema are checked. Both arms use the same exact dc5 compact writer,
same width16 scratch narrowed to S4, same source/table/metadata, original
stride-aware reducer and streams. No old BF16 native library is loaded.
Two N32 calls are **not** a performance denominator and do not occur in this pair.

## Bounded geometries and append semantics

Only two cases are prepared, both with Q pitch2560 and **K/V source pitch2560**:

|case|query rows|nonempty rows|base KV length|reserved pages after3 appends|spare of44769|
|---|---:|---:|---:|---:|---:|
|valid41|48|41|17408|44649|120|
|valid42|48|42|16896|44394|375|

Padding rows have zero length, all-minus-one tables and negative append slots.
`42*17408` is rejected. A deterministic nonoverlapping physical permutation of
all44769 pages reserves `ceil((base+3)/16)` pages per nonempty row. Each A/B/C
snapshot independently proves its accessed pages and append slots fit those
reservations. A uses base+1, changed B at most base+2 and side C at most base+3;
B/C also contain short nonempty tails. Each snapshot has a counter-defined
existing prefix and exactly one current append. This is not a claim of three
successive generated tokens: restoration changes the independently defined
prefix before each diagnostic snapshot. Timed replay repeats a fixed append
snapshot, as in the existing fullstep.

Q/K/V each have a separate readonly backing, with padding bytes retained and
checked in full. This verifies the actual2560 source row stride; it does not
cover a shared model QKV allocation. Initial BF16 source construction, complete
CPU encoded oracle, GPU upload, linear source packing, initial slot preparation,
initial writer JIT/load and full dc5 encode are all reported separately from
step timing. The writer never calls an old BF16 writer on live encoded K.

## Gates and preserved timing

The existing compact fullstep run body is adapted reversibly. Its `timed`,
`capture`, `reducer`, `poison`, `fingerprints`, `verify_inputs` and `step` functions
are byte-identical. The existing complete D128 I64-QK/FP64 golden math is called
unchanged. Both arms implement the same quantized algorithm, so one independent
golden per phase is used for both; cross-arm bitwise equivalence is not imposed.

Before timing, both arms must pass full partial/max/sum/final `.005` pointwise
and RRMSE gates, pagewise BF16-P implementation partial reference, ideal quantized
FP64 final, full outside canaries, all inactive/unused-segment poison, source row
padding/readonly checks, and streaming equality of **all44769 cache pages**.
Ideal-partial diagnostics remain separate and can fail. Changed Graph B precedes
fresh eager B; each arm's complete buffers must match its own fresh eager result.
Changed side C and pre/post-timing checks are retained.

Every actual JIT artifact and signature/constants/attrs/metadata is saved for the
initial encoder, TL append, native append, TL48 producer and reducer. These are
actual new stride2560 specializations. Exact writer typed ABI, TL signed-I8/F32
scale/BF16-V pointer ABI, live `.src.fn` identity, positive postlaunch register
count and zero spills are required before timing. Unknown resources stop the
component. Native static resources remain the separately reviewed actual SO
metadata, not occupancy. The complete new frontend source is copied to output.

The original warmed protocol is unchanged: warm128 graph replays per arm,
unroll8 steps per graph and ABBA with64 replays per measurement; GPU-event and
Graph-launch-plus-sync host-wall milliseconds are divided by8. Each step contains
dc5 append + producer + original reducer. No score or kernel benefit is inferred
from this CPU preparation.

## Actual small-consumer prerequisite

The hard gate binds all of the following actual files, before Torch/CUDA:

- consumer summary `493e210f8eeaac6006b25a94ed55e08fbe932337d6481fe3d8ec6e86d0033272`;
- full170-file archive manifest `51ba899cc0b0767750b6a7018027448b01a925a26ea7963ba21ddf5e647c250f`;
- independent array report `c07937c3d6c3ffc78a5c7b213271e4f864a9463775fe932f9f2aadbff08ef4d0`;
- actual compile/SO and Root review `14c8f26fa0d7afb7d00d480aad9ec5cf84f490d11cca8da8cb5f6a959a114a50`.

It rehashes all original archive/artifact files and decompressed snapshots,
compares executed consumer source to1c0, replays actual compile/SO admission, and
requires the independently issued report's real identity/count/scope fields.
The report has176 hard gates,32 BF16 proxy observations,96 first48 field pairs,
168 regenerated golden arrays,24 epochs and2 host refusals. Its48 ideal-partial
diagnostic failures are retained. This prerequisite only qualifies29 pages,
maxctx193 and the tested41/42-row padding patterns, not this long fullstep.
Evidence was separately shared at `ff2fde66096edb9d81fe66d1290d7b98ed9e84af`.

## CPU replay and Root-only execution

Default plan and local CPU source/data tests import no Torch/Triton:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/probe_biv150_native_k8_compact_n48_fullstep.py
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/check_biv150_native_k8_compact_n48_fullstep_cpu.py --cpu-data --out /private/tmp/n48-fullstep-cpu-NEW.json
```

The staged consumer proof directory below must contain the original `out/`,
`exports/` and byte-identical `download-manifest.json` tree. All raw original
paths/SHA/bytes must remain intact; there is no metadata-only substitute. Root
may stage the already completed original tree without rerunning its consumer.
The exact Git objects3eb/dc5/fc25 used for source provenance must resolve locally.

Use the following shared arguments first with `--preflight-cpu` (omit `--out`)
to verify all actual prerequisites using stdlib only. Root alone may then use
`--gpu --root-owned-exclusive` after its external owner/queue checks:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PY" -B -W error tools/probe_biv150_native_k8_compact_n48_fullstep.py \
  --gpu --root-owned-exclusive \
  --consumer-raw /workspace/experiments/minicpm-s2/exports/cjj-native112-n48-consumer-proof-v1-20261029 \
  --consumer-archive-sha256 51ba899cc0b0767750b6a7018027448b01a925a26ea7963ba21ddf5e647c250f \
  --consumer-review /workspace/experiments/minicpm-s2/exports/cjj-agent-native112-consumer-independent-review-20261029.json \
  --consumer-review-sha256 c07937c3d6c3ffc78a5c7b213271e4f864a9463775fe932f9f2aadbff08ef4d0 \
  --compile-summary /workspace/experiments/minicpm-s2/cjj-compact-native-n48-compile-v1-20261029/compile-summary.json \
  --assembly-review /workspace/experiments/minicpm-s2/exports/cjj-root-native112-n48-consumer-review-approved-20261029.json \
  --native-library /workspace/experiments/minicpm-s2/cjj-compact-native-n48-compile-v1-20261029/build/cjj_native_k8_compact_n48_s4_probe.so \
  --out /workspace/experiments/minicpm-s2/cjj-native112-n48-fullstep-v1-20261029
```

Default device selection is both cases; `--case valid41` or `--case valid42`
can select one without changing any numerical/timing gate. Output must be new.
No new native compilation, service installation, model run, CLI change, or
actual-service-frequency claim is part of this frontend. The pair's fixture
tensor allocation is approximately2.43GiB plus CUDA/JIT/runtime/allocator costs;
CPU cold/oracle arrays together are about1.37GiB before reference temporaries.
