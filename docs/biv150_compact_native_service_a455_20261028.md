# Compact40f9 service candidate on team main a455

The reviewed compact reader and existing dc5 writer now have a default-off
service entry. This candidate normally merges latest104 `f313f948` with108
development head `faf88f56` (which includes team main
`a455a3de364051fbea80226870c239de3a48e716`). It depends on104's host integration,
100's actual40f9 component, and108's fullstep evidence. It does not include103.

The change fills104's previously pending component source record with the real
SO review, four-case consumer and complete-step summaries/review. It binds
physical12544B format, CUDA3bd, writer a0cacf and views1ab738. Cold preflight
checks the real40f9 SO and installed writer/views bytes before Torch; the service
manifest and C3 retain the same identities. The compact installer now recognizes
main a455's native-GQA parent source. No GPU, codec, prefill, scheduling, sampling,
batch or CLI behavior is rewritten.

All compact CUDA/host/codec files remain byte-exact to the actually compiled
PR100 source51da. All six compact Triton GPU sources remain byte-exact to dc5.
The separate main8b86 late-V-wait decoder remains16KiB and is not used for this
arena. No late-wait changes entered the compact CUDA by merging main.

The actual fullstep summary is
`53880d7cad7beb47a07a4551056b616afdba45e6fca4703da98aaea9816f036f`;
Root's independent review is
`96262cd2414ce85a3648f0d45e1727f93b243e754310e31e1160fc30639ebdb0`.
It passed the original hard implementation reference for34276 pages, complete
cache bytes, Graph and side-stream execution. Its compact step means are about
0.39683/0.76895ms for4K/16K, compared with aace BF16 about0.41641/0.80549ms.
These are component timings, not model throughput or a comparison against8b86.

The K8 hard reference uses exact quantized operands; the BF16 arm has its own
original-BF16 reference. Neither is a model accuracy score. PR91's separate
original-BF16 quantization proxy failures remain valid historical evidence;
passing100's smaller proxy does not override them. PR107's model100/105,
supplement102/105 and service throughput are not inherited by this candidate.

42-layer spec/allocation/reshape/forward-binding hooks are unchanged except for
a one-time `BIV150_KONLY_COMPACT_ALLOCATION` metadata record after initialization.
It includes actual configured blocks and every actual layer storage extent.
The unchanged allocator calculates capacity from12544B pages; no block-count
override or extra memory is requested. Metadata still reports
`capacity_reclaimed=false` until Root compares actual service capacity.

Native scope remains only N64/S2 and N32/S4 with query pitch2048/2560. N40, N48
and other existing unsupported shapes select the same private compact Triton
reader. New bounded `BIV150_KONLY_COMPACT_NATIVE_FALLBACK` records retain actual
shape, segments, query strides and capture status so increased capacity's real
buckets can be measured before any expansion. Counters are host/capture events,
not Graph replay counts. No new native shape is silently qualified.

Use the same official service CLI with these explicit candidate flags:

```sh
export VLLM_FL_BIV150_KONLY_COMPACT_SERVICE=1
export VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_DECODE=1
export VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY=/absolute/actual40f9.so
export VLLM_FL_BIV150_KONLY_COMPACT_NATIVE_LIBRARY_SHA256=40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453
export VLLM_FL_BIV150_KONLY_ENCODED_SERVICE=0
export VLLM_FL_BIV150_KONLY_NATIVE_DECODE=0
export VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST=0
```

The ordinary main native-GQA parent remains enabled under its usual settings.
The compact stats path may be set with `VLLM_FL_BIV150_KONLY_COMPACT_STATS_PATH`;
otherwise the existing experiment-local Triton-cache parent determines it.
There is no new whole-token audit or per-input device readback admission gate.

CPU checks:

```sh
python3 -B tools/check_biv150_konly_compact_install_cpu.py
python3 -B tools/check_biv150_compact_native_qualified_service_cpu.py --library /absolute/actual40f9.so
```

The first executes the original42-layer metadata/spec/allocator/C3 mock path.
The second reads the actual ELF without loading it, exercises real preflight,
attachment/ownership routing through existing CPU recorders, preserves32/64
scope and40/48 private fallbacks, rejects wrong identities/aliases/epochs,
checks the actual package guard and distinct compile fingerprints, and verifies
unchanged GPU sources. Mocks do not establish actual device or model correctness.

Pending Root's serialized execution: real42-layer allocation and block counts,
full service prefill/decoder/writer behavior (including KV source pitch2560),
official105, supplemental quality, warmed fixed-CLI4K/16K throughput and TTFT.
Default-off status, all numerical thresholds and precision gate remain unchanged.

The qualified CPU checker also runs the native-ON installation in a fresh
subprocess using the existing42-layer framework mocks. It executes the real
installer, binds40f9, replays actual spec/config/allocator/reshape methods,
registers a writer-owned cache and obtains the real combined host/native C3
receipt. Its42 allocations and2 blocks/layer are mock metadata, not a device
capacity result. The second CPU report is `qualified-cpu-v2.json`.
