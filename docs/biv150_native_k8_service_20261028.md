# Default-off existing NativeK8 service adapter (CPU preparation)

This independent branch starts at main4229 (production b3) and normally merges PR89 dependency `2e3abf3f9bcae78e37d8cf3a63d8fcfc6a609db9`. It retains89's owned-format producer interception **before** stock BF16 NativeGQA. The new native choice runs only inside the private GQA facade, after its exact registered cache-pair lookup and typed-role reconstruction. Stock profiling/native/Triton behavior is unchanged. Armed unknown storage, aliases or compact12544 never reach the new native code or stock BF16 fallback.

No CUDA/C++/codec/reader/reducer math is changed. The six generated89 GPU source files remain byte-identical. Existing22e library is reused; new service smoke, model coverage and performance are **pending**. Native component fullstep3eb reports about1.053–1.054x complete step versus native BF16, not model gain. Its tensor cases/evidence do not qualify this new service source.

## Explicit interface

`VLLM_FL_BIV150_KONLY_NATIVE_DECODE` defaults0. Off does not import/load/install the new adapter. On requires the existing `VLLM_FL_BIV150_KONLY_ENCODED_SERVICE=1`, compact0 and both:

- `VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY`: existing absolute22e SO path.
- `VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256=22e4a5e2a0b9d9a36ea390a11126417e1bf58e3c27f6b96ba0913b27690f09dd`.

The actual file is hashed before and after `torch.ops.load_library`. Exact dispatcher schema/CUDA registration, loaded-library set, Torch2.10/CXX11-ABIfalse, actual imported module source path/SHA and manifest entry must match. An already registered operator without this loaded library refuses startup. Library, op, adapter/helper code and registry/epoch identities are rechecked by actual C3 receipts. The compile fingerprint includes the default-off feature and enabled `konly_encoded.native_decode` with enabled/library SHA/operator; the implementation digest includes all new private code and the source manifest. Worker preflight rejects incompatible format/flag combinations and its sole C3 runner pin is updated.

## Shape and memory fence

Only logicalN64/S2 orN32/S4, each with actualQ stride2048 or2560, may choose native. Both pitches have small-consumer evidence; only64/2048 and32/2560 have the recorded large fullstep pair. This distinction appears in each event. Query/table/metadata, source-alias and separate scratch backing checks execute without `.cpu()`, `.item()` or GPU data reads. Cache ownership/unchanged allocation/epoch and all role pointers/strides are verified again: K I8 and F32 scale both fullbase, V already+8192 bytes, physical16384B pages. Scalar launch values must match actual Tensor strides. Scratch is narrowed from backing width16; its leading capacity may exceed logicalN, as allowed by the unchanged C++ ABI. Data validity/freshness continues to inherit the pinned89 scheduler/writer/Graph lifetime contract; this is not a new live-metadata readback system.

Unverified logical shape, pitch, grid, options/table/scratch specialization or attention scale retains the **same private Triton** producer. Invalid ownership, pointer, backing, alias or scalar observations raise. A native invocation happens exactly once on the current stream; exceptions propagate and never retry/fall back after launch. Original reducer4 remains outside the native adapter and unchanged. No prefill, writer, cache format/allocator, threshold, official service CLI or upstream dependency changes.

## Controller and coverage

`BIV150_KONLY_NATIVE_DECODE_INSTALL` records actual torch/module file+SHA, library path/SHA, schema/operator and confirmed loaded set. Only after the dispatcher call returns does `..._SELECTED` or `..._CAPTURED` record logicalN/S, Q/table/partial/max/sum geometry, entry generation, format/page bytes, actual22e library and qualification scope. These are bounded host/capture events, **not Graph replay counters**. Final `stats.native_decode` exports selected host calls, private-Triton fallback reasons and all recorded geometries. Existing parent stats and Q32-hoist behavior remain.

The unchanged official Math105 protocol has parallel8. It can predominantly or entirely select private TL; even102/105 must not be called NativeK8 branch model qualification if only startup native capture is observed. Root must keep that original protocol and add independent eligible32/64-request smoke / supplemental Math105 diagnostics, with actual request/bucket evidence. Supplemental concurrency does not replace official105 or change vLLM CLI/thresholds. No unverifiedN8/S16 instance, math switch or bypass is added. A cumulative startup selected/captured count alone cannot prove measured-request native execution. Root owns controller changes, GPU smoke,105 and sufficiently warmed4K/16K score runs.

## CPU evidence

`tools/check_biv150_native_k8_service_cpu.py` parses the actual existing221904B22e ELF, its embedded registration schema and both launch symbols without loading it. It then executes the **actual attachment routine, private Runtime/Registry/facade/producer and C3 receipt** with explicitly labeled metadata/dispatcher recorders. Four shape/pitch cases, capture records, wrong pointers/epochs/aliases/scalars, installed-op identity, private TL fallback, stock poison and postlaunch no-retry are checked. The ordinary main stock-native boundary is independently replayed. This is stronger than a selector stub, but it is not a real Torch library load, Tensor math or service proof. Six GPU sources and all csrc are unchanged. Full actual22e compile/codec/small-consumer/fullstep evidence remains PR93's dependency with its original scope.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_k8_service_cpu.py
```

Default library location is the existing local Root compile-record copy. `--library /ABS/EXISTING22e.so` replays the same CPU ELF check elsewhere; no Torch/GPU import or execution occurs. No author SSH/JIT/GPU, dependency installation or service activation was performed. Keep Draft; no new source service accuracy/TTFT/throughput/capacity acceptance is claimed. Do not run89's historical materializer over this overlay: it reconstructs the old dependency and would remove these hooks.
