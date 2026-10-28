# Optional N48 runtime capacity query: exact2ad and5cf

This independent tool is CPU-prepared only. Actual SDK compilation, normal Torch SO loading, runtime pointer resolution and capacity queries are **not yet verified**. The default command only checks the two actual library hashes and writes a plan. It neither imports Torch nor compiles a helper. No production source, kernel, consumer, numeric gate, service parameter or benchmark is changed.

The tool starts from PR113 source `1dc7ce5c2dd9050712aa9d5e7cba59fe3699f8d4`. It extends the archived32-line `runtime_query_example.cpp` from geometry notes `a19e7a9`. The original example and2ad plan are retained byte-for-byte. `Root-candidate5cf-static-admission.json` is Root's actual5cf source/ELF/host-binding report, not an occupancy result. `bindings.json` records the exact report checksums and actual S4 registered host stubs, virtual offsets and SO hashes. It does not admit unused S2.

| Actual SO | Runtime query target | Static shared from ELF |
| --- | --- | ---: |
| `2ad51a98…` | exported `_Z49__device_stub__…ILj4EE…`, offset100800 | 8192B |
| `5cf698e8…` | exported `_Z58__device_stub__…ILj4EE…`, offset100800 | 6144B |

With explicit `--query`, the tool compiles only the host C++ helper against installed Corex headers and `libcudart` (the same SDK include/library locations recorded in actual113's build). No GPU source or target library is compiled. Both exact SOs are then normally loaded through `torch.ops.load_library` in the **same Python process**. The helper uses `RTLD_NOLOAD`, exact `dlsym`, and `dladdr`/realpath/symbol-start/virtual-offset checks to bind each registered host stub. It never calls the stub, the launch wrapper, a Torch operator, or a driver `CUfunction` API.

After binding, it records the current logical device, real device properties, runtime/driver versions and return codes. It requires the requested current device and never calls `cudaSetDevice`. It calls only runtime `cuda/ixFuncGetAttributes` and runtime `cuda/ixOccupancyMaxActiveBlocksPerMultiprocessor`, with64 threads and **0 dynamic shared bytes**. Static8192/6144 is not added again. Returned attributes and runtime active-block capacity are preserved without substituting static ELF estimates. Successful API queries are not measured residency, performance or numerical qualification; `numRegs` is not a new SGPR measurement. Shared-size agreement with ELF is a separate reported observation.

The helper and Python wrapper contain no explicit kernel launch, tensor/device allocation, attribute setter or device-switch call. The runtime itself may initialize internal context state; Torch's own initialization flag before/after is recorded and is not a claim that all runtime initialization was absent.

Compilation has a60-second limit. Unsupported headers/linkage, missing symbols, provenance mismatch, API errors or a current-device mismatch produce `diagnostic_unavailable`; errors and return codes are retained, with no invented values. The wrapper writes incremental `result.json` before runtime calls. Root should use a whole-process timeout as below in case a runtime call hangs. A refusal or timeout is **not an ABBA blocker**, and never bypasses the existing numerical gates. The script exits normally for handled diagnostic failures; read the JSON status rather than interpreting exit0 as occupancy success.

## Commands

CPU plan with local actual SOs:

```sh
python3 -B tools/query_biv150_native_n48_runtime_capacity.py --reference-library /private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_s4_probe.so --candidate-library /private/tmp/cjj-native113-n48-shared6k-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_shared6k_probe.so --out /private/tmp/new-capacity-plan
python3 -B docs/evidence/native113-runtime-capacity-query-20261029/check_source_cpu.py
```

Root-only future execution, after confirming the current service has ended and the device is idle:

```sh
timeout 90s /usr/local/bin/python -B tools/query_biv150_native_n48_runtime_capacity.py --query --reference-library /workspace/experiments/minicpm-s2/cjj-compact-native-n48-compile-v1-20261029/build/cjj_native_k8_compact_n48_s4_probe.so --candidate-library /workspace/experiments/minicpm-s2/cjj-native113-n48-shared6k-compile-v1-20261029/build/cjj_native_k8_compact_n48_shared6k_probe.so --sdk /usr/local/corex --compiler /usr/local/corex/bin/clang++ --device 0 --out /workspace/experiments/minicpm-s2/native113-runtime-capacity-query-NEW
```

Each output directory must be new. Preserve `result.json`, original compiler stdout/stderr, and the compiled helper with its recorded SHA. Do not chain this optional command as a required `&&` prerequisite for ABBA. If externally timed out, preserve the partial report and timeout return code; it remains an unavailable diagnostic. The reference remote path comes from original112's compile summary, and the candidate path from actual113's compile summary; no new or guessed SO is used.

The included static/CPU report verifies exact hashes, source API-call restrictions, runtime versus driver API distinction, explicit64/0, default no-query behavior and wrong-SO refusal. It does not claim actual SDK ABI compilation, dlsym/dladdr success or occupancy data.
