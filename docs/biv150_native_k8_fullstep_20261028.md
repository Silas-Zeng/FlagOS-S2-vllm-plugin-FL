# Native K8 fullstep component, prepared CPU entry

This entry compares existing native K8 SO `22e4a5e2…` with existing b3 native BF16 SO `aace693b…`. It compiles no C++ and installs no service. PR83's writer/contract are copied byte for byte; original 377/388/52c/9695 files, kernels, dtypes and thresholds remain unchanged. Dependencies: PR83, PR93 corrected consumer v2, and merged PR94's append/reducer/arena protocol.

Actual consumer v2 summary is pinned to `c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128`. Its four cases, artifact hashes, complete counter inputs, full O/M/L/output arrays, canaries, inactive poison and changed Graph-B/eager-B equality are replayed before importing Torch. This qualification stops at context193. The previous consumer v1 ideal-partial failure remains recorded; no historical failure is changed. The original v2 summary and Root's independent review are in `docs/evidence/native-k8-qk-feasibility-20261028/consumer-v2/`; all158 raw files were verified locally, their full archive is deferred.

The new cases are N64/context4096/S2/Qpitch2048 and N32/context16384/S4/Qpitch2560, each with34276 physical16KiB pages, seeded nonoverlapping physical mapping and8192-wide table. Q/K/V vary independently across all128 dimensions. Native scratch allocates width16 and narrows toS2/S4, preserving head/token strides and poisoning every unused segment. Both new-KV sources are contiguous pitch256; actual service KV sourcepitch2560 is outside this timing scope.

Initial counter construction, independent RN32/RNE packed-byte oracle, upload, existing initial-encoder JIT/load, BF16 copy and full K8 initial encode have separate clocks. Initial encode passes already-offset V; exact83 append passes full-page0 V. K/code and F32scale use the same full-page0 address. Layer storage starts at nonzero offset256B. Initial source/destination separation and every initial/cache byte are checked. Cache checks stream every byte, recording chunk hashes instead of archiving repeatedGB cold buffers; the counter, source pins and append inputs reconstruct them. Whole initial GPU source, query padding, input guards, output guards and scratch poison are also checked.

Every arm/context runs full independent FP64 O/M/L and final-output gates before timing. Partial follows page16 online updates with source-defined FP32 score/rho operands, BF16 P rounding at each page and independent FP64 accumulation. Ideal unrounded partial remains a separate diagnostic. Maximum/sum and final output keep ideal FP64 references, with BF16 or I64 quantized-QK/dequantized scores for the respective arm. All hard arrays retain pointwise `absdiff <= .005 + .005*abs(reference)` and `RRMSE <= .005`; empty outputs require positive-zero bits. No BF16 proxy qualification is borrowed from quantized references.

Changed Q/cache/table/length/new-KV Graph-B replay precedes fresh eager-B. Side-C and complete same-arm output bits are checked. Actual83 typed6-pointer+6-I64 ABI, helper/reducer postlaunch registers/spills, IR artifacts and actual tensor strides/offsets are recorded before timing. A single warmed ABBA then measures append+native producer+the same b3 reducer, using128 warm graph replays,64 measured graph replays and8 steps per replay, matching PR94. GPU-event and graph-launch/sync host-wall clocks remain distinct. Cache/source/OML/final checks run again afterwards. This is a component protocol, not service throughput or model qualification.

CPU commands (NumPy only in the checker; default plan is stdlib):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_native_k8_fullstep_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/probe_biv150_native_k8_fullstep.py
```

Root-only future GPU command, in the fixed target environment and a new output directory:

```sh
python3 -B tools/probe_biv150_native_k8_fullstep.py --gpu \
  --consumer-summary /workspace/experiments/minicpm-s2/cjj-native-k8-consumer-v2-20261028/consumer-summary.json \
  --consumer-proof-sha256 c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128 \
  --k8-library /workspace/experiments/minicpm-s2/cjj-native-k8-compile-v1-20261028/build/cjj_native_k8_qk_probe.so \
  --bf16-library /workspace/experiments/minicpm-s2/cjj-main-b3-native-gqa-build-v1-20261028/corex_native_gqa_s2_probe.so \
  --out /workspace/experiments/minicpm-s2/cjj-native-k8-fullstep-v1-20261028
```

No device run of this entry has occurred at preparation. Larger-context correctness, actual width16 ABI, graph/side and paired timing remain pending. `precision_gate`/`performance_acceptance` remain false; a single component ABBA cannot establish model or stable service gains.
