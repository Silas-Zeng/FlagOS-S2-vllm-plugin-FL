# Actual-service Q32 bounded bitwise/numerical result

Root ran immutable frontend e4dd64e453fbe1bdeba4d8b2a1b551167e9446c4, owner652604/start346185455 same-startZ. The original42 files/60998965B and download manifest are preserved exactly. Because original Torch `.pt.gz` files exceed2MiB, multipart archives contain raw byte fragments plus the small original files. Each of32 parts is itself a valid gzip tar under2MiB; `index.json` records every member/original offset/length/SHA and whole original SHA. Reassembly must cover each original with no gap/overlap. Total archives57097510B; indexSHAdd0361c56d1936574fcec31c6890f15bfc5a9c587f544cbb102bb40af9a6f6e0. No original tensor, result, metadata, owner or failed precision field is rewritten.

Default CPU-only byte-archive replay and six corruption faults:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/biv150_konly_service_q32_bitwise_archive.py --directory docs/evidence/konly-service-q32-hoist-20261027/bitwise-native-v1
```

Independent full scientific CPU replay needs NumPy, imports neither Torch nor Triton, and enters no CUDA:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/replay_biv150_konly_service_q32_bitwise_cpu.py --directory docs/evidence/konly-service-q32-hoist-20261027/bitwise-native-v1 --out FRESH_CPU_PROOF_JSON
```

The original snapshot and multipart-reassembled snapshot produce byte-identical independent proofd01bc7645b4358f61103e6339cf71aa2404a2da24465595de68ed0f9d70a47f3. The earlier immediate core proof f1c41b6cfa5e710b65c0ce8f8830be6a2324b3ef8fce144e4c37bce800dfdbce is also kept unchanged.

All actual compiler artifacts are rehashed and bound to the fresh6d native record: same original source/typed signature/constants/attrs/metadata and exact source TTIR/TTGIR/LLVM. Actual new native Q hoist is rerun, while cubin equality and ASM text remain separately reported, never assumed binary equivalence. Loaded device-handle resource observations are distinguished from prelaunch `n_regs/n_spills=None`:

| Actual role | Hash prefix | Registers | Spills | Shared | Geometry |
|---|---|---:|---:|---:|---|
| Encoded unhoisted | c0a523d9 |135|0|57344|B256/Q32/T64,w16/s1,warp64|
| Fresh hoisted |597cf3bd|122|0|33792|same|

These are recorded actual loaded-handle attributes checked before the four reader executions, not an occupancy or performance measurement. Both roles use the same359 source and CPU-encoded input. No GPU encoder was run.

The independent replay reads full before/after/out bit arrays with a whitelist-only CPU Torch-serialization reader; it does not import Torch or execute arbitrary pickle globals. It checks both full-storage canaries71, all pre-output NaNs, all post-output finite, output-copy/full-storage-interior equality, complete unhoisted/hoisted output byte equality and full readonly hashes. CPU original BF16 Q/cache/table/lengths/starts/descales independently reconstruct those hashes. It rebuilds all K codes/scales with exact rational RN32 repair and integer RNE, rebuilds every16KiB page including V and padding, and compares full uploaded/expected/cache readonly bytes. INT64 QK plus FP64 dequantization/global softmax are recomputed without using saved codes/scales as an oracle; saved CPU references match within5e-13 (different FP64 reduction order is reported, no original gate changed).

Scope remains two patterns, Q2048/KV6144/prefix4096/one sequence, four reader launches/zero encoder, same candidate profile/grid65x2. Pattern0 checks all2048 rows; pattern1 checks the established eight rows. Both roles have identical full outputs and hard finite/codec/native identity/resources/canary/readonly/pointwise-quantized-reference gates. Pointwise threshold remains0.005+0.005abs. Recomputed errors:

| Pattern | Quantized FP64 relative RMSE | Original BF16 FP64 relative RMSE | Original proxy0.005 |
|---|---:|---:|---|
|0|0.0017247786112862535|0.0017247786112862535|true|
|1|0.0022699815155583425|0.009644218032913768|false|

`original_BF16_quality_passed=false` is preserved. Explicit diagnostic allows this proxy failure only; hard gates are unchanged. Bitwise equality shows this hoist preserved this encoded algorithm's outputs on the requested cases. It does not establish original-BF16 equivalence or whole-model accuracy. `precision_gate=false`, `full_numerical_matrix_verified=false`, model/Graph/side/service=false and timing/performance=false remain. No new throughput or earlier6% recovery claim is made. The actual service adapter, complete model105 and formal throughput are separate Root-owned gates; no installer/production source was changed here.
