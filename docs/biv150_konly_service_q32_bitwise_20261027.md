# Fresh service Q32 native and bitwise falsifier

Actual6d compile now completed: new native mechanism is CPU-verified on hash597cf3bd3f7fda00c986b8af314f37fe59615a30b713e36dc6e40b804b3cfd05 (see fresh-native-v1 originals/audit). Loaded resources and outputs remain pending. This adds only tools/docs after6dcd3d8 was actually launched by Root. All6d files and production/kernel/installer sources remain unchanged. It neither installs nor enables the candidate. It depends on the new actual service359 compile, rather than treating historical PR80 artifacts as a service licence.

The default commands import no Torch/Triton/Numpy, enter no CUDA, compile nothing and launch nothing:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_service_q32_native_cpu.py
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_konly_service_q32_bitwise.py
```

The first is a historical native-grammar regression and33 protocol faults, explicitly `actual_new_service_native_verified=false`; the second prints the future bounded plan. Python3.12.14 and3.14 outputs match byte-for-byte: checker28bd38d6e5665b38020ec7654311d728a5e3653068326f39e0edd696e1a77de2; plan423abd0a5fcd69edaa1ec6edb8fc271ca993f6199fb96709d31ad96e8184d205.

Root must first review the new real compile without device activity:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/audit_biv150_konly_service_q32_fresh_cpu.py \
  --compile-out ACTUAL_V1_COMPILE_OUT --out FRESH_NATIVE_AUDIT_JSON
```

The CPU reader rehashes all saved compiler artifacts, checks the unchanged worker-cache/proposal binding and live source/typed signature/constants/attrs, verifies exact original-service unhoisted TTGIR/LLVM and unchanged source TTIR, then audits the actual newly compiled override. Unknowns fail. It never rewrites raw artifacts or relaxes the original6d reconstruction guard. No historical kernel hash, layout alias, block name or Q register number is an admission predicate.

Actual proposed TTGIR must equal the newly proposed source move except scoped SSA naming. LLVM independently finds the natural KV loop containing all16 I8-QK and32 BF16-PV calls, traces the MMA A roots to16 two-byte shared reads, requires the original16 BF16 global Q loads and48 RN32 divisions outside/dominating KV, then proves Q shared store/publish/read and K-bank reuse/store/acquire/read dominance. The ASM parser splits fallthrough blocks after conditional branches, rediscovers its natural KV loop, Q operand register pairs,32 byte roots/24 packs, and K input roots. It checks Q register immutability over the entire loop, all Q definitions dominate their uses,32 Q byte stores precede a publish barrier, and K reuse/acquire barriers dominate the corresponding stores/reads. It distinguishes the K operand loads from later PV shared loads by following I8-MMA B operands. It preserves unknown/unsupported results.

Native code proof alone says nothing about loaded resources or outputs. After that CPU review, Root may explicitly run:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/bench_biv150_konly_service_q32_bitwise.py \
  --gpu --commit EXACT_NEW_FRONTEND_HEAD \
  --compile-out ACTUAL_V1_COMPILE_OUT \
  --service-source /workspace/experiments/minicpm-s2/code/cjj-konly-encoded-smoke-v1-20261027/vllm_fl/patches/biv150_konly_encoded_q32.py \
  --numeric-diagnostic --out FRESH_NUMERIC_OUTSIDE_CHECKOUT
```

This is only Q2048/one sequence/prefix4096/KV6144/pages773/B256/BQ32/T64/w16/s1/grid65x2, two pre-existing patterns, four reader launches and zero encoder launches. Both roles are the actual original359 private source: current encoded unhoisted control and the new fresh override. There is no downgraded BF16 control or timing. It reconstructs independent CPU RN32/RNE encoded bytes and uploads them directly; K I8 and full-page0 FP32 scale share the live16KiB page storage, while V BF16 is already offset4096 elements. No stock BF16 consumer is used for encoded bytes.

Each actual recompile must retain reviewed hash/source/typed signature/constants/attrs/metadata and exact TTIR/TTGIR/LLVM. ASM instruction and control/synchronization sequences must match, and the new actual native grammar is rerun. Full cubin bytes and hashes are saved; ELF format is required and equality reported separately. Binary equivalence is not inferred from compilation or cubin bytes. Post-handle-init registers/spills must be known, spills zero, w16/s1/warp64 and shared<=65536 before any reader; actual values are retained. Import paths/SHA for loaded SDK/helper/JIT, actual original private source and Torch location are recorded. It does not force a native W8 library or optional `_C` import.

For every role/pattern the full guarded2050x16x128 BF16 buffer is copied to CPU before and after. Only CPU indices0/2049 determine canaries. Pre-output is poisoned with NaNs, all post-output is finite, the separate full output view equals the CPU full-buffer interior, every full output byte is identical between roles, and full read-only Q/cache/table/lengths/starts/descales hashes must stay unchanged. All originals, encoded bytes, independent codes/scales/references, full buffers and hashes are saved before an assertion can stop the run. The isolated negative-advanced-index defect from PR87 is not used as a canary authority.

INT64 QK/FP64 scale/global-softmax references remain independent of GPU cache/scales. Pattern0 checks all2048 uniform rows; pattern1 checks the established eight boundary/interior rows. Quantized-reference finite/pointwise0.005+0.005abs gates are hard. The original-BF16 relative-RMSE0.005 proxy is reported separately, with original pointwise/error metrics intact. There is no new BF16 GPU baseline: this proxy is deliberately distinguished from old adaptive BF16-control and model precision gates. `--numeric-diagnostic` may continue only when this original-quality proxy fails; it cannot bypass codec, nonfinite, native identity, resource, full output/canary/readonly, quantized-reference or bitwise failure. `precision_gate=false` always remains until independent whole-model acceptance.

No Graph/side-stream/timing/full numerical matrix/model/service validation is claimed. The exact service specialization must actually compile and run before it can be integrated. Earlier~6% component gain is not a new service gain. Latest main benchmark/control behavior is a separate Root-owned investigation.
