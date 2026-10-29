# N1 S1/S2 single ABBA after the complete numerical screen

This tools-only child preserves `efd012eb7f31a70ad0a37bc5a05b9aa9fc8f0efe` and all production/GPU source bytes. It binds the actual main899 S1 control (`53ef08b7`, cubin`7eadbdbf`) and original2e S2-preheader candidate (`2d893863`, cubin`dab726c4`). It keeps Q2048/N1/BQ32,773 physical16KiB pages, the complete typed ABI/options/resource/cache guards and the original source/module/debug/override roots. No reader compilation is allowed on cache miss. Default invocation is a stdlib CPU plan.

Root independently replayed the complete numeric review to SHA `fbf84f3552b2eb7350252c95d881e94eba38ec8775a42b85051ea9602274276d`. That report binds92 originals/193272247B, manifest`d59fc072`, summary`a87961d8`, all8 numerical stages, complete two-arm output bits/cache/codec/guard and Graph/side checks. Random6143 original-BF16 relative RMSE remains `.00961857552 > .005`; quantized FP64 relative RMSE is `.00228378310`. The original pointwise and relative gates are unchanged. The FP64 oracle is ideal global softmax, not tilewise BF16 P or online-softmax rounding. No official/model precision qualification follows from this component.

The new worker derives from the unchanged efd numeric worker, inserting only the original522c timing block and corresponding result/receipt labels. Byte inverses preserve every original numerical, writer, Graph-B-before-fresh-eager-B, side-stream, full-array and readonly check. It reruns both existing patterns: aligned KV6144/prefix4096 uniform, and valid-tail KV6143/prefix4095 random. All hard gates must pass again. The BF16 proxy failure is retained with explicit `--numeric-diagnostic`; it cannot bypass codec, finite, bitwise, FP64 implementation, guard, readonly or IR failures.

Only final patternB/KV6143 is timed. KV6144 has numerical coverage and no timing claim. There is no16K, N2 substitution, new descriptor, production installation or service benchmark.

| Protocol item | Exact scope |
|---|---|
| Modes | Complete reader; unchanged stride2560 writer plus complete reader |
| Captured work | Four complete steps per CUDA Graph |
| Warmup | Eight graph replays per arm per mode =32 complete steps |
| Order | One S1–S2–S2–S1 round per mode |
| Event block | Sixteen graph replays =64 complete steps |
| Event samples | Four per mode; eight total |
| Event value | `begin.elapsed_time(end)/64` |
| Host value | Wall time for16 replay submissions and end-event synchronization, divided by64 |
| Validation | Full guarded output/cache and readonly hashes checked after every block; readback/validation outside timers |

Reader mode includes Q quantization/staging, K/scale loads, QK, FP32 online softmax, BF16 P/PV and output. Append-reader mode adds the same complete K/V writer. The CUDA Graph measurements do not include normal Python JIT/helper/dispatcher or service costs. The protocol warms and repeatedly uses a12.66MiB fixture; it is not an HBM/whole-model throughput claim. One ABBA is diagnostic and does not establish a stable speedup. Actual counters are recorded by the worker; the inherited successful path has36 Python reader attempts,18 writer attempts and168 graph replays, including numerical, warmup and timed phases. These are not profiler hardware-launch counts.

CPU checks and the default plan:

```sh
python3 -B -S tools/check_biv150_konly_q32_s2_timing_cpu.py \
  --preflight /private/tmp/cjj-root-q32-S1-S2-numeric-prepared-20261030.json \
  --numeric-review /private/tmp/cjj-root-q32-s2-preheader-numeric-independent-review-20261030.json
python3 -B -S tools/probe_biv150_konly_q32_s2_timing.py
```

The same small preflight (`1fec73ce…`,102925B) and Root-replayed review are sufficient evidence inputs; large raw arrays/IR do not need upload. Required immutable Git objects remain522c and e4dd. The original A2 S1 source/output/cache and2e S2 source/output/cache must still exist and stay byte-exact.

Device execution requires an independently hashed new timing receipt from `admission_template(FRONTEND_HEAD)`. Its timing approval, final numeric-review confirmation and current fresh A4K→B4K pair closure/no-continuation confirmations all default false and must explicitly be true. Model/service/precision fields must remain false. This entry cannot be scheduled between that pair; Root controls exact predecessor identity and the double GPU/9031/owner fences in the later launcher. No automatic queue or remote operation is included here.

Only after Root reviews this source and admits the slot:

```sh
/usr/local/bin/python tools/probe_biv150_konly_q32_s2_timing.py --gpu --timing \
  --root-owned-exclusive --numeric-diagnostic --commit FRONTEND_HEAD \
  --preflight ROOT_PREPARED_JSON --preflight-sha256 1fec73ce1b965b82b33a4d1f27a6f051fadc34fb8448002707f63ca3c2a49f16 \
  --numeric-review ROOT_FINAL_NUMERIC_REVIEW --numeric-review-sha256 fbf84f3552b2eb7350252c95d881e94eba38ec8775a42b85051ea9602274276d \
  --admission ROOT_TIMING_RECEIPT --admission-sha256 ROOT_RECEIPT_SHA \
  --out NEW_ABSOLUTE_OUTPUT
```

No timing, device, model, precision, service or capacity result is claimed by this CPU preparation. Both numerical-quality failures and any later negative timing are retained.
