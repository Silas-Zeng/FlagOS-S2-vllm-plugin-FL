# PR103 compact Q32 hoist: actual compile evidence

This evidence-only commit starts at frozen `6c08ec2561b1c3cee96838bffd4e296b757ad877`. Production source remains bound to `deab3ab474925c08dba48c4c8cdb00bf108b3b82`; no `vllm_fl`, `csrc`, probe, threshold or source-manifest bytes change. Team main was `a455a3de364051fbea80226870c239de3a48e716` when this evidence was prepared, but this older isolated component has not incorporated or qualified that main. It is separate from the new compact native-service branch. No old 16 KiB, model, service or throughput qualification transfers.

The actual compile-v2 completed and its finite independent CPU review passed. The exact review is [the original report](cjj-pr103-actual-compile-review-20261028.json), SHA256 `045654195e6387e9aa4ffa7b132fc970496a7d3c8b37c3eac6800f24ab442ad8`. Its [original script](cjj-pr103-actual-compile-review-20261028.py) is preserved byte-for-byte with its required external ELF-reader dependencies. Component source, proposal and native-auditor dependencies remain in commit `6c08ec2`.

## Actual result and limits

The review checked all 50 transported originals (4,937,816 bytes), 20 source-identity entries against local immutable Git blobs, first-unhoisted/adapter/cache artifact hashes, complete typed ABI, identical source/TTIR, exact proposal replay, and actual compiler LLVM/ASM native-auditor replay. The target is CoreX arch71, 16 warps, 64 lanes, one stage; one sequence, Q1024, grid `(33, 2)`, 139 compact12544 pages, 256-byte layer offset. Sixteen LLVM Q shared reads moved out of the KV loop. Both cubins have matching complete 28-argument binary ABIs.

| Resource provenance | Unhoisted | Hoisted |
|---|---:|---:|
| Loaded `n_regs`; cubin `.vgpr_count` | 135 | 122 |
| Triton dynamic shared bytes | 57,344 | 33,792 |
| Loaded `n_spills` | 0 | 0 |
| Cubin `.sgpr_count` | 104 | 104 |
| Cubin `.sgpr_spill_count` | 99 | 96 |
| Cubin `.vgpr_spill_count` | 0 | 0 |
| Cubin private segment bytes | 0 | 0 |
| Cubin fixed shared bytes | 0 | 0 |

Loaded `n_spills=0` does **not** mean every spill counter is zero. Fixed ELF shared storage and Triton dynamic shared storage are separate fields. These observations do not prove occupancy, speedup or off-chip spill behavior.

Attention, writer, encoder, Graph replay and explicit frontend input-initialization counters are all zero. Actual CUDA initialization, allocations, JIT compilation and handle initialization occurred; `CUDA_initialized_after_torch_import=true`. Zero attention launches does **not** mean zero CUDA activity or a device-wide profiler proof.

The cubin SHA256, ELF machine/ABI/resource notes and matching compiler-cache artifacts were verified. The native proof replays the compiler's actual LLVM and ASM. These originals contain **no independent cubin instruction disassembly**; this review does not claim an ixobjdump-level instruction-to-binary comparison. Installed SDK/helper/JIT pins and loaded driver/binder values retain their source-bound runtime-attestation scope.

Numerical correctness, changed-input Graph/side execution, model105, 42-layer capacity, service and performance remain unqualified by this compile evidence. Owner38862's later numeric task was reported ended with raw collection in progress when packaging began; it is not included or granted any qualification here. The numeric entry point freshly reaudits its own compiled objects. Its Graph route uses unhoisted normalized2D128 and cannot qualify hoisted-Q32 capture.

## Attempt history and correction

Keep the [old preparation report](cjj-pr103-next-compile-review-20261028.json) together with its [separate erratum](cjj-pr103-next-compile-review-erratum-20261028.json), SHA256 `270d01d43e15d41d86d140c05b78eabf2995c70811c3935bbce30ff5e69cbf16`. Neither historical file was rewritten.

The old statement that the existing launch spec required no changes missed the outer generic launcher's production-diff gate. The original outer `base=4229b48` differs in 32 production files. Root corrected only that outer comparison reference to unchanged production binding `deab3ab`; probe-internal BASE, source manifest and thresholds stayed unchanged.

1. Compile-v1: owner38631/start354661608, same-start terminal `Z`. Its two originals total 1,931 bytes. The now-preserved original traceback explicitly ends at probe line89 `args.out.mkdir()` with `FileNotFoundError` for the missing `.../compile-v1.../build` parent, before framework imports and CUDA. There was no successful device compilation or result summary. The earlier erratum described this as Root-reported because those two originals were not yet available to that review; their later inclusion adds direct evidence without changing the erratum.
2. Compile-v2: owner38713/start354708474, same-start terminal `Z`, completed/compile_passed/fresh_actual_compact_IR_native_resources_verified true, errors empty. Same6c source and compiler arguments; Root used the new top-level `.../cjj-compact-q32-hoist-compile-v2-20261028` output whose parent exists. Outer base remains deab.

Root also reported that one local inspection used a nonexistent `passed` field and raised `KeyError`; it was corrected to the actual summary fields. This is not a device failure and is not represented as one in the raw originals. The recorded reviewed launch specs and original launcher source are preserved for provenance; the replay utility never executes that launcher.

## Complete packages and replay

- `raw-compile-v2.tar.gz`: all 50 original files; 618,850 compressed bytes. Its complete, unchanged original collector manifest is [compile-v2-download-manifest.json](compile-v2-download-manifest.json).
- `raw-compile-v1.tar.gz`: both failed-attempt originals; 854 compressed bytes. Its unchanged collector manifest is [compile-v1-download-manifest.json](compile-v1-download-manifest.json).
- [archive-index.json](archive-index.json) records every package member, byte count and SHA256. Each original occurs exactly once within its version. No original is sampled or truncated.
- [files-manifest.json](files-manifest.json) covers every evidence file except itself. Every individual file and package is at most 2 MiB; no package split was necessary.

From this repository, the lightweight verification below checks file hashes, exact archive membership, original collector manifests, non-overlap, size limits and the v1 exception boundary. This is the only verification rerun for the evidence commit:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/evidence/compact-q32-hoist-compile-20261028/replay.py
```

Optional full CPU rereview reuses the preserved script and exact6c source. It creates a temporary local shared clone, uses only local Git objects, extracts originals to temporary paths, and remaps filesystem literals in memory. It does not modify the original script or any predicates, run a compiler, import Torch/Triton, load a binary, contact a remote or execute the launcher. The requested output and its companion erratum must not already exist:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/evidence/compact-q32-hoist-compile-20261028/replay.py \
  --replay --out /absolute/new/compact103-review.json
```

A replay report has remapped provenance paths and therefore need not reproduce the original report's SHA. The original `045654…` report remains the immutable historical result. Full rereview was not repeated while preparing this evidence-only commit.
