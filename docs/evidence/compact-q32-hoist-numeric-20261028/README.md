# PR103 compact Q32 hoist: complete numeric review, limited component scope

This docs-only evidence starts at shared parent `7cc8040d07b2e0fc2c507855cef485d796dba3f3`. The tested source remains frozen `6c08ec2561b1c3cee96838bffd4e296b757ad877`, with production BASE `deab3ab474925c08dba48c4c8cdb00bf108b3b82`. No production, GPU, probe, threshold or source-manifest bytes change. This older component is independent of current main and the new compact native-service branch; old16KiB qualifications do not transfer.

The unchanged [independent CPU report](cjj-pr103-actual-numeric-review-20261028.json), SHA256 `50bd90b34cf9c373d7cd3eb221650c48fe64b4a1657d6546e5f0cd3cec89a2da`, rechecked the complete original arrays. Its [original script](cjj-pr103-actual-numeric-review-20261028.py), SHA256 `6ad0a48193b4ee5ce0afda38fbc12f48c1b11b0032d8d88e2cba23d09b1a746b`, is preserved byte-for-byte. These results are from owner38862/start354791020, same-start terminal `Z`; source6c and the exact compile-v2 summary receipt `b51d9424834a3172ac076abca2670e6a2bf9fc06d1f99ebe332fba570524ea1d` agree.

## What passed, and what did not

For three input states, the review regenerated all1024 rows ×16 heads ×128 dimensions using the existing independent INT64 QK/FP64 dequantization-softmax-PV reference. Cross-host FP64 reference difference was at most `5.995204332975845e-15`. The original quantized pointwise gate remains exactly `abs(error) <= .005 + .005 * abs(reference)`; all12 arms pass it against saved and regenerated references, with zero violating values. The separate `1e-12` check concerns only cross-host FP64 reference reproduction; it does not relax the device gate.

All six stages' complete guarded outputs are bitwise identical across their two arms. The review checked252 arrays, 25,165,824 final-output elements, 48 guard regions,144 canonical/after readonly arrays and24 complete cache arrays. Inputs/cache contents remain unchanged by the readers. Q32 exposes final output only; no O/M/L tensors are claimed.

| Input / stages | Original BF16 reference RRMSE, both arms | Original BF16 pointwise violations per arm |
|---|---:|---:|
| A: eager Q32, normalized capture/replay | 0.008315889525066372 | 40,027 |
| B: changed Graph, fresh normalized eager, fresh Q32 eager | 0.008309456017604069 | 38,868 |
| C: side-stream Q32 | 0.008633089629533170 | 42,421 |

**All12 original-BF16 proxy checks fail the unchanged RRMSE `.005` threshold.** The unhoisted and hoisted outputs are identical in this fixture; the proxy failure was retained under explicit `--numeric-diagnostic`. Precision/model/service gates remain false. This is not a105-question model pass, a42-layer capacity pass, a full validation-matrix pass or a throughput result.

Eager A/B and side C compare actual Q32 unhoisted versus freshly hoisted objects. Graph's two labels both execute **unhoisted normalized2D128 bypass**. Changed Graph B precedes and matches fresh eager B on that same normalized path; changed A→B and Q32 B→C outputs were checked. Graph bypass success does not qualify hoisted-Q32 capture. Chronology, stream and live-object attribution are source-bound original frontend records plus complete arrays, not a device-wide execution trace.

Scope is one sequence/Q1024, KV lengths1057/1056/1058,139 compact12544 pages and a256-byte layer offset. Original counters are10 Python attention invocations (including capture),4 Graph replays, zero writer/encoder launches and63,454,479 explicit initialized elements. CUDA context, allocations, compilation and execution are real; those counters do not claim an exact device-wide kernel count.

## Fresh actual compilation and resources

The numeric process freshly audited its own source/typed ABI/TTIR/override/LLVM/ASM and actual binder/target. The original compile-v2 receipt is a prerequisite, not a reused numeric compiler-object licence. Preserved/cache artifacts and cubin hashes/complete ABI/resource notes were independently checked. No independent cubin instruction disassembly is present, so no instruction-to-binary disassembly proof is claimed.

| Resource | Q32 unhoisted | Q32 hoisted | normalized2D128 Graph bypass |
|---|---:|---:|---:|
| Loaded `n_regs` / ELF VGPR count | 135 | 122 | 135 |
| Dynamic shared bytes | 57,344 | 33,792 | 40,960 |
| Loaded `n_spills` | 0 | 0 | 0 |
| ELF SGPR spill count | 99 | 96 | 93 |
| ELF VGPR spill count | 0 | 0 | 0 |
| ELF private segment bytes | 0 | 0 | 0 |
| Warps / lanes / stages | 16 /64 /1 | 16 /64 /1 | 8 /64 /1 |

Loaded `n_spills=0` is distinct from ELF SGPR spill counts; do not call every spill counter zero. Resource metadata establishes neither occupancy nor speedup. Full fresh metadata and both fixed/dynamic shared fields remain in the original report and artifacts.

## Git subset and complete local archive

The complete71-original manifest (374,807,676 bytes) is preserved unchanged as [download-manifest-complete71.json](download-manifest-complete71.json). The Git package contains **all65 non-NPZ originals** (6,793,606 uncompressed bytes): summary, owner/log, fresh compiler artifacts/cache, binder/ABI and prelaunch observations. `numeric-non-array-originals.tar.gz` is844,705 bytes, SHA256 `7814b32e118825b5d16297ea91a419ac9be7dec0c16044db31ae5137ba53f0e6`. Every Git evidence file/package is at most2MiB; no package split was needed.

The six huge NPZ files are **not shared in Git**. They were fully reviewed and remain in the complete archive at this persistent workspace-relative path:

```text
repos/experiment-raw-archives/2026-10-28/cjj-compact-q32-hoist-numeric-v1-raw-20261028.tar-stream-v1.tar.gz
```

Archive size: **369,081,417 bytes**. SHA256: **`149c2af1339ff47dfec8114c3ed68e5c073e511e8d8575cb80c2967aa4992f1a`**. Its persisted hash was checked while packaging. It contains all71 originals plus their collector manifest. [external-storage.json](external-storage.json) records this local-only location; it is not a public download link. A600-second JSON collection timeout was followed by binary gzip-tar collection of the same completed owner38862; no device task was rerun.

[archive-index.json](archive-index.json) records every included member and all six excluded NPZ names, sizes and hashes. [files-manifest.json](files-manifest.json) covers each evidence file except itself. [replay-dependencies.json](replay-dependencies.json) pins32 source dependencies to6c and links the immutable ELF-reader/compile-receipt dependencies already shared in parent7cc's [compile evidence](../compact-q32-hoist-compile-20261028/README.md). Original raw files and reports were not rewritten.

## Verification and optional full CPU replay

The following default is a lightweight package/dependency check. It verifies exact65-member archive content, complete71-manifest coverage, six explicitly omitted NPZ records, source/dependency hashes and2MiB limits. It **does not** recompute numeric or compile audits:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/evidence/compact-q32-hoist-numeric-20261028/replay.py
```

Optional `--archive /absolute/path/to/full.tar.gz` additionally verifies only the local full archive's size/SHA. Complete numerical rereview requires explicit `--replay`, Python with NumPy, that local archive and a new output path:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B docs/evidence/compact-q32-hoist-numeric-20261028/replay.py \
  --archive /absolute/path/to/full.tar.gz --replay --out /absolute/new/numeric-review.json
```

The optional replay uses a temporary local shared clone at exact6c and complete extracted originals. It remaps filesystem literals in memory while retaining all original predicates/thresholds. It never imports Torch/Triton, runs a compiler, loads GPU code, uses SSH/network or changes service configuration. The original script/report stay unchanged; remapped paths can change the replay report SHA. Full replay was **not run** while making this evidence-only commit; only lightweight package checks were run.
