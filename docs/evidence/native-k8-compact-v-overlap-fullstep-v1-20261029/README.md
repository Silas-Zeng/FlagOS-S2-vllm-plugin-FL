# PR110 compact late-V fullstep: actual complete-array review

The frozen `4d1166792a5247b774d6b1977e60466574e6a34c` run with actual library `7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4` passes the independent complete CPU replay. Original owner61630/start355834758 was observed same-start Z. The full report is preserved as `independent-review.json`, SHA `0cd4cb1682fa359f566ac7bf2f076c7861761a4cb8783600ea2cdf551acf051d`. Root read the full report and identity adapter and accepted the bounded component conclusion without repeating the full CPU replay.

The replay reuses PR108's exact `review_compact_fullstep.py` (`f0922d1c…`) and original3eb auditor (`1c527358…`). `replay_fullstep.py` makes only reversible path, run identity, source/namespace/library/consumer adaptations. The expected nested `compact_plan` is extracted from the actual 4d `variant.execute_source()`, so both the namespace and nested transformation are checked against what ran. All original numerical loops, compact remapping, writer checks, tolerances, cache reconstruction and ABBA arithmetic remain unchanged. The adapter proof and complete substitutions are in the report and `replay-identity-plan.json`.

## What passed

- All181 originals /145,951,897 bytes and every internal artifact identity.
- Existing four-case consumer complete replay against actual7892/Root90f6, including its176 hard gates,32 BF16 proxy and96 full-buffer byte comparisons; its48 ideal-partial diagnostic failures remain failures.
- 36 complete fullstep output snapshots and144 O/M/L/final hard numeric checks, all active rows/segments, inactive poison, width16 backing and guards.
- Changed-input Graph and side results before fresh eager, and full-field pre/post-timing bitwise comparisons.
- Actual unchanged dc5 writer/views, four JIT function source/artifact identities, writer six-pointer/six-I64 ABI, integer resources and zero reported spills; initial encoding and timed append both use the same dc5 writer.
- Full34276-page cold BF16 and canonical K8 reconstruction; both lengths/arms phase0 FP64 golden reconstruction; all phases' saved golden arrays checked by the original arithmetic gates.
- Every initial/appended cache chunk's expected hash for all phases, including compact12544 remapping, plus complete guarded linear K/V/slot source hashes before/after timing.
- Original warm128, replay64, unroll8 and BF16→K8→K8→BF16 order, positive finite GPU-event/host-wall measurements and exact mean/ratio arithmetic.

Fullstep ideal-unrounded-partial diagnostics retain **12 failures for4K and12 for16K**. Their pagewise BF16-P implementation gates pass; these diagnostics are not relabeled or substituted into the hard gate. All GPU tolerances remain `.005 + .005*abs(reference)` and relative RMSE `.005`. Independent host FP64 regeneration uses only the original `rtol=2e-12, atol=2e-12` allowance; saved metric arithmetic retains `rtol=4e-14`. No tolerance was relaxed.

## Actual component timing

| Fixed one-layer case | Old aace BF16 event mean ms | New7892 K8 event mean ms | Event aace/K8 | Host-wall aace/K8 |
|---|---:|---:|---:|---:|
| N64 /4K /S2 /Qpitch2048 | 0.4163990765810013 | 0.34223921597003937 | 1.2166901312018377 | 1.2166355864706224 |
| N32 /16K /S4 /Qpitch2560 | 0.8053560554981232 | 0.6600197553634644 | 1.2201999242501824 | 1.2202181799527474 |

These clocks measure append+producer+original reducer inside the fixed graph, with separate initial upload/linear-pack/encode costs retained in the ledger. **The denominator is the old aace component**, not latest-main stock, PR107's8b86 or model throughput. Historical PR108/40f9 ratios1.049348031/1.047515436 against that same named control are cross-run context only. There is no same-run7892-versus40f9/8b86 comparison, stable-effect conclusion, official105 precision result or latest-main gain here.

Scope remains one synthetic layer with34276 logical pages. NewKV writer pitch is256; actual-service writer pitch2560, N40/N48,42-layer allocation/lifetime, reclaimed capacity and model serving are unqualified. The compact arena is429,958,144 bytes versus BF16's561,577,984 bytes for the same page count; this is not service-capacity evidence.

GPU input/cache bodies were not saved by the original run. CPU reconstruction checks expected complete chunk hashes against the runtime full-byte comparison ledger; it is not a new GPU readback. All actual output buffers are saved and independently replayed. Only phase0 FP64 gold is independently regenerated; other phases' saved complete gold and all-phase cache expectations are checked. GPU clock-frequency telemetry and independent post-run whole-GPU idleness are not present; same-start owner Z plus source cleanup/completion are the available cleanup evidence. Archived `.pyc`/helper binaries are never imported by the reviewer.

## Complete originals and the small shared package

Root's complete durable archive was independently rehashed without making another big copy:

`/Users/chenjunjie/Documents/flagos/repos/experiment-raw-archives/2026-10-29/cjj-native110-fullstep-v1-raw-20261029.tar-stream-v1.tar.gz`

67,555,520 bytes, SHA `b8905e97768fb5d904c9ef8f295990b5134157a7ec957fd12e8c0d18352f722f`. See `full-archive-location.json`. The full181-file manifest is preserved as `original-download-manifest.json` (SHA `944578c93b8c5db2bed3fa7d1285c5bb5d7132259f9c99eecd52057defa49962`). The fullstep summary SHA is `f12332f0b2397ed17baaadbad5e5d189c326188e1b6b9a05c8d4084ef7667478`.

`original-metadata-only.tar.gz` shares37 unchanged original metadata/small compiler-artifact files plus the full manifest. It is1,399,506 bytes, SHA `8549963b054c306ea45be80ea3f9c6e132df75d4a07b324eb7ad3a0bfd9e67c8`, below2MiB. The138 numerical `.bin` bodies and6 runtime helper/bytecode files remain in the complete archive. `metadata-package.json` records every included/excluded path and SHA. No missing proof is inferred from this metadata-only package.

## Replay

Lightweight shared-package verification performs no numerical rerun or GPU operation:

```sh
python3 -B docs/evidence/native-k8-compact-v-overlap-fullstep-v1-20261029/verify_shared_evidence.py \
  --full-archive /Users/chenjunjie/Documents/flagos/repos/experiment-raw-archives/2026-10-29/cjj-native110-fullstep-v1-raw-20261029.tar-stream-v1.tar.gz
```

`replay_fullstep.py` defaults to an inert CPU plan. A full replay requires NumPy, the complete raw/dependency directories, a clean frozen4d checkout and original3eb checkout; the small package alone is insufficient. It creates/resets no checkout and never loads a SO. All path options are exposed by `--help`; defaults refer to the actual local inputs used here.

```sh
python3 -B docs/evidence/native-k8-compact-v-overlap-fullstep-v1-20261029/replay_fullstep.py
python3 -B docs/evidence/native-k8-compact-v-overlap-fullstep-v1-20261029/replay_fullstep.py \
  --run --regenerate --out /private/tmp/cjj-native110-fullstep-independent-replay.json
```

The referenced original audit files already remain shared under `../native-k8-compact-fullstep-v1-20261028/`; no second numerical framework or copied reference was added. Consumer and compile originals are shared/referenced in the preceding PR110 evidence directories. CPU elapsed fields and substituted local paths can change report hashes on repeat; the preserved report hash above identifies this completed review exactly. No repeat full audit was done during packaging.
