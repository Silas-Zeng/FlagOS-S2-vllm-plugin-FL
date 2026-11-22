# PR108 compact fullstep: preserved actual CPU review

Tested source: **fa92060af9399e60938ddee6dd7e0a603610cd8c**, compact reader **40f9** and actual writer **dc5fd2e5b86a3d57aceaad26e8556d169ec7462a**. Owner38381/start354308715 was independently observed same-start Z. The later normal merge **faf88f56ee6b0c11eaac2f1a0c59c2fbae488c1e** incorporates main **a455a3de364051fbea80226870c239de3a48e716**; that integration is not the source tested by this run. This evidence-only commit changes no production or benchmark code. Remote PR108 was still at fa9 at this handoff: publish the ordinary fast-forward sequence **fa9 → faf → this evidence commit**, without force/rebase.

`independent-review.json` is the exact completed CPU report, SHA **96262cd2414ce85a3648f0d45e1727f93b243e754310e31e1160fc30639ebdb0**. `independent-review.md` is the original detailed scope/limitations. The reviewed script and its otherwise external original 3eb auditor are preserved byte-for-byte, with their hashes in `evidence-files.json`. This packaging did not rerun their numerical checks.

The actual review verified181 files /145,937,040B, four-case compact-consumer full-array replay,36 complete fullstep snapshots /144 O-M-L-final hard checks, changed-input Graph/side before fresh eager, guards/poison, actual dc5 writer sources/typed6-pointer6-I64 ABI/compiler artifacts, width16 scratch and all12544B/V4352 cache gold chunks. Initial linear K/V/slot buffers and their individual costs remain recorded. Thresholds are unchanged. Both ideal-partial diagnostics retain **12 failures per case**; they are separate from the pagewise BF16-P implementation hard reference. Phase0 complete FP64 gold was independently regenerated for both arms; all phases' saved FP64 arrays were checked, and all phases' expected cache chunk hashes were regenerated. No independent phase1/2 FP64 regeneration is claimed.

| Fixed synthetic component | Old BF16 aace mean ms | Compact40f9 mean ms | Event aace/compact | Host wall ratio |
|---|---:|---:|---:|---:|
| N64/4K/S2/Qpitch2048 | .416410223 | .396827564 | 1.049348031 | 1.049392367 |
| N32/16K/S4/Qpitch2560 | .805491537 | .768954337 | 1.047515436 | 1.047502513 |

This is one fixed34276-page component ABBA (warm128/replay64/unroll8), with append+producer+reducer included. **aace is an old component control, not current main's stock library or8b86**. The approximately1.049/1.048 ratios do not establish current8b86 gains, model throughput, stable effects, official105 quality,42-layer capacity, or allocator reclamation. NewKV pitch256 timing does not cover real service writer pitch2560. The synthetic compact arena429,958,144B vs BF16 arena561,577,984B keeps the same page count.

GPU cache/input bodies were not saved by the original run: CPU reconstructed gold chunk hashes corroborate its runtime full-byte/readonly ledger, rather than a fresh GPU readback. Full saved output buffers were independently replayed. Cleanup is supported by same-start owner Z and completion after per-case deletes/empty_cache in the hash-bound source; no independent postrun whole-GPU-idle observation is present. No missing proof is filled in.

## What is actually shared

- `original-download-manifest.json`: exact full181-file transport manifest.
- `original-metadata-only.tar.gz`: **37 original metadata/small compiler-artifact files plus that full manifest**,1,397,201B, below2MiB. SHA **e563ccf402aba22462b777f761dae8f57971c15398b4cb624c32e7040a579994**. Original member bytes are unchanged; only deterministic tar uid/gid/mtime and gzip mtime are normalized.
- `metadata-package.json`: exact included/excluded paths and hashes. The138 `.bin` numerical bodies and6 runtime-helper/bytecode files remain in the full archive. Preserved `.source`/TTIR/TTGIR/LLVM/ASM/cubin are actual artifacts, not assumed generated output.
- Dependency manifests and `replay-dependencies.json`: the additional real compact-consumer, compiler, original-codec,22e library and frozen Git objects required by full replay. They are not all embedded data.

The **complete** raw archive remains outside this plugin Git repository:

`/Users/chenjunjie/Documents/flagos/repos/experiment-raw-archives/2026-10-28/cjj-native-k8-compact-fullstep-v1-raw-20261028.tar.gz`

66,705,999B; SHA **beffd8ad4e7b3e15536a1868fbe0e37e5e29edf438d58caeb6ca8fafdca5d03a**, reverified during packaging. Originals also remain in the local raw directory and remote `/workspace/experiments/minicpm-s2/cjj-native-k8-compact-fullstep-v1-20261028`. **The big arrays are not Git-shared by this evidence commit.** `full-archive-location.json` records this distinction.

## Replay

Lightweight stdlib verification, no array regeneration:

```sh
python3 -B verify_shared_evidence.py
```

`replay.py` defaults to a stdlib-only plan. Full NumPy/Python3.12 replay needs the complete raw/dependency archives and actual immutable fa9/3eb checkouts; the metadata tar alone is insufficient. Git objects fa9,3eb,dc5 and their referenced baseline objects must be available. Do not reuse a newer source tree as the tested source. Existing paths are never created/reset by the wrapper.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B replay.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error replay.py --run --regenerate \
  --tested-source /ABSOLUTE/FA9_CHECKOUT \
  --parent-source /ABSOLUTE/3EB_CHECKOUT \
  --raw-root /ABSOLUTE/COMPLETE_FULLSTEP_RAW \
  --compact-consumer-raw /ABSOLUTE/COMPLETE_COMPACT_CONSUMER_RAW \
  --compile-out /ABSOLUTE/COMPLETE_COMPACT_COMPILE_RAW/out \
  --codec-summary /ABSOLUTE/COMPLETE_CODEC_RAW/out/codec-summary.json \
  --original-library /ABSOLUTE/ORIGINAL_22E.so \
  --out /ABSOLUTE/NEW_REVIEW.json
```

The wrapper changes declared path literals in memory only, preserving every reviewed numerical/identity guard, and writes a separate path-map receipt. Existing outputs are refused. This packaging ran only lightweight checks, not the full CPU replay, SSH, JIT, GPU, service or PR publication.
