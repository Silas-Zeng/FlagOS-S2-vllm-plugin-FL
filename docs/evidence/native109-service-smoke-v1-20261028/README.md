# PR109 actual compact service smoke

This is a CPU review of the completed `cjj-agent-native109-service-smoke-v1-20261028` records. Tested source is **fc25c86a13fc273bb7368b83af4ca264524b4e12**. The evidence commit adds documentation only; it does not change that running source. Owner 39057/start354896408 was collected with the same start and state Z. No SSH, GPU execution, new model evaluation, or component numeric rerun was performed for this review.

The independent review passed:

- All 18 original files / 1,681,499 bytes match the original download manifest. All 268 source hashes match the immutable fc25 Git tree, following the recorded MoE symlink. The executed controller is d8a5bd435845bdb3dd1e4d3741f4ca38fe85b4cc152036517cbcf034f0eb4495 and helper source is the original 94db file.
- Both logged worker C3 fingerprints equal the expected fingerprint, `545ae8008b0bae2c98b52a781f41296e55d2db80039ba411edf0be5f54cabe1b`. The compact implementation digest is independently recomputed. Existing M2/Q32/normalization/Q-fragment/linear installers remain installed and there is no compute-order scope rejection.
- Saved actual load receipts bind compact **40f9**, operator `cjj_native_k8_compact_probe::produce`, native module source/schema and Torch import path. The distinct stock NativeGQA library is **0c66** with V overlap enabled; the W8A8 gate library is **5c**. These library bodies are not in this smoke collection: this review checks saved identity/load receipts, not a fresh remote SO hash.
- One actual allocation record covers 42 layers. Each owns 44,769 physical pages at 12,544 bytes/page, or **561,582,336 bytes/layer**; total **23,586,458,112 bytes**. Each carrier has shape `[44769,2,16,2,128]`, BF16-element strides `[6272,2176,256,128,1]`, and zero pointer offset. Source-pinned Host checks reject duplicate storage identities, require 42 independent SDK allocation-plan entries, and check the final bound caches. Raw storage identities and cache data bodies were not dumped.
- The SDK log reports **716,304 tokens**, equal to 44,769 × 16. The exact hash-verified PR107 throughput server log reports 548,320 tokens: the observed capacity difference is **+30.636124890574855%**. This comparison is allocation/token capacity, not throughput, model quality, or a performance result.
- The three short requests and bursts of 32/64/40/48 have **187 unique saved responses and 187 HTTP 200s**. Their recorded completion counts are all 16 tokens (all finish reason `length`), with total 11,613 prompt + 2,992 completion tokens. This is response/count arithmetic, not a content or all-token quality gate.
- State and owned cleanup completed without recorded errors; no owned process remained. The identified API exited with return code 0. There is no EngineDeadError in the saved log. No independent post-cleanup GPU-idle observation is added.

## Limits that remain

The 84 SELECTED and 84 CAPTURED native records represent 42 layers at each N64/S2 and N32/S4 startup bucket, in eager/capture preparation. They do **not** count actual Graph replays or establish actual-request Native frequency. Request-window records have no native host events. Unknown/unsupported shapes retain the private compact Triton reader; no BF16 reinterpretation fallback is inferred.

`compact_final_export_found=false` and `compact_final_exports=[]` remain unchanged. Final lifetime counters, overflow, and observer errors are unknown; no final stats are manufactured. No Graph observer bootstrap was enabled. `native_quality_covered`, model precision and throughput qualification all remain false in this smoke record.

Writer83 selection records save a kernel hash/profile/implementation, but not the K/V source strides. **Actual writer pitch2560 is not proven by this smoke**. The observed native query stride is 2048; its scratch keeps a 64-row capacity and max-segment-16 head/token strides. This metadata review does not repeat component numeric tests or add service-quality qualifications.

The original collection excludes generated kernel/tuning caches. It is not a complete out-directory or full-device-data snapshot. The archive includes all 18 collected originals and their original manifest, the exact old PR107 server log plus its original manifest solely for capacity comparison, and the actual helper94db source. The baseline archive subset is not represented as a complete PR107 collection.

## Replay

Use a Git checkout containing fc25 and Python 3.12+; only the standard library is used. Default verifies shared files/archive; explicit `--run` performs the same read-only source/record CPU audit after extracting the saved bytes into a temporary directory. The new report uses those temporary input paths, so path strings differ from the original report.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error docs/evidence/native109-service-smoke-v1-20261028/replay.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error docs/evidence/native109-service-smoke-v1-20261028/replay.py --run --out /private/tmp/cjj-native109-smoke-review-NEW.json
```

The original CPU report SHA256 is `c7820cf6f7d1f67a662d378b3236c5c913cb8ff1ce62c968d4cc93e41f022f21`; its exact script SHA256 is `de2af0073be25d399b54f70fcb902f4145596a33c50bbc1afe3ae8ab189e7c9f`. Log-to-JSON mirror comparison converts only the checked W8A8 `layers=('gate_up_proj',)` representation to its JSON list, preserving every field; raw records are untouched.

The deterministic archive is 107,861 bytes, SHA256 **7f2fd31197efa33af04ccd831d5f6e9b01532ff16745ec7d43d797dc92dc69ae**, below 2 MiB. `archive-members.json` lists all 22 members and hashes. `shared-files.json` pins the shared review files and archive (excluding the replay driver and the manifest itself).

Git preparation used a new isolated `codex/cjj-native109-smoke-evidence-20261028` worktree from fc25. A local fetch attempt failed because github.com could not resolve; no successful fresh fetch, remote update, PR edit or network push is claimed. Root controls later publication and model/throughput acceptance.
