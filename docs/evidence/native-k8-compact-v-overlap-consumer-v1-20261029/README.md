# PR110 actual 7892 consumer: complete saved-array replay

The actual four-case consumer completed on frozen commit `4d1166792a5247b774d6b1977e60466574e6a34c`, owner 61196/start355697119 observed with the same start identity in state Z. This independent CPU replay passes all required saved-array, numerical and bitwise checks using the already prepared `probe_biv150_native_k8_compact_v_overlap_fullstep.replay_consumer` adapter. The entire numeric replay body remains byte-identical to the existing PR100 replay; the original 3eb fixtures, pagewise BF16-P reference and thresholds are unchanged. No Torch, Triton, SSH, compile, JIT or new GPU operation was executed by this reviewer.

Bindings:

- Actual library: `7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4`.
- Actual consumer summary: `6f719fe84e83e547d0e42f7c2be5bdb408e64cdb107cc455994a681705e34c7c`.
- Full download manifest: `4435696c50214a8ba60f6a4e4fbf7ff1970ce3f8fb202cb13f7b82b4c4c5613a`.
- Unchanged Root-approved actual-SO receipt: `90f6e18e142813e732694f453779b876048daaa8c122fd3e2d985dc2969cdb17`.
- This independent replay report: `896c7c992aefcbfa36ed7445c3cf6fb1d40135bde610b5d83693ed27ec8fadf1`.

## Results and limits

| Check | Complete replay result |
|---|---:|
| FP64 hard checks across complete active O/M/L/final fields | 176 passed |
| Separate original-BF16 final-output proxy checks | 32 passed |
| Full O/M/L/final buffer cross-arm byte comparisons | 96 passed |
| Input/cache readonly snapshot checks | 320 passed |
| Guards and physical padding checks | 384 passed |
| Golden arrays regenerated from the unchanged fixture | 84 checked |
| Original ideal-unrounded-partial diagnostics | **48 failures retained** |

Graph B consumes changed query/cache/table/length data and poisoned scratch **before** any fresh eager B, and all Graph-B arrays equal the later eager-B arrays. Side-stream C arrays pass their numerical/input/guard checks. Producer-only O/M/L remain unchanged by the original reducer. Whole cache bytes—including page gaps, outer guards and layout—are replayed, along with pointer/stride/capture binding, all three epochs and the recorded closed-epoch rejection.

Both original22e and new7892 arms use the same original reducer artifacts. The hard reference is the existing pagewise BF16-P v2 for partial O and the unchanged ideal quantized-score FP64 final/M/L; the original BF16 final output is a separate proxy. Every hard gate still requires finite values, elementwise `absdiff <= .005 + .005 * abs(reference)` and relative RMSE `<= .005`. Maximum observed hard absolute error was 0.0018667997130373815 and maximum relative RMSE 0.0011831102609028818. The 48 failed ideal-unrounded-partial checks remain diagnostics, never relabeled as passed or substituted for the hard reference.

Saved GPU outputs are byte-identical between original22e and new7892 in the bounded cases. Independently regenerated host FP64 golden arrays are not all bit-identical to the remote host: maximum difference is 2.842170943040401e-14, within the unchanged host-regeneration-only tolerance (`rtol=1e-12`, `atol=1e-11`). That tolerance does not change any GPU numeric gate.

Scope remains exactly `(N64,S2,pitch2048)`, `(N64,S2,pitch2560)`, `(N32,S4,pitch2048)`, `(N32,S4,pitch2560)`, 29 scattered physical pages, maximum context 193, 128-byte layer offset, three input patterns and one synthetic layer. No writer/encoder, fullstep, 34276-page scale, N40/N48, 42-layer allocation, capacity reclamation, actual service, official model precision or throughput qualification is granted. Graph/side execution identity comes from the bound run/source records; the arrays were independently replayed without another GPU run.

## Complete retention and small shared originals

Every one of the 138 original files (32,796,687 bytes) and the unchanged 56,202-byte download manifest was copied and rehashed at this durable project directory:

`/Users/chenjunjie/Documents/flagos/artifacts/cjj-native110-consumer-v1-raw-20261029`

That is 139 files / 32,852,889 bytes including manifest. The supplied full transport archive was independently hashed as 30,994,101 bytes, SHA `a2dc01e72c195920f95dd1052a9c28076dfb0d6032d1f6f7b690e58b4b1c9e87`. The 114 array blobs remain intact in the durable directory; they are not duplicated in Git.

`small-original-records.tar.gz` contains all 24 original non-blob records plus the original download manifest: 25 regular files / 1,832,358 bytes expanded. Its compressed size is 221,931 bytes, SHA `948e39660ffac34948b7bfa6e7988354e70d73ac40b9ebd1946200dcfc4cdfea`. This includes original owner/log/summary/receipts and actual reducer JIT source/IR/ASM/cubin/helper records. Every packaged member was read back and checked against its original size/SHA; `small-originals-index.json` lists them and records the full-array location. The summary and manifest are also exposed directly beside this README for review. The small package alone is insufficient to replay full arrays; use the complete durable directory.

## CPU replay command

From this worktree (NumPy required), use the original frozen 3eb reference checkout and the previously validated original codec/22e artifacts:

```sh
python3 -B docs/evidence/native-k8-compact-v-overlap-consumer-v1-20261029/replay_consumer.py \
  --raw /Users/chenjunjie/Documents/flagos/artifacts/cjj-native110-consumer-v1-raw-20261029 \
  --compile-raw /private/tmp/cjj-native110-compile-v1-raw-20261029 \
  --parent-source /private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028 \
  --codec-summary /private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json \
  --original-library /private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so \
  --out /private/tmp/cjj-native110-independent-consumer-replay.json
```

The new compile originals are already shared in `../native-k8-compact-v-overlap-actual-so-20261029/compile-original-records.tar.gz`. The wrapper calls the frozen 4d adapter directly, which invokes original 3eb `prepare` and checks source/library/receipt/run/manifest ownership before the original complete replay. No simplified reference or new numerical validation framework is introduced.

Replay time is recorded in `CPU_seconds` and therefore the output report hash changes on repeat. Changing worktree/input paths also changes the adapter's explicit identity substitution strings; compare all numerical/bitwise/guard results and the declared identity bindings, without discarding unexpected differences. The recorded 10.55 seconds is CPU review time, not kernel or throughput performance.
