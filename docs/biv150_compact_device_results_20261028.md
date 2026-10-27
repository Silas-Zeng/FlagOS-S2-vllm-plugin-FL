# Compact12544 actual device result

Frozen source dc5fd2e; default-off service remains unqualified. The frontend authenticates real platform bootstrap; GPU kernels/references/thresholds equal354.

Compile-only:117 original files4157855B, completed without numeric launches. Numeric:296 original files28140150B, both4K64/S2 and16K32/S4 completed; implementation_numeric_verified=true, passed=false. Runtime codec141 vectors withM1/M8, edge writer, true12544B roles/nonzero offset/V+4352, complete cache bytes and sources, FP64 final output, changed Graph and side gates passed. Allocator was only1layer/2pages, not42-layer service/capacity.

Root checked all296 SHA and independently replayed6 complete NPZ/288 rows/864 final metric sets against their saved original and quantized FP64 references. Baseline and quantized-operand pointwise hard gates passed;188 rows failed the original-BF16 relativeRMSE proxy (typically.0088-.0089 versus.005). This is not a model accuracy score. Thresholds are unchanged. Partial arrays have no FP64 gold; cold FP64 references were not independently regenerated. Graph/side/cache/guard claims rest on frozen runtime assertions, not separately saved replay snapshots.

Keep Draft for model quality and diagnostic performance experiments. No performance, official105,42-layer capacity or live service result follows from the implementation gate. Full original raw remains in the experiment tags cjj-compact91-device-{compile,numeric}-v1-20261028 and matching local raw directories. Shared archives contain original metadata/manifests/logs/owners; full tensor NPZ files are not embedded in Git.

Replay complete numeric raw with Python+NumPy and a new output:

```sh
python3 -B -W error tools/check_biv150_compact_device_records.py --raw-root /absolute/complete-numeric-raw --out /absolute/new-review.json
```
