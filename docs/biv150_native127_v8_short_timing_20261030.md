# PR127 conditional short consumer timing (pending)

This child keeps cd2f numeric, fcf CUDA/host and the two actual SOs unchanged.
Root independently replayed parent328892/start367406093: report260414601892
(full SHA in the selected results) binds190originals/175022287B,40snapshots,
88FP64 hard checks and80cross-arm OML/final comparisons. All16original BF16
proxy checks pass; max RRMSE0.0048402022 is below the unchanged.005 component
gate. The24ideal-partial diagnostics still fail and are preserved separately
from the pagewise-BF16-P implementation reference. Selected text only is in
[evidence/native127-v8-numeric-selected-20261030/README.md](evidence/native127-v8-numeric-selected-20261030/README.md).

The source now binds the actual summary, manifest, Root review113107B and
Root's df52 timing-scope receipt. Exact real schema/counts/owner/source/roles
are checked. Root's receipt still requires review of this timing source and
an explicit serial launch. Default GPU admission is false; caller Root flags
and all exact proof inputs remain mandatory. No timing or service result is
supplied by this child. Development ancestry is main5f, and actual compile/
source/library identities retain their immutable meanings.

Reuse is narrow: `biv150_native127_v8_numeric.adapted_run` supplies the original
consumer definitions. Blob/snapshot/readonly/canary/FP64/producer/reducer
functions remain byte-exact. The full A/GraphB/sideC matrix is not rerun: its
Root-replayed original result is a prerequisite. The finite adapter
replaces that phase block with only the new timing-capture validation. It
constructs A/B inputs, saves only the B golden and current input/snapshot
blobs, and references original CPU-oracle authority from the prior numeric
job. It does not duplicate the old original-BF16 K/V oracle payloads or sideC.
Current per-arm summaries explicitly leave Graph/side/full-numeric status
false and record only new-capture-B verification. Historical qualification
and newly observed checks are kept separate.

For each Q pitch2048/2560, each role gets a separate consumer run frame with
`f.CASES=(case,)`. A single end hook retains this frame's real allocation,
callbacks and graph. This avoids Python loop-cell reuse redirecting a stored
pitch2048 callback to the later pitch2560 tensors. It is private component
ownership, not a service tensor cache. A is5898 with explicit BF16 dequant V;
B is138cac with private V codes/scales. No writer, encoder or allocator change
is introduced; producer and original reducer mathematics/thresholds stay put.

The new capture first executes8initial untimed steps to compile/warm the
original reducer, then captures8producer+reducer steps on A. It changes Q,
cache, table and lengths to B, poisons outputs, saves the poisoned-before
state, replays before any fresh B eager, and checks full guarded O/M/L/final,
FP64 references and all input/cache bytes. One fresh B step must match all
those arrays bitwise. That is the one necessary validation of this new graph;
it does not rerun the unchanged complete numerical matrix. Original BF16
Q/K/V proxy is still separately recorded; it cannot turn `precision_gate`
or model/service status true.

The event+sync-host `timed(fn,repeats)` source is exactly frozen3eb
`probe_biv150_native_k8_fullstep.py:55–58`. Keep its unroll8/warm128/replays64
protocol (fixture line15). Each arm gets128warm graph replays=1024complete
consumer steps. One AB→BA per pitch uses64replays×8steps=512steps/block:
8event blocks total across two pitches. Save event and monotonic host ms per
complete producer+reducer step, including graph submission/end synchronization
for host wall and excluding setup/validation/readback. Arm means and A/B ratios
are arithmetic; a gain is not a correctness requirement. Warm/capture/counter
scope is explicit; Python graph-replay counts are not hardware-profiler counts.

After each block, reuse the original snapshot on the same B data and
require all guarded output/OML/cache/readonly bytes equal that arm's fresh B
reference. Content-addressed blobs deduplicate repeated identical snapshots.
Raw timer endpoints are not independently saved; per-replay original timer
values and per-step values are both recorded. Positive finite clocks are
required. No flush/cache parameter sweep or extra numerical rounds is added.

The fixture remains389physical16KiB pages=6373632B, compact S2 scratch,
N64/S2, with empty/short rows and exactly one6143+one6144 row per phase. It is
not64long requests, width16 fullstep, writer-inclusive, model or service. One
warm resident-fixture ABBA does not show stable throughput/HBM behavior or
capacity savings. This screen only decides whether the measured operand
path merits later full-width integration. The parent's16BF16 proxies passed, with max RRMSE0.0048402022; that component
result still does not grant model precision. The ideal-partial diagnostics
remain separately visible and never replace the pagewise partial hard gate.

CPU replay (no framework/GPU):

```sh
python3 -B -S tools/check_biv150_native127_v8_timing_cpu.py \
  --parent-source /private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028 \
  --numeric-summary /private/tmp/cjj-native127-packed-v8-numeric-raw-20261030/out/v8-consumer-summary.json \
  --numeric-review /private/tmp/cjj-root-native127-v8-numeric-independent-review-20261030.json \
  --timing-admission /private/tmp/cjj-root-native127-timing-admission-20261030.json
python3 -B -S tools/probe_biv150_native127_v8_timing.py
```

The eventual GPU entry adds `--gpu --timing --root-owned-exclusive
--root-timing-approved`, all original127 source/compile/static/two-library/
codec arguments, and exact prior `--numeric-summary`, `--numeric-review`,
`--numeric-review-sha256`, `--timing-admission`, `--timing-admission-sha256`.
Root must review this source and separately confirm the real serial slot;
the scope receipt is not automatic scheduling permission. No SSH/GPU/JIT/network/GitHub
operation was performed to prepare this child.
