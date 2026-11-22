# Compact NativeK8 complete-step candidate

This component candidate combines the existing compact12544 NativeK8 reader
`40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453` with the
unchanged PR91/dc5 compact writer. It compares complete append + producer +
original stride-aware reducer against the original aace BF16 component.
There is no late-V-wait, Q32 hoist, service installation or capacity claim.

The branch starts at shared PR100 `ae734b954447d0e7074a862437f7b8a3c46d2b13`
and normally merges team main `1cbac531ca9afaa0e9a68624d5d10cb1d0412055`.
Use a separate frozen `3ebdbebf61e0e90735cd8ad83eb1605627b6ef46` reference tree:
the original aace BF16 source pins intentionally reject current PR105 sources.
No pin is relaxed to make that old comparison run in a current tree.

`tools/probe_biv150_native_k8_compact_fullstep.py` extracts the original 3eb
`run` function and applies a finite, reversible list of address/writer/identity
edits. The original FP64 reference, metrics, changed-input Graph-before-eager,
side stream, full arrays/poison/canaries, BF16 writer, reducer, ABBA order,
128 warm replays, 64 timed replays and unroll8 are retained. Source proof records
every edit and exact inversion. Original 34276 logical pages remain fixed.
The cases are 4K/N64/S2/query pitch2048 and 16K/N32/S4/query pitch2560.
**New K/V source pitch remains 256. Actual service K/V pitch2560 is not covered.**

Both initial encoding and append use the exact writer at dc5. Its source and
the existing compact views module are materialized unchanged from their Git
objects into the output artifacts; no new encoder is introduced. The cold
source is interleaved by page, so guarded linear K and V source buffers are
packed separately to satisfy that writer's constant token stride. Their pack
cost is reported outside warmed timing; their complete contents, slots and
guards are checked after initial packing and after timing.

The canonical CPU cache/oracle stays 16KiB. Only its byte layout is remapped:
K/scales `[0,4224)` are copied, reserved `[4224,4352)` stays A5, and V moves
from8192 to4352. Every compact cache byte is compared at all original stages.
Initial and append writer pointers all refer to the full page0 base; consumer
V is offset4352 exactly once. The existing PR83 source/slot/alias qualifier is
reused with only PAGE and FORMAT globals changed in an isolated function.
The original module stays untouched; compact roles use dc5's exact views API.

Before Torch, the frontend reuses ae734's existing complete compact consumer
saved-array replay with path substitutions only. It requires the exact real
9d2b summary, complete download/raw artifacts, actual SO review/compile archive,
old codec and22e library. The original c2e consumer and aace gates remain too.

CPU default and bounded checks:

```sh
python3 -B tools/probe_biv150_native_k8_compact_fullstep.py
python3 -B tools/check_biv150_native_k8_compact_fullstep_cpu.py \
  --parent-source /absolute/frozen-3eb
```

Root-only serialized device invocation, after review:

```sh
python3 -B tools/probe_biv150_native_k8_compact_fullstep.py --gpu --root-owned-exclusive \
  --parent-source /absolute/frozen-3eb \
  --compact-consumer-raw /absolute/complete-compact-consumer-raw \
  --compile-summary /absolute/compact-compile/out/compile-summary.json \
  --codec-summary /absolute/original-codec/out/codec-summary.json \
  --original-library /absolute/original22e.so \
  --consumer-summary /absolute/original-consumer/out/consumer-summary.json \
  --consumer-proof-sha256 c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128 \
  --k8-library /absolute/compact40f9.so --bf16-library /absolute/aace.so \
  --out /absolute/new-fullstep-out
```

The complete compact raw includes the original138-file download manifest and
owner metadata; the compile archive includes `build/cjj_native_k8_compact_probe.so`.
No regenerated receipt or summary-only pass substitutes for those saved arrays.
The fullstep output retains original full-array blobs, FP64 gold, artifact
hashes, reconstructible full-cache chunk hashes and initial cold-source seeds.

Pending: actual full-sized writer+reader execution, helper ABI/resources,
all fullstep numeric/Graph/side gates, timing and independent raw review.
Separately pending: source pitch2560, live service prefill/routes,42-layer
allocation, reclaimed capacity, official accuracy, TTFT and throughput.
CPU checks do not grant any of those device/service qualifications.

Local CPU review completed on the first adapter: the finite layout/slot checks
passed; exact writer/views source hashes were checked. The unchanged compact
consumer saved-array replay also completed: all4 cases,176 hard numeric checks,
96 full-field cross-arm comparisons and32 original-BF16 proxy checks. Its report
SHA is `2d9388538f4b4ea794740c388c6ac859dfb5b770e1fdcb56c07477f919edae45`.
This is replay of the prior29-page consumer, not execution of this fullstep.
The pending gates above remain pending.
