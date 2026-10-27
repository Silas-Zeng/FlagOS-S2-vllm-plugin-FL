# Compact NativeK8 small consumer frontend — device pending

This adds a private frontend to the address-only c2ae sources without changing
CUDA, C++, codec, typed host, factory, or production files. It is not a service or
performance runner. PR93's original consumer_v2 fixture/reference sources are
checked against immutable3eb: full nonperiodic D128 counter values, all16 distinct
Q-head scales, zero and PAGE16 boundary lengths up to193, three data phases, and
the corrected pagewise BF16-P partial reference. The final, M and L retain the
original ideal quantized FP64 gates. Original BF16-proxy error stays separate.
An implementation pass cannot turn `precision_gate` or model qualification true.

The four cases are N64/S2 and N32/S4, each Q pitch2048/2560. Every scratch arena
really has batch64/width16 backing, nonzero guard offsets and the shipping strides
O `(32768,2048,128,1)`, M/L `(256,16,1)`; active views narrow N and S. All physical
padding, unused rows/segments, prefix/suffix canaries, inactive poison and exact
positive-zero output for empty rows are checked. This is small-context consumer
evidence only; it does not establish 4K/16K, append, full-cost or service coverage.

For each phase the exact same canonical Q/K-code/scale/BF16-V, block table and
lengths are uploaded to old22e and newcompact namespaces. CPU remapping copies
K4096/scale128 and V8192 into actual12544B pages, leaving reserved128B untouched.
Actual compact roles are I8 fullbase/page12544, F32 samebase/page3136 with internal
+1024 once, BF16 V pre-offset4352/page6272. Each arm keeps the original FP64,
finite/guard/readonly gates, then **entire physical O/M/L/final buffers** must be
bit identical across arms at producer-only/eagerA/GraphB/freshEagerB/sideC phases.
No allclose replaces that comparison. Full inputs, gold, actuals, failures and
reducer IR/resources are saved losslessly with SHA-addressed gzip blobs; no
necessary arrays are truncated for transport.

`LiveEpoch` holds the real raw tensor and role tensors strongly. A complete actual
GPU readback must equal the CPU upload before it grants a new process-local
identity lease. Activation invalidates the old lease before readback, including
on failure. Every dispatch checks current allocation/storage/view identity,
typed offsets, format and lease; source/output backing overlap rejects. The
Graph path captures A, uploads changed Q/cache/table/length B and poisons output,
grants B's lease and rechecks captured pointer/layout identity **before replay**,
then compares replay with fresh eager B. Side C repeats admission on the side
stream. Closing the lease rejects further dispatch. JSON metadata alone never
registers or authenticates live GPU storage. These are implemented runtime gates,
still untested on a device; CPU mock tensors only test their rejection behavior.

Before Torch import the frontend validates its own compact compile summary and
every artifact SHA, new shader/header bytes, flags, final-SO SASS/IR inspection,
library SHA and a separate Root actual-SO review sidecar. Review defaults false;
all four actual mechanism/binary flags and known S2/S4 resource integers are
mandatory. Unknown/bool resources reject; spills are recorded and not silently
treated as zero. The original inspection's `actual_SO_audit_passed:false` remains
unchanged, and Root's new sidecar binds its exact SHA. The old22e library must
match its fixed SHA and its existing codec snapshot receipt must independently
replay. This does not transfer old attention qualification to compact.

CPU-only commands (stdlib; no Torch/Triton/NumPy import):

```sh
python3 -B tools/check_biv150_native_k8_compact_consumer_cpu.py
python3 -B tools/probe_biv150_native_k8_compact_consumer.py
```

After Root alone performs the new compile and independently reviews the actual
SO, create an **unapproved** template and fill it only from those actual artifacts:

```sh
python3 -B tools/probe_biv150_native_k8_compact_consumer.py \
  --review-template --compile-summary /absolute/compile-out/compile-summary.json
```

Future Root-only device command (all paths are required; review SHA is the exact
saved approved sidecar SHA). No device or target compilation was run while
developing this frontend:

```sh
python3 -B tools/probe_biv150_native_k8_compact_consumer.py \
  --gpu --root-owned-exclusive --out /absolute/new/consumer-out \
  --compile-summary /absolute/compile-out/compile-summary.json \
  --assembly-review /absolute/approved-review.json \
  --assembly-review-sha256 APPROVED_REVIEW_SHA256 \
  --compact-library /absolute/new-compact.so \
  --original-library /absolute/original22e.so \
  --codec-summary /absolute/original-codec-out/codec-summary.json \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263
```

Root's external serial scheduling is the exclusivity authority. There is no PID
namespace inference, fake host identity, background queue or service fallback.
No timing path exists. Counters distinguish eager dispatches, capture node
registrations and replay nodes. Actual compiler/mechanism/resources, live epoch,
bitwise/numeric/Graph/side, model, 42-layer allocator/service and capacity benefits
remain pending. N40 and the shipping S16 policy are unchanged.
