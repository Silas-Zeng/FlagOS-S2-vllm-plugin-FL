# PR118 executable short consumer, before the mixed complete-step benchmark

This child starts at ff10718980f35a2275771b12eb02b9d4adf20d24. It adds a real
opt-in numeric entry; it does not change that parent's kernel, host admission,
CUDA/codec, C3, default-OFF behavior or None service qualification pins. No actual
mixed SO exists in this source record, and no real device has been used here.

`probe_biv150_native_k8_mixed_consumer.py` reuses the frozen3eb v2 consumer's full
run body, helpers and hard thresholds. Exact reversible source edits only select
the namespace/library, build true width16 scratch, expose all unused segments to
the original poison check, and record actual strides. The original metadata,
FP64/pagewise-BF16-P partial and ideal-final references, full input/output blobs,
changed-Graph-before-fresh-eager and side phases remain. BF16 proxy and ideal
partial diagnostic failures remain separate; a hard implementation pass does
not turn precision_gate true.

Each of29/30/31→S4 and60/61/63→S2 runs at Q pitch2048 and2560. Candidate rows
are exactly N. The accepted8b reference uses N32/N64 with3/2/1 or4/3/1 extra
zero-length rows. Its entire dummy table remains -1, and original zero-output /
inactive-poison checks test those dummy rows. Only the real N rows are compared
for full O/M/L/final bits. Every original snapshot stage is checked, including
producer-only output poison, Graph-first B and fresh eager B. Cache bytes and
all real-row query/length/table inputs must agree between roles.

The short consumer retains29 pages and max context193: it is a tail/empty/offset
numeric test, not a34276-page long-context result. Physical GPU table pitch is
8192. The old FP64 reference reads the equivalent original16-column logical
mapping; all extra uploaded columns remain -1. Full scratch backing has width16
with strides32768/2048/128/1 and256/16/1; Native and reducer receive narrow
same-stride views. These are actual device assertions when Root executes it.

Default CPU commands:

```sh
python3 -B tools/probe_biv150_native_k8_mixed_consumer.py
python3 -B -W error tools/check_biv150_native_k8_mixed_numeric_cpu.py \
  --parent-source /absolute/frozen-3eb-checkout
```

The CPU checker verifies the exact source inverse and twelve case descriptors,
including every dummy zero length and invalid table column, without NumPy,
Torch, Triton or device compilation. It exercises resource/approval/schema
refusals using a clearly labeled in-memory schema fixture; that fixture is not
written as an actual approval receipt. Original ff107 CPU routing still applies.

Root-only future numeric command after actual compilation/static approval:

```sh
python3 tools/probe_biv150_native_k8_mixed_consumer.py \
  --gpu --root-owned-exclusive \
  --parent-source /absolute/frozen-3eb-checkout \
  --compile-summary /absolute/new-mixed/compile-summary.json \
  --assembly-review /absolute/root-actual-mixed-review.json \
  --assembly-review-sha256 ACTUAL_REVIEW_SHA256 \
  --main8b-library /absolute/actual8b86.so \
  --library /absolute/new-mixed/build/cjj_native_k8_mixed_probe.so \
  --codec-summary /absolute/original-codec/codec-summary.json \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263 \
  --out /absolute/new-consumer-out
```

The compile summary must bind the exact six-row ff107 host/CUDA sources and all
actual artifacts, final-SO IR/SASS and seven symbol renames, successful no-launch
compile, and CUDA-initialized before/after false. Root's externally hash-pinned
review must approve all31/63,29/30,60/61 host-ABI flags, actual final-SO binding,
normalized8b mathematics and full/tail/wait/codec. Both S2/S4 actual CoreX resource
rows must have positive integer `.vgpr_count`/`.sgpr_count`, shared8192, warp64,
and known zero spill/private fields; None/bool/missing values reject. No candidate
SO SHA or actual resources are invented in this checkout.

The original actual codec receipt is fully replayed by the unchanged3eb gate
before Torch. The original8b SHA is fixed. Static-only dual-library qualification
is not fabricated: the entry loads both real libraries and checks both exact
namespaces, schemas, CUDA registration and loaded sets before its first attention
call, retaining that actual runtime receipt. A wrong library/dispatch stops the
original consumer. The compile and validator source roots are recorded separately;
new tools do not pretend to be the old compiler checkout.

This first executable entry is consumer-only: it has no benchmark or service
qualification. The following complete-step frontend must compare original TL
S4/S8 with Native S2/S4 for N61/L4K and N30/L16K, including real K256/V2560
all-token append, original width16 reducer, unchanged paired prefill, metadata
setup and actual stream completion. The old117 graph-only append/decode/reducer
benchmark is insufficient: natural mixed service runs Graph NONE and has a
prefill suffix. Same-S TL and padded8b are numerical controls, not a substitute
for the shipping performance baseline. No timing is exposed here under a
misleading full-step label.117 lookahead/5898 and120/121 are not imported.
