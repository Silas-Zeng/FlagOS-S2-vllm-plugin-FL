# Q32 changed-input Graph/side result and transport-only recovery

Root's runtime d2ceb3bd8301ba846792e3370803832dc64270ca completed owner628883 /
start344599705, same-start Z. Graph-side and numeric summaries both completed,
errors=[], Graph_verified=true and side_stream_verified=true. No timing ran.
The original90-second collector timeout was a transport failure, not a kernel
failure. Root used scoped read-only transport62d3ca5… (zlib1/240seconds) to retrieve
the same finished owner; the verified manifest says read_only=true and
model_rerun=false. There was no kernel rerun for collection.

Independent CPU replay covers all twelve full saved mode outputs: eager0 and
Graph1/eager1/side1 for BF16, unhoisted I8 and hoisted I8. Each has4194304 BF16
elements. For each role Graph1/eager1/side1 are byte-identical; unhoisted/hoisted
are byte-identical in all four modes, and saved numeric-artifact outputs match
the corresponding mode arrays. All modes are finite. Full encoded bytes/layout,
actual Q/table/length control digests and all selected-row FP64 metrics replay.

The changed control has2048 queries and6143 runtime KV tokens, within the original
6144-KV/4096-prefix capacity. Q, table and length digests all change before first
replay. The table is rotated and runtime length changes6144→6143; no new geometry,
batchTokens setting or source/kernel expression is introduced.

| Pattern1 comparison | Relative RMSE | Max absolute |
| --- | --- | --- |
| BF16 semantic control vs original FP64 |0.0022874587115760603|0.00033584525499748275|
| I8 vs quantized FP64 |0.0022542700142202956|0.00030156309991787034|
| I8 vs original BF16 FP64 |0.009626741068106567|0.0011051162578886997|

All pointwise checks pass. The original-BF16 relative proxy remains over0.005:
eager_numeric_passed=false and precision_gate=false. This is not a Math105 result.
The per-record parent eager-policy Graph/side=false labels are historical scope
fields; the separate actual mode arrays/outer Graph-side summary establish the
new Graph/side checks. No archived JSON is rewritten to reconcile those labels.

Twelve resource observations come directly after real attention executions, from
the same CompiledKernel objects as warmup. Every mode records BF16101regs/0spills/
98304shared, unhoisted135/0/57344, hoisted122/0/33792. They are observations, with
no zero-spill runtime requirement and no speed inference. All three source/ABI/
key-IR exactness and actual ASM mechanism audits replay. Actual cubin SHA values
are retained without asserting cross-compilation binary equality. The stale
baseline nested exact-cubin helper label remains explicitly invalid historical
metadata. The fifteen extra native mode checks retain one owned PID1881841.

All98 raw files/117797007B plus manifest are preserved exactly. Content-identical
parts reuse20098950B from the previously verified476 eager archive; the remaining
data is in52 new SHA-bound archives, each<2MiB. The checker reconstructs all99
files, checks every original SHA and optionally replays CPU arrays using NumPy;
Torch, CUDA, JIT and network access are unnecessary.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_biv150_konly_q32_graph_evidence.py --arrays --extract FRESH_EXTRACT
```

References are the saved independent selected-row FP64 results; this replay
recomputes metrics but does not regenerate global attention. Canary and read-only
checks remain recorded device guards because their complete live guard buffers
were not serialized. The canonical d6 BF16 control is a semantic control, not
accepted service performance. A later diagnostic must materialize the actual
main BF16 Q-hoisted control before timing.16K, full cost, model105, throughput,
service/TTFT and KV capacity recovery remain unverified.
