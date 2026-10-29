# PR118 executable decode-step diagnostic, not the complete mixed step

This separate child starts at e6c5a576805e35a355af5a8f1a72ff6ffdd1562b; ff107 and
e6c5 worktrees remain frozen. Root explicitly approved this narrower first
benchmark. No new CUDA, codec, host admission, service hook, CLI or numerical
threshold changes are included. Actual device validation is pending.

| case | Real decode rows | KV length | Q pitch | Performance A | Candidate C |
|---|---:|---:|---:|---|---|
|4k|61|4096|2048|current-main private K8 TL/S4|new mixed Native/S2|
|16k|30|16384|2048|current-main private K8 TL/S8|new mixed Native/S4|

The source of A is the exact accepted-main a455
`biv150_konly_encoded_decode.py`, not Native8b and not a same-S TL substitute.
The only import adaptation selects the exact original quantizer under a private
module name, avoiding a service/package installation. It reverses byte-exactly.
The actual loaded TL JIT source/IR/ASM/cubin/metadata and resources are preserved
by the original3eb exporter before timing. Root must still review those actual
artifacts; the CPU source proof is not actual IR qualification.

Both arms use the original PR83 K8 append and b3 stride-aware reducer,34276
16KiB pages, table8192, true width16 scratch. New K is contiguous pitch256 and
V is a real pitch2560 view with readonly padding. Each arm narrows the original
scratch to its own S; its reducer and its independent original golden receive
that same S. No cross-partition O/M/L/final bit equality is required between A
and C. Same-arm Graph/eager and pre/post timing full arrays must remain bitwise
identical, and each arm passes the original all-element FP64/pagewise-BF16-P
partial/M/L and ideal-final .005 gates. Ideal-partial and original-BF16 proxy
metrics remain separate, with precision_gate=false.

The full3eb run body is reused with exact reversible host substitutions. Its
`timed` function is byte-identical. The canonical full-D128 cold cache, whole
encoded-cache comparisons/chunk hashes, complete readonly input/guard checks,
Graph B-before-fresh-eager, side C, full arrays, fresh ABI/resource gates and
post-timing checks remain. The actual two append handles, TL producer and reducer
must have known positive n_regs and integer n_spills=0. ELF scalar spill fields
are not silently equated to runtime n_spills. Native static resources come only
from the actual externally hash-pinned Root SO review.

Timing remains one TL/Native/Native/TL event+host-wall ABBA, unroll8, warm128,
64 timed replays. It measures **decode-N append + producer + reducer only**.
Initial CPU construction, encoding/upload and JIT are recorded separately.
It excludes all2048-token append, paired prefill, natural mixed host eligibility,
metadata/view/ticket preparation, framework, projection and sampling. Graph
captures here are isolated component captures; the natural service mixed path
still requires Graph NONE. `full_mixed_step_timing=null`,
`paired_prefill_included=false`, `host_metadata_ticket_included=false` and
`all2048_token_writer_included=false` are explicit. No result licenses full mixed,
service throughput, model accuracy, capacity or stable performance.

## Actual consumer gate before framework import

The new benchmark requires the actual six-N, both-Q-pitch consumer summary/SHA
from the e6c5 entry. It reuses the original3eb receipt function with exact
width16/identity/case-count adaptations, regenerates all three phases and original
FP64 references, replays every full saved snapshot and compares all real-row
candidate/reference bits. It also checks actual role/schema/layout/zero-row and
dual-library receipts. Summary booleans alone cannot license timing. Consumer and
benchmark validator roots may differ; compile/library/source identities must
match, and both roots remain recorded. Exact plan comparison uses its JSON wire
representation; a wrong-N mutation still rejects. No consumer or SO receipt exists yet in
this checkout, and no production qualification pin is filled.

## CPU and Root-only device commands

```sh
python3 -B tools/probe_biv150_native_k8_mixed_decode_step.py
python3 -B -W error tools/check_biv150_native_k8_mixed_decode_step_cpu.py \
  --parent-source /absolute/frozen-3eb-checkout
# Explicit optional tiny NumPy helper consistency, no Torch/Triton/SDK/GPU:
python3 -B -W error tools/check_biv150_native_k8_mixed_decode_step_cpu.py \
  --parent-source /absolute/frozen-3eb-checkout --arrays
```

The optional array test checks only a tiny seven-page/three-row counter: K256 /
V2560 layout and padding, unchanged original K8/BF16 golden functions and exact
patched-cache helper. It is not one of the actual device shapes and grants no
device numeric qualification. Default CPU entry/check remain stdlib-only.

After Root's real compile/static review and successful full short-consumer:

```sh
python3 tools/probe_biv150_native_k8_mixed_decode_step.py \
  --gpu --root-owned-exclusive \
  --parent-source /absolute/frozen-3eb-checkout \
  --compile-summary /absolute/actual-six-scope/compile-summary.json \
  --assembly-review /absolute/actual-Root-review.json \
  --assembly-review-sha256 ACTUAL_ROOT_REVIEW_SHA256 \
  --main8b-library /absolute/actual8b86.so \
  --library /absolute/actual-six-scope/build/cjj_native_k8_mixed_probe.so \
  --codec-summary /absolute/original-codec/codec-summary.json \
  --codec-proof-sha256 3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263 \
  --consumer-summary /absolute/actual-consumer/mixed-consumer-summary.json \
  --consumer-sha256 ACTUAL_CONSUMER_SUMMARY_SHA256 \
  --case 4k --out /absolute/new-decode-step-out
```

Use `--case 16k` in a separate fresh run for the other major case. Omitting it
runs both serially. Result filename is `decode-step-summary.json`, deliberately
distinct from a full mixed-step summary. Both requested cases retain all original
hard checks before timing. A negative ABBA result is retained as a diagnostic,
not hidden or converted into a qualification. No117 lookahead/5898 or120/121 is
imported, no SO identity/resources are fabricated, and no GPU/network/publish work
was performed while preparing these tools.
