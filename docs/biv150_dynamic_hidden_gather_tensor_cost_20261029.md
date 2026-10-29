# Private full-call Tensor OFF/ON component (CPU prepared)

Parent: `8e65fcc27ac26a6ab08403b9e2208d3acf9eeb39`. Production files,
the original private helper worker, source pins, 805a cubin/afc cache admission,
5124 launcher admission, numerical checks and refusal paths stay unchanged.
This is a local child for Root review. No device work or performance result.

The helper-v2 independent raw replay checked all 57 original files and all eight
complete uint16 outputs (573,440 words), four independently regenerated hidden
fixtures (8,531,968 words), GPU indices, actual source/ABI/cache/handler bindings.
Its report SHA is `05be726183f7048d0ac1948f62e8f1c461a9b51398e0f7948aa59ee4a55a4109`.
That run checked full hidden readonly on device readback but did not separately
save GPU hidden-after arrays. Constructed Plan, no injected helper output
canaries, no Graph numeric/side/performance/service qualification remain explicit.

## Real control and dispatch policy

The previous optional timing control used an `IndexSelectProxy`. It does not
represent production `hidden_states[logits_indices]` (runner line 4503).
The new adapter returns the **same actual Tensor in both arms**. Feature OFF
naturally reaches the original Tensor advanced-index seam. Feature ON follows
the original execute/prepare/receipt/allocation/launch path. The old direct
`index_select` call remains solely a numeric comparator in the reused worker.

Service-policy evidence is the archived PR118 B4K run:

- source `69827253756a0ef9e618ae0322e9ff361e93504f`;
- manifest `bff4dcb835c5c0de0bf9eeef29829da6f777059558f402aaa1b7e33bce2da16e`;
- executed controller `637d29e30c86a56e0536548b5678067954c8900046dc2d1a9dd78a96f4fa7199`;
- server log `e13480803a70ce788665ff2820cadae2b4fdc6b0ad04d974d5755a33afce27c2`.

Controller lines 518–525 clear deployment overrides; the actual cleared list is
empty. Worker/utils/env/config source hashes equal this child. `use_flaggems`
defaults true, the platform blacklist is empty, and Worker lines 246–278 select
`flag_gems.enable(record=True, once=True, path=...)` for rank 0. The adapter uses
that API with a private output recording path. It does **not** override installed
vendor exclusions or pretend the whole Worker constructor ran.

FlagGems cache/log paths are set before framework/runner imports. Contradictory
inherited dispatch/config flags reject. The installed FlagGems/index source and
actual same-hidden advanced-index call must be observed in numeric profiling;
if registration selects an unknown implementation, it stops before timing.
The archived service's 75 index IR files do not independently attribute calls to
this seam; the new run supplies that attribution, not an assumption.

## Minimal reuse and boundaries

`run_biv150_dynamic_hidden_gather_tensor_cost.py` reuses the existing helper
worker/CLI and all four ordinary/Plan cases. It replaces only its controlled
model and numeric observer. Eight extra OFF numeric calls compare full results
on the same logical inputs; actual GPU hidden-after and GPU index bytes are now
saved. Profiling records actual dispatch function, source, descriptors and
compiled objects. Complete IR/cubin/ABI/options/resources/loaded launcher bytes
are exported. Unknown actual dispatch or binary admission cannot license timing.

There are three reversible edits to the **private** timing function: 8 warmups
instead of 2, a before-block hook, and an after-block hook. The derived function
is saved; the CPU checker reverses all three edits byte for byte. Original
numeric, cold, graph-refusal and exact candidate cache/ABI/resource gates remain.

Each existing ABBA block measures 32 complete controlled `execute_model` calls.
First baseline compilation and eight warmups per block are outside timing.
During the measured block, backend compilation rejects; observed runtime-cache
keys/objects and tracked shader/generated-code/SQLite(+WAL/SHM) bytes must remain
unchanged. Inventory/profiling/export work is outside timing. All 4 case loops
remain, with ordinary and constructed Plan. Model/head arithmetic and genuine
`make_plan`/C3/service are excluded; receipt, new allocation, host work and
original input preparation/H2D/event wait are included. A single ABBA remains
a component diagnostic, with `performance_verified=false`, not service gain.

## Local CPU and later Root-only CLI

```bash
/opt/homebrew/bin/python3 -B -S tools/check_biv150_dynamic_hidden_gather_tensor_cost_cpu.py
/opt/homebrew/bin/python3 -B -S tools/run_biv150_dynamic_hidden_gather_tensor_cost.py
```

Later, only after Root serial admission, use the same reviewed read-only 8d
compile/v1 production directories and three Root-reviewed small proofs:

```bash
python -B tools/run_biv150_dynamic_hidden_gather_tensor_cost.py \
  --run-helper-cost --root-owned-exclusive --expected-source-commit CHILD_HEAD \
  --compile-out ORIGINAL_8D_OUT --actual-review BDB_REVIEW_JSON \
  --bits-review NINE_FE_REVIEW_JSON --production-out ORIGINAL_HELPER_V1_OUT \
  --production-review THIRTY_NINE_EF_REVIEW_JSON --out FRESH_OUT --timing
```

Dropping `--timing` runs only the new numeric/actual-dispatch stage. These are
placeholders for already reviewed paths, not an executable admission or queue.
No launcher, SSH, GPU work or service activation is performed by this delivery.
