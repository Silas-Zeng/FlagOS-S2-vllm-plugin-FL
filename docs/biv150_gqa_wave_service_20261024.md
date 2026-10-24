# Physical-B40 decode segmentation on the latest-team residual gate

Parent e3a0497adbd6570dd522877131bc47058350da83 already incorporates team
main1ea082f (PR22+PR23) and the compressed residual gate. The parent's full
model acceptance is still running; it has no credited model result yet.

## Measured operator evidence

The independent shipping-kernel probe fe097dd completed96/96 numerical cases.
For physical B40, width16 vs4 observed uniform4K1.2956×/KV16896 1.4343× and
ragged1.3319×/1.3860×. B32/64 prefer their current widths; B56/63 have regressions
for some lengths. This supports a bounded dispatch, not a uniform width16 policy
or an end-to-end gain claim. Full evidence is on the shared wave-probe branch.

## Shipping change and compatibility

Only `(num_seqs,kv_heads,builder_segments)==(40,2,16)` chooses16 segments.
Every other input retains the exact former host policy. The producer, reducer,
math, masks, allocated scratch bounds, model, service and scheduler are unchanged.
CPU evaluation covered7168 chooser tuples; all outputs stay within builder
allocation and all unselected tuples equal the original.

GQA source SHA2596cc09caf9b5b38b49755ae9af41ebdcf59a1d5ad889fc8e9319d1584438d1.
The Q-fragment EXPECTED_GQA pin is updated to this exact source; its resulting
module SHA is3f90de854adeb7d3584b19bad1f317cb527f16f14bec15d40f55ffa3e13b4d8a.
Compute-order SOURCES pins for these two modules are correspondingly updated.
All other helper and Q-fragment logic and the whole parent runner remain exact.
These are explicit new audited identities, not disabling or weakening checks.

## Required gates before full service

A new standalone48-case test uses physical40 rows with live0/1/8/32 and33..40,
KV4096/5119/16384/17407 (start/end with partial-page tails), actual packed strides, random page tables, invalid sentinel pages
and NaN scratch for padded rows. It requires full-output comparison, CPU FP64
references, exact zero padded output and Graph replay after changing live rows
40→33→0→original without changing addresses. It saves3 alternating warmed
producer+reducer timing rounds. Reachable live33..40 cases must have minimum
median speedup>=0.97 and geometric mean>=1.03; all48 numerical cases must pass.
A failed gate stops the queue before building/loading a real model.

Then private native build and9 gate checks must pass, followed by fresh105-item
accuracy and the fixed warmed4K/16K benchmarks with complete requests, token
counts and TTFT. Existing acceptance requires actual compute-order selection
and Q-fragment installation/selection with no recorded fallback. The tiny host
policy is chosen during Graph capture; these microtests validate its padded
shapes, not real service shape frequencies. Only full service results can credit
the combination, and a separate trace would be needed for per-shape attribution.

Source and outputs are isolated; the waiting queue pins its V-liveness probe
predecessor PID/start ticks and refuses busy GPU/port. No running checkout is
modified. Local validation: Ruff, syntax, source-pin/AST identity and chooser
bounds. GPU padding results and all model/service results remain pending.
