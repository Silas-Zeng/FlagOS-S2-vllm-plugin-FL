# Same-main native / Triton GQA operator diagnostic

Fresh team mainb3b87d2, independent own branch. Root-only future device execution;
author performs no SSH/JIT/GPU/build. Draft prepares one bounded pair, not model
profiling. Main4K fullwarm6449.65 versus score5655.19 is an observation, not a
proven28-second JIT explanation: only two new prefill compiler file spans total
about2.8s, selected SQLite rows also changed and timestamps do not isolate cost.

Two arms use the actual same b3 producer math and original reducer: actual native
S2/S4 SO content SHAaace693… versus direct exact b3 Triton GQA (no native dispatch).
Both include the same previously tested ENCODE=False BF16 append. K8 is excluded
from the pair; its old measurements cannot be mixed into this new native control.

One fixed34276-page physical16KiB arena matches the548416-token layer budget.
The deterministic no-overlap permutation seed20261028 maps active4K/B64 pages
16384 and16K/B32 pages32768 across that arena; table row stride8192. Each step
appends from length-1 to post-append length, including page/tail boundaries.
Resident inputs/cache, actual ABI, full outputs/guards/readonly and independent
CPU FP64 gates precede one warmed ABBA. Same-arm repeat/Graph outputs must be
bit exact; cross-arm bit equality is recorded, while both must independently
pass original.005+.005|reference| pointwise and.005RRMSE gates. Initial BF16 pack
copy cost is separate; no model/TTFT/service gain is inferred.

Initial Draft currently contains a CPU-only contract; device runner is pending.
