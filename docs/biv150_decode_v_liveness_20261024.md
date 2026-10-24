# Decode V live-range experiment

Team base1ea082f79db65a66f78fdd7b1d958335450fa9f4 (PR22+PR23). No service,
attention wrapper, scheduler or global package is changed. Standalone diagnostic.

## Evidence before proposing a change

The existing personal-instance compiled decode artifact matches shipping source
SHA2561a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a.
It reports93 registers,0 spills and8192 bytes shared for B32/KV16384, one64-lane
warp and4 segments. The archived TTGIR has Q local_alloc at line130, Q local_load
at160, and scf.while at162. Q is **already loaded outside the loop**: simply
copying the teammate's prefill Q-hoist would target work this decode lacks.

Inside the loop, K and V local_alloc occur at185/188, QK dot at190, softmax and
rescale at191–214, probability convert_layout at215, V local_load at216, PV dot
at217. V is globally loaded/staged before QK although its fragment is consumed
after softmax. The LLVM loop also contains shared stores/loads and CTA barriers.
Their presence alone does not prove any is redundant or the bottleneck. No
barrier, mask, floating arithmetic, quantization or cache policy is removed.

## One structural hypothesis

Delay only V's global load until immediately before its PV dot, in the full-page
loop and partial-page tail. This might shorten temporary allocation/live ranges
or change registers; it might lose useful memory overlap, or the compiler might
produce the same schedule. The exact source transform asserts both unique
anchors and reverses byte-for-byte. It is not a tile/warp parameter sweep.

The independent probe records new TTIR/TTGIR/LLVM and resource metadata for both
paths on actual packed KV strides8192/256/128/1. Cases: B8/32/40/64,
KV4096/16896, uniform/ragged (16 total). It compares the complete producer plus
reducer, keeps existing segment policy, performs three alternating warmed Graph
timing rounds, requires bit-exact output against the original, CPU FP64 sampled
references, NaN scratch, changed/zero-query Graph replay. Registers/SMEM changes
and any speedup must be tied to the newly produced IR rather than inferred from
source. Existing93-register metadata is historical and may differ with strides.

Queue waits for the latest-team gate acceptance by PID/start ticks and refuses
busy GPU/port. New source and artifacts have independent paths. No measured gain,
model accuracy or service benefit is claimed before results. Full-model gates
remain required before any shipping hook. Local checks: Ruff, syntax, exact
transform inversion and git whitespace checks.
