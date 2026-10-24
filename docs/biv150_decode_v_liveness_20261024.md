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

## Historical correction and scope of the recheck

The earlier c2b4215 (2026-10-23) already implemented
`_gqa_decode_attention_low_shared`: late V loads with .cg full-page K/V loads,
paired with target384 instead of256 and a small-batch fallback. The old combined
candidate failed a full stochastic precision run99/105; that does not isolate
the late-V operator as the cause, since accuracy batch8 retained the old kernel.
Its standalone uniform-shape evidence reported4–5% gains for the combined policy.

An AST comparison now confirms this probe's delayed-V kernel is identical to that
legacy low-shared kernel after normalizing function name and removing cache
modifier keywords. Therefore this is **not a newly invented kernel**. It is a
component recheck on the current shipping256-program policy, ordinary loads,
real packed KV strides, independent CPU references and changed-input Graph tests.
Its metrics must not be credited as a novel algorithm or borrowed from the old
combined configuration. Frozen queued source remains2170077; this is a later
report-only correction.

## Prefill composition audit before scheduling another experiment

The running latest-team gate service selected the exact Q-fragment override
SHA2568ae2258568db275c2f3b5e4fc69709431744f08de43343d82a35cf312bfb405a.
Its full TTGIR was copied read-only from the live temporary artifact and archived.
At252 it allocates Q128×128×BF16=32768 bytes; Q local_load is291, before the
unique loop292. K and V each allocate128×64×BF16=16384 bytes at337/343.
Although their lifetimes overlap, simply moving V later cannot be assumed to
halve current32KiB SMEM: the Q initial staging allocation itself is32KiB and
remains in this hypothetical change. The old prefill late-V probe cf0c770 reduced
64KiB→48KiB but was about2% slower at16K before Q-hoist. Neither that negative
result nor arithmetic32→16KiB is proof of the new combination's behavior.
No additional prefill experiment is queued solely on that unsupported capacity
claim. Eliminating or changing Q's initial shared staging would require a distinct
layout/communication design plus new compiler evidence and full validation.
