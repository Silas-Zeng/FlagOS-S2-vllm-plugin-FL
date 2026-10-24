# GQA launch-grid wave probe, 2026-10-24

Team integration base: cae4e34b92254acb5883d01612052c50c35cb86b (PR22).
No serving hook or scheduler changes. The shipping GQA source remains SHA256
1a81409ba23a149081f7fc257460ea1512cbfdceeb0af32a2698994958749b9a;
therefore the team's compute-order source guard remains unchanged.

The preceding 48-record diagnosis (archived with this commit) passed all numerical
checks. Three alternating Graph timing rounds observed B33 splits4→16 at about
1.233× / 1.324× for KV4096 / 16896; B40 **ragged** lengths at 1.324× / 1.372×.
B32 and B64 prefer the current widths4 and2. These are complete attention operator
measurements, not service throughput or a diagnosis of the 16K regression.
FULL_DECODE_ONLY can pad real request counts to other captured shapes.

The next probe covers B32/33/36/40/48/56/63/64, uniform and ragged lengths,
KV4096/16896, and the original width plus8 and16 (96 records).
It preserves actual packed KV strides, random pages, NaN scratch and independent
CPU FP64 reference rows, and measures producer plus reducer together with warmed
Graph replay and alternating timing order. Numerical comparisons include all
output rows against the original width. Candidate widths never exceed the
builder's allocated16. It installs nothing into a real service.

The queue pins predecessor PID and process start ticks, then refuses an occupied
GPU or port. It waits behind the existing bounded-head acceptance, keeping the
personal GPU tests serial. The remote source and output are separate from all
other candidates. A complete service candidate requires these results, runtime
shape evidence, and fresh full accuracy and both official workloads; none is
claimed by this diagnostic.

Local validation: Ruff, syntax compilation and git diff whitespace checks only.
GPU results for the expanded probe remain pending.
