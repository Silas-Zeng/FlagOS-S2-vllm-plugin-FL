# Projection provider and output planning — component plan

Dependency: PR58, based on current team main34942f7. No production install,
dispatch policy or default is changed. This is a prepared component experiment,
not a measured speedup. PR58 retains every existing small-row BF16 protection;
this helper refuses SDK rows1–8 rather than weakening those protections.

PR58 exact-source candidate/control4K is5760.27/5624.27 tok/s (+2.42%) with
precision102/105 and103/105. Candidate16K4049.28; matched control16K still runs.
Both TTFTs remain under the working competition limit. Neither these data nor
old confounded runs prove that one GEMM provider is best at every row count.

The pinned FlagGems linear wrapper calls an existing linear kernel with14
positional arguments, bias disabled, GROUP_M8 and SPLIT_K1. It allocates its
output internally. The component helper supplies that output directly with the
same ABI, source and dispatch kernel; no SDK symbol is replaced and no new math
or upstream update is introduced. Before any device launch it validates the
selected Iluvatar module/source and LibEntry source. Existing eager fallback
and unsupported shapes are not silently rewritten.

Next device experiment: compare existing MM/out, existing functional linear,
and provided-output linear on actual QKV/O/down/all42layer weights and the8
uncompressed edge gate layers. Use rows32/64/2048 for the full weight working
set, with numerical and changed-input poisoned Graph checks before timing.
Supplement smaller/awkward rows9/16/24/31/33/40/48 as scope checks. Do not pick a
production row threshold before measured evidence. A single repeatedly hot
weight is insufficient to rank a memory-bound full-model projection path.

The current personal GPU remains exclusive to PR58's full matched control.
No new GPU job is queued. Any subsequent component win still needs actual
service selection, full105, sufficient warmup, both fixed benchmarks/TTFT and
post-warm correctness before merging.
