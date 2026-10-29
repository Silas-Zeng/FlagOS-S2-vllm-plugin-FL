# PR124: preheader Q32 stage2 diagnostic result

Close this specific stage2 variant with its negative result preserved. One warm
S1–S2–S2–S1 round found no improvement over the actual accepted main899 hoisted
S1 reader. This is an isolated component observation, not a service or model
result and not a verdict on every stage2 design. Exact samples, identities and
checks are in [results.json](results.json).

Source887217375d113d62829e19b5794fce6f70458a88 completed on the original
owner267007/start365746448, same-start terminalZ. The independent full audit
was replayed byte-identically by Root (report SHA `ba88f548000b7538ded3e2e8b9091a402529860888b34c72ec5a3034dd3c0493`).
All92 originals/193296438B passed SHA verification. Arrays, caches, binaries,
logs and the full auditor/report stay private; these two files are selected
text results only. The earlier CPU preparation and numeric source are unchanged.

| Complete component | S1 mean ms | S2 mean ms | S1/S2 | S2 extra latency |
|---|---:|---:|---:|---:|
| Reader |2.220137596130371|2.2602272033691406|0.9822630188774778|+1.80573%|
| Append writer + reader |2.2427358627319336|2.2849892377853394|0.9815082826847978|+1.88401%|

Synchronized host ratios were0.9828359986873827/0.9816897909420151, respectively.
Only random-pattern KV6143/prefix4095 was timed: Q2048/N1, BM256/BQ32/K64/w16,
773 physical16KiB pages, Qpitch2048, table8192 and unchanged writer stride2560.
The control is actual main899 N1 hoisted S1 (`53ef08b7`, cubin`7eadbdbf`), not
raw S2 or an N2 substitute. The candidate is preheader-hoisted S2 (`2d893863`,
cubin`dab726c4`), with identical GPU source mathematics and complete typed ABI.
Actual resources were S1:122VGPR/33792B shared, S2:124/41216. Both runtime
`n_spills` were0, while actual ELF scalar spills were96/93; these are distinct
fields. No occupancy or slowdown cause follows from these counts alone.

Each CUDA Graph contains four complete reader or writer+reader steps. Each arm
was warmed with eight replays per mode; each of eight event blocks used sixteen
replays and divided event/wall duration by64. Observed168 replays comprise
8 numerical,32 warm and128 timed replays. Python attempts were36reader/18writer,
including capture; they are not profiler hardware-launch counts. Host timing
includes replay submission and end-event synchronization, excluding validation.
This repeats a12.66MB cache fixture and excludes normal Python JIT/helper,
host dispatch, other model operators and service scheduling. One ABBA does not
establish statistical stability. No6144 timing,16K or capacity result exists.

The full eight numerical stages passed implementation hard checks: both arms'
4194304 BF16 output elements and the entire12664832B cache per stage, independent
codec/code/scale bits, canaries, readonly checks, changed-input GraphB before
fresh eagerB, and nondefault-stream execution. FP64 covers all2048 uniform rows
and ten selected random rows; full random outputs are S1/S2 bitwise equal.
The ideal global FP64 reference does not model BF16-P or online-softmax rounding.
Original-BF16 relative RMSE remains0.009618575519632055, above the unchanged
0.005 gate; quantized-reference RMSE is0.002283783103268205. Finite and original
pointwise checks pass, but `precision_gate=false`; explicit numeric-diagnostic
mode retained that failure rather than bypassing any implementation hard gate.

After every timed block the frozen worker synchronously read back and checked
complete output/guards/cache/readonly, then saved their hashes. The independent
audit compares all eight blocks with hashes regenerated from archived freshB
full arrays and original inputs. Per-block GPU arrays, readonly-after tensors
and raw timer endpoints were not separately archived; this boundary is retained.

A different S2 proposal must address useful K/scale-pipeline and V-carry overlap
that outweighs its extra staging, synchronization and live values: this version
already moved the Q operand before KV but still uses7424B more shared memory and
two more VGPR than S1. These are concrete resource differences, not a measured
causal attribution. Merely issuing prefetch does not prove hidden latency.
Any changed buffering or schedule needs fresh actual compiler/synchronization
evidence, the same numerical/Graph/readonly gates, and a fair complete-reader
comparison against accepted hoisted S1; beating raw S2 is insufficient. This
result closes only the measured preheader-stage2 configuration.
