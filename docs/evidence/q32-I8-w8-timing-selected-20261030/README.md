# PR126: Q32 I8 w8 preheader diagnostic result

Close this measured variant after review. One warm w16–w8–w8–w16 round per
mode found w8 slower than the actual accepted main899 hoisted w16/S1 control.
The complete implementation checks passed, while the original BF16 quality
gate remains false. This version does not advance to service. The conclusion
covers this BM256/BQ32/w8/S1 exact-preheader configuration, not every w8 design.
Exact samples and selected identities are in [results.json](results.json).

| Complete component | w16 mean ms | w8 mean ms | w16/w8 | w8 extra latency |
|---|---:|---:|---:|---:|
| Reader | 2.218609929084778 | 2.4827897548675537 | 0.8935955711655219 | +11.90745% |
| Append writer + reader | 2.2421488761901855 | 2.5064316987991333 | 0.8945581390725431 | +11.78703% |

Synchronized host ratios were 0.8941141418326065 and 0.8945840857060723.
Source `edbd30da11c6d56f2e16c9d2b0c6f3722935c55e` completed on original
owner 322147/start366942236, same-start terminal Z. Root replayed the full
independent audit (SHA `93e7d0d5f0ba8d9966c91fa902b0a740050e95280d11c1d35dc65b43ab7984ff`).
All 92 originals/194141359B were SHA-verified. Arrays, caches, binaries, logs,
and the full auditor/report remain private; this directory contains only
these two selected text files. No production or experimental kernel changes
are part of this result commit.

Only random-pattern KV6143/prefix4095 was timed: N1/Q2048, Q16/KV2/D128,
BM256/BQ32/K64, Q pitch2048, table pitch8192 and writer stride2560. Both arms
use the same 773 physical16KiB pages (12664832B). Aligned KV6144 has numerical
coverage only. The control is actual main899 N1 hoisted w16/S1, hash `53ef08b7`,
cubin `7eadbdbf`; the candidate is actual preheader w8/S1, hash `773d51c9`,
cubin `03d37332`. Source/typed ABI/options/IR and loaded executable identities
were verified against their respective reviewed compile artifacts.

Actual loaded resources were w16:122 VGPR/33792B shared/1024 threads and
w8:237/33792/512. Runtime `n_spills` was0 for both, while actual ELF scalar
spills were96/478. These are distinct fields; no occupancy or causal slowdown
claim follows from the counts alone.

Each CUDA Graph contains four complete reader or writer+reader steps. Each
arm received eight warm replays per mode. Each of the eight timing blocks
used sixteen replays, with event and synchronized host duration divided by64.
Observed168 replays comprise8 numerical,32 warm and128 timed replays. Python
attempts were36 reader/18 writer including capture, not profiler hardware
launch counts. Host timing includes submissions and end-event synchronization;
validation/readback are excluded. This repeats a warm12.66MB fixture and
excludes normal Python helper/JIT/host dispatch, paired prefill, other model
operators and service scheduling. One ABBA does not establish stability.

All eight numerical stages passed implementation hard checks: complete
4194304-element outputs and12664832B cache, independent codec/code/scale bits,
canaries, readonly checks, changed-input GraphB before fresh eagerB, and
nondefault-stream execution. Full w8 outputs had zero error and were also
bitwise equal to w16 in every stage; bitwise equality was diagnostic, while
the unchanged full pointwise gate was mandatory. FP64 covers all2048 uniform
rows and ten selected random rows. It is ideal global softmax, without
BF16-P/online intermediate rounding; the full random reference is w16 control,
not full random FP64. Original-BF16 random relative RMSE remains
0.009618575519632048, above the unchanged0.005 gate; quantized-reference RMSE is
0.002283783103268182. `precision_gate=false`, model and service remain unverified.
Explicit numeric-diagnostic mode preserved that failure and all hard gates.

After each timed block the frozen worker synchronously checked complete
output/guards/cache/readonly and saved hashes. The independent audit matches
all eight blocks against hashes of archived freshB full arrays and original
inputs. Separate per-block GPU arrays, readonly-after tensors and raw timer
endpoints were not archived; this limit is preserved.

The source-level Q preheader move and reduced thread count did not improve
these complete component modes. Any different w8 mapping or staging proposal
must establish fresh actual resource/synchronization evidence and outperform
the accepted hoisted w16/S1 control under the same hard gates. This result
closes only the measured variant; it provides no16K, model, service, capacity,
or generally stable performance conclusion.
