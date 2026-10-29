# Frozen PR118 decode step: selected results

Independent CPU review and Root replay passed for source
`01dfe84028560b1a982dac33bc788e875ce9cf1b` and mixed library
`ebbc35d028bb7c719895c8b3569f3120387380888f23c4bb772b7176f0856bba`.
Root's private review SHA is
`db7a21308aa07b89b8e35493a9c05d2ccf8fc3028c173d1dbd2df37824b0cc49`.
[results.json](results.json) contains selected scalar results and artifact identities.

| Case | Reader segments (TL / mixed) | TL mean ms/step | Mixed mean ms/step | TL / mixed time ratio |
| --- | --- | ---: | ---: | ---: |
| 4K, N61 | 4 / 2 | 0.430973709 | 0.340525776 | **1.26561×** |
| 16K, N30 | 8 / 4 | 0.793356687 | 0.633914739 | **1.25152×** |

Each case uses the original single TL/mixed/mixed/TL ABBA, warmup 128,
64 replays and unroll 8. The corresponding host launch/synchronize time ratios
are 1.26554× and 1.25148×. These are complete **decode-component** steps:
N-row K/V append, reader and matching reducer. They exclude paired prefill,
the all-2048-token writer, host metadata preparation, QKV projection and RoPE.
Initial cache construction, upload and initial JIT costs were recorded separately.
The baseline is the current mixed-path TL reader; this is not a comparison
against main117's pure-decode 5898 library or a model throughput result.

Both cases use 34,276 physical 16KiB pages, Q/K/V pitches 2048/256/2560,
scratch width 16 and independent Q/K/V backing. All 232 originals were rehashed;
36 complete output snapshots and 144 original hard array gates passed. All twelve
case/arm/phase FP64 references were regenerated. Same-arm changed-input Graph
before fresh eager, side-stream, and pre/post-timing comparisons passed.
Each arm retains its own segment-count oracle: **different-S partial outputs
are not required to be cross-arm bitwise equal**. The original mathematical
thresholds remain unchanged. BF16 proxy is 24/24; ideal-partial diagnostic 0/24
is retained and is not substituted for the original partial hard gate.

The complete cache check is a CPU reconstruction of every expected phase and
chunk digest matched to the recorded runtime byte-comparison ledger. GPU cache
bodies were not saved: this is **not a new GPU readback or memory-sanitizer run**.
Output snapshots and guards are separately audited from their saved artifacts.
Both actual reducer roles are bound independently to source, JIT artifacts and
resources. Runtime reducer registers are S2: 11, S4: 11, S8: 14; runtime
`n_spills=0` is not a claim about every static ELF spill field. A same-S reducer
can legitimately have the same compiled hash across roles.

This frozen component result does not qualify the merged HEAD, model shared-QKV
allocation, natural mixed service Graph path, full-model capacity, Math105,
official score or service throughput. One ABBA per case does not establish
statistical stability. Production qualification and defaults are unchanged by
this documentation commit. The preceding [consumer result](../native118-consumer-selected-20261029/README.md)
remains separately scoped.

Only this README and selected results JSON are included. The full private
232-file archive (164,245,898 bytes; gzip SHA recorded in JSON), arrays, caches,
execution logs, binaries and complete audit remain local. Root independently
replayed the audit; this documentation step did not rerun GPU work or the full
CPU numerical replay.
