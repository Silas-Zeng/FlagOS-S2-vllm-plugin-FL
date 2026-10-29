# PR123 multi2 Q-hoist: selected 4K A/B/A2 service evidence

This service version has **not demonstrated a stable throughput gain**. Do not
merge it on these results. Keep the bounded positive component result for a
possible later revision; no 16K service run or further long run is requested.
This evidence-only child of `3e43fc40753a13000f52b967ff8f9d9456622e29` changes
neither production nor the test protocol. Root handles publication and closure.

## Formal results and unchanged protocol

A and A2 use main `899ce77470b0b539aebc108fd122f668c2d5894d` with controller
`637d29e3…`, encoded implementation `ed27276e…`; B uses candidate `3e43fc4…`,
controller `eb1cd254…`, implementation `6ca0be90…`. All retain the accepted
5898 Native decode and ebbc mixed libraries, Q32, v6 tuning and the original
94db helper. The benchmark function remains `e2363389…`.

Each fresh service ran the original 67 precheck requests, three short warmups,
one full-workload warmup and one formal score: 4096 input / 1024 output tokens,
256 requests, concurrency 64, seed 0. The extra two plain-smoke coverage requests,
statistics RPC, Graph observer and profiling were off. The full warmup is never
counted as a score.

| Arm | Formal total tok/s | Mean TTFT ms | Duration s | Completed | Unscored full warmup tok/s |
|---|---:|---:|---:|---:|---:|
| A, main | 6999.844342 | 3530.986318 | 187.249878 | 256/256 | 6939.336836 |
| B, multi2 hoist | 6875.699498 | 3583.003829 | 190.630786 | 256/256 | 6853.686408 |
| A2, main repeat | 6468.325718 | 3662.189524 | 202.636672 | 256/256 | 6988.007730 |

B is **1.7735% below A** and **6.2980% above A2**. The same main's A2 score is
7.5933% below A; B lies between the two baseline endpoints. The ratio to their
arithmetic mean is recorded in JSON as descriptive arithmetic only, not a causal
estimate or acceptance criterion. This sequence neither establishes a stable
benefit nor proves that all differences are random noise. It also does not
establish a reliable regression magnitude for the candidate.

## What the complete audits established

A, B and A2 each passed full collected-file SHA checks, frozen production and
controller identities, actual service CLI, all five benchmark commands/results,
67 complete precheck responses, cache reconstruction and owned cleanup. Their
local independent review hashes are in [results.json](results.json). B's overall
review remains explicitly partial for the separate scope reason below; benchmark
completion is not confused with that qualification gap.

Actual cache observations were A 359→362, B 360→363 and A2 358→364 metadata files.
A/B each added index shapes 1774, 1420 and 469. A2 added six index metadata files:
1420, 469, two for 1800 and two for 1714. Existing metadata changed/removed sets
were empty; the selected Triton 3.6 tuning hashes changed in every arm. The
window includes benchmark client preflight. No trace isolates compilation cost,
so neither the extra artifacts nor A2's score/full-warmup difference is assigned
as the cause of the longer measured duration.

## Precision and actual candidate coverage remain separate

The complete official Math105 audit verified **103/105** unique responses under
the original parallel-8 command. That worker had no multi2-hoist selection; this
is the candidate configuration's model result, not a per-kernel coverage proof.

B's throughput worker really prepared and selected a multi2-hoist binding, but
its actual suffix was **Q2047 / N2 / grid65×2**, while the reviewed component was
Q2048 / N2 / grid66×2. Binding `864c2106…` and hoisted cubin `8955350f…` are recorded.
The original scope comparison rejected the new geometry before the final ELF
comparison. Preserve that rejection: current-worker hard numeric/Graph and final
ELF equivalence remain unqualified. No numeric failure is inferred from a shape
scope mismatch. The benchmark/source/cleanup review passed independently.

The first successful host selection is present. Sparse same-worker timestamps
cannot place it exactly within a benchmark stage. Combined-log ordering suggests
short warmup 0, but does not prove score-window calls, GPU time or Graph replay
frequency. An initial marker is not a call counter.

## Positive component evidence retained

The earlier frozen `ce905346…` complete-reader ABBA measured **2.134541 ms →
1.465148 ms, 1.456877×**, for Q2048/N2, pitch2048, BQ32/K64/w16/s1. It includes
Q quantization, loads/staging, QK, softmax, PV and epilogue; it excludes the writer.
All original hard numerical, full-output, Graph and side-stream gates passed.
Both arms' original BF16-proxy relative-RMSE failures remained explicit diagnostic
failures; no threshold was changed. That component result motivates preserving
the direction, but it does not override this service A/B/A2 outcome.

Only this selected README and compact results JSON are added. Full original
arrays, caches, logs, binaries, manifests and audit reports remain in local
archives; they are not included in this commit. No production edit, additional
GPU run, new profile, precision relaxation, model qualification transfer or 16K
measurement was performed while preparing this evidence.
