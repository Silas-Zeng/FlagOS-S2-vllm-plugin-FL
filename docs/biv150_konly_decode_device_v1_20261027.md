# K-only I8 codec and first full4K/full16K component evidence

Runtime source is frozen at `f28783e15eaf8593305c3e6a057b76bb2f7189cc`, with the decoder kernel/support boundary still `ae00f58629ad85850fec1b8b8d2da28757af5f8b`. This evidence-only change leaves all f287 runtime, driver, numerical gates, and production files byte-for-byte unchanged. The original results and logs retain their original status.

Both runs used the private BF16 append control and the original current-main GQA producer/reducer for comparison. The control is not the production FlagGems writer ABI. No service hook, built-in quantization mode, upstream upgrade, or physical capacity reclamation is included.

## Codec/writer prefix

`cjj-konly-codec-prefix-v1-20261027`, owner PID `625037`, start ticks `343141089`, was terminal with the same owner identity. Its original result is `completed=true`, `passed=true`, `status=codec_and_writer_only`, with no errors. M1 and M8 each cover 141 finite BF16 vectors: all 18,048 codes and 141 FP32 scale bit patterns match the independent integer-rational RN32→RNE reference. The CPU fast oracle examined 851 rational half-boundary cells. Inputs and output canaries pass.

The separate three-page initial encoder matches all 49,152 destination bytes. BF16 and K8 append controls pass invalid/unmapped slots, reserved-byte preservation, unchanged inputs, and original V signed-zero/subnormal/max-value bits. Source and destination allocations are separate. Actual post-launch resources are M1 10 registers, zero spills, zero shared bytes; M8 60 registers, zero spills, 32 shared bytes. This proves only the recorded codec/writer cases. It does not prove all IEEE inputs or an installed native division ISA mapping. Historical frontend metadata keeps `device_numeric_verified=false` because that helper records frontend availability, not the later bounded numerical result; the original JSON is not rewritten.

## Full4K component

`cjj-konly-inline-full4k-v1-20261027`, owner PID `625234`, start ticks `343163134`, was terminal with the same owner identity. Its original result is `completed=true`, `passed=false`, `precision_gate=false`, `errors=[]`. All nine hard checks pass: finite valid outputs, independent baseline reference, codec, complete cache bytes, unchanged sources, scratch/canaries, changed-input Graph/eager, side-stream/eager, and actual IR. There are three controls covering original inputs, ragged/tail/empty/invalid-slot metadata, and zero Q. Inline/preQ scratch, maxima, sums, and outputs are recorded bitwise equal. The NPZ independently permits checking all output bits; the other three buffers are recorded device assertions rather than independently archived arrays.

The original error limits remain unchanged. Candidate output passes the quantized-operand FP64 pointwise gate, but fails the original BF16 relative-RMS gate. For the three original full-length rows:

| Comparison | Relative RMS error range |
| --- | ---: |
| BF16 device baseline vs original BF16 FP64 | 0.002249–0.002296 |
| Candidate vs quantized-operand FP64 | 0.002153–0.002280 |
| Quantized-operand FP64 vs original BF16 FP64 | 0.009074–0.009550 |
| Candidate vs original BF16 FP64 | 0.009399–0.009771 |

The quantized FP64 reference already exceeds the unchanged 0.005 limit before implementation rounding. This supports quantization error as the main explanation for this failure. It is not proof that the device implementation has no error. Ragged lengths 15–100 also have original-reference pointwise violations; empty/one-token controls and zero Q pass. The CPU replay recomputes every reported metric for 21 selected rows from saved BF16 output bits and SHA-verified FP64 references, with exact agreement to the original JSON. The cold BF16 cache was not archived, so this replay does not independently regenerate the complete attention references.

Both actual producer IRs independently replay successfully: logical 8-row I8 QK, K32 native encoding, eight I8 QK MMA calls across full/tail source regions, original BF16 PV shapes and sixteen BF16 PV machine calls, complete I8 C chains and all 64 rescaled native PV C scalars, and exact AST score-conversion sites. No primitive proof is substituted for these actual decoder artifacts. Actual post-launch producer resources differ:

| Producer | Registers | Spills | Shared bytes |
| --- | ---: | ---: | ---: |
| BF16 baseline | 93 | 0 | 8,192 |
| Inline Q8 | 144 | 0 | 6,144 |
| PreQ | 145 | 0 | 6,144 |

The explicitly requested diagnostic performs one alternating timing round after all hard checks pass. Complete inline closure includes Kcode/Kscale plus original BF16 V append, inline-Q8 producer, and the original reducer. Complete baseline closure includes BF16 K/V append, original BF16 producer, and the same reducer.

| Recorded cost | Milliseconds |
| --- | ---: |
| Complete baseline step | 0.5277451873 |
| Complete inline step | 0.4180153906 |
| Baseline append, separately timed | 0.0027509616 |
| Candidate append, separately timed | 0.0032894232 |
| Baseline producer, separately timed | 0.5319432616 |
| Candidate producer, separately timed | 0.4165014327 |
| Original reducer, separately timed | 0.0077649038 |
| Initial encoder, separately timed | 1.7508586645 |

The complete-step diagnostic ratio is **1.26250181×**. Separate stage measurements are not summed to reconstruct the complete closure. Initial encoding is recorded separately and is not service-free. PreQ has numerical evidence in this run, but no complete-step timing measurement. This is a single component diagnostic with `precision_gate=false`, not model accuracy, official service throughput, TTFT, or a measured service gain. Keep PR79 Draft; do not merge or discard this candidate solely from the conservative component error proxy. All prefill/generic readers, the production writer ABI and metadata lifecycle, real prefill/initialization costs, and full model acceptance remain future work.

## Full16K component, subsequent evidence-only update

`cjj-konly-inline-full16k-v1-20261027`, owner PID `625590`, start ticks `343262174`, was terminal with the same owner identity. It uses the same f287 runtime with B32, KV16384 and the actual four-segment policy. Original status remains `completed=true`, `passed=false`, `precision_gate=false`, `errors=[]`. All nine hard gates, changed-input Graph, side stream and post-timing poisoned complete step pass. Inline/preQ output bits independently match for all three controls; scratch/maxima/sums equality is preserved as the original recorded device assertion. Both actual producer IRs independently replay, and actual post-launch resources match the corresponding full4K values above.

For the three original full-length 16K rows:

| Comparison | Relative RMS error range |
| --- | ---: |
| BF16 device baseline vs original BF16 FP64 | 0.002230–0.002294 |
| Candidate vs quantized-operand FP64 | 0.002191–0.002263 |
| Quantized-operand FP64 vs original BF16 FP64 | 0.008806–0.009718 |
| Candidate vs original BF16 FP64 | 0.009076–0.009954 |

All metrics for 21 selected rows independently recompute exactly from the NPZ witnesses. The same reference-regeneration boundary applies: saved FP64 references are SHA-verified but not regenerated from an absent cold-cache dump. Quantized-reference error alone exceeds the original 0.005 limit, while candidate-vs-quantized implementation error is similar in scale to baseline-vs-original. This supports the same bounded quantization explanation, not a proof of implementation equivalence or model accuracy. Neither numerical gate nor primary runtime changes.

The one-round complete-step diagnostic records **1.0318504572ms baseline vs 0.8284283876ms inline**, ratio **1.245551785×**. Separate stage costs are baseline/candidate append 0.0026682692/0.0032115385ms, baseline/candidate producer 1.0306370258/0.8224605322ms, reducer 0.0058230767ms, and initial encoder 3.4984474182ms. These separate stages are not the complete closure and initial encoding is not free. PreQ is numerically exercised, not complete-step timed. Retain Draft for all-reader work and real model evaluation; this remains diagnostic component evidence with `precision_gate=false`, not service +20% or formal acceptance.

## Immutable archive and CPU replay

`docs/evidence/konly-i8-qk-20261027/numeric-device-v1-provenance.json` enumerates every archive SHA/size, original download manifest SHA/size, owner, and omission. All 85 prefix, 222 full4K and 222 full16K downloaded files were verified before packaging (1,396,602 / 8,634,643 / 7,844,259 bytes respectively). Only two `__pycache__/*.pyc` files per run are excluded; their original paths/sizes/SHAs remain in both the untouched original manifest and explicit exclusion lists. The original root download directories are unchanged. All actual source, TTIR, TTGIR, LLVM, ASM, cubin, NPZ, result, owner and log files remain present. Each independent tar.gz is at most 2 MiB. No archive is extracted or imported by the checker.

`numeric-runtime-f287-part1.tar.gz` additionally preserves the exact f287 local Python import closure, explicitly loaded support/probe/input modules, and original main GQA source. Its source manifest records every byte hash. The evidence checker verifies these archived files still match the current worktree; it never executes them.

Run from repository root (stdlib only by default):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_decode_numeric_evidence.py --out /tmp/konly-device-v1-replay.json
```

With existing NumPy, independently replay all codec codes/scale bits and NPZ numerical metrics on CPU:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -W error tools/check_biv150_konly_decode_numeric_evidence.py --arrays --out /tmp/konly-device-v1-arrays-replay.json
```

Neither command imports Torch/Triton, JIT-compiles, uses SSH, launches GPU work, changes an original artifact, or reruns a benchmark. Both preserve `full_decoder_precision_passed=false`, `diagnostic_only=true`, and `service_verified=false`.

Before this evidence commit, `git fetch origin main` timed out. A separate live GitHub CLI query verified team main remained `56f1838192abc940b3fb9accef490022ac125000`, and verified CLI/PR79 author `CJJ1008`, open Draft, remote head f287. This is a fresh API check, not a claim that the failed fetch synchronized refs.
