# Full-vocabulary projection hypothesis

Base f1cf5d7 includes team main 3283672, reverified on 2026-10-24. This branch
adds isolated probes only; there is no model hook, scheduling change or new
upstream framework code. The current SDK calls LM head through
`UnquantizedEmbeddingMethod.apply`, separately from the linear method patched
by the gate candidate. It does not yet optimize the LM head.

Measure synthetic BF16 weights with the full 130560 x 2048 vocabulary geometry
at sampling row counts 1, 8, 16, 24, 32, 40 and 64. These are selected cases,
not a captured production shape frequency distribution. Compare FlagGems BF16
mm against the existing native compressed INT8 linear primitive. Both weight
copies coexist only for diagnosis; packing is excluded from timing, while
activation quantization, vendor GEMM and output epilogue are included.

For each row count, sample 32 columns spanning the full vocabulary, compare to
CPU FP64 dense and CPU INT64 quantized references, and report both relative
RMSEs. Transfer to CPU before FP64 conversion to avoid the documented CoreX
conversion issue. Native quantized output allows a BF16 rounding tolerance;
quantization error against dense BF16 is reported separately, not hidden by
that primitive check. Three alternating graph timing rounds compare actual
full-vocabulary operations. Changed-input and all-zero graph replay check that
captured execution reads current hidden states.

This does not establish real-model accuracy, real hidden-state error,
perplexity, KV capacity, serving integration or end-to-end speedup. Any useful
microbenchmark result still requires a new model candidate and full acceptance.

The queue waits on an explicitly supplied predecessor PID plus start ticks,
checks port 9031 and GPU idle without stopping other processes, and verifies
the audited native library hash both before waiting and before probing. Use
the existing validated SO from residual source 20c991a, SHA256
`cf966eda1802da08bfdc07cc0b9be1e2688f80518fbbd519a6ce10a16949c562`.
It never builds, installs or replaces a global library.

Local syntax, Ruff and diff checks pass. The completed GPU microbenchmark is reported below; model acceptance is pending.

## Completed isolated GPU result

Source 7c1ba7d6058a3f7e68d50b090cc2dbe2fbfdaeed passes all seven sampled
primitive checks and both changed-input/zero graph replays at every M.

| M | Median speedup | Quantized relative RMSE | BF16 relative RMSE |
|---|---:|---:|---:|
| 1 | 1.5908x | 0.012593 | 0.002093 |
| 8 | 1.6040x | 0.012336 | 0.001668 |
| 16 | 1.6352x | 0.012775 | 0.001662 |
| 24 | 1.6312x | 0.011561 | 0.001687 |
| 32 | 1.6639x | 0.011574 | 0.001658 |
| 40 | 1.6873x | 0.012041 | 0.001654 |
| 64 | 1.5951x | 0.012076 | 0.001720 |

The native projection wins this synthetic microbenchmark at all measured M,
including activation quantization and the output epilogue. Sampled INT64
quantized references match exactly, but dense error is materially larger than
BF16 (about 1.2% versus 0.17% relative RMSE). Actual model hidden-state outliers
and token changes remain untested. This supports preparing an isolated LM-head
model candidate, not merging a model hook or claiming end-to-end speedup.
