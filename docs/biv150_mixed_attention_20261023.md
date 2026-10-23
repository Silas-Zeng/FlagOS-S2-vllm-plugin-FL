# BI-V150 mixed attention candidate (validation pending)

Base: team integration `3283672`; implementation `3789b2a`, tail dispatch `e444c9c`.
The existing natural-prefix split remains installed. This candidate handles residual mixed
operator calls, including scattered single-query rows and prefill-first input layouts.
A scheduler-level mixed execution does not prove the old operator receives a mixed call:
this corrects the earlier diagnostic inference. Installation alone is not selection evidence.

The normalized prefill clone skips qlen=1 rows; two offset-aware GQA kernels compute those
rows using GPU cumulative query offsets. No CPU readback, sorting, scheduler change, KV
quantization, model-weight change, or installed-package mutation is involved. Original
paths handle unsupported geometry/features, pure decode, fewer than 8 or more than 64
sequences, and large balanced multi-prefill calls. Short tails are included. Builder scratch
is reused when compatible; otherwise one bounded scratch set per CUDA stream is owned.

## Operator validation

`tools/check_biv150_mixed_attention.py` at e444c9c: 10/10 pass, six select the new
operator and four fall back. Includes true packed KV strides, padded/random block tables,
empty rows, multiple/scattered prefills, prefill-first placement, independent FP32 checks,
and one captured graph replayed with 0/15/31 decode rows. Feature exclusion assertions
cover causal=false, pure decode, softcap, windowing, TD, and sinks.
Three alternating timing rounds (11 samples, graph unroll 4) give median ratios:

| Case | Speedup over normalized generic operator |
| --- | ---: |
| 63 decode + prefill, KV 4096 | 1.3872x |
| 31 decode + prefill, KV 16384 | 1.1973x |
| 31 decode + 17-query tail, KV 16384 | 1.8962x |
| Scattered rows and prefills | 1.0734x |

These are **operator microbenchmarks**, not model-throughput gains, and the generic
reference does not include the existing natural-prefix split. Do not extrapolate these
ratios to whole-service gains. Four fallback cases are ~0.998–1.000x and bit exact.
Raw results are in `docs/evidence/biv150-mixed-attention-20261023/`.

## Service validation in progress

Personal instance only, source `/workspace/experiments/minicpm-s2/code/mixed-attention-20261023`.
Runner `tools/run_biv150_mixed_attention_acceptance.py`, output
`/workspace/experiments/minicpm-s2/mixed-attention-20261023/full`.
Unchanged official serve and benchmark arguments; full 105-question Level-3 test first,
then stabilized warmup and one scored run per 4K/16K case. It requires real workload
selection evidence before reporting complete service acceptance.
At 21:55 CST the service was live, installation confirmed, KV capacity 523056 tokens,
and accuracy evaluation was running. This is a historical snapshot; inspect live processes
before continuing. No completed precision or throughput score exists yet. Do not merge.

## Historical native candidate distinction

The separate `beb3518` configuration passed 100/105 precision and achieved 5318.13 total
tokens/s at 4K (+10.6%), TTFT 3978.19 ms. Its 16K was 3094.24 (-8.02%), TTFT160217.19 ms.
Its precision result remains valid for that configuration. Later combinations and the
gate-only variant failed precision; those failures do not retroactively invalidate beb3518.
PR14 is still an unmerged draft. This mixed-attention branch contains no native INT8 path.

## Read-only recovery audit

`python tools/audit_biv150_acceptance.py SAVED_FULL_DIR --require-mixed` reads saved
summary evidence only and never launches workloads. It checks exact successful
request/token totals and concurrency, finite consistent token/duration throughput,
105-question precision/error accounting, stable final warmup pair, working TTFT
references and both 20% targets. Mixed selection log evidence is required with
the flag. Missing data remains incomplete. This is not a replacement for review
of raw predictions, commands, source/SO identity and actual process termination.
CPU checks reproduced the historical adaptive candidate's 4K TTFT rejection and
rejected missing data, changed token accounting, inconsistent timing, and missing
mixed dispatch logs. Synthetic passing values were explicitly fixtures only.

The user confirmed the instance is unavailable until 2026-10-24 06:00 Asia/Shanghai.
An authenticated SSH observation failure does not prove the old jobs terminated.
The one-shot in-chat automation `06-00` is scheduled for 06:00; inspect the old
processes and outputs first. No new GPU workload was started during this outage.
