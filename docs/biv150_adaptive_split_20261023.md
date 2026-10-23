# Low-batch GPU adaptive split-K decode candidate

Base: team main `3283672b92fa30f93f652b242193999836a95f88` (verified through remote API).
Code: `d52cae6`; self-contained verification: `474b337`.
Status: operator/wrapper checks passed; full model acceptance running. Do not merge yet.

The existing decode split policy is capped by builder scratch capacity16. At
captured batches1/2/3-4 this candidate uses persistent operator-owned FP32 scratch
of capacity128/64/32. Both GPU kernels read the live sequence length and choose
16 segments at length<=512,32 at<=2048,64 at<=8192,128 otherwise, bounded by
capacity. Host code only chooses capacity from captured tensor geometry. This
preserves runtime length changes under CUDA Graph. Batches>=5 keep their old
width policy. KV remains BF16 with the same physical layout; no framework or
benchmark arguments change.

Inactive reduction weights now use exp(-inf)=0. Previously an extremely negative
valid maximum could make an invalid segment's exp(0-max) overflow and produce
NaN through 0*inf. Controlled query=-2,key=4 tests show8192 nonfinite baseline
outputs; the stable reduction passes independent FP32 checks at all four widths.
This is a controlled stress case, not evidence of an observed model failure.

Validation:

- Extended static-split probe:76/76 cases passed comparison to original and
  independent FP32 sampled rows; negative-logit stress passes widths16/32/64/128.
- GPU adaptive probe:20/20 cases passed, using interleaved
  [blocks,2,page,kv_head,head] cache with K/V stride[8192,256,128,1]. A single
  captured graph was replayed across length thresholds with NaN-poisoned scratch,
  including empty graph padding and ragged rows.
- Production wrapper:18/18 cases passed at batches1/2/3/4/8/64, lengths100/4096/16384.
  Graph replay changes lengths and K/V contents; output checked against independent
  FP32. Sinks, quantized KV and multi-token queries fall back.
- Wrapper 16K speedups for batches1/2/3/4:2.773/2.721/1.725/1.641x; B8 and B64
  essentially unchanged. At length100 B1 is0.609x, B2 is0.802x and B3/4 about0.91x:
  the larger fixed launch/reduction capacity has overhead on short sequences.
  These are operator-only numbers, not model throughput gains.

Full acceptance started on the owned machine, serially after wrapper completion:
`/workspace/experiments/minicpm-s2/adaptive-split-20261023/full`.
Runner performs all105 Level-3 questions and requires precision>=0.95 before
stable warmup (two to four rounds, adjacent TPOT within3%) and one official scored
round per 4K/16K case. Driver and source manifest record exact revisions/import
path. No performance/20% success is claimed while this is pending.

## Complete precision gate

The full105-question run completed with103 correct:0.981 reported precision,
105 unique questions and no failed predictions. The acceptance runner has moved
to throughput warmup. This is a precision pass, not performance acceptance;
throughput and TTFT are pending. Sampling remains temperature1/top_p0.95, so this
single result does not establish a precision improvement over the parent.
