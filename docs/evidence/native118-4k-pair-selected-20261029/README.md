# PR118: selected matched 4K throughput pair

The two completed runs used the same 67 startup requests, three short warmups,
one complete warmup and one scored benchmark. Both scored runs completed all
256 requests without failure. A is main `56f8873` with native 5898; B is frozen
`6982725` with native 5898 plus the PR118 mixed reader. Q32 hoist remains on;
stats RPC and Graph observer remain off. Official service parameters and the
shipping benchmark function are unchanged.

| Metric | A main | B candidate | B relative to A |
| --- | ---: | ---: | ---: |
| **Scored total tokens/s** | **6509.843429** | **6961.164300** | **+6.932899%** |
| Scored TTFT, ms | 3632.204514 | 3523.234031 | -3.000120% |
| Scored TPOT, ms | 45.325089 | 42.240233 | -6.806068% |
| Scored duration, s | 201.344320 | 188.290341 | -6.483411% |
| Unscored complete-warmup tokens/s | 6950.703766 | 6917.674984 | Not used as a score |

This is one matched scored pair. The warmup observations are retained and are
close to each other; they are not substituted for the scored A result or pooled
into an average. The positive scored difference does not yet establish a stable
or entirely causal operator improvement. Both source/library identities, exact
benchmark commands, token arithmetic, original 67 responses, private-cache
snapshots and owned cleanup passed independent review and Root's exact replay.

Cache observations still differ. A adds six `_index_linearized_jit_function`
metadata records: BF16 input rows 1695, 1782, 1860, each with 8x128 and 16x128 compiled
tiles. B adds three: rows 469, 1420, 1774, each with 8x128 tiles. Input width is 2048.
There are no shared newly added metadata keys or changed existing metadata, and
both selected-tuning digests change. These observations cover the whole benchmark
invocation, including client preflight. Metadata counts are not compile durations
or call counts and cannot explain the score difference by themselves.

The final linear config tables contain 211/215 rows. Their 207 shared keys have
identical final configurations; A has four additional M=1696 keys, while B has
eight additional M=480/M=1440 keys. Final linear benchmark tables have 2911/2975
rows with 2499 identical rows. This is final history, not evidence assigning work
to warmup or score; benchmark-table history was outside the existing before/after
observer.

B's post-ready logs record 29/30/31 with S4 and 60/61/63 with S2 from worker 156253.
They prove bounded host selection during the run, not exact scoring-stage calls,
per-request paths or Graph replay frequency. No FINAL_STATS was saved. The
original `mixed_same_worker_smoke_verified=false` and empty final worker list are
preserved.

**16K remains pending; full service acceptance is not granted by this record.**
The earlier [Math105 result](../native118-math105-selected-20261029/README.md)
is separate evidence for the same B source, not a precision rerun in these jobs.
[results.json](results.json) contains selected metrics, cache-shape summaries and
identities. Full raw archives, request bodies, event lists, generated code and
complete audit files remain local. This commit changes documentation only.
