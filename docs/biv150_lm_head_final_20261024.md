# Pure LM head full acceptance: rejected

Actual source319dc3024c8fca104c77df1b4bb8dab5b7a644c9; output on personal instance:
`/workspace/experiments/minicpm-s2/lm-head-prefixfix-20261024/full`.
Full evidence JSON is archived alongside this report.

| Metric | 4K | 16K |
|---|---:|---:|
| Total tokens/s | 4913.056040 | 2825.245987 |
| vs original4809.36/3364.03 | +2.156% | −16.016% |
| Mean TTFT ms | 4779.329579 | 186667.375028 |
| Complete, failed | 256,0 | 128,0 |
| Input/output tokens | 1048576/262144 | 2097152/131072 |

Full Math500 Level3:101/105 (0.9619),105 unique, no prediction errors.
The fixed official serving/benchmark parameters were retained. Each workload had
stable adjacent warmup TPOT before its single formal run. Both TTFT gates failed,
and16K throughput also failed. The queue completed with exit1 and cleaned only its
own service. This candidate must not merge into main on these results.

Actual service capacity504784 KV tokens, compared with the earlier BF16 baseline
522960. The startup head log reports M256, not2048; its INT32 tile is127.5MiB.
M2048 would require roughly1GiB, but that is a hypothetical operator case and
cannot explain current service memory consumption directly. Loader temporary
allocations are also cleared/reset before profiling. Vendor non-Torch workspace
or other profile allocations remain hypotheses.

A separately shared bounded-head candidate d5f5f581 limits the INT32 tile to64MiB
and records peak allocator usage, including M256. It requires separate primitive,
Graph, hook, full precision and service acceptance; this report credits it no gain.

Team main advanced independently to cae4e34b (PR22) after this frozen experiment
started. These measurements describe319dc30 on the earlier3283672 integration;
they do not measure the new team baseline or the gate/mixed-attention candidates.
