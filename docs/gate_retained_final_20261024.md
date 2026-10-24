# Retained BF16 gate:4K preserved,16K incomplete

Frozen source33d3d85b7458e2d14bb783adbeb792b404b6cce7. Full accuracy103/105,
105 unique,0 prediction errors.4K5401.515649 total tok/s (+12.31257%), mean
TTFT4054.801963ms,256/256 success, correct input/output totals. Both workloads
had stable adjacent warmup TPOT; these are single-round research measurements.

At16:29:50,16K failed inside the SDK async scheduler:
`async_scheduler.py:68 _update_request_with_output`,
`assert request.num_output_placeholders >= 0`.
The engine died and benchmark completed117/128 with11 failures. Its saved partial
JSON has only1916899 input and113107 output tokens. Any throughput or percentage
from it is **invalid** as a full16K score; no16K speedup/regression is reported.
This was an observed engine assertion, not an agent-requested service stop or a
benchmark timeout. The original driver detected incomplete requests, exited1 and
cleaned its own service. No complete acceptance.json exists.

Final sampled metrics before the assertion observed7 cumulative preemptions,
29 running/11 waiting and KV fraction0.988905. These are sampled exporter values;
the actual count at the exact failure may differ. Source-level cause of the
placeholder error is unresolved; timing and memory pressure do not prove causal
attribution. No scheduler code or parameters were changed in response.

The complete CPU metrics and failure-tail evidence are archived in
`docs/evidence/gate-retained-bf16-20261024/final/`; full1.28MB server log remains
in the local/remote experiment archives with its SHA256 recorded. Original
accuracy and4K evidence remain valid within their own scope. Keep this branch
for follow-up, but do not merge it or credit a two-workload acceptance.

The bounded-head queue automatically started after this queue ended and the GPU
was free. Its separate later early-stop decision is unrelated to this failure.
The latest-team compressed-residual candidate remains the next full-service
route; it requires its own105-question and complete4K/16K validation.
