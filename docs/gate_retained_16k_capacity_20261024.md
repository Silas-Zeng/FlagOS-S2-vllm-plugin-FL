# Mid-run16K KV-capacity observation (not final acceptance)

Runtime source33d3d85, personal instance only. Existing HTTP metrics were sampled
on CPU every15s, bound to engine41838 and its process start ticks, finite2400s
maximum. No inference requests, GPU profiling or service changes were introduced.
The collector source and early raw JSONL snapshot are archived with this report.
Sampling itself is diagnostic observation; no zero-observer-overhead claim.

At16:18 a one-off metrics read observed total preemptions0 during warmup. The
collector then identified its benchmark phase from the existing driver's child
command `--label score-16384`. Within that phase it observed:

| CST | Running | Waiting | KV fraction | Cumulative preemptions |
|---|---:|---:|---:|---:|
|16:20:54|30|34|0.991815|0|
|16:21:09|29|35|0.975221|1|
|16:21:24|29|35|0.991335|1|
|16:21:39|27|36|0.933431|2|

These are exporter observations, not per-kernel row counts or exact attribution
of the requests whose work was recomputed. They establish preemption and capacity
pressure in this new run; they do not explain the whole older compressed
candidate's16K regression. Its metrics were not sampled under the same protocol.

One independent full-length request needs ceil((16384+1024)/16)*16=17408 KV-token
slots (no shared-prefix assumption). The corresponding rough capacity budgets
are28 completed-length contexts at500432 tokens (retained candidate),31 at547536
(old compressed combo),30 at522960 (earlier BF16 baseline). Admission counts can
exceed these as requests grow, and the approximation is not a scheduler model.
The64-request4K workload only needs64*(4096+1024)=327680 slots, below all three
capacities. This provides a concrete reason why extra gate BF16 storage can
preserve4K throughput yet pressure16K; a controlled follow-up remains necessary.

The latest-team gate candidate e3a0497 keeps compressed weights and small-M
residual correction, preserving PR22 compute-order and PR23 Q-fragment paths. It
is already queued for independent full validation. The current retained candidate
continues unchanged to its full16K result; no throughput/TTFT success is inferred
from these mid-run measurements.
