# Latest-team residual gate: full accuracy passed, throughput pending

Actual frozen source e3a0497adbd6570dd522877131bc47058350da83 on the personal
BI-V150 instance completed105 unique Math500 Level3 questions with102 correct,
reported0.9714, no failed predictions. This is one stochastic full evaluation.

The code includes team main1ea082f (PR22 compute-order/H2D and PR23 prefill
Q-fragment) plus the compressed middle34-layer residual gate. Actual installation
metadata confirms gate-only, compressed storage, small-M residual1..16. Service
KV capacity547536, and actual Q-fragment selection confirms the expected
8ae22585 override and65536→32768 shared memory with no observed fallback.

At17:02 the owned acceptance driver58886 and engine59032 remained live and
started the throughput stage; no complete4K/16K score exists yet. Do not combine
older gate percentages or teammate throughput with this precision result.

Finite existing-metrics collection65820 samples CPU exporter values every15s,
maximum2400s, bound to this engine identity and existing benchmark phase. It
starts before throughput and changes no inference request, GPU work, scheduling
or serving parameters. The earlier retained-BF16 assertion and its measurements
belong to a separate source and output directory.

The running source remains e3a0497. This subsequent report/evidence commit is
not deployed into the active checkout and credits no unmeasured service gain.
