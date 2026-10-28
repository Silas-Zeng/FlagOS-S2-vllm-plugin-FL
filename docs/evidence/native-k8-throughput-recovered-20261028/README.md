# Recovered original NativeK8 throughput — a975 only

The original personal-instance job completed at 2026-10-27 22:00:21 UTC, just after the SSH reservation cutoff. Renewal on October 28 allowed collection of the closed original records. This is not a rerun. Its source is `a975e8715beb752d6a76b456f417ba2011ea917b`, based on team main4229, with W8A8 5c, stock GQA aace, private K8 22e, and initial tuning snapshot v6.

| Original scored workload | Total tok/s | Mean TTFT ms | Completed/failed |
| --- | ---: | ---: | ---: |
| 4K | 6512.178795942298 | 3519.073296462011 | 256/0 |
| 16K | 5012.049282572983 | 100996.92515456627 | 128/0 |

Each configuration used three short warmups, one complete workload warmup, then one scored workload with the unchanged e236 benchmark. The 16K full-warm value4436.339088084973 is not its score. Original owner712697/start348367040 is gone; the original state and cleanup both closed successfully.

`original-records.tar.gz` contains all66 original diagnostic files plus the collection manifest; it excludes the generated private caches. Raw files total4039331B. Archive SHA256 `6748db80cf374a66809338fbdd32eaa1c825e6f30c0c39fb9dda6430dd379bc7`. The independent CPU replay verifies every raw SHA,252 production-source hashes, helper/controller/protocol identities, all10 warm/score commands, token/duration arithmetic, actual import and compile semantics, parent installations,168 NativeK8 host/capture records, and cleanup. Root replay SHA256 `c42ff79bcbc70aca60673f6041eb94702a484e4f5f52bed958f647828319c481`; `root-review.py` retains the exact local replay paths to the extracted archive, immutable a975 source and existing helper.

This is a complete one-round throughput result, not full acceptance. Model precision was not run in that job. Host/capture records do not prove request-frequency or Graph-replay quality coverage. All observed native events used query pitch2048; component coverage for other strides remains separate. The new original-source105 evaluation is running separately.

Team main has since advanced to PR105/1cbac531. Its published four-round-discard-first averages are6627.816667/4832.66 with different W8A8/GQA libraries and slightly different capacity. The descriptive deltas−1.7447%/+3.7120% are not a controlled gain. PR105's summary omits TTFT, so this record does not qualify latest-main TTFT. Against the older local b3 control point, 4K TTFT is2.6963% higher; that relative+1% gate would fail, although the established original competition working ceiling passes. Both scopes are retained in the report.

The branch now contains a normal merge of main105 and a strict installer-pin update. These historical a975 scores do not qualify that new source. Rebuild the overlap-enabled stock library and perform same-environment quality/throughput/TTFT acceptance before any merge.
