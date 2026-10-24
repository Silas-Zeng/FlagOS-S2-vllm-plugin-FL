# Residual gate plus remaining mixed attention — 16K rejected

Base: f1cf5d7, framework ancestor team main 3283672 (reverified with GitHub API
after normal fetch timed out). No new upstream framework code is merged.

The exact standalone gate source 20c991a passed one full Level-3 run at 102/105 and
a warmed single 4K run at 5258.354991 total tokens/s, TTFT 4244.359266 ms, 256/256.
Its completed 16K is 3113.626945 total tokens/s (-7.44%), with TTFT
161138.989054 ms exceeding the working ceiling. Standalone gate is rejected. The separate mixed operator source e444c9c passed one
Level-3 run at 102/105 and 4K at 5017.400661, but its 16K was interrupted by SIGTERM.
These individual results do not validate this combination or predict additive
speedup. The mixed operator code is copied byte-for-byte from the tested source.

The combination installs residual gate fusion first and mixed attention second,
after the existing normalization/continuous-prefix decode paths. All existing
fallback and selection logic remains. There are no scheduler, serve parameter,
max_num_batched_tokens, benchmark, weight-scope or KV-dtype changes.

The serial helper builds a private SO, then requires all 9 residual dispatch
checks and all 10 mixed wrapper checks, before a new full 105-question accuracy
run and stable warmup plus one scored round per official workload. It verifies
all three install markers and records exact source/library hashes. The complete
mode additionally requires post-ready mixed serving selection evidence; this
covers real evaluation/warmup/scored serving, not isolated probe execution.

This candidate has passed local Python syntax/lint checks only. It is not
installed, queued or GPU-tested yet. Wait for the current acceptance and both
queued probes, then choose the next model candidate from their actual results.
Never copy separate candidates' accuracy, throughput or TTFT to this version.

The complete mode now saves explicit working performance gates and exits with
failure for a bad throughput/TTFT/request result. These conservative research
TTFT limits are 4770.3209 / 156938.85 ms; original throughput references are
4809.36 / 3364.03, with a 99% floor and a separately reported 120% goal.
Raw measurement completion is distinct from acceptance. The gate function
rejects the actual completed residual run, accepts a passing fixture, and
rejects NaN/Inf, missing requests, failed requests, missing output and changed
concurrency fixtures. This is runner validation, not GPU candidate validation.


## Complete model result, source 8d9f56d

Private build, all 9 residual dispatch and all 10 mixed checks passed.
Full Level-3 passed 101/105 (0.9619), 105 unique questions, zero failures.
The warmed single-round results are:

| Workload | Total tokens/s | Gain vs original | Mean TTFT ms | Requests |
|---|---:|---:|---:|---|
| 4K | 5404.542700 | +12.3755% | 4096.593504 | 256/256, zero failed |
| 16K | 3145.573683 | -6.4939% | 159183.320265 | 128/128, zero failed |

Both token totals match the official workloads. Warmups stabilized. There are
384 post-ready mixed-selection log records, proving actual service integration.
Working 4K gates pass, but 16K throughput and TTFT both fail. The owned runner
exits 1 after saving evidence and cleans its own server. Do not merge. This
combination improves 4K beyond the historical 10.6%, but does not preserve 16K.
The 20% objective remains unmet. Earlier paragraphs describe pre-test state.
