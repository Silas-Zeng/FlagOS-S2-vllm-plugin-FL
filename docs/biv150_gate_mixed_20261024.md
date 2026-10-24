# Residual gate plus remaining mixed attention — unvalidated combination

Base: f1cf5d7, framework ancestor team main 3283672 (reverified with GitHub API
after normal fetch timed out). No new upstream framework code is merged.

The exact standalone gate source 20c991a passed one full Level-3 run at 102/105 and
a warmed single 4K run at 5258.354991 total tokens/s, TTFT 4244.359266 ms, 256/256.
Its 16K is still running. The separate mixed operator source e444c9c passed one
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
