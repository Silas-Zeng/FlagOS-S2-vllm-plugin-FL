# Recovered serving results: accuracy/4K pass, 16K interrupted

Recovered from the personal BI-V150 on 2026-10-24 after reservation recovery.
Actual source e444c9c, clean worktree, exact serving command and import hashes are
recorded in the saved manifest. This branch remains unmerged and unaccepted.

- Full Level-3 accuracy: 102/105 (reported 0.9714), 105 unique questions, zero
  failed predictions.
- 4K: 256/256 requests, zero failures, 5017.400661 total tokens/s,
  TTFT 4756.173660 ms. Approximately +4.33% against the original 4809.36
  reference, below the requested +20%. Warmup stabilized before this single run.
- 16K warmup stabilized, but the scored run ended with 114/128 completed and
  14 failures. Input/output counts are incomplete; its throughput and TTFT are
  not valid acceptance results. The top-level scores file consequently only
  contains 4K. No 16K score is reconstructed from incomplete data.
- Actual serving logs contain BIV150_MIXED_ATTENTION_SELECTED across many
  geometries, proving workload integration rather than only standalone speedup.

The server logged SIGTERM at 2026-10-23 22:43:32, followed by EngineDeadError in
the API process. No preceding kernel exception stack was found in the retained
server log. The signal sender is unknown; this is not proof of a kernel crash
or of an external cause. The recovered queue exit status was 1. The server and
queue are now absent and GPU was independently confirmed idle before new work.

Evidence: [manifest and partial scores](evidence/biv150-mixed-attention-20261024-recovered/),
including raw scored result JSONs, commands, accuracy verdict, warmups and gzip
of the complete server log with its uncompressed integrity hash. Full EvalScope
reports remain in the original remote result directory and local recovered
archive; the compact evidence folder does not duplicate evaluation predictions.

Source result directory:
/workspace/experiments/minicpm-s2/mixed-attention-20261023/full.
Local recovered archive: /private/tmp/flagos-recovered-20261024.tgz.

Required before merge: determine interruption cause where possible, run a fresh
complete 16K with stable warmup and all requests/tokens, and recheck both
throughput and TTFT together with the candidate's current accuracy evidence.
