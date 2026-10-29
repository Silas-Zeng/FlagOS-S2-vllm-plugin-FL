# PR118 plain-service smoke: selected observations

The frozen `69827253756a0ef9e618ae0322e9ff361e93504f` smoke completed all
**163 requests**: the original 3 sequential + 64 burst requests, and 96 added
overlap requests. New request groups were 61 keepers + 3 arrivals and 30 keepers
+ 2 arrivals; each keeper completed 512 tokens and each arrival 16. Response IDs,
usage arithmetic and all recorded request outcomes were independently checked;
both added groups have empty coverage gaps. Owned cleanup passed.

In the server-log window beginning after readiness and before the first request,
actual mixed post-op markers record **61/S2, 30/S4 and 31/S4**. Their worker PID
132998 matches the archived owned epoch `start_ticks=361618503`, session 132970.
The same PID has the accepted 5898 installer, the expected encoded implementation
digest, and 32/S4 plus 64/S2 native records: 42 selected and 42 captured records
per bucket. This independently closes same-worker two-library dispatch identity.
These host records do not measure score frequency or Graph replay count, and do
not attribute each marker to one particular added-request group.

**The original final-stats result remains false.** The controller recorded
`mixed_final_evidence.workers=[]` and `mixed_same_worker_smoke_verified=false`.
The independent audit's original strict summary also remains false. Its separate
`request_window_operational_checks_passed=true` uses complete request results and
the existing same-worker log evidence; it does not invent a lifetime counter.

Shutdown logs show API/EngineCore abort mode with zero timeout and a process
manager report of one forced termination. No final-stats marker or file was
collected. The resource tracker also warned about 69 semaphore objects. This
supports a plausible incomplete-finalizer explanation; the exact interrupted or
uncalled statement is not established. The successful requests were not rerun
for this missing summary, and the original results were not edited.

The mixed path bypasses `Runtime.note`; its marker does not export full Q/K/V
allocation descriptors. The saved pure-decode query descriptors have pitch 2048
and offset 0 for both 32/64 buckets, including capture and eager records. Those
are adjacent evidence and are not relabeled as mixed-specific shared-QKV proof.
The selected source retains its bounds, alias and stride checks. Smoke response
completion is not an independent output oracle.

Root read and exactly replayed the independent audit, SHA
`b3103957e8f34ba29439ef20b0cd813f8380eada54c22856758bf48466c96a06`.
[results.json](results.json) contains selected scalars and identities only.
This record grants no Math105 precision, model-throughput or full service
acceptance. It changes no production code or running source. The full private
2,374-file archive, 168 original native events, response bodies and complete audit
remain local and are not included in this documentation commit.
