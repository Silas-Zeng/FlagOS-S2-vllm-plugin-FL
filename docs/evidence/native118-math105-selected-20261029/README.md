# PR118 Math105: selected official and supplemental results

Frozen source `69827253756a0ef9e618ae0322e9ff361e93504f` completed the official
Math105 test and a separate 64-concurrent diagnostic. Independent recounting of
all original per-question evaluator labels gives:

| Run | Questions | Correct | Exact accuracy | Reported | Errors / max-token truncations / empty raw responses |
| --- | ---: | ---: | ---: | ---: | --- |
| Official, batch 8 | 105 | 101 | 101/105 | 0.9619 | 0 / 0 / 0 |
| Supplemental, batch 64 | 105 | 102 | 102/105 | 0.9714 | 0 / 0 / 0 |

Both metrics exceed 0.95 (at least 100/105); the supplemental run does not replace
or augment the official score. The same 105 question IDs, prompts and reference
metadata were checked. Original service CLI, generation settings, grader and
threshold were preserved. Response/review pairing, token arithmetic, original
67 smoke requests, all 277 unique response IDs and owned cleanup passed. The
extra 96 requests used by the prior plain-smoke run were not added to this run.

The first independent audit stopped at an extra assumption inherited from the
PR117 audit: every extracted answer must be nonempty. In the official run,
`test/prealgebra/1761.json` (index 16) has a nonempty 42-character raw response,
12 output tokens, `stop_reason=stop`, no request error, and an empty extracted
answer. The original evaluator marked this response incorrect (`accuracy=0`).
It is one of the four official wrong answers, not a maximum-token truncation.

The original failed audit is retained. The narrow v2 change counts empty
extractions and requires any such answer to retain its original zero grade;
nonempty raw responses and no request errors remain hard checks. This one answer
still contributes to the denominator of 105. No question was dropped, rescored or
regenerated, and no grader or 0.95 threshold was changed. Reversing the two explicit
result-block substitutions reproduces the original block byte for byte. The
v1 extra nonempty-extraction gate remains false. The supplemental run has no
empty extraction.

The supplemental log window contains mixed post-op host markers for **60/S2,
61/S2 and 63/S2**, from worker 139146. Its owned epoch is start_ticks 361755827,
session 139118. The same PID has the accepted 5898 installer and encoded
implementation digest, plus 32/64 native selected/captured records from the run.
Those records include setup and capture; they are not request counts or Graph
replay counts. The markers carry no response IDs, so they do not prove which
math questions used Native. The official interval has no new mixed marker;
absence of a first-selected marker is not proof of no execution. S4 request
coverage is not established by this Math105 record.

**The original coverage flags remain false:** `native_quality_covered`,
`native_model_precision_verified` and `mixed_same_worker_smoke_verified`.
No FINAL_STATS was saved; `mixed_final_evidence.workers=[]` and the missing
lifetime counter are preserved. This observation gap does not alter the official
Math105 metric. It also does not grant per-question Native precision, model
throughput or full service acceptance.

Root read the v2 diff and exactly replayed the full independent audit, SHA
`15e48733f2f94120d65f5bbb3bfb828480f9419c2dc022e0256b85c1ed7f7e7c`.
All 4,091 original files (102,130,418 bytes) passed SHA/size checks. The selected
[results.json](results.json) records scalars, identities, the narrow audit
correction and limitations. Full 210-question text/responses, raw artifacts,
event lists and the complete audit remain private local evidence. This commit
adds documentation only and does not change the running source.
