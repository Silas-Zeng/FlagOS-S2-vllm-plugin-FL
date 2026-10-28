# PR111 a6d failed service: preserve the rejection and complete raw evidence

This original service **failed acceptance**. It finished four benchmark rounds and cleaned up, but the unchanged production evidence gate rejected `BIV150_COMPUTE_ORDER_SCOPE_REJECTED`. The original `completed=false`, `phase=failed`, single error, missing final statistics and absence of Math105 remain unchanged. These results cannot qualify this candidate or the later pin fix.

Source `a6d654a7444bba72cbbd48d3d9bfd33172cfd129`; controller `ae99bfb55c69dc1bf9464ffe0a1b7e627882bd4bc9d185d4177f9a902c29ea56`; owner PID62772/start356555561 ended with the same start identity in Z state. The variant uses compact12544 Native7892 for N32/S4 and N64/S2, Native2ad for N48/S4, and Q32 hoist ON. Helper94db, e236 benchmark, official service CLI, concurrency64, output1024 and the fixed one-full-warm/one-score protocol are preserved.

| Geometry | Diagnostic total tok/s | Mean TTFT ms | Successful score requests |
| --- | ---: | ---: | ---: |
| 4K | 6253.253144 | 3980.753697 | 256/256 |
| 16K | 4379.683211 | 89332.315798 | 128/128 |

Both warm rounds also completed all requests. The independent review recomputes all token/duration arithmetic and checks all four exact commands. The archived official performance bounds are separately reported as diagnostic comparisons; they do not override the failed production gate or missing precision evaluation. No latest-main improvement, significance or single-change causal attribution is claimed. 4K score added three index metadata entries and changed selected tuning rows; 16K added/changed none. That observation includes the client's preflight, not only timed score execution.

## Failure and evidence boundaries

The whole original controller `evidence()` rejects again when replayed against the complete untouched log. Its real earlier check, after all four diagnostic bursts, reproduces `manifest.actual_runtime` from the exact first 954760 bytes. No rejection line is filtered to obtain a passing full-run result.

The log at line1030 reports a runner source rejection. The actual a6d worker hash is `d444d8f2...`, while the compute-order source still requires PR109's `9ee00d1f...`. The generic message also covers missing modules, so it does not establish a missing import. The immutable source diagnosis proves that the full `ModelRunnerFL` class and all116 methods are byte-identical in main a455, PR109 fc25 and PR111 a6d. Reversing the module-tail Q32 admission change reconstructs the entire109 runner. The raw-byte identity gate remains a real failed contract even though those method bodies did not change.

The rejection occurs before compute-order staging writes; its existing catch returns the original route and disables that optional plan for the runner. No compute-order FIRST_SELECTED event is present. This does not establish how much throughput was lost. The controller's later final source/library/tuning-snapshot checks were skipped after the error; their execution is not claimed. Early source manifests, immutable Git bytes, actual runtime receipts and complete saved artifacts are checked independently.

Native wiring has42 selected and42 captured records for each of N32/S4, N48/S4 and N64/S2, all query pitch2048, on44769 physical pages. These are host/capture observations, **not actual request or Graph replay frequencies**. The allocation receipt records42 layers and23,586,458,112 bytes. There is no final compact statistics file. The three smoke requests and184 burst requests are separately checked and never counted as benchmark or Math105 scores.

One actual fresh Q32 pair is preserved. Existing proposal/native-IR checks were replayed, with all artifacts/ABI/source hashes and identical TTIR checked. Runtime registers/shared change135/57344→122/33792 and both runtime `n_spills` are0. Actual cubin metadata, independently parsed with the existing ELF/msgpack helpers and matched to saved ASM metadata, reports SGPR spill99→96, VGPR spill0 and private bytes0. Static shared is0; dynamic shared is reported separately. No independent cubin disassembly or new numerical/model qualification is claimed.

The complete3516 raw files (90,754,674 bytes) are preserved in the exact original15,932,799-byte gzip, split into eight nonoverlapping byte parts, each≤2MiB. `archive-parts.json` binds every part, the full original gzip SHA `f11bd3480c76f7ca15d4909ddbc843eed9b934f738eb7098644db5d8ff3c754a`, and the complete download manifest. The identical unsplit archive is also retained under the workspace's `repos/experiment-raw-archives/2026-10-29/`. All raw files were rehashed during the independent review.

## Replay

Run the review against a separate frozen a6d checkout. A future fixed checkout intentionally has different production hashes and must not be substituted. The two actual SOs come from the already shared110/112 compile archives; pass their extracted paths if the historical local defaults are unavailable. The helper and ELF parser are already in a6d.

```sh
python3 -B docs/evidence/native111-a6d-failed-service-20261029/restore_raw.py
python3 -B docs/evidence/native111-a6d-failed-service-20261029/restore_raw.py --extract-to /private/tmp/new-a6d-failed-raw
python3 -B docs/evidence/native111-a6d-failed-service-20261029/review_failed_service.py --raw /private/tmp/new-a6d-failed-raw --repo /absolute/frozen-a6d-checkout --launch docs/evidence/native111-a6d-failed-service-20261029/original-launch.json --base-so /absolute/actual7892.so --n48-so /absolute/actual2ad.so --out /private/tmp/new-a6d-failed-review.json
```

`original-prepared-success-review.py` is preserved unmodified. The first failure-review attempt incorrectly chose the first burst's start instead of the last burst's end and failed its early-prefix comparison; it produced no passing report. `audit-attempts.json` retains that audit mistake and the later limited corrections. The final independent report explicitly sets `service_acceptance_passed=false`, `original_state_completed=false` and `precision_gate=false`.

## Later fix CPU evidence, separate from this failed run

Commit `069838d6fbb7591de4384aac8caef49324c43aef` changes only the exact runner pin and dependent manifest digest. Its author and Root CPU reports are retained here as later evidence, not as changes to a6d or a replacement for a new real service run. Root's byte-loop and combination reports pass. Relative to the author reports, Python3.12/3.14 only differ in the AST dump serialization digest and five Python installation paths; `later-fix-Root-replay-diff.json` lists the exact differences. No later service result is included in this evidence commit.
