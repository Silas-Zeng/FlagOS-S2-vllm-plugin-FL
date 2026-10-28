# PR111 compute-order runner identity correction

Parent: `a6d654a7444bba72cbbd48d3d9bfd33172cfd129`. This change grants no service, Math105, or throughput qualification.

The compact+Q32 merge changed only the worker module tail: compact mode no longer rejects the reviewed Q32 combination. `ModelRunnerFL` itself (all 116 methods), every other top-level runner definition and the worker loader are byte-identical in main a455, PR109 fc25 and PR111 a6d. Reversing the one module-tail region reconstructs the entire fc25 runner byte-for-byte. The source diagnosis is reproducible from immutable Git objects.

The compact source manifest already identified a6d's actual worker as `d444d8f2b477a979a81a079087070f61d4785d646abe00d08c37d4afd93ec96a`, but compute-order still required fc25's `9ee00d1facb70e942871a5780d467df14c8ded2dbfd7efdadf0106187fd6051d`. Its error text covers both absent modules and hash mismatch; that wording alone does not establish a missing import. The guard runs before compute-order staging writes; rejection disables this optional plan on that runner and returns the original route.

Production changes are exactly two digest substitutions: the precise `EXPECTED_RUNNER` constant, and the compact manifest's checksum of that compute-order file. `_cold`, its source allowlist, all computation functions, runner bytes, mathematical kernels, libraries, official CLI, benchmark, warmup, controller and quality thresholds remain byte-identical. No hash allowlist was widened, no helper was extracted and no fallback gate was removed.

## CPU validation

`tools/check_biv150_compute_order_runner_bytes_cpu.py` executes the exact first source-verification loop AST from `_cold`, with explicit module/file metadata fixtures and an empty dependency map **only in the test namespace** to isolate its dynamic runner entry. It uses the original runtime `Path(...).read_bytes()` check; it does not claim to naturally import the full GPU runner. Current raw bytes pass; the old a6d stale pin, fc25 bytes, appended bytes, CRLF-to-LF-only change, absent module and absent pin are rejected. Every complete production function is byte-identical to a6d. The test also verifies the compact cross-pins and unchanged local pinned dependency files.

The existing `check_biv150_compact_native_n48_service_cpu.py` subsequently passed serially, including real component qualification, both actual SO hashes, CPU namespace/schema routing and the Q32 combination. These checks do not execute a GPU. A fresh real service must still verify natural imports, remaining cold scope and live closures, then rerun the fixed warm/score protocol. No old result becomes accepted after this source correction.

The older `check_biv150_compute_order_identity_cpu.py` normalizes runner line endings via `read_text()` before hashing and carries older exact-inverse assumptions. It is preserved unmodified and was not used to claim this runtime-byte admission.

```sh
python3 -B docs/evidence/native111-compute-order-pin-fix-20261029/review_source_diagnosis.py --repo "$PWD" --out /private/tmp/new-source-diagnosis.json
python3 -B tools/check_biv150_compute_order_runner_bytes_cpu.py --out /private/tmp/new-runner-byte-regression.json
python3 -B tools/check_biv150_compact_native_n48_service_cpu.py --out /private/tmp/new-dual-library-q32-cpu.json
```

The original a6d service failure and all of its raw results are retained separately; this commit contains only the source diagnosis, correction and CPU checks, with no pending runtime result.
