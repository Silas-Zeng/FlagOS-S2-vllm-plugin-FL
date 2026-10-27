# DAC1 throughput warm-state review (CPU only)

The DAC1 run is a valid **first-full-geometry throughput diagnostic**. The saved
data do not demonstrate a source regression relative to the later PR66 score.
This report changes no production source, controller, kernel, or numerical gate.
No SSH, JIT, GPU execution, or queue operation was performed for this review.

## Originals and replay

Current originals: `/private/tmp/cjj-konly-main-dac-throughput-records-raw-v1-20261027`.
Owner `643740/start345932200` is recorded with the same start and terminal `Z`.
All 34 original files / 354614 bytes match the download manifest. Its SHA256 is
`912f8a26640d77adb4f9539461d0e043edd40dc13b09c69d35ce4eee1170eaf3`.

PR66 originals: `/private/tmp/cjj-multi2-final-raw-snapshot-v1-20261027/pair`.
`baseline` is `241220723a09061c182764f8f738e5ec01190367`; `candidate` is
`e08c79456f9b943dbc4d9d02d0a4e030b4eceac1`. All files consumed from both arms
have their exact absolute path, byte count, and SHA256 in `report.json`.
The originals remain unchanged in those locations; this directory stores the
CPU report rather than another copy of those runs.

From this worktree, the following command reads only the local originals and
reproduces `report.json` byte for byte. It imports neither Torch nor Triton.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/review_biv150_main_dac_warm_state_cpu.py
```

## Exact comparison

All entries below use 16K input, 1024 output, 128 measured requests, concurrency
64, seed0, and the same benchmark CLI after removing only output path and label.

| Source | First complete workload (tok/s) | Subsequent scored workload (tok/s) |
| --- | ---: | ---: |
| PR66 baseline `2412207` | 3972.738444587674 | 4479.567553447986 |
| PR66 candidate `e08c794` | 3958.939027045799 | 4517.191826643136 |
| Main DAC1 `dac1b8f` | 3972.417815604062 (scored) | Not run |

The new result is **−0.0081%** versus the old baseline's first full workload,
and **+0.3405%** versus the old candidate's first full workload. Comparing it
with 4517.19 mixes different warm states. The old candidate's first 16K full
workload took 562.833624 seconds; its subsequent score took 493.276373 seconds.
The new first full workload took 560.923876 seconds.

The old driver explicitly adds a full workload after three short warmups:
`tools/run_biv150_small_out_acceptance.py:87–113`, especially lines96–100.
Its comment identifies late mixed/tail shapes not visited by short warmups.
The DAC1 controller instead used the shipping adaptive short-warm function:
three output128 warmups happened to meet its TPOT condition in each length,
then it scored the first output1024 workload. Old runs also performed mixed
diagnostics and Math105 before throughput; the current run did not.

The 4K new score is 5909.673194325466 tok/s, TTFT3262.458100918593 ms;
the 16K new score is 3972.417815604062 tok/s, TTFT120624.73473017963 ms.
Both have complete counts, zero failures, exact recorded input/output totals,
finite positive metrics, and token/duration arithmetic agreement. Per-request
input lengths are absent because this CLI was not given `--save-detailed`.

## Validity checks and remaining attribution limits

- The new log has zero `COMPUTE_ORDER_SCOPE_REJECTED`,
  `PREFILL_MULTI2_FALLBACK`, or encoded-selection markers. All nine parent
  installers report installed; both API/worker compile fingerprints equal the
  expected DAC1 fingerprint. The corrected symmetric evidence guard also
  accepts the unchanged original log.
- GQA width64, Q32 and M2 actual-selection markers are present. DAC1 additionally
  reports `HEAD_TEMPERATURE_READY`. The old and new GQA, mixed-decode, Q32, M2,
  and normalization source leaves are unchanged; compiled selection hashes need
  not match after source/debug/fingerprint changes. PR84 adds head-temperature
  code and runner hooks; this report does not attribute a performance cost to it.
- Both old arms and the new run allocate **548416 KV tokens / 34276 page16
  blocks**. Service CLI, native library SHA
  `5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe`, and the
  two initial v6 database hashes match. No preemption warning was observed.
- Equal initial databases do not imply equal live tuned configurations. Old
  16K scored cache inventory has no added kernels and no selected-row changes;
  the new run saved no equivalent per-phase inventory. Kernel/config observations
  cannot by themselves bound every compilation/tuning cost.
- Old full/score 16K log windows show prefix-cache hit rates at most0.1%; the new
  score window shows0%. These logs do not support assigning the difference to
  prefix reuse. The warm-state comparison also does not strictly identify JIT,
  tuning, scheduler, or hardware contributions.
- `max_concurrent_requests=68/65` in the new JSON does not invalidate B64:
  vLLM0.24 `serve.py:688–701` counts overlapping **integer-second buckets**, so
  requests finishing and starting within one second may both enter one bucket.

## Review of the opt-in full-workload mode

Root's separate controller change uses `--full-workload-warmup
--throughput-case 16384`; this worktree does not implement or launch it.
The reviewed plan is sufficient to test whether the first/second full-workload
difference recurs on unchanged DAC1, without needing Math105 again:

1. Keep one owned service, source/native/initial snapshot, and service CLI.
2. Run the old fixed three short warmups: length16384, output128, count64,
   seeds920000/920001/920002.
3. Call the pinned, AST-exact shipping `benchmark` with
   `full-warm-16384 / 16384 / 1024 / 128 / seed0`.
4. Validate and preserve that warm result, then call the same function once with
   `score-16384` and identical numerical parameters; validate it independently.
5. Preserve raw JSON/commands, source/controller identity, cleanup evidence, and
   read-only selected-row/Triton metadata snapshots around the score.

The new adapter must explicitly accept `full-warm-16384` with the same fixed
1024/128/seed0 arguments as the score. The default branch remains the original
shipping `throughput` function. The opt-in branch calls the original `benchmark`
but uses its own disclosed control flow, so it must not claim the original
`throughput` body ran. This fixed-three-warm mode does not apply the default
adaptive3% TPOT test.

The reviewed `private_cache_state` uses read-only SQLite connections and exactly
the old `tuning_config_fingerprint` algorithm: ignore benchmark/sqlite tables,
sort serialized rows, serialize table mapping, SHA256. Its before/after samples
are **after** the full warm and therefore describe score-phase changes only.
They do not alone attribute the full-warm latency. The new launcher names its
controller by SHA rather than overwriting the previous shared fixed path.

This is still a single within-process warm-state diagnostic, not a stable
performance estimate, a full formal wrapper run, service correctness acceptance,
or evidence of an encoded-cache speedup. Default short-warm evidence remains
valid under its own recorded protocol.
