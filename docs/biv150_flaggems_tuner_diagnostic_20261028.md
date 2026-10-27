# Temporary source-pinned FlagGems cache diagnosis

CPU-only preparation based on actual target8-source snapshot. libentry SHA414e8a3f… and SQL6dc19cb5… exactly match the historical host-gap source. This branch starts at actual main4229 and changes no production file, framework source, CLI parameter, cache or kernel. No PR/service/device run is authorized by this preparation. Every receipt says `formal_score_eligible=false`; instrumented scores cannot serve as performance acceptance.

The old warm-to-score gap is not explained yet. Warnings and new generated index Python prove module/constructor activity, not benchmarking. TRITON_PRINT_AUTOTUNING covers LibTuner.run1054, but ConfigCache misses can print and persist a best-config row while every BenchmarkCache result hits. `bench_time` excludes ConfigCache reads/prune, final config write/re-read and final launch/JIT. Actual Autotuner._bench154 additionally deduplicates seen_tuned_metas183; even its entry count is not the benchmark-callback count. Its do_bench207 invokes the resolved helper only after that test. Modern protocol treats measurement_ms as total budget, divides by10 retries before do_bench_cudagraph. The actual triton.testing helper internals have not been read; no detailed timing-algorithm claim is made.

SQL get_sql_model187 checks pool by physical short_name, but newly created classes are saved under name238. Absent tables can therefore trigger whole-database automap.prepare on miss-read, insert/DDL and immediate re-read. This is a source-level reason to time reflection, not evidence that it caused the historical28.55s gap. Existing tables/rows behave differently; new index files cannot be multiplied by an assumed reflection cost. Model lock is a threading.Lock; RollbackSession.commit catches IntegrityError and rolls back.

## Scope and CPU checks

The default entry imports only standard library and prints a plan:

```sh
python3 -B tools/probe_biv150_flaggems_tuner_diag.py
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_flaggems_tuner_diag_cpu.py \
  --source-root /private/tmp/cjj-flaggems-tuner-source-live-20261028
```

The checker compiles actual full modules without executing/importing them, extracts exact CodeTypes, then executes selected methods with fake cache/import/driver/SQL objects. All CodeType fields and nested constants stay strict, including future flags, qualname, line and exception tables. It proves opaque-handle identity, original return/exception identity even on recorder failure, lazy driver access, cached-property assignment/deletion/scoped restore, static SQL reflection, two _bench entries/one callback, import-loader order/forwarding, idempotent install/uninstall and wrong-source/code rejection. No Torch/Triton/SQLAlchemy import or device work occurs.

## Opt-in target bootstrap, Root scheduling only

The temporary `sitecustomize.py` enables a narrow import watcher only with explicit env. It does not import frameworks or initialize a driver. SQL/libentry modules are wrapped only after their normal loader executes; the loader's other methods forward unchanged. Their exact source bytes and full-module method code, inherited Autotuner code and selected benchmark-protocol source must match. Unknown sources fail import instead of claiming a diagnostic receipt. Existing modules can be instrumented late; required-worker coverage still has to be established by Root.

Hooks are LibTuner.run total wall, inherited _bench entry, lazy benchmark callback and SQLPersistantModel.build_sql_model_by_db reflection. A data descriptor delegates the original cached_property at its original access point; it preserves the original instance cache and setter/deleter behavior, including use_benchmark_protocol's save/restore. It adds no synchronization, kernel calls, cache clear, retry, reset, DB query or benchmark-protocol change. Callback counts mean the callable was invoked, not that a successful GPU measurement necessarily completed. Counts/time distinguish deduplication, selection/cache work and reflection.

Root may prefix the original fixed serve command in a separately labelled diagnostic run (the command arguments stay exact):

```sh
CJJ_FLAGGEMS_TUNER_DIAGNOSTIC=1 \
CJJ_FLAGGEMS_TUNER_DIAGNOSTIC_OUT=NEW_OWNED_DIAGNOSTIC_OUT \
PYTHONPATH=ABSOLUTE_FROZEN_WT/tools/diagnostics/flaggems_tuner:ORIGINAL_PYTHONPATH \
  ORIGINAL_FIXED_SERVE_COMMAND_AND_ARGS
```

Use the actual prior PYTHONPATH, not the literal placeholder. Do not alter official request/benchmark counts, warmup, precision or TTFT rules. The bootstrap refuses to shadow an existing downstream sitecustomize; inspect that admission before starting. No service is started by the tools. The caller must arrange its normal owned cleanup.

Each process has one atomic JSON receipt, inherited hooks reset counters/locks after fork, and no monitor thread or signal handler is added. Nested events carry PID/thread/parent IDs and monotonic+wall timestamps. Join them with existing fullwarm/score command timestamps; do not sum inclusive parent and child time. Records flush at outer-call boundaries at most every0.5s and at ordinary exit. Per-process8192-event overflow or recording error marks the trace incomplete; missing SQL/libentry hooks also mark it incomplete. Recording errors preserve original return/exception and cannot become a pass. Abrupt kills may lose final records. Root must verify actual serving worker PID coverage and source identities; a controller-only/partial trace cannot rule out unobserved retuning. Instrumentation perturbs host time, so only causal/count attribution is intended.

## Read-only SQLite phase snapshots (design only, collector not implemented)

Before score after fullwarm, and after score, use the actual observed `libcache.db_url`. Read via SQLite `mode=ro`/query_only with bounded busy timeout; use online backup into a new owned file to capture committed WAL consistently, never immutable=1 on live WAL or a plain .db-only copy. Do not checkpoint/change journal mode. Do this between requests and diff offline copies. Record per-table schema/xinfo/PK, full typed rows, row counts and digests. Encode exact ints, REAL binary64 bits, text and BLOB separately. Compare by complete unique PK; use full-row multisets when no unique PK and report ambiguity. Preserve row payloads, not only file SHA/counts. Classify actual ConfigCache versus BenchmarkCache namespaces from observed tuner/SQL keys; both can have "benchmark" in the name, so substring classification is insufficient. Config-only inserts do not imply fresh measurements. Correlate new benchmark rows with observed callback events/time. The old runs lack row snapshots, so their exact mutations remain unknown.
