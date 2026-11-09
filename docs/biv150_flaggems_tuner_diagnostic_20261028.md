# Temporary source-pinned FlagGems cache diagnosis

CPU-only preparation based on actual target8-source snapshot. libentry SHA414e8a3f… and SQL6dc19cb5… exactly match the historical host-gap source. This branch starts at actual main4229 and changes no production file, framework source, CLI parameter, cache or kernel. The CPU tools are shared as Draft97; this preparation does not authorize a service/device run. Every receipt says `formal_score_eligible=false`; instrumented scores cannot serve as performance acceptance.

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

## Read-only SQLite phase snapshots and offline diff

The standalone collector is now implemented. Its default command prints a standard-library plan and opens no database:

```sh
python3 -B tools/diagnostics/flaggems_tuner/cjj_sqlite_typed_snapshot.py
python3 -B tools/diagnostics/flaggems_tuner/cjj_sqlite_typed_snapshot.py snapshot \
  --db ACTUAL_OBSERVED_CACHE_DB --out NEW_BEFORE_JSON
python3 -B tools/diagnostics/flaggems_tuner/cjj_sqlite_typed_snapshot.py snapshot \
  --db SAME_ACTUAL_CACHE_DB --out NEW_AFTER_JSON
python3 -B tools/diagnostics/flaggems_tuner/cjj_sqlite_typed_snapshot.py diff \
  --before NEW_BEFORE_JSON --after NEW_AFTER_JSON --out NEW_DIFF_JSON
```

These are explicit CPU diagnostics, scheduled outside formal score windows. No service or hook is started. One invocation handles one database; multiple databases are not globally atomic. Each successful database snapshot uses URI mode=ro and one BEGIN/read/ROLLBACK transaction, including sqlite_schema, every ordinary table's table_xinfo, and all explicitly named columns/rows. All cursors/connections close on success and failure. No DDL, source write, permission change, backup, immutable flag, checkpoint or write PRAGMA is issued. Read-only table_xinfo is the only PRAGMA. Missing, zero-byte, initialized-but-tableless, locked, unsupported virtual-table and oversized databases fail explicitly; no success/empty-cache artifact is published. Existing ordinary zero-row tables remain represented. All SQLite internal tables, indexes, views, triggers and generated columns are inventoried; views are schema-only, not executed.

Live WAL reads honor SQLite's normal locking/snapshot rules; they are not a plain .db copy. WAL mode requires existing -wal and -shm files before admission, avoiding the known missing-sidecar creation path. Normal SQLite SHM lock/read-mark handling still applies; do not treat this as an immutable byte copy or a guarantee against concurrent sidecar lifecycle races. A disappearing/changing source or sidecar can cause an explicit read failure. Never pass immutable=1 to bypass that failure. Root should collect between phases/requests and verify serving-worker ownership separately.

Limits are explicit and configurable per invocation: 1,000,000 total rows, 4096 tables, 512 columns/table, 8MiB schema/column metadata, 64MiB encoded output, 256MiB source file and 4MiB/cell; read-lock wait defaults to1000ms with a hard10000ms maximum. Oversize failure is not sampling. One row/cell is materialized by sqlite3 before its output-size check; these are inventory/output limits rather than a strict process-RSS bound. No partial snapshot is labelled complete. Output must be new and is linked atomically only after complete collection; it never overwrites an earlier artifact. Source byte SHA/size/inode/mtime and sidecar metadata are labelled nontransactional auxiliary observations, not row/schema equality. Logical hashes exclude those fields and physical rootpage placement. Raw rootpages remain saved and their changes are reported separately.

Each value retains its SQLite result type: NULL; exact signed64-bit integer decimal; REAL IEEE754 binary64 bits; TEXT bytes/base64 (including invalid UTF8); BLOB bytes/base64. Complete rows form a sorted multiset with positive duplicate counts, not rowid/PK assumptions. WITHOUT ROWID tables and duplicate unkeyed rows are handled the same way. Offline validation rejects missing tables, incomplete row counts, malformed typed cells and mismatching digests. A row update appears as typed rows removed/added; no uncertain PK pairing is invented. Diff inventories added/removed/changed schema definitions and all table/column/row changes. All tables and deltas remain unattributed. Neither a benchmark substring nor p50/p20/p80 columns prove ConfigCache/BenchmarkCache identity or that a benchmark callback ran. Link exact observed tuner/SQL table identities from the temporary hook before assigning a cache role.

### Actual namespace evidence and limits

The actual LibTuner._make_config_table_name524-539 returns its base unchanged for protocol[0]==triton_do_bench; other protocols append _benchmark_PROTOCOLHASH. Actual ConfigCache.get/set229-233 pass that exact name to model.get_config/put_config, and SQL get_sql_model187-214 appends the key-column-name hash to obtain the physical name. The separate CPU checker extracts these actual full-module CodeTypes and records the exact get/put/reflection arguments:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_config_cache_namespace_cpu.py \
  --source-root /private/tmp/cjj-flaggems-tuner-source-live-20261028
```

Its custom/replay protocol cases demonstrate a constructible ConfigCache name containing benchmark; they are fake protocol inputs, not observations from a live serving worker. Actual benchmark_protocol191-215 falls back to event timing when its active backend is not in the replay implementation map. No claim is made about the historical worker's resolved protocol without its receipt.

Initial v6 replay has58 tables/3859 rows in3_2 and894 tables/32110 rows in3_6. The old substring filter excludes29 and426 tables respectively; all excluded tables in these initial databases have p50/p20/p80. **No initial-v6 ConfigCache-with-benchmark-name instance has been established.** After-score databases have not been collected, so no现场漏Config mutation or exact historical row delta is claimed. The prior selected-table aggregate cannot support an all-table unchanged claim. Neither a changed aggregate nor an added benchmark row alone proves real benchmarking. Hook callback/entry/protocol receipts and full before/after rows are needed together; the old warm-to-score gap remains unexplained.

### CPU replay

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -W error tools/check_biv150_sqlite_typed_snapshot_cpu.py \
  --v6-root /private/tmp/cjj-native-k8-service-smoke-v1-raw-20261028/out/flaggems-cache/config_cache
```

Fixtures cover quoted table/column names, typed equal representations, duplicate rows, WITHOUT ROWID, generated columns/internal tables, schema-only/add/remove changes, a Config-labelled benchmark name without filtering, missing/zero/empty/readonly/locked inputs, all size/row bounds, new-output/no-overwrite CLI, malformed artifact rejection and all-path connection closure. A concurrent WAL commit after the schema read proves that table rows stay in the original transaction snapshot; its unchanged main-file SHA alongside a real row change also demonstrates why byte SHA is insufficient. The two actual initial v6 databases are each read twice, with complete logical multiset equality and unchanged source bytes/permissions. Tests import no Torch/Triton/SQLAlchemy/FlagGems and do not attribute performance, precision or a historical cause.
