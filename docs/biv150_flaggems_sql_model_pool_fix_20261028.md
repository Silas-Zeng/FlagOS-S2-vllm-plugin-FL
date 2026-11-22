# Newly created FlagGems SQL model pool key (default-off prototype)

Pinned target: current FlagGems5.3.5 SQLPersistantModel source SHA6dc19cb581d70154f14b95ed9112d682c6b5919b6cf3f8dea01a78a75ec0efd7, team main4229. This does not upgrade the competition framework.

After a new tuning table is created, get_sql_model currently stores its mapped class at sql_model_pool[name]. Lookups use short_name (name plus an MD5 of key field names, with PostgreSQL truncation rules). The next lookup can miss this in-memory entry and run AutoBase.prepare(engine) again, reflecting the DB. The proposed change stores only that newly created class at short_name, matching the already-existing-table branch. It does not alter cache keys, DB data, table creation, tuning results or GPU mathematics.

The tool is explicitly default OFF, has no automatic worker hook and imports no FlagGems, SQLAlchemy, Torch or Triton. An explicit caller supplies the already-loaded module and enabled=True. It verifies the whole original source SHA, all loaded method CodeType fields and original globals, then changes only the exact AST insertion index. The whole-source inverse is checked. Original method signature/defaults/closure/annotations/override flags are retained. Reinstall is idempotent; uninstall refuses to overwrite another runtime change. No source/site-packages file is edited.

CPU tests execute the actual source method with an opaque DB adapter, not a copy of its algorithm. Create→read causes2 reflections before versus1 after; miss→create→read causes3 versus2. SQLite/PostgreSQL naming, two distinct key schemas, identical returned classes, DDL exception identity, context closure,12 concurrent callers, default-off and source/code/binding rejection are checked. Real SQLAlchemy/target worker/actual wall time/model accuracy/throughput are not yet tested. This does not establish that reflection caused the prior4K warm/score gap. Initialv6 has no proved Config table misclassification; typed-row and live timing diagnostics are separate Draft97 work.

Replay (Python3.12, standard library):

```sh
python3 -B -W error tools/check_flaggems_sql_model_pool_fix_cpu.py --source /absolute/pinned/flag_gems/utils/models/sql.py
python3 -B tools/flaggems_sql_model_pool_fix.py
```

A future opt-in worker experiment must install after loading the genuine pinned module, record its receipt, keep the fixed service CLI, and measure source-qualified service performance. Do not infer performance qualification from the opaque-adapter reflection counts.
