#!/usr/bin/env python3
"""Local SQLite fixtures and optional read-only real-v6 replay. No frameworks."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent / 'diagnostics/flaggems_tuner'))
import cjj_sqlite_typed_snapshot as s


def expect_error(fn, phrase=None):
    try:
        fn()
    except s.SnapshotError as error:
        if phrase:
            assert phrase.lower() in str(error).lower(), (phrase, error)
        return
    raise AssertionError('Expected explicit failure')


def run(v6_root=None):
    original_connect = sqlite3.connect
    connections = []
    sql_trace = []
    inject = None
    class Tracked(sqlite3.Connection):
        closed = False
        def execute(self, statement, *args, **kwargs):
            cursor = super().execute(statement, *args, **kwargs)
            if inject is not None:
                inject(statement)
            return cursor
        def close(self):
            super().close()
            self.closed = True
    def connect(*args, **kwargs):
        assert kwargs['uri'] is True and '?mode=ro' in args[0]
        assert 'immutable=' not in args[0] and kwargs['isolation_level'] is None
        connection = original_connect(*args, factory=Tracked, **kwargs)
        connection.set_trace_callback(sql_trace.append)
        connections.append(connection)
        return connection
    def snap(path, limits=s.Limits()):
        with patch.object(s.sqlite3, 'connect', connect):
            return s.snapshot(path, limits)
    def closed():
        assert connections and all(c.closed for c in connections)
        for c in connections:
            try:
                c.execute('SELECT 1')
            except sqlite3.ProgrammingError:
                pass
            else:
                raise AssertionError('Connection leaked')
    with tempfile.TemporaryDirectory(prefix='cjj-sqlite-typed-') as tmp:
        root = Path(tmp)
        db = root / 'quoted ? # data.db'
        name = 'Config_benchmark_"quoted"; DROP TABLE impossible;'
        column = 'value"quoted'
        writer = original_connect(db)
        writer.execute('CREATE TABLE ' + s.quote(name) + '(' + s.quote(column) + ')')
        writer.executemany('INSERT INTO ' + s.quote(name) + ' VALUES(?)',
                          [(1,), (1.0,), ('1',), (b'1',), (None,), ('duplicate',), ('duplicate',),
                           (float('inf'),), (-float('inf'),), (b'\x00\xff',)])
        writer.execute('INSERT INTO ' + s.quote(name) + " VALUES(CAST(x'80' AS TEXT))")
        writer.executescript('''CREATE TABLE "without rowid"(k TEXT, n INTEGER, PRIMARY KEY(k,n)) WITHOUT ROWID;
INSERT INTO "without rowid" VALUES('k',2);
CREATE TABLE generated(a INTEGER, b INTEGER GENERATED ALWAYS AS (a*2) VIRTUAL,
 c TEXT GENERATED ALWAYS AS (CAST(a AS TEXT)) STORED);
INSERT INTO generated(a) VALUES(3);
CREATE TABLE aut(id INTEGER PRIMARY KEY AUTOINCREMENT);
INSERT INTO aut DEFAULT VALUES;
CREATE VIEW view_only AS SELECT a FROM generated;
CREATE INDEX idx_generated ON generated(a);
CREATE TRIGGER tg AFTER INSERT ON aut BEGIN SELECT 1; END;''')
        writer.commit()
        writer.close()
        raw = db.read_bytes()
        before_mode = stat.S_IMODE(db.stat().st_mode)
        a = snap(db)
        s.validate(a)
        b = snap(db)
        assert s.diff(a, b)['logical_equal'] and not s.diff(a, b)['table_changes']
        assert db.read_bytes() == raw and stat.S_IMODE(db.stat().st_mode) == before_mode
        table = next(t for t in a['tables'] if t['name'] == name)
        types = {r['row'][0]['type'] for r in table['rows']}
        assert types == {'null', 'integer', 'real', 'text', 'blob'}
        assert table['row_count'] == 11 and table['distinct_rows'] == 10
        assert any(r['count'] == 2 for r in table['rows'])
        assert any(r['row'] == [{'type': 'text', 'base64': 'gA=='}] for r in table['rows'])
        assert all(t['attribution'] == 'unattributed' for t in a['tables'])
        assert {r['type'] for r in a['schema']} >= {'table', 'view', 'trigger', 'index'}
        assert any(t['name'] == 'sqlite_sequence' for t in a['tables'])
        assert any(c['hidden'] == 2 for t in a['tables'] for c in t['columns'])
        assert any(c['hidden'] == 3 for t in a['tables'] for c in t['columns'])
        closed()
        assert all(q in ('BEGIN', 'ROLLBACK') or q.startswith('SELECT ') or
                   q.startswith('PRAGMA main.table_xinfo(') for q in sql_trace)
        assert not any('=' in q for q in sql_trace if q.startswith('PRAGMA'))
        # Physical file metadata and rootpage movement cannot masquerade as a row diff.
        aux = deepcopy(a)
        aux['auxiliary_before']['byte_sha256'] = '0' * 64
        aux['schema'][0]['rootpage'] += 100
        delta = s.diff(a, aux)
        assert delta['logical_equal'] and delta['physical_rootpage_changes'] and not delta['schema_changed']
        writer = original_connect(db)
        writer.execute('DELETE FROM ' + s.quote(name) + ' WHERE typeof(' + s.quote(column) + ")='integer'")
        writer.execute('INSERT INTO ' + s.quote(name) + ' VALUES(?)', ('1',))
        writer.commit()
        writer.close()
        changed = snap(db)
        d = s.diff(a, changed)
        assert len(d['table_changes']) == 1
        assert d['table_changes'][0]['rows_added'] == [dict(row=[dict(type='text', base64='MQ==')], count=1)]
        assert d['table_changes'][0]['rows_removed'] == [dict(row=[dict(type='integer', decimal='1')], count=1)]
        writer = original_connect(db)
        writer.execute('CREATE INDEX schema_only ON aut(id)')
        writer.commit()
        writer.close()
        schema_only = snap(db)
        d = s.diff(changed, schema_only)
        assert [r['name'] for r in d['schema_added']] == ['schema_only'] and not d['table_changes']
        writer = original_connect(db)
        writer.execute('DROP INDEX schema_only')
        writer.execute('DROP TABLE "without rowid"')
        writer.execute('CREATE TABLE new_empty(v)')
        writer.commit()
        writer.close()
        dropped = snap(db)
        d = s.diff(schema_only, dropped)
        assert any(r['name'] == 'schema_only' for r in d['schema_removed'])
        assert any(t['name'] == 'without rowid' and t['table_removed'] for t in d['table_changes'])
        assert any(t['name'] == 'new_empty' and t['table_added'] for t in d['table_changes'])
        writer = original_connect(db)
        writer.execute('ALTER TABLE new_empty ADD COLUMN x TEXT')
        writer.commit()
        writer.close()
        d = s.diff(dropped, snap(db))
        assert d['schema_changed'] and any(t['columns_changed'] for t in d['table_changes'])
        db.chmod(0o444)  # Fixture setup only; collector never chmods a source.
        assert s.diff(snap(db), snap(db))['logical_equal']
        assert stat.S_IMODE(db.stat().st_mode) == 0o444
        db.chmod(before_mode)
        for limits in [replace(s.Limits(), max_rows=1), replace(s.Limits(), max_tables=1),
                       replace(s.Limits(), max_columns=1), replace(s.Limits(), max_schema_bytes=1),
                       replace(s.Limits(), max_output_bytes=1), replace(s.Limits(), max_cell_bytes=1),
                       replace(s.Limits(), max_source_bytes=1)]:
            expect_error(lambda: snap(db, limits), 'limit')
            closed()
        missing = root / 'does-not-exist.db'
        expect_error(lambda: snap(missing), 'missing')
        assert not missing.exists()
        empty = root / 'zero.db'
        empty.touch()
        expect_error(lambda: snap(empty), 'initialized')
        empty.unlink()
        writer = original_connect(empty)
        writer.execute('PRAGMA user_version=1')  # Fixture initialization only.
        writer.close()
        expect_error(lambda: snap(empty), 'no tables')
        writer = original_connect(db)
        writer.execute('BEGIN EXCLUSIVE')
        try:
            expect_error(lambda: snap(db, replace(s.Limits(), timeout_ms=20)), 'locked')
            closed()
        finally:
            writer.rollback()
            writer.close()
        sentinel = RuntimeError('fixture sentinel, not an SQLite failure')
        def fail(statement):
            if statement.startswith('PRAGMA main.table_xinfo'):
                raise sentinel
        inject = fail
        try:
            snap(db)
        except RuntimeError as error:
            assert error is sentinel
        else:
            raise AssertionError('Exception swallowed')
        inject = None
        closed()
        # A WAL writer remains open, so sidecars already exist. Commit after
        # sqlite_schema establishes the reader's snapshot: table rows stay old.
        waldb = root / 'wal.db'
        writer = original_connect(waldb)
        assert writer.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
        writer.execute('CREATE TABLE t(v)')
        writer.execute('INSERT INTO t VALUES(1)')
        writer.commit()
        committed = []
        def commit_during_read(statement):
            if not committed and statement.startswith('SELECT type,name'):
                writer.execute('INSERT INTO t VALUES(2)')
                writer.commit()
                committed.append(True)
        inject = commit_during_read
        wal_a = snap(waldb)
        inject = None
        wal_b = snap(waldb)
        assert wal_a['total_rows'] == 1 and wal_b['total_rows'] == 2
        assert wal_a['auxiliary_before']['byte_sha256'] == wal_b['auxiliary_after']['byte_sha256']
        assert not s.diff(wal_a, wal_b)['logical_equal']
        closed()
        writer.close()
        expect_error(lambda: snap(waldb), 'existing -wal and -shm')
        virtual = root / 'virtual.db'
        writer = original_connect(virtual)
        try:
            writer.execute('CREATE /*fixture*/ VIRTUAL TABLE vt USING fts5(v)')
            writer.commit()
        except sqlite3.OperationalError as error:
            assert 'no such module' in str(error)
            virtual_result = 'not available in this SQLite build'
        else:
            expect_error(lambda: snap(virtual), 'virtual table')
            virtual_result = 'explicitly rejected'
            closed()
        finally:
            writer.close()
        # Corrupt/incomplete artifacts cannot produce a reassuring zero diff.
        faults = []
        def fault(f):
            v = deepcopy(a)
            f(v)
            faults.append(v)
        fault(lambda v: v.update(completed=False))
        fault(lambda v: v.update(tables=[None]))
        fault(lambda v: v['tables'][0]['rows'].pop())
        fault(lambda v: v['tables'][0]['rows'][0].update(count=0))
        fault(lambda v: v['tables'][0]['rows'][0]['row'].append({'type': 'null'}))
        fault(lambda v: v['tables'][0]['rows'][0]['row'][0].update(type='unknown'))
        fault(lambda v: v.update(schema_sha256='0' * 64))
        fault(lambda v: v.update(logical_sha256='0' * 64))
        for bad in faults:
            expect_error(lambda: s.diff(a, bad))
        # Explicit CLI/new-output paths, default plan and failures.
        cli = Path(s.__file__)
        def invoke(*args):
            return subprocess.run([sys.executable, '-B', '-W', 'error', str(cli), *map(str, args)],
                                  capture_output=True, text=True, timeout=15)
        assert json.loads(invoke().stdout)['default'].startswith('stdlib CPU plan')
        out = root / 'new.json'
        result = invoke('snapshot', '--db', db, '--out', out)
        assert result.returncode == 0, result.stderr
        saved = out.read_bytes()
        assert invoke('snapshot', '--db', db, '--out', out).returncode == 2
        assert out.read_bytes() == saved
        failed_out = root / 'failed.json'
        assert invoke('snapshot', '--db', missing, '--out', failed_out).returncode == 2
        assert not failed_out.exists()
        diff_out = root / 'diff.json'
        result = invoke('diff', '--before', out, '--after', out, '--out', diff_out)
        assert result.returncode == 0 and json.loads(diff_out.read_bytes())['logical_equal']
        assert all(c.closed for c in connections)
    report = dict(completed=True, no_framework_or_GPU_import=True,
                  typed_null_int_real_text_blob_and_invalid_UTF8=True,
                  quoted_identifiers_no_SQL_injection=True, duplicate_multiset_and_WITHOUT_ROWID=True,
                  views_triggers_indexes_internal_and_generated_columns_recorded=True,
                  Config_named_benchmark_never_filtered_or_classified=True,
                  typed_row_update_table_add_remove_and_schema_only_diff=True,
                  physical_metadata_not_a_semantic_diff=True,
                  URI_readonly_single_transaction_consistent_with_concurrent_WAL_commit=True,
                  all_connections_closed_and_transactions_rolled_back=True,
                  source_bytes_and_permissions_unchanged=True,
                  missing_zero_empty_locked_and_WAL_sidecar_gap_explicit_failures=True,
                  unsupported_virtual_table=virtual_result,
                  seven_limits_enforced=True, incomplete_snapshot_faults_rejected=len(faults),
                  default_plan_and_new_output_CLI=True, actual_bench_execution_proven=False,
                  formal_score_eligible=False)
    if v6_root:
        replay = []
        for path in sorted(Path(v6_root).glob('*.db')):
            first = snap(path)
            second = snap(path)
            delta = s.diff(first, second)
            assert delta['logical_equal'] and not delta['table_changes']
            tables = first['tables']
            filtered = [t for t in tables if 'benchmark' in t['name']]
            replay.append(dict(name=path.name, source_sha256=first['auxiliary_before']['byte_sha256'],
                               tables=len(tables), rows=first['total_rows'],
                               logical_sha256=first['logical_sha256'], schema_sha256=first['schema_sha256'],
                               old_name_filter_excluded_tables=len(filtered),
                               excluded_tables_with_p50_p20_p80=sum({'p50', 'p20', 'p80'} <= {c['name'] for c in t['columns']} for t in filtered),
                               classification='unattributed; column names are observations, not execution proof',
                               same_source_two_read_transactions_equal=True,
                               initialized_v6_only_not_after_score=True))
        assert len(replay) == 2
        closed()
        report['actual_v6_readonly_replay'] = replay
    assert not {'torch', 'triton', 'sqlalchemy', 'flag_gems'}.intersection(sys.modules)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--v6-root', type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.v6_root), sort_keys=True, indent=2))
