#!/usr/bin/env python3
"""Explicit, bounded CPU SQLite read transactions and offline typed multiset diff."""
import argparse
import base64
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import time

FORMAT = 'cjj-sqlite-typed-multiset-v1'


class SnapshotError(ValueError):
    pass


class SQLiteText(bytes):
    """Distinguish SQLite TEXT (including invalid UTF8) from BLOB without casts."""


@dataclass(frozen=True)
class Limits:
    max_rows: int = 1_000_000
    max_tables: int = 4096
    max_columns: int = 512
    max_schema_bytes: int = 8 * 1024 * 1024
    max_output_bytes: int = 64 * 1024 * 1024
    max_source_bytes: int = 256 * 1024 * 1024
    max_cell_bytes: int = 4 * 1024 * 1024
    timeout_ms: int = 1000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise SnapshotError('Limit must be a positive integer: ' + name)
        if self.timeout_ms > 10_000:
            raise SnapshotError('Read lock timeout is capped at 10000ms')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False, separators=(',', ':')).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def cell(value, limit):
    if value is None:
        return {'type': 'null'}
    if type(value) is int:
        return {'type': 'integer', 'decimal': str(value)}
    if type(value) is float:
        return {'type': 'real', 'ieee754_binary64_be': struct.pack('>d', value).hex()}
    if isinstance(value, (SQLiteText, bytes)):
        if len(value) > limit:
            raise SnapshotError('Cell byte limit exceeded')
        return {'type': 'text' if isinstance(value, SQLiteText) else 'blob',
                'base64': base64.b64encode(value).decode('ascii')}
    raise SnapshotError('Unsupported SQLite value type: ' + type(value).__name__)


def identifier(value):
    if not isinstance(value, SQLiteText):
        raise SnapshotError('Expected TEXT schema identifier')
    try:
        return value.decode('utf-8')
    except UnicodeDecodeError as error:
        raise SnapshotError('Non-UTF8 schema identifier is unsupported') from error


def quote(name):
    if type(name) is not str or '\0' in name:
        raise SnapshotError('Invalid SQLite identifier')
    return '"' + name.replace('"', '""') + '"'


def auxiliary(path, limit):
    """Not a logical snapshot; changes here never imply typed-row changes."""
    stat = path.stat()
    if stat.st_size > limit:
        raise SnapshotError('Source file byte limit exceeded')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    sidecars = {}
    for suffix in ('-wal', '-shm', '-journal'):
        candidate = Path(str(path) + suffix)
        try:
            st = candidate.stat()
            sidecars[suffix] = {'exists': True, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
        except FileNotFoundError:
            sidecars[suffix] = {'exists': False}
    return dict(size=stat.st_size, byte_sha256=h.hexdigest(), device=stat.st_dev,
                inode=stat.st_ino, mtime_ns=stat.st_mtime_ns, sidecars=sidecars,
                transactional=False, semantic_diff_evidence=False)


def rows(cursor):
    try:
        yield from cursor
    finally:
        cursor.close()


def snapshot(path, limits=Limits()):
    requested = Path(path).absolute()
    try:
        path = requested.resolve(strict=True)
    except FileNotFoundError as error:
        raise SnapshotError('Missing source database: ' + str(requested)) from error
    if not path.is_file():
        raise SnapshotError('Source must be an existing regular database file')
    before = auxiliary(path, limits.max_source_bytes)
    with path.open('rb') as stream:
        header = stream.read(100)
    if len(header) < 100 or header[:16] != b'SQLite format 3\0':
        raise SnapshotError('Not an initialized SQLite database (empty is not missing=zero)')
    # mode=ro can otherwise create WAL shared-memory sidecars. Refuse rather
    # than using immutable=1 (which could silently ignore committed live WAL).
    if 2 in header[18:20] and not all(before['sidecars'][s]['exists'] for s in ('-wal', '-shm')):
        raise SnapshotError('WAL database needs existing -wal and -shm; refusing sidecar creation')
    conn = None
    stage = 'connect'
    started = time.time_ns()
    try:
        conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True,
                               timeout=limits.timeout_ms / 1000, isolation_level=None,
                               detect_types=0)
        conn.text_factory = SQLiteText
        stage = 'begin'
        with closing(conn.execute('BEGIN')):
            pass
        stage = 'schema'
        schema = []
        schema_bytes = 0
        for raw in rows(conn.execute('SELECT type,name,tbl_name,rootpage,sql FROM main.sqlite_schema ORDER BY type,name')):
            record = dict(type=identifier(raw[0]), name=identifier(raw[1]),
                          table=identifier(raw[2]), rootpage=raw[3],
                          sql=None if raw[4] is None else identifier(raw[4]))
            schema_bytes += len(canonical(record))
            if schema_bytes > limits.max_schema_bytes:
                raise SnapshotError('Schema byte limit exceeded')
            schema.append(record)
        definitions = [r for r in schema if r['type'] == 'table']
        if not definitions:
            raise SnapshotError('Database has no tables; refusing an empty-cache success')
        if len(definitions) > limits.max_tables:
            raise SnapshotError('Table count limit exceeded')
        tables = []
        total_rows = 0
        encoded_bytes = schema_bytes
        for definition in definitions:
            name = definition['name']
            prefix = re.sub(r'/\*.*?\*/|--[^\r\n]*', ' ', definition['sql'] or '', flags=re.S)
            if re.match(r'^\s*CREATE\s+VIRTUAL\s+TABLE\b', prefix, flags=re.I):
                raise SnapshotError('Virtual table needs explicit module qualification: ' + name)
            stage = 'table_xinfo:' + name
            columns = []
            # Read-only PRAGMA only; never journal_mode, query_only, checkpoint,
            # busy_timeout or any other PRAGMA assignment.
            for raw in rows(conn.execute('PRAGMA main.table_xinfo(' + quote(name) + ')')):
                columns.append(dict(cid=raw[0], name=identifier(raw[1]),
                                    declared_type=identifier(raw[2]), notnull=raw[3],
                                    default=None if raw[4] is None else identifier(raw[4]),
                                    pk=raw[5], hidden=raw[6]))
            if not columns or len(columns) > limits.max_columns:
                raise SnapshotError('Missing columns or column limit exceeded: ' + name)
            column_bytes = len(canonical(columns))
            schema_bytes += column_bytes
            if schema_bytes > limits.max_schema_bytes:
                raise SnapshotError('Schema/column metadata byte limit exceeded')
            encoded_bytes += column_bytes
            if encoded_bytes > limits.max_output_bytes:
                raise SnapshotError('Encoded output byte limit exceeded')
            stage = 'rows:' + name
            counts = Counter()
            query = 'SELECT ' + ','.join(quote(c['name']) for c in columns) + ' FROM main.' + quote(name)
            for raw in rows(conn.execute(query)):
                total_rows += 1
                if total_rows > limits.max_rows:
                    raise SnapshotError('Total row limit exceeded')
                key = canonical([cell(value, limits.max_cell_bytes) for value in raw])
                if key not in counts:
                    encoded_bytes += len(key) + 64
                    if encoded_bytes > limits.max_output_bytes:
                        raise SnapshotError('Encoded output byte limit exceeded')
                counts[key] += 1
            multiset = [dict(row=json.loads(key), count=counts[key]) for key in sorted(counts)]
            table = dict(name=name, attribution='unattributed', columns=columns,
                         row_count=sum(counts.values()), distinct_rows=len(counts), rows=multiset)
            table['multiset_sha256'] = digest(multiset)
            tables.append(table)
        schema_semantic = [{k: v for k, v in r.items() if k != 'rootpage'} for r in schema]
        logical = dict(schema=schema_semantic, tables=tables)
        result = dict(format=FORMAT, completed=True, consistent_read_transaction=True,
                      connection='URI mode=ro; one BEGIN/ROLLBACK; no immutable/backup/write PRAGMA',
                      requested_path=str(requested), source_path=str(path), limits=asdict(limits),
                      sqlite_version=sqlite3.sqlite_version, started_unix_ns=started,
                      schema=schema, tables=tables, total_rows=total_rows,
                      schema_sha256=digest(schema_semantic), logical_sha256=digest(logical),
                      auxiliary_before=before, formal_score_eligible=False,
                      attribution='unattributed; join actual observed hook identities offline')
    except sqlite3.Error as error:
        raise SnapshotError(f'SQLite failure at {stage}: {type(error).__name__}: {error}') from error
    finally:
        if conn is not None:
            try:
                if conn.in_transaction:
                    conn.rollback()
            finally:
                conn.close()
    result['auxiliary_after'] = auxiliary(path, limits.max_source_bytes)
    after = result['auxiliary_after']
    if (before['device'], before['inode']) != (after['device'], after['inode']):
        raise SnapshotError('Database file identity changed during collection')
    if len(canonical(result)) + 1 > limits.max_output_bytes:
        raise SnapshotError('Final encoded output byte limit exceeded')
    return result


def require(ok, message):
    if not ok:
        raise SnapshotError('Invalid snapshot: ' + message)


def validate_cell(value):
    require(type(value) is dict, 'cell must be typed object')
    kind = value.get('type')
    if kind == 'null':
        require(value == {'type': 'null'}, 'null fields')
    elif kind == 'integer':
        require(set(value) == {'type', 'decimal'} and type(value['decimal']) is str, 'integer fields')
        try:
            number = int(value['decimal'])
        except ValueError as error:
            raise SnapshotError('Invalid integer') from error
        require(str(number) == value['decimal'] and -(2**63) <= number < 2**63, 'integer range/canonical encoding')
    elif kind == 'real':
        require(set(value) == {'type', 'ieee754_binary64_be'}, 'real fields')
        bits = value['ieee754_binary64_be']
        require(type(bits) is str and len(bits) == 16 and all(c in '0123456789abcdef' for c in bits), 'real bits')
    elif kind in ('text', 'blob'):
        require(set(value) == {'type', 'base64'} and type(value['base64']) is str, 'byte fields')
        try:
            raw = base64.b64decode(value['base64'], validate=True)
        except ValueError as error:
            raise SnapshotError('Invalid base64') from error
        require(base64.b64encode(raw).decode('ascii') == value['base64'], 'base64 canonical encoding')
    else:
        raise SnapshotError('Invalid snapshot: unknown SQLite cell type')


def validate(value):
    require(type(value) is dict and value.get('format') == FORMAT and value.get('completed') is True
            and value.get('consistent_read_transaction') is True, 'format/completion/transaction')
    schema, tables = value.get('schema'), value.get('tables')
    require(type(schema) is list and type(tables) is list, 'schema/tables')
    seen_schema = set()
    for r in schema:
        require(type(r) is dict and set(r) == {'type', 'name', 'table', 'rootpage', 'sql'}, 'schema record')
        require(all(type(r[k]) is str for k in ('type', 'name', 'table')) and type(r['rootpage']) is int
                and (r['sql'] is None or type(r['sql']) is str), 'schema values')
        key = r['type'], r['name']
        require(key not in seen_schema, 'duplicate schema object')
        seen_schema.add(key)
    require(all(type(t) is dict for t in tables), 'table objects')
    require([t.get('name') for t in tables] == [r['name'] for r in schema if r['type'] == 'table'], 'all tables in schema order')
    total = 0
    for t in tables:
        require(type(t) is dict and t.get('attribution') == 'unattributed', 'table attribution')
        columns = t.get('columns')
        require(type(columns) is list and bool(columns), 'columns')
        for c in columns:
            require(type(c) is dict and set(c) == {'cid', 'name', 'declared_type', 'notnull', 'default', 'pk', 'hidden'}, 'column record')
            require(type(c['name']) is str and type(c['declared_type']) is str and
                    all(type(c[k]) is int for k in ('cid', 'notnull', 'pk', 'hidden')) and
                    (c['default'] is None or type(c['default']) is str), 'column values')
        previous = None
        count = 0
        require(type(t.get('rows')) is list, 'row multiset')
        for entry in t['rows']:
            require(type(entry) is dict and set(entry) == {'row', 'count'}, 'multiset record')
            require(type(entry['count']) is int and entry['count'] > 0, 'positive duplicate count')
            require(type(entry['row']) is list and len(entry['row']) == len(columns), 'row width')
            for v in entry['row']:
                validate_cell(v)
            key = canonical(entry['row'])
            require(previous is None or previous < key, 'sorted distinct multiset rows')
            previous = key
            count += entry['count']
        require(type(t.get('row_count')) is int and t['row_count'] == count and
                type(t.get('distinct_rows')) is int and t['distinct_rows'] == len(t['rows']), 'row counts')
        require(t.get('multiset_sha256') == digest(t['rows']), 'multiset SHA')
        total += count
    semantic = [{k: v for k, v in r.items() if k != 'rootpage'} for r in schema]
    require(value.get('total_rows') == total and type(value.get('total_rows')) is int, 'total rows')
    require(value.get('schema_sha256') == digest(semantic), 'schema SHA')
    require(value.get('logical_sha256') == digest(dict(schema=semantic, tables=tables)), 'logical SHA')
    return value


def diff(before, after):
    before, after = validate(before), validate(after)
    def objects(v):
        return {(r['type'], r['name']): r for r in v['schema']}
    a, b = objects(before), objects(after)
    additions = [b[k] for k in sorted(b.keys() - a.keys())]
    removals = [a[k] for k in sorted(a.keys() - b.keys())]
    changed, physical = [], []
    for k in sorted(a.keys() & b.keys()):
        if {x: y for x, y in a[k].items() if x != 'rootpage'} != {x: y for x, y in b[k].items() if x != 'rootpage'}:
            changed.append(dict(before=a[k], after=b[k]))
        elif a[k]['rootpage'] != b[k]['rootpage']:
            physical.append(dict(before=a[k], after=b[k]))
    at, bt = ({t['name']: t for t in v['tables']} for v in (before, after))
    table_changes = []
    for name in sorted(at.keys() | bt.keys()):
        old, new = at.get(name), bt.get(name)
        ac = {canonical(r['row']): r['count'] for r in old['rows']} if old else {}
        bc = {canonical(r['row']): r['count'] for r in new['rows']} if new else {}
        added = [dict(row=json.loads(k), count=bc[k] - ac.get(k, 0)) for k in sorted(bc) if bc[k] > ac.get(k, 0)]
        removed = [dict(row=json.loads(k), count=ac[k] - bc.get(k, 0)) for k in sorted(ac) if ac[k] > bc.get(k, 0)]
        columns_changed = old is not None and new is not None and old['columns'] != new['columns']
        if old is None or new is None or added or removed or columns_changed:
            table_changes.append(dict(name=name, attribution='unattributed',
                                      table_added=old is None, table_removed=new is None,
                                      columns_changed=columns_changed,
                                      columns_before=None if old is None else old['columns'],
                                      columns_after=None if new is None else new['columns'],
                                      rows_added=added, rows_removed=removed,
                                      mutation_pairing='not inferred; rows are multisets'))
    return dict(format='cjj-sqlite-typed-diff-v1', completed=True,
                before_logical_sha256=before['logical_sha256'], after_logical_sha256=after['logical_sha256'],
                logical_equal=before['logical_sha256'] == after['logical_sha256'],
                schema_added=additions, schema_removed=removals, schema_changed=changed,
                physical_rootpage_changes=physical, table_changes=table_changes,
                classification='unattributed; names and p50 columns do not prove hook identity or benchmark execution',
                actual_bench_execution_proven=False, formal_score_eligible=False)


def read_snapshot(path, limit):
    if Path(path).stat().st_size > limit:
        raise SnapshotError('Input snapshot byte limit exceeded')
    return validate(json.loads(Path(path).read_text()))


def write_new(path, value, limit):
    path = Path(path)
    raw = canonical(value) + b'\n'
    if len(raw) > limit:
        raise SnapshotError('Output byte limit exceeded')
    if path.exists() or path.is_symlink():
        raise SnapshotError('Output must be new: ' + str(path))
    # Hard-link a completed temporary file so existing outputs are never replaced.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.sqlite-snapshot-', delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(raw)
        os.link(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def plan():
    return dict(default='stdlib CPU plan; no database opened', format=FORMAT,
                snapshot='explicit snapshot --db EXISTING --out NEW_JSON',
                diff='offline diff --before SNAPSHOT --after SNAPSHOT --out NEW_JSON',
                limits=asdict(Limits()), source_mutations=False,
                source_pragmas='read-only table_xinfo only',
                wal='no immutable; existing WAL+SHM required, no sidecar creation intended',
                typed_values=['null', 'integer decimal int64', 'real IEEE754 binary64 bits', 'text bytes base64', 'blob bytes base64'],
                duplicate_rows='full-row multiset counts; no rowid or primary-key assumptions',
                tables='all ordinary tables including sqlite internal; tableless/virtual databases fail qualification',
                classification='unattributed until joined to actual hook identity',
                formal_score_eligible=False, GPU_or_framework_imports=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('snapshot')
    p.add_argument('--db', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    for name, default in asdict(Limits()).items():
        p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    p = sub.add_parser('diff')
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-output-bytes', type=int, default=Limits().max_output_bytes)
    args = parser.parse_args()
    if args.command is None:
        print(json.dumps(plan(), sort_keys=True, indent=2))
        return 0
    try:
        if args.out.exists() or args.out.is_symlink():
            raise SnapshotError('Output must be new: ' + str(args.out))
        if args.command == 'snapshot':
            limits = Limits(**{k: getattr(args, k) for k in asdict(Limits())})
            result = snapshot(args.db, limits)
        else:
            if args.max_output_bytes <= 0:
                raise SnapshotError('Output bound must be positive')
            result = diff(read_snapshot(args.before, args.max_output_bytes), read_snapshot(args.after, args.max_output_bytes))
        sha = write_new(args.out, result, args.max_output_bytes)
        print(json.dumps(dict(completed=True, output=str(args.out), sha256=sha, formal_score_eligible=False)))
        return 0
    except (SnapshotError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(json.dumps(dict(completed=False, error_type=type(error).__name__, error=str(error), formal_score_eligible=False)), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
