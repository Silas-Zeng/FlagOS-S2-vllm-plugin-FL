#!/usr/bin/env python3
"""Pure CPU controller contracts. Synthetic logs/Math rows are never service evidence."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def reject(call):
    try:
        call()
    except (RuntimeError, AssertionError, sqlite3.DatabaseError, ValueError):
        return
    raise AssertionError('Expected contract failure')


def run():
    path = ROOT / 'tools/run_biv150_stock_decode_105.py'
    spec = importlib.util.spec_from_file_location('stock105_cpu_only', path)
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    original = ROOT / 'tools/run_biv150_normalization_accuracy.py'
    tree = ast.parse(original.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    command = next(n.value for n in ast.walk(main) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'command' for t in n.targets))
    assert tool.SERVICE_COMMAND[1:] == ast.literal_eval(command)[1:]
    assert Path(tool.SERVICE_COMMAND[0]).name == 'vllm'
    assert hashlib.sha256(tool.protocol_source(ROOT).encode()).hexdigest() == tool.PROTOCOL_SHA
    plan = json.loads(subprocess.check_output([sys.executable, str(path)], text=True))
    assert plan['cpu_only'] and not plan['service_started'] and not plan['gpu_executed'] and not plan['throughput_prepared']
    # CoreX may import as triton without a canonical 'triton' distribution.
    class VendorMetadata:
        @staticmethod
        def packages_distributions(): return {'triton': ['corex-triton']}
        @staticmethod
        def version(name):
            if name == 'triton': raise tool.importlib.metadata.PackageNotFoundError(name)
            assert name == 'corex-triton'
            return 'CPU-vendor-version-fixture'
    versions, provenance = tool.package_metadata(('triton',), metadata=VendorMetadata,
        find_spec=lambda name: SimpleNamespace(origin='/CPU-vendor/triton/__init__.py', submodule_search_locations=None))
    assert versions == {'triton': None}
    assert provenance['distributions']['triton']['requested']['status'] == 'missing'
    assert provenance['distributions']['triton']['providers']['corex-triton']['version'] == 'CPU-vendor-version-fixture'
    assert provenance['modules']['triton'] == dict(origin='/CPU-vendor/triton/__init__.py', search_locations=[], module_executed=False)
    inherited = dict(VLLM_FL_BIV150_PREFILL_MULTI2='0', VLLM_FL_BIV150_PREFILL_Q32='0',
                    VLLM_FL_BIV150_NATIVE_W8A8='0', VLLM_FL_BIV150_GQA_DECODE='0',
                    VLLM_FL_PREFER='wrong', BIV150_NATIVE_W8A8_LIBRARY='wrong.so', FLAGTUNE='1',
                    PYTHONPATH='old-source', KEEP_CPU_FIXTURE='yes')
    untouched = dict(inherited)
    env, cleared = tool.environment(inherited, Path('/source'), Path('/private-out'), Path('/library5c'), Path('/real-cert.json'))
    assert inherited == untouched and env[tool.ENABLE] == '1' and env[tool.CERTIFICATE] == '/real-cert.json'
    assert env['BIV150_NATIVE_W8A8_LIBRARY'] == '/library5c' and env['PYTHONPATH'] == '/source'
    assert env['FLAGGEMS_CACHE_DIR'] == '/private-out/flaggems-cache'
    assert not any(k in env for k in ('VLLM_FL_BIV150_PREFILL_MULTI2', 'VLLM_FL_BIV150_PREFILL_Q32',
                                     'VLLM_FL_BIV150_NATIVE_W8A8', 'VLLM_FL_BIV150_GQA_DECODE', 'VLLM_FL_PREFER', 'FLAGTUNE'))
    assert set(cleared) == set(inherited) - {'PYTHONPATH', 'KEEP_CPU_FIXTURE'}
    # Exercise actual readonly SQLite code with a synthetic DB and test-only expected digest.
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); db = root / 'unit.db'
        with contextlib.closing(sqlite3.connect(db)) as connection:
            connection.execute('CREATE TABLE unit(x INTEGER)'); connection.execute('INSERT INTO unit VALUES(1)')
            connection.commit()
        expected = {db.name: (tool.sha(db), db.stat().st_size)}
        (root / 'manifest.json').write_text(json.dumps(dict(databases=[dict(name=db.name, sha256=tool.sha(db))])))
        before = db.read_bytes(); rows = tool.snapshot_inventory(root, expected)
        assert rows[0]['quick_check'] == 'ok' and db.read_bytes() == before
        reject(lambda: tool.snapshot_inventory(root, {db.name: ('0' * 64, db.stat().st_size)}))
        Path(str(db) + '-wal').touch(); reject(lambda: tool.snapshot_inventory(root, expected)); Path(str(db) + '-wal').unlink()
    # Same PID with a different start time must never be signalled, including cleanup.
    observed = {101: dict(pid=101, start_ticks=1, ppid=10, pgid=101, sid=101, state='R'),
                102: dict(pid=102, start_ticks=2, ppid=101, pgid=101, sid=101, state='R'),
                999: dict(pid=999, start_ticks=3, ppid=10, pgid=999, sid=999, state='R')}
    signals = []
    registry = tool.Owned(scan=lambda: dict(observed), read=lambda pid: observed.get(pid),
                          send=lambda row, sig: signals.append((row['pid'], row['start_ticks'], sig)))
    registry.add_launcher(SimpleNamespace(pid=101)); assert set(registry.members) == {101, 102}
    observed[102] = dict(observed[102], start_ticks=20)
    registry.terminate(signal.SIGTERM)
    assert signals == [(101, 1, signal.SIGTERM)]
    observed[101] = dict(observed[101], state='Z')
    assert not registry.cleanup()['complete']  # Unknown reused birth in same session is reported, never killed.
    expected = dict(schema=1, sha256='CPU-log-fixture-on', native_enabled=True)
    markers = ('BIV150_NATIVE_W8A8_INSTALL', 'BIV150_STABLE_SMALL_O_INSTALL', 'BIV150_SMALL_OUT_LOWERING_INSTALL',
        'BIV150_Q_FRAGMENT_INSTALL', 'BIV150_PREFILL_NORMALIZATION_INSTALL', 'BIV150_GQA_DECODE_INSTALL',
        'BIV150_MIXED_DECODE_INSTALLED', 'BIV150_PREFILL_Q32_INSTALL', 'BIV150_PREFILL_MULTI2_INSTALL', 'BIV150_STOCK_QK8_INSTALL')
    log = 'BIV150_COMPILE_SEMANTICS ' + json.dumps(expected) + '\n'
    log += '\n'.join(m + ' ' + repr(dict(installed=True, large_linear=True)) for m in markers) + '\n'
    selection = 'BIV150_STOCK_QK8_SELECTED ' + repr(dict(host_submissions=1, actual_kernel_hash='CPU-fixture-only', original_fallbacks=2, diagnostic_only=True))
    evidence = tool.runtime_evidence(log + selection, expected)
    assert evidence['selected_host_submissions_lower_bound'] == 1 and evidence['original_generic_fallback_lower_bound'] == 2
    reject(lambda: tool.runtime_evidence(log, expected))
    reject(lambda: tool.runtime_evidence((log + selection).replace("'installed': True", "'installed': False", 1), expected))
    reject(lambda: tool.runtime_evidence(log + selection, dict(expected, sha256='CPU-off')))
    reject(lambda: tool.runtime_evidence(log + selection + '\nBIV150_COMPUTE_ORDER_SCOPE_REJECTED', expected))
    # The exact original accuracy function runs only against this in-memory fake client.
    class FakeEval:
        STDOUT = -2
        def __init__(self, out, count=105, duplicate=False, error=False):
            self.out, self.count, self.duplicate, self.error = out, count, duplicate, error
        def run(self, command, *, stdout, stderr, check, timeout):
            assert command == tool.accuracy_command(self.out) and check and timeout == 5400
            report = self.out / 'math500-level3/unit/reports/minicpm/math_500.json'
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps(dict(metrics=[dict(identity=dict(name='accuracy'), score=1., num=self.count)])))
            predictions = report.parents[2] / 'predictions/minicpm/math_500_Level 3.jsonl'
            predictions.parent.mkdir(parents=True)
            rows = [dict(index=i, metadata=dict(question_id=0 if self.duplicate else i),
                         model_output=dict(choices=[] if self.error and i == 0 else [dict(text='CPU fixture only')])) for i in range(self.count)]
            predictions.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    with tempfile.TemporaryDirectory() as folder:
        for index, kwargs in enumerate(({}, dict(count=104), dict(duplicate=True), dict(error=True))):
            out = Path(folder) / str(index); out.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                if index == 0:
                    tool.official_accuracy(ROOT, out, FakeEval(out, **kwargs))
                    tool.validate_verdict(json.loads((out / 'accuracy-verdict.json').read_text()))
                    inventory = tool.prediction_inventory(out)
                    assert inventory['count'] == inventory['unique_question_ids'] == inventory['unique_indices'] == 105
                    assert not inventory['failed_rows'] and not inventory['parse_errors']
                else:
                    reject(lambda out=out, kwargs=kwargs: tool.official_accuracy(ROOT, out, FakeEval(out, **kwargs)))
    reject(lambda: tool.validate_verdict(dict(score=float('nan'), metric_count=105, predictions=105, unique_questions=105, failed_predictions=[])))
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', tool.FROZEN_PRODUCTION, '--', 'vllm_fl', 'csrc'], check=True)
    assert not {'torch', 'triton', 'numpy', 'vllm', 'evalscope'}.intersection(sys.modules)
    return dict(passed=True, cpu_only=True, official_service_cli_preserved=True, official_accuracy_function_exact=True,
                missing_canonical_vendor_distribution_is_recorded_not_fatal=True,
                default_plan_no_service=True, env_cleanup_then_explicit_flag=True, production_defaults_not_disabled=True,
                sqlite_readonly_and_sha_guards=True, no_unowned_or_reused_pid_signal=True, no_group_signal=True,
                no_selection_or_wrong_on_fingerprint_refused=True, synthetic_105_rows_only=True,
                frozen_867_production_unchanged=True, real_math105_executed=False, service_started=False, gpu_executed=False)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
