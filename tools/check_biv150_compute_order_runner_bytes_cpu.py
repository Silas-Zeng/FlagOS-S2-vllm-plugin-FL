#!/usr/bin/env python3
"""Replay the unchanged cold source loop on explicit CPU module/file fixtures.

This imports neither the full GPU runner nor a framework. SOURCES is empty only
inside the test namespace to isolate the loop's dynamic runner entry; the shipped
SOURCES map and every shipped function must match the immutable a6d parent.
"""
import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'a6d654a7444bba72cbbd48d3d9bfd33172cfd129'
PR109 = 'fc25c86a13fc273bb7368b83af4ca264524b4e12'
RUNNER = 'vllm_fl/worker/model_runner.py'
ORDER = 'vllm_fl/patches/biv150_compute_order.py'
MANIFEST = 'vllm_fl/patches/biv150_konly_compact_sources.json'
NAME = 'vllm_fl.worker.model_runner'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def original(ref, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', ref + ':' + path])


def constant(raw, key):
    return next(ast.literal_eval(n.value) for n in ast.parse(raw).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == key for t in n.targets))


def definitions(raw):
    lines = raw.splitlines(keepends=True)
    return {n.name: b''.join(lines[n.lineno - 1:n.end_lineno]) for n in ast.parse(raw).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    runner = (ROOT / RUNNER).read_bytes()
    order = (ROOT / ORDER).read_bytes()
    old_order = original(PARENT, ORDER)
    expected = constant(order, 'EXPECTED_RUNNER')
    old_expected = constant(old_order, 'EXPECTED_RUNNER')
    assert runner == original(PARENT, RUNNER)
    assert expected == sha(runner) != old_expected
    assert order.count(expected.encode()) == 1
    assert order.replace(expected.encode(), old_expected.encode(), 1) == old_order
    assert definitions(order) == definitions(old_order)  # Includes all of _cold.
    assert constant(order, 'SOURCES') == constant(old_order, 'SOURCES')
    local_sources = []
    for name, pin in constant(order, 'SOURCES').items():
        if name.startswith('vllm_fl.'):
            assert sha((ROOT / (name.replace('.', '/') + '.py')).read_bytes()) == pin
            local_sources.append(name)
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    old_manifest = json.loads(original(PARENT, MANIFEST))
    assert manifest['binding_files'][RUNNER.removeprefix('vllm_fl/')] == expected
    assert manifest['binding_files'][ORDER.removeprefix('vllm_fl/')] == sha(order)
    restored = copy.deepcopy(manifest)
    restored['binding_files'][ORDER.removeprefix('vllm_fl/')] = sha(old_order)
    assert restored == old_manifest
    assert (ROOT / MANIFEST).read_bytes().replace(sha(order).encode(), sha(old_order).encode(), 1) == original(PARENT, MANIFEST)
    cold = next(n for n in ast.parse(order).body if isinstance(n, ast.FunctionDef) and n.name == '_cold')
    assert isinstance(cold.body[0], ast.For)
    loop = copy.deepcopy(cold.body[0])
    # Exact original loop AST, including sys.modules.get, Path(...).read_bytes,
    # hash comparison and original ValueError; no production helper refactor.
    wrapper = ast.parse('def check(runner):\n    pass\n').body[0]
    wrapper.body = [loop]
    namespace = dict(SOURCES={}, EXPECTED_RUNNER=expected, sys=SimpleNamespace(modules={}),
                     hashlib=hashlib, Path=Path)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(ROOT / ORDER), 'exec'), namespace)
    fixture_class = type('ModelRunnerFLMetadataFixture', (), {'__module__': NAME})
    fixture = fixture_class()
    module = ModuleType(NAME)
    module.__file__ = str(ROOT / RUNNER)
    namespace['sys'].modules[NAME] = module
    check = namespace['check']
    assert check(fixture) is None
    negatives = []

    def reject(label):
        try:
            check(fixture)
        except ValueError as exc:
            assert str(exc) == '未自然加载的固定compute-order源：' + NAME
            negatives.append(dict(case=label, error=str(exc)))
        else:
            raise AssertionError('Unexpected source admission: ' + label)

    namespace['EXPECTED_RUNNER'] = old_expected
    reject('original_a6d_stale_pin_rejects_current_runner_bytes')
    namespace['EXPECTED_RUNNER'] = expected
    with tempfile.TemporaryDirectory(prefix='compute-order-source-bytes-CPU-') as tmp:
        cases = {
            'old_PR109_runner': original(PR109, RUNNER),
            'current_runner_with_appended_comment': runner + b'\n# CPU negative fixture\n',
            'only_CRLF_to_LF_changed': runner.replace(b'\r\n', b'\n'),
        }
        assert all(sha(value) != expected for value in cases.values())
        for label, raw in cases.items():
            path = Path(tmp) / (label + '.py')
            path.write_bytes(raw)
            module.__file__ = str(path)
            reject(label)
    module.__file__ = str(ROOT / RUNNER)
    namespace['sys'].modules.pop(NAME)
    reject('module_absent')
    namespace['sys'].modules[NAME] = module
    namespace['EXPECTED_RUNNER'] = None
    reject('pin_absent')
    namespace['EXPECTED_RUNNER'] = expected
    assert check(fixture) is None
    assert not {'torch', 'triton', 'vllm', 'vllm_fl', 'numpy'} & set(sys.modules)
    report = dict(
        schema='compute-order-runner-raw-bytes-CPU-v1', passed=True, parent=PARENT,
        runner_sha256=expected, rejected_stale_pin=old_expected,
        compute_order_sha256=sha(order), compact_manifest_sha256=sha((ROOT / MANIFEST).read_bytes()),
        actual_cold_source_loop_ast_sha256=sha(ast.dump(loop, include_attributes=False).encode()),
        unchanged_complete_function_names=sorted(definitions(order)),
        all_runner_bytes_unchanged=True, all_source_allowlist_entries_unchanged=True,
        local_pinned_modules=local_sources, only_two_production_hash_literals_changed=True,
        source_loop_uses_runtime_read_bytes=True, current_raw_bytes_positive=True, negatives=negatives,
        newline_normalized_runner_sha256=sha(runner.replace(b'\r\n', b'\n')),
        GPU_executed=False, full_GPU_runner_natural_import_verified=False,
        remaining_cold_scope_and_live_closures_executed=False,
        test_scope='Exact original _cold first-loop AST; isolated runner entry via empty SOURCES in test namespace and explicit module/file metadata fixtures. Full live natural-import scope awaits a new service run.',
        precision_gate=False, service_acceptance_passed=False)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(args.out), sha256=sha(args.out.read_bytes()), passed=True)))


if __name__ == '__main__':
    main()
