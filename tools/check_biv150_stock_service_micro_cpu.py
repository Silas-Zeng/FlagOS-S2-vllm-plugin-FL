#!/usr/bin/env python3
"""Default-mode and real-header ABI replay. No framework, kernel compilation or GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from biv150_stock_service_abi import compare

ROOT = Path(__file__).resolve().parents[1]


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Changed typed ABI was accepted')


def run():
    folder = ROOT / 'docs/evidence/stock-qk8-service-diagnostic-20261027'
    dependency = json.loads((folder / 'micro-dependency-provenance.json').read_text())
    assert hashlib.sha256((ROOT / 'tools/biv150_benchmark_utils.py').read_bytes()).hexdigest() == dependency['input_factory_sha256']
    for item in dependency['files']:
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    assert hashlib.sha256((ROOT / 'tools/biv150_stock_service_micro_common.py').read_bytes()).hexdigest() == dependency['common_sha256']
    original = json.loads((folder / 'actual-79-abi-headers.json').read_text())['headers']
    actual = compare(original['baseline'], original['stock'])
    assert actual['typed_abi_match'] and len(actual['ttir_parameters']) == 16 and len(actual['llvm_parameters']) == 18
    mutants = []
    for kind, old, new in (
        ('ttir', '%key_cache_ptr: !tt.ptr<bf16>', '%key_cache_ptr: !tt.ptr<i8>'),
        ('ttir', '%key_cache_ptr:', '%wrong_cache_ptr:'),
        ('ttir', '%k_stride_block: i64', '%k_stride_block: i32'),
        ('llir', 'float %8', 'i64 %8'),
        ('llir', 'i64 %15', 'i32 %15'),
        ('llir', '%17)', '%18)'),
    ):
        changed = copy.deepcopy(original['stock'])
        assert old in changed[kind]; changed[kind] = changed[kind].replace(old, new, 1)
        mutants.append(changed)
    for changed in mutants:
        reject(lambda changed=changed: compare(original['baseline'], changed))
    plan = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'tools/bench_biv150_stock_service_micro.py')], text=True))
    assert plan['cpu_only'] and not plan['certificate_available'] and not plan['timing_prepared']
    assert not any(plan[k] for k in ('torch_imported', 'triton_imported', 'numpy_imported', 'gpu_executed', 'service_installed'))
    expected = [(b, 129) for b in (1, 8, 16, 32, 40, 64)] + [(1, 4096)]
    assert [(c['batch'], c['kv']) for c in plan['cases']] == expected
    assert not {'torch', 'triton', 'numpy'}.intersection(sys.modules)
    return dict(passed=True, cpu_only=True, immutable_reference_modules_verified=len(dependency['files']),
                actual_79_ttir_llvm_entry_headers_replayed=True, typed_abi_negatives=len(mutants),
                default_cli_no_framework=True, cases=plan['cases'], device_micro_verified=False,
                certificate_available=False, service_installed=False, gpu_executed=False)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
