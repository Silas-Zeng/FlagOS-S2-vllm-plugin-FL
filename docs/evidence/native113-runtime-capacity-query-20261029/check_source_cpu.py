#!/usr/bin/env python3
"""Finite static/CPU-plan checks; does not compile SDK code or load either SO."""
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = Path(__file__).resolve().parent
PYTHON = ROOT / 'tools/query_biv150_native_n48_runtime_capacity.py'
CPP = PYTHON.with_suffix('.cpp')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = json.loads((D / 'bindings.json').read_text())
ref = Path('/private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_s4_probe.so')
new = Path('/private/tmp/cjj-native113-n48-shared6k-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_shared6k_probe.so')
assert [sha(ref), sha(new)] == [x['sha256'] for x in pins['libraries']]
assert all('ILj4EE' in x['host_stub'] and '__device_stub__' in x['host_stub'] and x['host_stub_virtual_offset'] == 100800 for x in pins['libraries'])
compile(ast.parse(PYTHON.read_bytes()), str(PYTHON), 'exec')
cpp = CPP.read_text()
api_calls = sorted(re.findall(r'\b(cuda[A-Za-z]+)\s*\(', cpp))
assert api_calls == sorted(['cudaGetDevice', 'cudaGetDeviceProperties', 'cudaRuntimeGetVersion', 'cudaDriverGetVersion', 'cudaFuncGetAttributes', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor', 'cudaGetErrorString', 'cudaGetErrorString'])
assert 'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, stub, 64, 0)' in cpp
assert all(x in cpp for x in ('RTLD_NOLOAD', 'dlsym(handle, symbol)', 'dladdr(stub, &info)', 'info.dli_saddr != stub', 'offset != expected_offset', 'realpath(info.dli_fname'))
assert not re.search(r'\b(?:cuOccupancy\w*|cudaSet\w*|cudaMalloc\w*|cudaLaunch\w*|ixLaunch\w*|stub)\s*\(|<<<|__global__', cpp)
with tempfile.TemporaryDirectory(prefix='native113-runtime-query-plan-CPU-') as tmp:
    reports = []
    for label, candidate in [('valid', new), ('wrong_SO', ref)]:
        out = Path(tmp) / label
        process = subprocess.run([sys.executable, '-B', str(PYTHON), '--reference-library', str(ref), '--candidate-library', str(candidate), '--out', str(out)], capture_output=True, check=True)
        result = json.loads((out / 'result.json').read_text())
        assert result['query_requested'] is False and result['runtime_API_queried'] is False
        assert result['runtime_capacity_queried'] is False and result['runtime_capacity_observed'] is False
        assert result['helper_compiled'] is False and result['blocks_other_validation'] is False
        assert not (out / 'host-runtime-capacity.so').exists()
        assert result['status'] == ('CPU_plan' if label == 'valid' else 'diagnostic_unavailable')
        if label == 'wrong_SO':assert 'Exact library SHA mismatch' in result['errors'][0]['message']
        reports.append(dict(case=label, status=result['status'], library_hashes_checked=True, no_runtime_query=True))
assert not {'torch', 'triton', 'vllm', 'numpy'} & set(sys.modules)
report = dict(CPU_static_checks_passed=True, default_and_negative_plans=reports,
              runtime_API_queried=False, runtime_capacity_observed=False, SDK_helper_compiled=False,
              source_sha256=sha(PYTHON), helper_source_sha256=sha(CPP), bindings_sha256=sha(D/'bindings.json'),
              allowed_runtime_API_calls=api_calls, full_SDK_ABI_verified=False, GPU_numeric_verified=False,
              source_limits='Static source + normal CPU plan only. Actual SDK compilation, dlsym/dladdr and runtime capacity remain pending.')
print(json.dumps(report, indent=2))
