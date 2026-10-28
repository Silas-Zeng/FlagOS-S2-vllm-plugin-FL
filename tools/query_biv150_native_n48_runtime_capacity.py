#!/usr/bin/env python3
"""Optional runtime-capacity diagnostic; default is CPU plan, never an ABBA gate."""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/evidence/native113-runtime-capacity-query-20261029'
CPP = Path(__file__).with_suffix('.cpp')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', action='store_true', help='Explicitly compile host helper, import Torch and make runtime queries')
    parser.add_argument('--reference-library', type=Path, required=True)
    parser.add_argument('--candidate-library', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, default=Path('/usr/local/corex'))
    parser.add_argument('--compiler', type=Path, default=Path('/usr/local/corex/bin/clang++'))
    parser.add_argument('--device', type=int, default=0, help='Require this CURRENT logical device; never switch it')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be a new directory')
    args.out.mkdir(parents=True)
    pins = json.loads((DATA / 'bindings.json').read_text())
    report = dict(schema='native-N48-runtime-capacity-diagnostic-v1', status='CPU_plan', query_requested=args.query,
                  runtime_API_queried=False, runtime_capacity_queried=False, runtime_capacity_observed=False, helper_compiled=False,
                  measured_residency=False, numeric_verified=False, performance_verified=False,
                  blocks_other_validation=False, explicit_kernel_launch_calls=0,
                  explicit_device_allocation_calls=0, attribute_set_calls=0, device_switch_calls=0,
                  pid=os.getpid(), python=sys.executable, platform=platform.platform(),
                  source_sha256=sha(Path(__file__)), helper_source_sha256=sha(CPP),
                  bindings_sha256=sha(DATA / 'bindings.json'), bindings=pins, results=[], errors=[])
    def save():
        (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    try:
        paths = [args.reference_library.resolve(strict=True), args.candidate_library.resolve(strict=True)]
        for path, entry in zip(paths, pins['libraries']):
            if sha(path) != entry['sha256']:
                raise ValueError('Exact library SHA mismatch: ' + str(path))
        helper = (args.out / 'host-runtime-capacity.so').resolve()
        command = [str(args.compiler), '-std=c++17', '-O2', '-shared', '-fPIC', '-D__ILUVATAR__',
                   '-I' + str(args.sdk / 'include'), str(CPP), '-L' + str(args.sdk / 'lib64'),
                   '-lcudart', '-ldl', '-Wl,-rpath,' + str(args.sdk / 'lib64'), '-o', str(helper)]
        report.update(library_paths=[str(x) for x in paths], compile_command=command,
                      requested_threads=64, requested_dynamic_shared_bytes=0,
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        save()
        if not args.query:
            return
        if platform.system() != 'Linux':
            raise RuntimeError('Runtime query requires the target Linux Torch/Corex environment')
        compiler = args.compiler.resolve(strict=True)
        report['compiler'] = dict(path=str(compiler), sha256=sha(compiler))
        header = args.sdk / 'include/cuda_runtime_api.h'
        report['SDK_runtime_header'] = dict(path=str(header.resolve(strict=True)), sha256=sha(header))
        try:
            compiled = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        except subprocess.TimeoutExpired as exc:
            (args.out / 'compile.stdout.txt').write_bytes(exc.stdout or b'')
            (args.out / 'compile.stderr.txt').write_bytes(exc.stderr or b'')
            report.update(compile_returncode=None, compile_timed_out=True)
            raise RuntimeError('SDK host-helper compilation exceeded60s; diagnostic stopped') from exc
        (args.out / 'compile.stdout.txt').write_bytes(compiled.stdout)
        (args.out / 'compile.stderr.txt').write_bytes(compiled.stderr)
        report['compile_returncode'] = compiled.returncode
        if compiled.returncode:
            raise RuntimeError('SDK host-helper compilation unavailable; see original compiler logs')
        report.update(helper_compiled=True, helper_sha256=sha(helper))
        save()
        # Normal registration in this one process; no tensor, CUDA init helper or kernel call.
        import torch
        report['torch'] = dict(version=torch.__version__, file=torch.__file__, file_sha256=sha(Path(torch.__file__)),
                               cuda_initialized_before=torch.cuda.is_initialized())
        report['status'] = 'loading_exact_libraries'
        save()
        for path in paths:
            torch.ops.load_library(str(path))
        for path in paths:
            if str(path) not in torch.ops.loaded_libraries:
                raise RuntimeError('Normal Torch library registration missing: ' + str(path))
        query_helper = ctypes.CDLL(str(helper), mode=os.RTLD_LOCAL | os.RTLD_NOW)
        query = query_helper.cjj_native_capacity_query
        query.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulonglong, ctypes.c_int]
        query.restype = ctypes.c_char_p  # JSON from SDK-compiled structs, no guessed ctypes CUDA ABI.
        for path, entry in zip(paths, pins['libraries']):
            if sha(path) != entry['sha256']:
                raise RuntimeError('Library changed before query: ' + str(path))
            report.update(status='querying_runtime', current_query=entry['label'])
            save()
            value = query(os.fsencode(path), entry['host_stub'].encode(), entry['host_stub_virtual_offset'], args.device)
            if value is None:
                raise RuntimeError('Host helper returned no report')
            result = json.loads(value.decode())
            result.update(label=entry['label'], library_sha256=sha(path), expected_static_shared_bytes=entry['ELF_static_shared_bytes'])
            if result['library_sha256'] != entry['sha256']:
                raise RuntimeError('Library changed after query: ' + str(path))
            attrs = result.get('attributes')
            result['runtime_static_shared_matches_ELF'] = None if attrs is None else attrs['static_shared_bytes'] == entry['ELF_static_shared_bytes']
            report['results'].append(result)
            report['runtime_API_queried'] |= bool(result.get('binding_verified'))
            report['runtime_capacity_queried'] |= bool(result.get('runtime_capacity_queried'))
            save()
        report['current_query'] = None
        report['torch']['cuda_initialized_after'] = torch.cuda.is_initialized()
        report['runtime_capacity_observed'] = all(x.get('status') == 'queried' for x in report['results'])
        report['status'] = 'queried' if report['runtime_capacity_observed'] else 'diagnostic_unavailable'
    except Exception as exc:
        report['status'] = 'diagnostic_unavailable'
        report['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        save()
        print(json.dumps(dict(result=str(args.out / 'result.json'), status=report['status'],
                              runtime_capacity_observed=report['runtime_capacity_observed'], blocks_other_validation=False)))
    # Diagnostic refusal is not permission to bypass numerical gates, nor an ABBA blocker.


if __name__ == '__main__':
    main()
