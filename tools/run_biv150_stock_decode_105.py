#!/usr/bin/env python3
"""Default CPU plan; explicit owned pure-decode Math105 diagnosis, never throughput."""
import argparse
import ast
import contextlib
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PRODUCTION = '867b725c08765fd8775d77de59a466da20bfe155'
NATIVE_SHA = '5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
PROTOCOL_SHA = '27937c1566e4b8680fc4feb2e18d6f12e0a4c157416ac9840fab6cc19b221bd2'
ENABLE = 'VLLM_FL_BIV150_STOCK_QK8_DIAGNOSTIC'
CERTIFICATE = 'VLLM_FL_BIV150_STOCK_QK8_MICRO_CERTIFICATE'
SNAPSHOT = {
    'TunedConfig_iluvatar_triton_3_2.db': ('5764f125233211d8ed8369f0bcdc6a08a58a00c9a3470bfaa36be0af7665ca55', 1605632),
    'TunedConfig_iluvatar_triton_3_6.db': ('2c5aa448065453188f1ff1794d8e929bacb63dcad84a600e086ae05165821314', 15712256),
}
SERVICE_COMMAND = ['/usr/local/bin/vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
    '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}', '--served-model-name', 'minicpm',
    '--gpu-memory-utilization', '0.85', '--max-model-len', '131072']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n'); temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def protocol_source(source):
    text = (source / 'tools/run_biv150_normalization_accuracy.py').read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'accuracy')
    exact = ''.join(text.splitlines(keepends=True)[fn.lineno - 1:fn.end_lineno])
    if hashlib.sha256(exact.encode()).hexdigest() != PROTOCOL_SHA:
        raise RuntimeError('Official Math105 function differs from the reviewed protocol')
    return exact


def accuracy_command(out):
    return ['evalscope', 'eval', '--model', 'minicpm', '--api-url', 'http://127.0.0.1:9031/v1/chat/completions',
        '--api-key', 'EMPTY', '--eval-type', 'openai_api', '--datasets', 'math_500', '--dataset-args',
        '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}',
        '--eval-batch-size', '8', '--timeout', '3600', '--generation-config',
        '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}', '--work-dir', str(out / 'math500-level3'), '--ignore-errors']


def environment(original, source, out, library, certificate):
    env = dict(original)
    cleared = []
    for key in list(env):
        if key.startswith(('VLLM_FL_', 'BIV150_')) or key in (
            'FLAGTUNE', 'USE_C_EXTENSION', 'USE_FLAGGEMS', 'VLLM_BATCH_INVARIANT', 'VLLM_ATTENTION_BACKEND'):
            cleared.append(key); env.pop(key)
    env.update(VLLM_PLUGINS='fl', PYTHONPATH=str(source), PYTHONUNBUFFERED='1',
        VLLM_DISABLE_COMPILE_CACHE='1', TRITON_CACHE_DIR=str(out / 'triton-cache'),
        TORCHINDUCTOR_CACHE_DIR=str(out / 'inductor-cache'), FLAGGEMS_CACHE_DIR=str(out / 'flaggems-cache'),
        BIV150_NATIVE_W8A8_LIBRARY=str(library), **{ENABLE: '1', CERTIFICATE: str(certificate)})
    # M2, large-linear, Q32, multi2, GQA and native enable flags stay unset:
    # the shipping source's production defaults select them.
    return env, sorted(cleared)


def snapshot_inventory(snapshot, expected=SNAPSHOT):
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    rows = manifest['databases']
    if len(rows) != len(expected) or {r['name'] for r in rows} != set(expected):
        raise RuntimeError('The same accepted initial tuning snapshot is required')
    result = []
    for row in rows:
        name = row['name']
        if Path(name).name != name or not name.endswith('.db'):
            raise RuntimeError('Unsafe database name')
        path = snapshot / name
        digest, size = expected[name]
        if row['sha256'] != digest or sha(path) != digest or path.stat().st_size != size:
            raise RuntimeError('Tuning database differs from the accepted initial snapshot')
        if any(Path(str(path) + suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
            raise RuntimeError('Snapshot is not a standalone immutable SQLite backup')
        with contextlib.closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            check = db.execute('PRAGMA quick_check').fetchall()
        if check != [('ok',)]:
            raise RuntimeError('SQLite quick_check failed')
        result.append(dict(name=name, sha256=digest, bytes=size, quick_check='ok'))
    return sorted(result, key=lambda x: x['name'])


def identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=int(pid), start_ticks=int(fields[19]), ppid=int(fields[1]),
                    pgid=int(fields[2]), sid=int(fields[3]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def scan_proc():
    return {int(p.name): row for p in Path('/proc').iterdir() if p.name.isdigit() and (row := identity(int(p.name)))}


def signal_exact(owner, sig):
    current = identity(owner['pid'])
    if not current or current['start_ticks'] != owner['start_ticks'] or current['state'] == 'Z':
        return False
    fd = None
    try:
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            try:
                fd = os.pidfd_open(owner['pid'])
                current = identity(owner['pid'])
                if not current or current['start_ticks'] != owner['start_ticks']:
                    return False
                signal.pidfd_send_signal(fd, sig)
                return True
            except OSError:
                pass  # Kernel/container may not support pidfd; never skip the identity recheck.
        current = identity(owner['pid'])
        if not current or current['start_ticks'] != owner['start_ticks']:
            return False
        os.kill(owner['pid'], sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


class Owned:
    def __init__(self, scan=scan_proc, read=identity, send=signal_exact):
        self.scan, self.read, self.send = scan, read, send
        self.members, self.launchers = {}, []
        self.signal_errors = []

    def add_launcher(self, child):
        owner = self.read(child.pid)
        if not owner or owner['sid'] != child.pid:
            raise RuntimeError('New owned session identity was not captured')
        self.members[child.pid] = owner; self.launchers.append(owner)
        self.remember()
        return owner

    def remember(self):
        observed = self.scan()
        valid = {pid for pid, old in self.members.items() if pid in observed and observed[pid]['start_ticks'] == old['start_ticks']}
        sessions = {observed[pid]['sid'] for pid in valid}
        changed = True
        while changed:
            changed = False
            for pid, row in observed.items():
                if pid in self.members:
                    continue
                if row['ppid'] in valid or row['sid'] in sessions:
                    self.members[pid] = row; valid.add(pid); changed = True

    def alive(self):
        return [current for pid, old in self.members.items() if (current := self.read(pid))
                and current['start_ticks'] == old['start_ticks'] and current['state'] != 'Z']

    def terminate(self, sig):
        self.remember()
        for row in reversed(self.alive()):
            try:
                self.send(row, sig)
            except OSError as error:
                self.signal_errors.append(dict(pid=row['pid'], start_ticks=row['start_ticks'], error=repr(error)))

    def cleanup(self):
        self.terminate(signal.SIGTERM)
        deadline = time.monotonic() + 90
        while self.alive() and time.monotonic() < deadline:
            self.remember(); time.sleep(1)
        if self.alive():
            self.terminate(signal.SIGKILL)
            deadline = time.monotonic() + 30
            while self.alive() and time.monotonic() < deadline:
                time.sleep(1)
        known = {(r['pid'], r['start_ticks']) for r in self.members.values()}
        sessions = {r['sid'] for r in self.launchers}
        unknown = [r for pid, r in self.scan().items() if r['sid'] in sessions and (pid, r['start_ticks']) not in known and r['state'] != 'Z']
        return dict(complete=not self.alive() and not unknown, remaining_owned=self.alive(),
                    unobserved_same_session=unknown, identities=list(self.members.values()),
                    signal_errors=self.signal_errors, pid_reuse_never_signalled=True, group_signal_used=False)


def idle():
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(('127.0.0.1', 9031)) == 0:
            raise RuntimeError('9031 occupied; no unowned process will be stopped')
    text = subprocess.check_output(['/usr/local/corex/bin/ixsmi'], text=True, timeout=15)
    if 'No running processes found' not in text:
        raise RuntimeError('GPU busy; no workload will be started or stopped')
    return text


def package_resolution(env, cwd):
    code = '''import importlib.util,json,os,sys
s=importlib.util.find_spec("vllm_fl")
print(json.dumps(dict(python=sys.executable,sys_path=sys.path,cwd=os.getcwd(),PYTHONPATH=os.environ.get("PYTHONPATH"),origin=s.origin if s else None,package_executed=False)))'''
    return json.loads(subprocess.check_output([sys.executable, '-c', code], env=env, cwd=cwd, text=True, timeout=30))


def package_metadata(names, metadata=importlib.metadata, find_spec=importlib.util.find_spec):
    """Observational vendor metadata; never import modules or alter ABI fingerprints."""
    versions, records = {}, dict(distributions={}, modules={})
    try:
        providers = metadata.packages_distributions()
    except Exception as error:
        providers = {}
        records['provider_lookup_error'] = dict(type=type(error).__name__, message=str(error))
    def lookup(name):
        try:
            return dict(version=metadata.version(name), status='found')
        except importlib.metadata.PackageNotFoundError:
            return dict(version=None, status='missing')
        except Exception as error:
            return dict(version=None, status='lookup_error', error=dict(type=type(error).__name__, message=str(error)))
    for name in names:
        requested = lookup(name)
        versions[name] = requested['version']
        records['distributions'][name] = dict(requested=requested,
            providers={provider: lookup(provider) for provider in providers.get(name, [])})
        try:
            spec = find_spec(name)
            records['modules'][name] = dict(origin=spec.origin if spec else None,
                search_locations=list(spec.submodule_search_locations or []) if spec else [], module_executed=False)
        except Exception as error:
            records['modules'][name] = dict(origin=None, module_executed=False,
                lookup_error=dict(type=type(error).__name__, message=str(error)))
    return versions, records


def abi_probe(env, cwd):
    cpu_env = dict(env, CUDA_VISIBLE_DEVICES='', VLLM_PLUGINS='')
    cpu_env.pop(ENABLE, None); cpu_env.pop(CERTIFICATE, None)
    code = 'import torch,json;print(json.dumps(dict(torch=str(torch.__version__),cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))))'
    return json.loads(subprocess.check_output([sys.executable, '-c', code], env=cpu_env, cwd=cwd, text=True, timeout=60))


def expected_fingerprint(source, env, abi):
    semantics = module(source / 'vllm_fl/compile_semantics.py', 'stock105_expected_semantics')
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(**semantics._GEOMETRY)))
    return semantics.build_fingerprint(config, 'iluvatar', package_root=source / 'vllm_fl', environ=env, abi=abi)


def runtime_evidence(logs, expected):
    for marker in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED', 'BIV150_PREFILL_MULTI2_FALLBACK'):
        if marker in logs:
            raise RuntimeError('Main runtime scope failed: ' + marker)
    fingerprints = [json.loads(line.split('BIV150_COMPILE_SEMANTICS ', 1)[1])
                    for line in logs.splitlines() if 'BIV150_COMPILE_SEMANTICS ' in line]
    if not fingerprints or any(value != expected for value in fingerprints):
        raise RuntimeError('Actual loaded compile semantics is not the diagnostic-on source')
    markers = ('BIV150_NATIVE_W8A8_INSTALL', 'BIV150_STABLE_SMALL_O_INSTALL', 'BIV150_SMALL_OUT_LOWERING_INSTALL',
        'BIV150_Q_FRAGMENT_INSTALL', 'BIV150_PREFILL_NORMALIZATION_INSTALL', 'BIV150_GQA_DECODE_INSTALL',
        'BIV150_MIXED_DECODE_INSTALLED', 'BIV150_PREFILL_Q32_INSTALL', 'BIV150_PREFILL_MULTI2_INSTALL', 'BIV150_STOCK_QK8_INSTALL')
    installations = {}
    for marker in markers:
        values = [ast.literal_eval(line.split(marker + ' ', 1)[1]) for line in logs.splitlines() if marker + ' ' in line]
        if not values or any(value.get('installed') is not True for value in values):
            raise RuntimeError('Production/diagnostic installer missing or declined: ' + marker)
        installations[marker] = values
    if any(not value.get('large_linear') for value in installations['BIV150_SMALL_OUT_LOWERING_INSTALL']):
        raise RuntimeError('Main large-linear default is not active')
    selected = []
    for line in logs.splitlines():
        if 'BIV150_STOCK_QK8_SELECTED ' in line:
            value = ast.literal_eval(line.split('BIV150_STOCK_QK8_SELECTED ', 1)[1])
            if value.get('host_submissions', 0) <= 0 or not value.get('actual_kernel_hash') or value.get('diagnostic_only') is not True:
                raise RuntimeError('Malformed diagnostic kernel selection')
            selected.append(dict(prefix=line.split('BIV150_STOCK_QK8_SELECTED ', 1)[0], event=value, raw=line))
    if not selected:
        raise RuntimeError('No actual worker kernel selection; no valid diagnosis')
    return dict(installations=installations, loaded_compile_semantics=fingerprints, diagnostic_on_bit_verified=True,
        selection_events=selected, selected_host_submissions_lower_bound=max(v['event']['host_submissions'] for v in selected),
        original_generic_fallback_lower_bound=max(v['event'].get('original_fallbacks', 0) for v in selected),
        actual_kernel_hashes=sorted({v['event']['actual_kernel_hash'] for v in selected}), graph_replays_not_counted=True)


def validate_verdict(verdict):
    score = verdict['score']
    complete = verdict['metric_count'] == verdict['predictions'] == verdict['unique_questions'] == 105
    if not complete or verdict['failed_predictions'] or type(score) not in (int, float) or not math.isfinite(score) or not .95 <= score <= 1:
        raise RuntimeError('Math105 failed/incomplete/nonfinite; not a valid qualified diagnostic')


def prediction_inventory(out):
    files = list((out / 'math500-level3').glob('*/predictions/minicpm/math_500_Level 3.jsonl'))
    rows, parse_errors = [], []
    for file in files:
        for number, line in enumerate(file.read_text().splitlines(), 1):
            if not line:
                continue
            try:
                value = json.loads(line); output = value.get('model_output', {})
                choices = output.get('choices') or []
                rows.append(dict(index=value.get('index'), question_id=value.get('metadata', {}).get('question_id'),
                    choices=len(choices), error=output.get('error'), empty_choices=not bool(choices)))
            except (ValueError, TypeError, AttributeError) as error:
                parse_errors.append(dict(file=str(file), line=number, error=repr(error)))
    failed = [row for row in rows if row['error'] or row['empty_choices']]
    return dict(files=[str(p) for p in files], rows=rows, count=len(rows),
        unique_question_ids=len({repr(row['question_id']) for row in rows}),
        unique_indices=len({repr(row['index']) for row in rows}), failed_rows=failed, parse_errors=parse_errors)


class EvalAdapter:
    STDOUT = subprocess.STDOUT

    def __init__(self, out, env, registry):
        self.out, self.env, self.registry = out, env, registry

    def run(self, command, *, stdout, stderr, check, timeout):
        if command != accuracy_command(self.out) or check is not True or timeout != 5400:
            raise RuntimeError('Math105 protocol changed')
        child = subprocess.Popen(command, env=self.env, stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        self.registry.add_launcher(child)
        deadline = time.monotonic() + timeout
        while child.poll() is None:
            self.registry.remember()
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(2)
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
        return subprocess.CompletedProcess(command, child.returncode)


def official_accuracy(source, out, adapter):
    namespace = dict(json=json, subprocess=adapter)
    exec(compile(protocol_source(source), '<pinned-official-Math105-function>', 'exec'), namespace)
    namespace['accuracy'](out)


def run(args):
    source, out = args.source.resolve(), args.out.resolve()
    if out == source or source in out.parents or out == args.tuning_snapshot.resolve() or args.tuning_snapshot.resolve() in out.parents:
        raise RuntimeError('Output/cache must be independent of source and original snapshot')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != args.commit:
        raise RuntimeError('Source HEAD does not match the frozen controller invocation')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Source is not clean')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', FROZEN_PRODUCTION, '--', 'vllm_fl', 'csrc'], check=True)
    protocol_source(source)
    if sha(args.gate_library) != NATIVE_SHA:
        raise RuntimeError('Native library differs from accepted5c')
    tuning = snapshot_inventory(args.tuning_snapshot.resolve())
    hook = module(source / 'vllm_fl/patches/biv150_stock_qk8_diagnostic.py', 'stock105_certificate_contract')
    cert_bytes = args.micro_certificate.read_bytes(); certificate = json.loads(cert_bytes)
    hook.certificate_contract(certificate)
    initial_idle = idle()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'idle-before.txt').write_text(initial_idle)
    private_cert = out / 'micro-certificate.json'; private_cert.write_bytes(cert_bytes)
    target = out / 'flaggems-cache/config_cache'; target.mkdir(parents=True)
    for item in tuning:
        shutil.copy2(args.tuning_snapshot / item['name'], target / item['name'])
        if sha(target / item['name']) != item['sha256']:
            raise RuntimeError('Private initial tuning copy differs')
    original_env = dict(os.environ)
    env, cleared = environment(original_env, source, out, args.gate_library.resolve(), private_cert)
    original_resolution = package_resolution(original_env, Path.cwd())
    installed_env = dict(original_env); installed_env.pop('PYTHONPATH', None)
    installed_resolution = package_resolution(installed_env, out)
    actual_resolution = package_resolution(env, out)
    if not actual_resolution['origin'] or Path(actual_resolution['origin']).resolve() != source / 'vllm_fl/__init__.py':
        raise RuntimeError('Service PYTHONPATH resolves another plugin checkout')
    abi = abi_probe(env, out)
    expected_on = expected_fingerprint(source, env, abi)
    expected_off = expected_fingerprint(source, dict(env, **{ENABLE: '0'}), abi)
    if not expected_on or expected_on == expected_off or not expected_on['native_enabled']:
        raise RuntimeError('Diagnostic on/off or native compile semantics is not separated')
    paths = subprocess.check_output(['git', '-C', str(source), 'ls-files', '--', 'vllm_fl', 'csrc'], text=True).splitlines()
    source_hashes = {p: sha(source / p) for p in paths}
    packages, package_provenance = package_metadata(('torch', 'vllm', 'flag_gems', 'triton', 'evalscope'))
    manifest = dict(command=SERVICE_COMMAND, accuracy_command=accuracy_command(out), protocol_function_sha256=PROTOCOL_SHA,
        source=str(source), source_commit=args.commit, frozen_production=FROZEN_PRODUCTION, source_hashes=source_hashes,
        controller_sha256=sha(__file__), native_library_sha256=NATIVE_SHA, tuning_snapshot=tuning,
        tuning_manifest_sha256=sha(args.tuning_snapshot / 'manifest.json'), micro_certificate_sha256=sha(private_cert),
        original_package_resolution=original_resolution, actual_package_resolution=actual_resolution,
        installed_package_resolution_without_PYTHONPATH=installed_resolution,
        original_PYTHONPATH=original_env.get('PYTHONPATH'), actual_PYTHONPATH=env['PYTHONPATH'], cleared_switch_names=cleared,
        env={k: env.get(k) for k in ('VLLM_PLUGINS', 'PYTHONPATH', 'PYTHONUNBUFFERED',
            'VLLM_DISABLE_COMPILE_CACHE', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
            'FLAGGEMS_CACHE_DIR', 'BIV150_NATIVE_W8A8_LIBRARY', ENABLE, CERTIFICATE)},
        expected_compile_semantics_on=expected_on, expected_compile_semantics_off=expected_off, abi=abi,
        production_defaults_preserved=True, diagnostic_only=True, full_prefill_quantization=False,
        packages=packages, package_metadata_provenance=package_provenance)
    atomic(out / 'manifest.json', manifest)
    state = dict(completed=False, passed=False, valid_pure_decode_diagnostic=False, precision_gate=False,
                 diagnostic_only=True, full_prefill_precision_passed=False, encoded_cache_precision_passed=False,
                 throughput_measured=False, cleanup_complete=False, errors=[], owner=identity(os.getpid()))
    registry = Owned(); server = None; log = None
    def save(phase, **values):
        state.update(phase=phase, updated_utc=now(), **values); atomic(out / 'state.json', state)
    try:
        idle()  # Recheck immediately before spawn after all file/ABI metadata work.
        log = (out / 'server.log').open('x')
        server = subprocess.Popen(SERVICE_COMMAND, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        owner = registry.add_launcher(server)
        save('waiting_service', server_owner=owner)
        deadline = time.monotonic() + 1200
        while True:
            registry.remember()
            if server.poll() is not None:
                raise RuntimeError('Owned server exited before ready')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models', timeout=3) as response:
                    json.load(response)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Service readiness deadline exceeded')
                time.sleep(3)
        logs = (out / 'server.log').read_text()
        manifest['runtime_before_eval'] = runtime_evidence(logs, expected_on)
        manifest['owned_initial_environment'] = {}
        for pid, row in registry.members.items():
            if (current := identity(pid)) and current['start_ticks'] == row['start_ticks']:
                try:
                    pairs = Path('/proc', str(pid), 'environ').read_bytes().split(b'\0')
                    values = dict(v.decode(errors='replace').split('=', 1) for v in pairs if b'=' in v)
                    manifest['owned_initial_environment'][str(pid)] = {k: values.get(k) for k in ('PYTHONPATH', 'VLLM_PLUGINS', ENABLE, CERTIFICATE, 'BIV150_NATIVE_W8A8_LIBRARY')}
                except (FileNotFoundError, PermissionError):
                    pass
        atomic(out / 'manifest.json', manifest)
        evaluation_offset = (out / 'server.log').stat().st_size
        save('Math105', actual_installation_and_selection=True)
        client_env = dict(env, CUDA_VISIBLE_DEVICES='', VLLM_PLUGINS='')
        client_env.pop(ENABLE, None); client_env.pop(CERTIFICATE, None)
        official_accuracy(source, out, EvalAdapter(out, client_env, registry))
        verdict = json.loads((out / 'accuracy-verdict.json').read_text()); validate_verdict(verdict)
        after = (out / 'server.log').read_text()
        manifest['runtime_after_eval'] = runtime_evidence(after, expected_on)
        with (out / 'server.log').open('rb') as stream:
            stream.seek(evaluation_offset)
            manifest['evaluation_window_selection_lines'] = [line for line in stream.read().decode(errors='replace').splitlines() if 'BIV150_STOCK_QK8_SELECTED ' in line]
        manifest['evaluation_window_counts_scope'] = 'Host events only; FULL_DECODE_ONLY Graph replay does not execute Python counters'
        if sha(args.gate_library) != NATIVE_SHA or snapshot_inventory(args.tuning_snapshot.resolve()) != tuning:
            raise RuntimeError('Original library/snapshot changed during run')
        if any(sha(source / p) != digest for p, digest in source_hashes.items()) or sha(private_cert) != manifest['micro_certificate_sha256']:
            raise RuntimeError('Source or private certificate changed during run')
        atomic(out / 'manifest.json', manifest)
        save('Math105_complete', accuracy=verdict, pure_decode_math105_gate_passed=True,
             protocol_rows_complete=True, actual_kernel_selection=True)
    except BaseException as error:
        state['errors'].append(dict(type=type(error).__name__, message=str(error)))
        (out / 'error-traceback.txt').write_text(traceback.format_exc())
        if (out / 'accuracy-verdict.json').exists():
            state['accuracy'] = json.loads((out / 'accuracy-verdict.json').read_text())
        save('failed')
    finally:
        cleanup = registry.cleanup()
        if server is not None:
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cleanup['complete'] = False
        if log is not None:
            log.close()
        atomic(out / 'owned-cleanup.json', cleanup)
        if (out / 'server.log').exists():
            final_lines = (out / 'server.log').read_text().splitlines()
            try:
                manifest['final_worker_counts'] = [dict(prefix=line.split('BIV150_STOCK_QK8_FINAL ', 1)[0],
                    counts=ast.literal_eval(line.split('BIV150_STOCK_QK8_FINAL ', 1)[1]))
                    for line in final_lines if 'BIV150_STOCK_QK8_FINAL ' in line]
                manifest['counter_scope'] = 'Final host/capture counts if emitted; SELECTED events are lower bounds; Graph replay is not counted'
                atomic(out / 'manifest.json', manifest)
            except (ValueError, SyntaxError) as error:
                state['errors'].append(dict(type='CounterParseFailure', message=repr(error)))
        inventory = prediction_inventory(out)
        atomic(out / 'prediction-inventory.json', inventory)
        state['prediction_rows'] = dict(count=inventory['count'], unique_questions=inventory['unique_question_ids'],
            unique_indices=inventory['unique_indices'], failed=inventory['failed_rows'], parse_errors=inventory['parse_errors'])
        if state.get('pure_decode_math105_gate_passed') and not (
            len(inventory['files']) == 1 and inventory['count'] == inventory['unique_question_ids'] == inventory['unique_indices'] == 105
            and not inventory['failed_rows'] and not inventory['parse_errors']):
            state['errors'].append(dict(type='PredictionClosureFailure', message='Final standard105 rows/errors do not close'))
        state['cleanup_complete'] = cleanup['complete']
        if not cleanup['complete']:
            state['errors'].append(dict(type='CleanupIncomplete', message='Owned identities remain; no unowned PID was signalled'))
        valid = not state['errors'] and bool(state.get('pure_decode_math105_gate_passed')) and cleanup['complete']
        save('completed' if valid else 'failed', completed=valid, passed=valid,
             valid_pure_decode_diagnostic=valid, precision_gate=False)
    if not state['valid_pure_decode_diagnostic']:
        raise RuntimeError('Pure-decode105 diagnostic failed; see immutable raw/state/errors')
    return state


def plan():
    return dict(cpu_only=True, service_command=SERVICE_COMMAND, accuracy_command=accuracy_command(Path('FRESH_OUT')),
        required_inputs=['source', 'commit', 'out', 'gate-library', 'tuning-snapshot', 'micro-certificate'],
        run_requires_explicit_flag='--run-service', defaults_preserved=['M2', 'Q32', 'multi2', 'native5c'],
        timing_prepared=False, throughput_prepared=False, pure_decode_only=True, full_prefill_precision_passed=False,
        service_started=False, framework_imported=False, gpu_executed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-service', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT); parser.add_argument('--commit')
    for name in ('out', 'gate-library', 'tuning-snapshot', 'micro-certificate'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.run_service and any(getattr(args, name) is None for name in ('commit', 'out', 'gate_library', 'tuning_snapshot', 'micro_certificate')):
        parser.error('Explicit service mode requires every frozen input path and commit')
    print(json.dumps(run(args) if args.run_service else plan(), indent=2))
