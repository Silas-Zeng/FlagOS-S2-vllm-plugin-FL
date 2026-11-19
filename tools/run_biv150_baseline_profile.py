#!/usr/bin/env python3
"""Owned BI-V150 baseline diagnostics; preserves the accepted workload and warmups.

This driver has no torch/vllm imports. Only main() launches the explicitly owned
service. Profiled results are diagnostic and are never official score evidence.
"""
import argparse
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import types
import urllib.request

MAIN = '0bf9d2efeaa10144dd7f0ff4597298df029f9328'
LIBRARY_SHA256 = '5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
SERVICE_ARGS = [
    'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
    '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85',
    '--max-model-len', '131072', '--profiler-config',
    '{"profiler":"cuda","delay_iterations":0,"max_iterations":0}',
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def proc_identity(pid):
    """PID plus Linux process start ticks guards against PID reuse."""
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), start_ticks=int(fields[19]), pgid=int(fields[2]),
                    sid=int(fields[3]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_process(saved):
    current = proc_identity(saved['pid'])
    return current if current and current['start_ticks'] == saved['start_ticks'] else None


def session_members(sid):
    result = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            member = proc_identity(int(p.name))
            if member and member['sid'] == sid:
                result.append(member)
    return result


def checked_output(command, **kwargs):
    return subprocess.check_output(command, text=True, timeout=30, **kwargs).strip()


def source_identity(args):
    source = args.source.resolve()
    head = checked_output(['git', '-C', str(source), 'rev-parse', 'HEAD'])
    if head != args.commit:
        raise RuntimeError(f'HEAD mismatch: expected {args.commit}, got {head}')
    if checked_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no']):
        raise RuntimeError('Source has tracked modifications')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', MAIN, '--',
                    'vllm_fl', 'csrc'], check=True, timeout=30)
    library = args.library.resolve()
    if sha256(library) != LIBRARY_SHA256:
        raise RuntimeError('Native library does not match the completed control')
    paths = [
        'tools/run_biv150_gate_team_acceptance.py', 'tools/biv150_profile_service.py',
        'tools/biv150_graph_probe.py', 'tools/biv150_profile_scopes.py',
        'tools/biv150_profile_output_guard.py',
        'vllm_fl/worker/worker.py', 'vllm_fl/compile_semantics.py',
        'csrc/iluvatar/native_w8a8.cpp', 'csrc/iluvatar/native_w8a8.cu',
    ]
    return dict(source=str(source), commit=head, production_equivalent_to=MAIN,
                library=str(library), library_sha256=sha256(library),
                source_hashes={p: sha256(source / p) for p in paths},
                driver=str(Path(__file__).resolve()), driver_sha256=sha256(__file__))


def idle_preflight(out):
    with socket.socket() as sock:
        sock.settimeout(2)
        if sock.connect_ex(('127.0.0.1', 9031)) == 0:
            raise RuntimeError('Port 9031 already has a listener; nothing will be stopped')
    result = subprocess.run(['/usr/local/corex/bin/ixsmi'], capture_output=True,
                            text=True, check=True, timeout=30)
    (out / 'idle-ixsmi.txt').write_text(result.stdout + result.stderr)
    if 'No running processes found' not in result.stdout:
        raise RuntimeError('ixsmi does not confirm an idle GPU; nothing will be stopped')


def clean_env(original):
    env = original.copy()
    for key in list(env):
        if key.startswith(('VLLM_FL_BIV150_', 'BIV150_NATIVE_', 'BIV150_PROFILE_')) or key in (
            'USE_C_EXTENSION', 'FLAGTUNE', 'NVTX_SCOPES', 'VLLM_NVTX_SCOPES'
        ):
            env.pop(key)
    return env


def make_environment(args, bootstrap):
    common = clean_env(os.environ)
    common.update({
        'VLLM_PLUGINS': 'fl', 'PYTHONPATH': str(args.source),
        'VLLM_DISABLE_COMPILE_CACHE': '1',
        'TORCHINDUCTOR_CACHE_DIR': str(args.out / 'inductor-cache'),
        'PYTHONUNBUFFERED': '1', 'VLLM_FL_BIV150_LIVE_CARRY_GATE': '0',
        'BIV150_NATIVE_W8A8_LIBRARY': str(args.library),
    })
    service = common.copy()
    service.update({
        'PYTHONPATH': os.pathsep.join(map(str, [bootstrap, args.source, args.source / 'tools'])),
        'BIV150_PROFILE_DIR': str(args.out / 'observations'),
        'BIV150_PROFILE_BOOTSTRAP': '1',
    })
    return service, common


def environment_evidence(env):
    # Explicit runtime whitelist; unrelated credentials never enter the evidence bundle.
    keys = {
        'PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME',
        'PYTHONUNBUFFERED', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'COREX_HOME',
        'COREX_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'USE_FLAGGEMS',
        'VLLM_PLUGINS', 'VLLM_USE_V1', 'VLLM_WORKER_MULTIPROC_METHOD',
        'VLLM_DISABLE_COMPILE_CACHE', 'VLLM_LOGGING_LEVEL',
        'VLLM_ATTENTION_BACKEND', 'VLLM_USE_FLASHINFER_SAMPLER',
        'VLLM_FL_BIV150_LIVE_CARRY_GATE', 'BIV150_NATIVE_W8A8_LIBRARY',
        'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
        'BIV150_PROFILE_DIR', 'BIV150_PROFILE_BOOTSTRAP',
        'NVTX_SCOPES', 'VLLM_NVTX_SCOPES', 'USE_C_EXTENSION', 'FLAGTUNE',
    }
    return {key: env[key] for key in sorted(keys) if key in env}


def make_bootstrap(args):
    bootstrap = args.out / 'bootstrap'
    bootstrap.mkdir()
    (args.out / 'observations').mkdir()
    site_source = '''import importlib.abc, importlib.machinery, os, sys
if os.environ.get("BIV150_PROFILE_DIR") and not os.environ.get("BIV150_PROFILE_BOOTSTRAP"):
    _WORKER_MODULE = "vllm_fl.worker.worker"

    def _wrap_worker(module):
        import functools
        original_init = module.WorkerFL.__init__
        if getattr(original_init, "_biv150_deferred_install", False):
            return

        @functools.wraps(original_init)
        def initialize_worker(self, *args, **kwargs):
            # Executed only when a real WorkerFL is constructed, before its
            # initialization, device/model loading, capture, and step_fn binding.
            import json
            from pathlib import Path
            import biv150_profile_service
            biv150_profile_service.install()
            package = sys.modules["vllm_fl"]
            record = {"pid": os.getpid(), "python": sys.executable,
                      "bootstrap_kind": "worker_init",
                      "instrumentation": biv150_profile_service.__file__,
                      "vllm_fl_origin": getattr(package, "__file__", None),
                      "worker_origin": module.__file__,
                      "native_library": os.environ.get("BIV150_NATIVE_W8A8_LIBRARY")}
            target = Path(os.environ["BIV150_PROFILE_DIR"]) / ("bootstrap-%d.json" % os.getpid())
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(target)
            return original_init(self, *args, **kwargs)

        initialize_worker._biv150_deferred_install = True
        module.WorkerFL.__init__ = initialize_worker

    class _WorkerLoader(importlib.abc.Loader):
        def __init__(self, loader):
            self.loader = loader

        def create_module(self, spec):
            create = getattr(self.loader, "create_module", None)
            return None if create is None else create(spec)

        def exec_module(self, module):
            self.loader.exec_module(module)
            _wrap_worker(module)

        def __getattr__(self, name):
            return getattr(self.loader, name)

    class _WorkerFinder(importlib.abc.MetaPathFinder):
        _biv150_worker_bootstrap = True

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _WORKER_MODULE:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _WorkerLoader(spec.loader)
            return spec

    if not any(getattr(finder, "_biv150_worker_bootstrap", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _WorkerFinder())
'''
    (bootstrap / 'sitecustomize.py').write_text(site_source)
    launcher_source = '''import json, os, sys
from pathlib import Path
try:
    os.setsid()
except PermissionError:
    if os.getsid(0) != os.getpid():
        raise
pid = os.getpid()
if os.getpgrp() != pid or os.getsid(0) != pid:
    raise RuntimeError("service launcher must own its session and group")
fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
record = {"pid": pid, "pgid": os.getpgrp(), "sid": os.getsid(0),
          "start_ticks": int(fields[19]), "exec_command": sys.argv[2:]}
target = Path(sys.argv[1])
temporary = target.with_name(target.name + ".tmp")
temporary.write_text(json.dumps(record, indent=2))
temporary.replace(target)
os.environ.pop("BIV150_PROFILE_BOOTSTRAP", None)
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
'''
    launcher = bootstrap / 'owned_service.py'
    launcher.write_text(launcher_source)
    compile(site_source, str(bootstrap / 'sitecustomize.py'), 'exec')
    compile(launcher_source, str(launcher), 'exec')
    return bootstrap, launcher


def expected_benchmark_command(run_dir, name, input_len, output_len, count, seed):
    return [
        'vllm', 'bench', 'serve', '--backend', 'vllm', '--model', 'minicpm',
        '--tokenizer', '/workspace/MiniCPM5-2B', '--endpoint', '/v1/completions',
        '--host', '127.0.0.1', '--port', '9031', '--dataset-name', 'random',
        '--ignore-eos', '--random-input-len', str(input_len),
        '--random-output-len', str(output_len), '--max-concurrency', '64',
        '--num-prompts', str(count), '--seed', str(seed), '--disable-tqdm',
        '--save-result', '--result-dir', str(run_dir / name), '--label', name,
    ]


def original_benchmark(source, observed_run):
    """Compile only the existing benchmark definition, without importing its module."""
    path = source / 'tools/run_biv150_gate_team_acceptance.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'benchmark']
    if len(matches) != 1:
        raise RuntimeError('Expected one original benchmark()')
    module = ast.Module(body=matches, type_ignores=[])
    proxy = types.SimpleNamespace(**vars(subprocess))
    proxy.run = observed_run
    namespace = {'json': json, 'subprocess': proxy}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['benchmark']


def validate_result(value, input_len, output_len, count):
    expected = dict(completed=count, num_prompts=count, failed=0, max_concurrency=64,
                    total_input_tokens=input_len * count, total_output_tokens=output_len * count)
    differences = {k: {'expected': v, 'actual': value.get(k)} for k, v in expected.items()
                   if value.get(k) != v}
    if differences:
        raise RuntimeError('Workload integrity failure: ' + json.dumps(differences))


def validate_profile_windows(folder, previous):
    paths = sorted(set(folder.glob('worker-*-window-*.json')) - previous)
    errors, records = [], []
    if not paths:
        errors.append('No new worker diagnostic window; HTTP 200 is insufficient')
    for path in paths:
        try:
            value = json.loads(path.read_text())
            scope = value.get('scope_diagnostics', {})
            if value.get('errors') != []:
                errors.append(f'{path.name}: graph probe errors: {value.get("errors")}')
            if scope.get('errors') != [] or scope.get('error_total') != 0 or scope.get('callback_disabled') is not False:
                errors.append(f'{path.name}: scope errors or disabled callbacks: {scope}')
            samples = value.get('samples', [])
            if not samples:
                errors.append(f'{path.name}: no sampled Graph replays; eager-only trace is insufficient')
            for index, sample in enumerate(samples):
                observations = sample.get('observations', [])
                if {x.get('variant') for x in observations} != {'boundary', 'full'} or len(observations) != 2:
                    errors.append(f'{path.name}: sample {index} lacks boundary/full calibration')
                for observation in observations:
                    label = f'{path.name}: sample {index} {observation.get("variant")}'
                    if observation.get('numeric_exact') is not True:
                        errors.append(label + ': output equality not proven')
                    duration = observation.get('graph_total_ms')
                    if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
                        errors.append(label + ': invalid graph event duration')
                    if observation.get('all_api_calls_succeeded') is not True:
                        errors.append(label + ': native event calls not proven successful')
                    if observation.get('variant') == 'full' and not observation.get('kernels'):
                        errors.append(label + ': no per-kernel graph event records')
            records.append(dict(path=str(path), pid=value.get('pid'), epoch=value.get('epoch'),
                                samples=len(samples), scheduled_steps=len(value.get('scheduled_steps', []))))
        except Exception as error:
            errors.append(f'{path.name}: cannot validate: {error!r}')
    return dict(passed=not errors, windows=records, errors=errors)


class Driver:
    def __init__(self, args):
        self.args = args
        self.collector = None
        self.collector_identity = None
        self.owner = None
        self.known_members = {}
        self.state = dict(created=now(), complete=False, diagnostic_only=True,
                          official_score=False, phase='created', steps=[])
        self.current_benchmark = None
        self.server_log = None

    def update(self, phase, **extra):
        event = dict(time=now(), phase=phase, **extra)
        self.state.update(phase=phase, updated=event['time'])
        self.state['steps'].append(event)
        self.remember_service()
        atomic_json(self.args.out / 'state.json', self.state)
        print(json.dumps(event), flush=True)

    def remember_service(self):
        owner_file = self.args.out / 'service-owner.json'
        if self.owner is None and owner_file.exists():
            self.owner = json.loads(owner_file.read_text())
            if self.owner['pid'] != self.owner['pgid'] or self.owner['pid'] != self.owner['sid']:
                raise RuntimeError('Invalid owned service session')
            self.state['service_owner'] = self.owner
        if self.owner and same_process(self.owner):
            for member in session_members(self.owner['sid']):
                self.known_members[member['pid']] = member
            self.state['owned_service_members'] = list(self.known_members.values())

    def start(self):
        identity = source_identity(self.args)
        self.update('source_verified')
        idle_preflight(self.args.out)
        self.update('idle_confirmed')
        bootstrap, launcher = make_bootstrap(self.args)
        self.service_env, self.bench_env = make_environment(self.args, bootstrap)
        command = [
            '/usr/local/corex/bin/ixsys', '-t', 'cuda,nvtx', '--profile-from-start', 'off',
            '--cuda-trace-scope', 'process-tree', '-o', str(self.args.out / 'baseline.ixsys'),
            sys.executable, str(launcher), str(self.args.out / 'service-owner.json'),
            '/usr/local/bin/vllm', *SERVICE_ARGS,
        ]
        atomic_json(self.args.out / 'service-env.json', environment_evidence(self.service_env))
        atomic_json(self.args.out / 'benchmark-env.json', environment_evidence(self.bench_env))
        os.chmod(self.args.out / 'service-env.json', 0o600)
        os.chmod(self.args.out / 'benchmark-env.json', 0o600)
        atomic_json(self.args.out / 'manifest.json', dict(
            **identity, diagnostic_only=True, official_score=False,
            collector_command=command, service_command=['/usr/local/bin/vllm', *SERVICE_ARGS],
            bootstrap_sha256=sha256(bootstrap / 'sitecustomize.py'),
            launcher_sha256=sha256(launcher), python=sys.executable,
            ixsys='/usr/local/corex/bin/ixsys',
            env_files=['service-env.json', 'benchmark-env.json'],
        ))
        self.server_log = (self.args.out / 'server.log').open('w')
        self.collector = subprocess.Popen(command, env=self.service_env,
                                          stdout=self.server_log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
        self.collector_identity = proc_identity(self.collector.pid)
        self.state['collector_owner'] = self.collector_identity
        self.update('collector_started', pid=self.collector.pid)
        deadline = time.monotonic() + self.args.startup_timeout
        next_progress = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.remember_service()
            if time.monotonic() >= next_progress:
                self.update('waiting_service_ready')
                next_progress = time.monotonic() + 30
            if self.collector.poll() is not None:
                raise RuntimeError(f'Collector exited before readiness: {self.collector.returncode}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models', timeout=3) as response:
                    models = json.load(response)
                if self.owner is None or not same_process(self.owner):
                    raise RuntimeError('Ready endpoint without a live owned service identity')
                bootstrap_records = list((self.args.out / 'observations').glob('bootstrap-*.json'))
                if not bootstrap_records:
                    raise RuntimeError('Service ready without instrumentation bootstrap evidence')
                for record in bootstrap_records:
                    origin = json.loads(record.read_text())['vllm_fl_origin']
                    if origin is None or not Path(origin).resolve().is_relative_to(self.args.source):
                        raise RuntimeError(f'Unexpected vllm_fl import origin: {origin}')
                atomic_json(self.args.out / 'models.json', models)
                self.update('service_ready')
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                time.sleep(2)
        raise TimeoutError('Owned service readiness timed out')

    def observed_run(self, command, *args, **kwargs):
        current = self.current_benchmark
        if current is None:
            raise RuntimeError('Unexpected benchmark subprocess')
        original = list(command)
        expected = expected_benchmark_command(self.args.out / 'load', **current['geometry'])
        if original != expected:
            raise RuntimeError('Original benchmark command changed: ' + json.dumps(original))
        actual = original + (['--profile'] if current['profile'] else [])
        folder = Path(actual[actual.index('--result-dir') + 1])
        atomic_json(folder / 'command.json', actual)
        with (self.args.out / 'benchmark-commands.jsonl').open('a') as log:
            log.write(json.dumps(dict(time=now(), original_command=original,
                                      actual_command=actual, profile=current['profile'])) + '\n')
        kwargs['env'] = self.bench_env
        return subprocess.run(actual, *args, **kwargs)

    def stop_profile(self):
        if self.owner is None or not same_process(self.owner):
            return
        try:
            request = urllib.request.Request('http://127.0.0.1:9031/stop_profile',
                                             data=b'', method='POST')
            with urllib.request.urlopen(request, timeout=10) as response:
                self.state['last_stop_profile_http'] = response.status
        except Exception as error:
            self.state['last_stop_profile_error'] = repr(error)

    def workloads(self):
        benchmark = original_benchmark(self.args.source, self.observed_run)
        load = self.args.out / 'load'
        load.mkdir()
        warmups, diagnostics = {}, {}
        for length, count in ((4096, 256), (16384, 128)):
            warm_count = 128 if length == 4096 else 64
            warmups[str(length)] = []
            cases = [(f'warm-{length}-{index}', 128, warm_count, 920000 + index, False)
                     for index in range(3)]
            cases.append((f'diagnostic-{length}', 1024, count, 0, True))
            for name, output_len, num_prompts, seed, profile in cases:
                geometry = dict(name=name, input_len=length, output_len=output_len,
                                count=num_prompts, seed=seed)
                self.current_benchmark = dict(geometry=geometry, profile=profile)
                previous_windows = set((self.args.out / 'observations').glob('worker-*-window-*.json'))
                self.update('benchmark_started', name=name, profile=profile)
                try:
                    result = benchmark(load, **geometry)
                    validate_result(result, length, output_len, num_prompts)
                finally:
                    if profile:
                        self.stop_profile()
                if profile:
                    validation = validate_profile_windows(self.args.out / 'observations', previous_windows)
                    atomic_json(self.args.out / f'{name}-profile-validation.json', validation)
                    diagnostics[str(length)] = result
                    atomic_json(self.args.out / 'diagnostic_results.json', dict(
                        diagnostic_only=True, official_score=False, results=diagnostics))
                    if not validation['passed']:
                        raise RuntimeError('Diagnostic observation failed: ' + json.dumps(validation['errors']))
                else:
                    warmups[str(length)].append(result)
                    atomic_json(self.args.out / 'warmups.json', warmups)
                self.update('benchmark_completed', name=name, profile=profile)
        self.current_benchmark = None
        self.update('workloads_completed')

    def cleanup(self):
        self.remember_service()
        self.stop_profile()
        self.update('stopping_owned_service')
        # Never signal the collector's process group. The target owns a separate session.
        if self.owner and same_process(self.owner):
            current = same_process(self.owner)
            if current and current['pgid'] == self.owner['pid'] and current['sid'] == self.owner['pid']:
                try:
                    os.killpg(self.owner['pgid'], signal.SIGTERM)
                    self.state['service_sigterm_sent'] = True
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + self.args.service_stop_timeout
        while time.monotonic() < deadline:
            self.remember_service()
            living = [v for v in self.known_members.values()
                      if (same_process(v) or {}).get('state') not in (None, 'Z')]
            if not living:
                break
            time.sleep(1)
        # If graceful shutdown left children, signal only identities observed in our session.
        for saved in self.known_members.values():
            current = same_process(saved)
            if current and current['state'] != 'Z' and current['sid'] == self.owner['sid']:
                try:
                    os.kill(saved['pid'], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.update('waiting_collector_flush')
        if self.collector is not None:
            try:
                self.state['collector_returncode'] = self.collector.wait(timeout=self.args.collector_flush_timeout)
            except subprocess.TimeoutExpired:
                self.state['collector_flush_pending'] = True
                self.update('collector_flush_pending', pid=self.collector.pid)
                raise RuntimeError('Collector is still flushing; left alive, see recorded owner identity')
        if self.server_log is not None:
            self.server_log.close()
        traces = [dict(path=str(p), size_bytes=p.stat().st_size)
                  for p in sorted(self.args.out.glob('baseline*')) if p.is_file()]
        self.state['trace_files'] = traces
        self.update('cleanup_completed')
        if self.collector is not None and not any(x['size_bytes'] > 0 for x in traces):
            raise RuntimeError('Collector exited without a nonempty trace artifact')
        accepted_exit_codes = {0}
        if self.state.get('service_sigterm_sent'):
            accepted_exit_codes.add(128 + signal.SIGTERM)
        if self.collector is not None and self.state['collector_returncode'] not in accepted_exit_codes:
            raise RuntimeError(f"Collector exit status {self.state['collector_returncode']}; inspect trace and log")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--startup-timeout', type=int, default=1200)
    parser.add_argument('--service-stop-timeout', type=int, default=90)
    parser.add_argument('--collector-flush-timeout', type=int, default=1800)
    args = parser.parse_args(argv)
    args.source = args.source.resolve()
    args.out = args.out.resolve()
    args.library = args.library.resolve()
    if len(args.commit) != 40 or any(c not in '0123456789abcdef' for c in args.commit):
        parser.error('--commit must be the full lowercase 40-character HEAD')
    if min(args.startup_timeout, args.service_stop_timeout, args.collector_flush_timeout) <= 0:
        parser.error('timeouts must be positive')
    return args


def main(argv=None):
    args = arguments(argv)
    os.umask(0o077)
    args.out.mkdir(parents=False, exist_ok=False)
    driver = Driver(args)
    error = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        driver.start()
        driver.workloads()
    except BaseException as exc:
        error = exc
        driver.state['error'] = repr(exc)
        driver.update('failed')
    finally:
        try:
            driver.cleanup()
        except BaseException as exc:
            driver.state['cleanup_error'] = repr(exc)
            error = error or exc
        driver.state['complete'] = error is None
        driver.update('complete' if error is None else 'failed')
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
