"""Independent K-lookahead: unchanged v4/v5 protocols, fixed5898 identity and private Graph bootstrap."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import threading
import biv150_native118_service_bridge as mixed_bridge

ENABLE = 'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE'
HOIST_ENABLE = 'VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'
HELPER_SHA = '94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
NATIVE_SHA = '5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
THROUGHPUT_FUNCTION_SHA = 'e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d'


class OwnedBenchmarkAdapter:
    STDOUT = subprocess.STDOUT

    def __init__(self, out, env, registry):
        self.out, self.env, self.registry = out, env, registry

    def run(self, command, *, stdout, stderr, check, timeout):
        label = command[-1]
        warm = re.fullmatch(r'warm-(4096|16384)-([0-3])', label)
        score = re.fullmatch(r'(?:score|full-warm)-(4096|16384)', label)
        if warm:
            length, index = map(int, warm.groups()); output, count, seed = 128, 64, 920000 + index
        elif score:
            length = int(score.group(1)); output, count, seed = 1024, 256 if length == 4096 else 128, 0
        else:
            raise RuntimeError('Unknown shipping warmup/score label')
        exact = ['vllm', 'bench', 'serve', '--backend', 'vllm', '--model', 'minicpm',
                 '--tokenizer', '/workspace/MiniCPM5-2B', '--endpoint', '/v1/completions',
                 '--host', '127.0.0.1', '--port', '9031', '--dataset-name', 'random', '--ignore-eos',
                 '--random-input-len', str(length), '--random-output-len', str(output),
                 '--max-concurrency', '64', '--num-prompts', str(count), '--seed', str(seed),
                 '--disable-tqdm', '--save-result', '--result-dir', str(self.out / label), '--label', label]
        if command != exact or check is not True or timeout != 2400 or stderr != subprocess.STDOUT:
            raise RuntimeError('Shipping warmup/score benchmark parameters changed')
        child = subprocess.Popen(command, env=self.env, cwd=self.out, stdout=stdout, stderr=stderr,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        self.registry.add_launcher(child)
        deadline = time.monotonic() + timeout
        while child.poll() is None:
            self.registry.remember()
            if time.monotonic() >= deadline: raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(2)
        if child.returncode: raise subprocess.CalledProcessError(child.returncode, command)
        return subprocess.CompletedProcess(command, child.returncode)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded_parent_evidence(logs, expected, *, native_gqa_sha=None):
    for marker in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED', 'BIV150_PREFILL_MULTI2_FALLBACK'):
        if marker in logs:
            raise RuntimeError('Existing production scope/fallback gate: ' + marker)
    parent_installations = {}
    for marker in ('BIV150_NATIVE_W8A8_INSTALL', 'BIV150_STABLE_SMALL_O_INSTALL',
                   'BIV150_SMALL_OUT_LOWERING_INSTALL', 'BIV150_Q_FRAGMENT_INSTALL',
                   'BIV150_PREFILL_NORMALIZATION_INSTALL', 'BIV150_GQA_DECODE_INSTALL',
                   'BIV150_MIXED_DECODE_INSTALLED', 'BIV150_PREFILL_Q32_INSTALL',
                   'BIV150_PREFILL_MULTI2_INSTALL'):
        values = [ast.literal_eval(line.split(marker + ' ', 1)[1]) for line in logs.splitlines() if marker + ' ' in line]
        if not values or any(value.get('installed') is not True for value in values):
            raise RuntimeError('Existing production installer declined: ' + marker)
        parent_installations[marker] = values
    if any(not value.get('large_linear') for value in parent_installations['BIV150_SMALL_OUT_LOWERING_INSTALL']):
        raise RuntimeError('Existing large-linear production default is inactive')
    fps = [json.loads(line.split('BIV150_COMPILE_SEMANTICS ', 1)[1]) for line in logs.splitlines()
           if 'BIV150_COMPILE_SEMANTICS ' in line]
    if not fps or any(fp != expected for fp in fps):
        raise RuntimeError('Actual worker compile fingerprint differs from the reviewed ON source')
    native_gqa = None
    if native_gqa_sha is not None:
        installs = [json.loads(line.split('BIV150_NATIVE_GQA_INSTALL ', 1)[1]) for line in logs.splitlines()
                    if 'BIV150_NATIVE_GQA_INSTALL ' in line]
        if not installs or any(x.get('installed') is not True or x.get('library_sha256') != native_gqa_sha
                               for x in installs):
            raise RuntimeError('New main NativeGQA installer/library differs or is absent')
        events = []
        for marker in ('BIV150_NATIVE_GQA_SELECTED', 'BIV150_NATIVE_GQA_CAPTURED'):
            for line in logs.splitlines():
                if marker + ' ' in line:
                    value = json.loads(line.split(marker + ' ', 1)[1])
                    if value.get('library_sha256') != native_gqa_sha:
                        raise RuntimeError('Actual NativeGQA selection uses another library')
                    events.append(dict(marker=marker, **value))
        native_gqa = dict(installs=installs, events=events, request_frequency_proven=False)
    if 'konly_encoded' not in expected:
        if 'BIV150_KONLY_ENCODED_SELECTED ' in logs:
            raise RuntimeError('Encoded kernel observed in the stock main control')
        return dict(actual_compile_semantics=fps, preserved_parent_installations=parent_installations,
                    stock_main_control=True, native_gqa=native_gqa)
    installs = [ast.literal_eval(line.split('BIV150_KONLY_ENCODED_INSTALL ', 1)[1]) for line in logs.splitlines()
                if 'BIV150_KONLY_ENCODED_INSTALL ' in line]
    digest = expected['konly_encoded']['implementation_sha256']
    if not installs or any(i.get('installed') is not True or i.get('implementation_sha256') != digest for i in installs):
        raise RuntimeError('Actual encoded installer is absent, declined or has another implementation')
    events = [json.loads(line.split('BIV150_KONLY_ENCODED_SELECTED ', 1)[1]) for line in logs.splitlines()
              if 'BIV150_KONLY_ENCODED_SELECTED ' in line]
    if not events or any(not e.get('hash') or e.get('implementation_sha256') != digest for e in events):
        raise RuntimeError('Actual private kernel selection/hash is missing or inconsistent')
    profiles = {e['profile'] for e in events}
    if 'writer83' not in profiles or not profiles.difference({'writer83'}):
        raise RuntimeError('Need both encoded writer and an encoded attention reader')
    if expected['konly_encoded'].get('q32_hoist_enabled'):
        if not any(e.get('profile') == 'q32' and e.get('I8_fragment_hoist_verified') is True for e in events):
            raise RuntimeError('Requested fresh Q32 hoist has no actually selected verified kernel')
    return dict(actual_compile_semantics=fps, actual_installs=installs,
                preserved_parent_installations=parent_installations, selected=events,
                host_capture_events_only=True, graph_replays_counted=False, native_gqa=native_gqa)


def validate_score(result, input_length, output_length, count):
    """Independent result arithmetic; the shipping benchmark remains unchanged."""
    if result.get('completed') != count:
        raise RuntimeError('Incomplete measured requests')
    for key in ('duration', 'total_token_throughput', 'output_throughput', 'mean_ttft_ms', 'mean_tpot_ms'):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError('Invalid measured metric: ' + key)
    inputs, outputs = result.get('total_input_tokens'), result.get('total_output_tokens')
    if type(inputs) is not int or type(outputs) is not int or inputs <= 0 or outputs != count * output_length:
        raise RuntimeError('Invalid token totals or incomplete ignore-EOS output')
    # Do not infer tokenizer input length from the command alone: preserve totals
    # and check request-level lengths separately when supplied by this CLI build.
    details = result.get('input_lens')
    if details is not None and (len(details) != count or sum(details) != inputs):
        raise RuntimeError('Input request lengths do not close the total')
    seconds = result['duration']
    if not math.isclose(result['total_token_throughput'], (inputs + outputs) / seconds, rel_tol=1e-6):
        raise RuntimeError('Total tokens/s differs from saved tokens/duration')
    if not math.isclose(result['output_throughput'], outputs / seconds, rel_tol=1e-6):
        raise RuntimeError('Output tokens/s differs from saved tokens/duration')
    return dict(result_arithmetic_verified=True, requested_input_length=input_length,
                recorded_input_tokens=inputs, complete_ignore_eos_output=True,
                request_level_input_lengths_saved=details is not None)


def private_cache_state(out):
    """Observe metadata/selected SQLite rows between requests; never change caches."""
    metadata = {str(p.relative_to(out / 'triton-cache')): sha(p)
                for p in sorted((out / 'triton-cache').rglob('*.json'))
                if not p.name.startswith('__grp__')}
    tuning = {}
    for p in sorted((out / 'flaggems-cache/config_cache').glob('*.db')):
        connection = sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                     if 'benchmark' not in row[0].lower() and not row[0].startswith('sqlite_')]
            tables = {}
            for name in names:
                quoted = '"' + name.replace('"', '""') + '"'
                rows = connection.execute('SELECT * FROM ' + quoted).fetchall()
                tables[name] = sorted(json.dumps(row, default=str, sort_keys=True) for row in rows)
            tuning[p.name] = hashlib.sha256(json.dumps(tables, sort_keys=True).encode()).hexdigest()
        finally:
            connection.close()
    return dict(triton_json_sha256=metadata, selected_tuning_rows_sha256=tuning,
                limitation='Metadata/config observations do not bound every compile or tuning cost.')


def warmup_plan_metadata(a):
    """An explicit scheduling choice, never evidence of sufficient/stable warmth."""
    name=getattr(a,'warmup_plan','legacy')
    if name not in ('legacy','full-only'):raise ValueError('Unknown warmup plan')
    full=getattr(a,'full_workload_warmup',False)
    if name=='full-only' and (not getattr(a,'throughput',False) or not full
            or getattr(a,'graph_coverage_bootstrap_dir',None) is not None):
        raise ValueError('full-only requires --throughput + --full-workload-warmup and rejects Graph diagnosis')
    return dict(name=name,explicit_opt_in=name=='full-only',
      optional_short_warmups_requested=0 if name=='full-only' else (3 if full else 'original TPOT-stability loop'),
      complete_full_workload_warmups_per_case=1 if full else 0,score_runs_per_case=1,
      complete_warm_request_finite_arithmetic_required=True,warm_formal_TTFT_or_TP_gate_applied=False,
      formal_score_gates_unchanged=True,stability_guaranteed_by_plan=False,
      skipped_calls_are_not_faked_success=True,official_score_function_sha256=THROUGHPUT_FUNCTION_SHA,
      parent_controller_sha256='4a35a01a38ddd7015dde533a24caed3ed914086f65a506c96c305e7c31e498aa')


def full_workload_only_throughput(out, benchmark, selected_case):
    """One complete unscored full workload then one score; no stability claim."""
    cases = ((4096, 256), (16384, 128))
    if selected_case != 'both':
        cases = tuple(pair for pair in cases if str(pair[0]) == selected_case)
    if not cases:
        raise RuntimeError('Unknown full-workload diagnostic case')
    scores, warmups = {}, {}
    for length, count in cases:
        history = []
        value = benchmark(out, f'full-warm-{length}', length, 1024, count, 0)
        validate_score(value, length, 1024, count)
        history.append(dict(value, full_workload=True, scored=False))
        warmups[str(length)] = history
        (out / 'warmups.json').write_text(json.dumps(warmups, indent=2) + '\n')
        before = private_cache_state(out)
        (out / f'score-{length}-cache-before.json').write_text(json.dumps(before, indent=2) + '\n')
        scores[str(length)] = benchmark(out, f'score-{length}', length, 1024, count, 0)
        validate_score(scores[str(length)], length, 1024, count)
        after = private_cache_state(out)
        report = dict(before=before, after=after,
                      new_metadata=sorted(after['triton_json_sha256'].keys() - before['triton_json_sha256'].keys()),
                      changed_metadata=sorted(k for k in after['triton_json_sha256'].keys() & before['triton_json_sha256'].keys()
                                              if after['triton_json_sha256'][k] != before['triton_json_sha256'][k]),
                      selected_tuning_rows_changed=before['selected_tuning_rows_sha256'] != after['selected_tuning_rows_sha256'],
                      scope='Complete benchmark including client preflight; not isolated timed-scoring cost.')
        (out / f'score-{length}-cache-observation.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    return scores



def full_geometry_throughput(out, benchmark, selected_case):
    """Match the prior full-warm acceptance geometry, then one diagnostic score."""
    cases = ((4096, 256), (16384, 128))
    if selected_case != 'both':
        cases = tuple(pair for pair in cases if str(pair[0]) == selected_case)
    if not cases:
        raise RuntimeError('Unknown full-workload diagnostic case')
    scores, warmups = {}, {}
    for length, count in cases:
        history = []
        for index in range(3):
            value = benchmark(out, f'warm-{length}-{index}', length, 128, 64, 920000 + index)
            validate_score(value, length, 128, 64)
            history.append(value)
        value = benchmark(out, f'full-warm-{length}', length, 1024, count, 0)
        validate_score(value, length, 1024, count)
        history.append(dict(value, full_workload=True, scored=False))
        warmups[str(length)] = history
        (out / 'warmups.json').write_text(json.dumps(warmups, indent=2) + '\n')
        before = private_cache_state(out)
        (out / f'score-{length}-cache-before.json').write_text(json.dumps(before, indent=2) + '\n')
        scores[str(length)] = benchmark(out, f'score-{length}', length, 1024, count, 0)
        validate_score(scores[str(length)], length, 1024, count)
        after = private_cache_state(out)
        report = dict(before=before, after=after,
                      new_metadata=sorted(after['triton_json_sha256'].keys() - before['triton_json_sha256'].keys()),
                      changed_metadata=sorted(k for k in after['triton_json_sha256'].keys() & before['triton_json_sha256'].keys()
                                              if after['triton_json_sha256'][k] != before['triton_json_sha256'][k]),
                      selected_tuning_rows_changed=before['selected_tuning_rows_sha256'] != after['selected_tuning_rows_sha256'],
                      scope='Complete benchmark including client preflight; not isolated timed-scoring cost.')
        (out / f'score-{length}-cache-observation.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    return scores


def graceful_api_shutdown(server, owner, *, timeout=20, read_start=None):
    """Let this owned API close its engine before the scoped force-cleanup pass."""
    if server is None or server.poll() is not None:
        return dict(attempted=False, reason='API already terminal or not started')
    if read_start is None:
        def read_start(pid):
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
            return int(fields[19])
    if owner.get('pid') != server.pid or read_start(server.pid) != owner.get('start_ticks'):
        raise RuntimeError('Owned API identity changed before graceful shutdown')
    server.send_signal(signal.SIGTERM)
    try:
        code = server.wait(timeout=timeout)
        return dict(attempted=True, identity_verified=True, signal='SIGTERM',
                    API_exited=True, returncode=code, force_cleanup_follows=True)
    except subprocess.TimeoutExpired:
        return dict(attempted=True, identity_verified=True, signal='SIGTERM',
                    API_exited=False, timeout_seconds=timeout, force_cleanup_follows=True)



PARENT_CONTROLLER_SHA='6e2dca1b401b219946c59e294e456317abe0cdf0274cdd1d765d0b41afaba85e'
K8_ENABLE='VLLM_FL_BIV150_KONLY_NATIVE_DECODE'
K8_PATH='VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY'
K8_HASH='VLLM_FL_BIV150_KONLY_NATIVE_LIBRARY_SHA256'
K8_SHA='5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'
K8_OPERATOR='cjj_native_k8_klookahead_probe::produce'
K8_SCHEMA='cjj_native_k8_klookahead_probe::produce(Tensor query, Tensor key_code, Tensor scale_full, Tensor value_offset, Tensor lengths, Tensor table, Tensor(a!) partial, Tensor(b!) maximum, Tensor(c!) summation, float scale_log2e, int segments) -> ()'
K8_MODULE='vllm_fl/patches/biv150_konly_encoded_native.py'



GRAPH_FLAG='CJJ_NATIVE_GRAPH_COVERAGE'
GRAPH_DIR='CJJ_NATIVE_GRAPH_BOOTSTRAP_DIR'
GRAPH_OUT='CJJ_NATIVE_GRAPH_COVERAGE_OUT'
GRAPH_MANIFEST_SHA='8d4330a7a202153765e3d30554137f9ef8d6f30dae9ca32199eb5aeabece70cc'


def clear_graph_environment(original):
    env=dict(original);directory=env.pop(GRAPH_DIR,None);env.pop(GRAPH_FLAG,None);env.pop(GRAPH_OUT,None)
    if directory is not None:
        env['PYTHONPATH']=os.pathsep.join(x for x in env.get('PYTHONPATH','').split(os.pathsep) if x and Path(x).resolve()!=Path(directory).resolve())
    return env


def graph_bootstrap_environment(original,directory,out):
    env=clear_graph_environment(original)
    if directory is None:return env,dict(enabled=False,formal_score_eligible=False,quality_covered=False)
    directory=Path(directory)
    if not directory.is_absolute() or env.get(K8_ENABLE)!='1' or env.get(ENABLE)!='1':raise RuntimeError('Graph coverage requires explicit Native/encoded candidate and absolute bootstrap')
    directory=directory.resolve();manifest=directory/'bootstrap-sources.json'
    if sha(manifest)!=GRAPH_MANIFEST_SHA:raise RuntimeError('Graph bootstrap manifest differs from CPU-reviewed freeze')
    value=json.loads(manifest.read_text())
    if set(value['files'])!={'sitecustomize.py','cjj_native_graph_coverage_bootstrap.py','cjj_native_graph_coverage_probe_20261028.py'}:raise RuntimeError('Graph bootstrap source set')
    if {x.name for x in directory.iterdir()}!=set(value['files'])|{'bootstrap-sources.json'}:raise RuntimeError('Graph bootstrap needs a clean source-only directory; no stale pyc')
    for name,digest in value['files'].items():
        if Path(name).name!=name or sha(directory/name)!=digest:raise RuntimeError('Graph bootstrap source changed')
    env.update({GRAPH_FLAG:'1',GRAPH_DIR:str(directory),GRAPH_OUT:str(out/'native-graph-coverage'),
                'PYTHONPATH':str(directory)+os.pathsep+env.get('PYTHONPATH',''),'PYTHONDONTWRITEBYTECODE':'1'})
    return env,dict(enabled=True,path=str(directory),manifest_sha256=GRAPH_MANIFEST_SHA,source_hashes=value['files'],formal_score_eligible=False,quality_covered=False,actual_worker_install_and_capture_order_pending=True)


def native_env(original,library=None):
    env=mixed_bridge.clear_environment(clear_graph_environment(original))
    for name in (K8_ENABLE,K8_PATH,K8_HASH):env.pop(name,None)
    env[K8_ENABLE]='0'
    if library is not None:
        path=Path(library)
        if not path.is_absolute() or env.get(ENABLE)!='1' or env.get('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','0')=='1':
            raise RuntimeError('Native needs explicit absolute library, encoded ON, compact OFF')
        env.update({K8_ENABLE:'1',K8_PATH:str(path.resolve()),K8_HASH:K8_SHA})
    return env


def torch_identity_probe(env,out):
    off=native_env(dict(env,**{ENABLE:'0',HOIST_ENABLE:'0'}));off.update(CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS='')
    code='import torch,json,hashlib;from pathlib import Path;p=Path(torch.__file__).resolve();print(json.dumps(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),version=str(torch.__version__),cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),cuda_initialized=bool(torch.cuda.is_initialized()))))'
    value=json.loads(subprocess.check_output([sys.executable,'-c',code],env=off,cwd=out,text=True,timeout=60))
    if value['version']!='2.10.0' or value['cxx11_abi'] is not False or value['cuda_initialized'] is not False:
        raise RuntimeError('Native Torch identity/CPU ABI preflight differs')
    return value


def native_events(logs,library):
    result=[]
    for marker,capture in (('BIV150_KONLY_NATIVE_DECODE_SELECTED',False),('BIV150_KONLY_NATIVE_DECODE_CAPTURED',True)):
        for line in logs.splitlines():
            if marker+' ' not in line:continue
            e=json.loads(line.split(marker+' ',1)[1]);n,seg=e.get('shape'),e.get('segments')
            if type(n) is not int or type(seg) is not int or (n,seg) not in ((64,2),(32,4)):
                raise RuntimeError('Native event shape/segments')
            if e.get('library')!=library or e.get('operator')!=K8_OPERATOR or e.get('capture') is not capture:
                raise RuntimeError('Native event library/operator/capture mismatch')
            if e.get('format')!='cjj-konly-rn32-rne-v2' or e.get('physical_page_bytes')!=16384:
                raise RuntimeError('Native event physical cache format')
            if type(e.get('entry_generation')) is not int or e['entry_generation']<=0 or type(e.get('cache_pages')) is not int or not 0<e['cache_pages']<=2**31-1:
                raise RuntimeError('Native event generation/pages')
            if e.get('native_launch_returned') is not True or e.get('host_or_capture_event_not_Graph_replay_count') is not True or e.get('service_numeric_verified') is not False:
                raise RuntimeError('Native event execution/claim boundary')
            def geom(name,dtype,tail,strides):
                x=e.get(name,{})
                if x.get('dtype')!=dtype or x.get('device')!='cuda:0' or type(x.get('offset')) is not int or x['offset']<0:
                    raise RuntimeError('Native '+name+' dtype/device/offset')
                shape=x.get('shape');pitch=x.get('strides')
                if not isinstance(shape,list) or not all(type(v) is int and v>0 for v in shape):raise RuntimeError('Native '+name+' shape')
                if name=='table':
                    if len(shape)!=2 or shape[0]!=n or not 0<shape[1]<=8192:raise RuntimeError('Native table extent')
                elif name=='query':
                    if shape!=[n,16,128]:raise RuntimeError('Native Q extent')
                elif len(shape)!=len(tail)+1 or shape[0]<n or shape[1:]!=tail:raise RuntimeError('Native scratch extent')
                if pitch not in strides:raise RuntimeError('Native '+name+' strides')
            geom('query','torch.bfloat16',[16,128],[[2048,128,1],[2560,128,1]])
            geom('table','torch.int32',[],[[8192,1]])
            geom('partial','torch.float32',[16,seg,128],[[32768,2048,128,1]])
            for key in ('maximum','summation'):geom(key,'torch.float32',[16,seg],[[256,16,1]])
            qualified='large_fullstep' if (n,seg,e['query']['strides'][0]) in ((64,2,2048),(32,4,2560)) else 'small_consumer_stride_only'
            if e.get('component_qualification')!=qualified:raise RuntimeError('Native component scope promoted')
            result.append(dict(marker=marker,event=e))
    return result


def evidence(logs,expected,*,native_gqa_sha=None,native_library=None,native_identity=None):
    result=encoded_parent_evidence(logs,expected,native_gqa_sha=native_gqa_sha)
    config=expected.get('konly_encoded',{}).get('native_decode',{})
    native_markers=('BIV150_KONLY_NATIVE_DECODE_INSTALL','BIV150_KONLY_NATIVE_DECODE_SELECTED','BIV150_KONLY_NATIVE_DECODE_CAPTURED')
    if native_library is None:
        if config.get('enabled') or any(marker+' ' in logs for marker in native_markers):raise RuntimeError('Native appeared in OFF/control mode')
        result['native_decode']=dict(enabled=False,quality_covered=False);return result
    library=dict(path=str(Path(native_library).resolve()),sha256=K8_SHA)
    if config.get('enabled') is not True or config.get('library_sha256')!=K8_SHA or config.get('operator')!=K8_OPERATOR:
        raise RuntimeError('Native compile semantics missing/wrong')
    encoded=result['actual_installs']
    if any(e.get('native_decode_enabled') is not True or e.get('native_decode_library')!=library for e in encoded):
        raise RuntimeError('Encoded installer did not bind requested native library')
    installs=[json.loads(line.split(native_markers[0]+' ',1)[1]) for line in logs.splitlines() if native_markers[0]+' ' in line]
    if not installs:raise RuntimeError('Native runtime installer absent')
    for x in installs:
        if any(x.get(k) is not True for k in ('installed','enabled','loaded_library_confirmed')) or x.get('runtime_compile') is not False:
            raise RuntimeError('Native actual load/install declined or runtime compiled')
        if x.get('library')!=library or x.get('operator')!=K8_OPERATOR or x.get('schema')!=K8_SCHEMA:
            raise RuntimeError('Native runtime library/operator/schema mismatch')
        if x.get('format')!='cjj-konly-rn32-rne-v2' or x.get('physical_page_bytes')!=16384:
            raise RuntimeError('Native installer cache format')
        if native_identity is None or Path(x.get('module_file','')).resolve()!=Path(native_identity['module_file']).resolve() or x.get('module_sha256')!=native_identity['module_sha256']:
            raise RuntimeError('Native runtime source/module differs')
        if Path(x.get('torch_file','')).resolve()!=Path(native_identity['torch']['file']).resolve():raise RuntimeError('Native actual Torch import differs')
    events=native_events(logs,library)
    if not events:raise RuntimeError('Native no actual selected/captured ABI evidence')
    result['native_decode']=dict(enabled=True,installations=installs,events=events,actual_imports_verified=True,
      host_or_capture_only=True,actual_request_frequency_proven=False,Graph_replay_observed=False,quality_covered=False)
    return result


def concurrent_burst64(request_fn=None):
    if request_fn is None:
        def request_fn(index,payload):
            request=urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=240) as response:return json.load(response)
    gate=threading.Barrier(64)
    def worker(index):
        payload=dict(model='minicpm',messages=[dict(role='user',content=f'编号{index}。请按顺序继续列出自然数：1 2 3 4 5 6 7 8。')],max_tokens=16,temperature=0,top_p=1,ignore_eos=True)
        gate.wait(timeout=30);value=request_fn(index,payload)
        if len(value.get('choices',[]))!=1 or value.get('usage',{}).get('completion_tokens')!=16:
            raise RuntimeError('Burst64 incomplete ignore-EOS request')
        return dict(index=index,request=payload,response=value)
    with ThreadPoolExecutor(max_workers=64) as pool:rows=list(pool.map(worker,range(64)))
    if len(rows)!=64 or {r['index'] for r in rows}!=set(range(64)):raise RuntimeError('Burst64 identity/count mismatch')
    return dict(completed=64,concurrency_requested=64,records=rows,diagnostic_only=True,
      actual_native_bucket_proven=False,actual_request_frequency_proven=False,quality_covered=False)


def runtime_window(log,offset,out,label,library):
    with log.open('rb') as stream:stream.seek(offset);raw=stream.read()
    (out/(label+'-runtime-window.log')).write_bytes(raw)
    return dict(log_byte_offset=offset,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
      native_events=[] if library is None else native_events(raw.decode(),dict(path=str(Path(library).resolve()),sha256=K8_SHA)),
      capture_or_host_events_only=True,Graph_replays_counted=False,quality_covered=False)




def supplemental64_command(helper,parent_out,supplemental_out):
    original=helper.accuracy_command(Path(parent_out));command=helper.accuracy_command(Path(supplemental_out))
    parallel=command.index('--eval-batch-size')+1;work=command.index('--work-dir')+1
    if command[parallel]!='8' or original[parallel]!='8':raise RuntimeError('Original official parallelism changed')
    command[parallel]='64'
    differences=[i for i,(a,b) in enumerate(zip(original,command)) if a!=b]
    if len(original)!=len(command) or differences!=sorted((parallel,work)) or command[work]!=str(Path(supplemental_out)/'math500-level3'):
        raise RuntimeError('Supplemental changed more than parallel/output directory')
    return dict(original_official_command=original,actual_supplemental_command=command,changed_indices=differences,
      official_parallel8_unchanged=True,supplemental_parallel64=True,diagnostic_only=True,quality_covered=False)


class SupplementalMath64Adapter:
    STDOUT=subprocess.STDOUT
    def __init__(self,helper,parent_out,out,env,registry):
        self.helper,self.parent_out,self.out,self.env,self.registry=helper,parent_out,out,env,registry
    def run(self,command,*,stdout,stderr,check,timeout):
        if command!=self.helper.accuracy_command(self.out) or check is not True or timeout!=5400:
            raise RuntimeError('Supplemental helper template/protocol differs')
        proof=supplemental64_command(self.helper,self.parent_out,self.out)
        self.helper.atomic(self.out/'supplemental-command-adaptation.json',proof)
        folder=self.out/'math500-level3'
        (folder/'helper-original-command.json').write_text(json.dumps(command,indent=2)+'\n')
        actual=proof['actual_supplemental_command']
        (folder/'command.json').write_text(json.dumps(actual,indent=2)+'\n')
        child=subprocess.Popen(actual,env=self.env,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,start_new_session=True)
        self.registry.add_launcher(child);deadline=time.monotonic()+timeout
        while child.poll() is None:
            self.registry.remember()
            if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(actual,timeout)
            time.sleep(2)
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,actual)
        return subprocess.CompletedProcess(actual,child.returncode)


def run(a):
    mixed_bridge.options(a)
    warm_plan=warmup_plan_metadata(a)
    if sha(a.helper) != HELPER_SHA:
        raise RuntimeError('The previously verified stdlib owner/protocol helper differs')
    spec = importlib.util.spec_from_file_location('encoded_smoke_owned_helpers', a.helper)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    source, out, snapshot = a.source.resolve(), a.out.resolve(), a.tuning_snapshot.resolve()
    if out.exists() or source == out or source in out.parents or snapshot == out or snapshot in out.parents:
        raise RuntimeError('Need a fresh independent output')
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != a.commit:
        raise RuntimeError('Source HEAD differs')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Source is not clean')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', a.base, '--', 'csrc'], check=True)
    m.protocol_source(source)
    if sha(a.gate_library) != NATIVE_SHA:
        raise RuntimeError('Native 5c library differs')
    if a.native_gqa_library is not None and sha(a.native_gqa_library) != a.native_gqa_sha256:
        raise RuntimeError('Pinned new NativeGQA library differs')
    if a.native_k8_library is not None and sha(a.native_k8_library)!=K8_SHA:raise RuntimeError('Fixed existing native K8 library changed')
    bridge=mixed_bridge.validate(source,a.commit,a.mixed_source_bridge,a.mixed_source_bridge_sha256,a.mixed_native_library)
    tool=source/'tools/biv150_native118_service_bridge.py'
    if Path(mixed_bridge.__file__).resolve()!=tool or tool.read_bytes()!=subprocess.check_output(['git','-C',str(source),'show',a.commit+':tools/biv150_native118_service_bridge.py']):raise RuntimeError('Mixed bridge tool differs from frozen source')
    entry=source/'tools/run_biv150_native118_service.py'
    if Path(__file__).resolve()!=entry or entry.read_bytes()!=subprocess.check_output(['git','-C',str(source),'show',a.commit+':tools/run_biv150_native118_service.py']):raise RuntimeError('Mixed controller is not frozen source entry')
    bridge_tool_sha=sha(tool)
    tuning = m.snapshot_inventory(snapshot)
    before_idle = m.idle()
    out.mkdir(parents=True)
    (out / 'idle-before.txt').write_text(before_idle)
    target = out / 'flaggems-cache/config_cache'; target.mkdir(parents=True)
    for item in tuning:
        shutil.copy2(snapshot / item['name'], target / item['name'])
        assert sha(target / item['name']) == item['sha256']
    env = dict(os.environ)
    cleared = []
    for key in list(env):
        if key.startswith(('VLLM_FL_', 'BIV150_')) or key in ('FLAGTUNE', 'USE_C_EXTENSION', 'USE_FLAGGEMS', 'VLLM_BATCH_INVARIANT', 'VLLM_ATTENTION_BACKEND'):
            cleared.append(key); env.pop(key)
    env.update(VLLM_PLUGINS='fl', PYTHONPATH=str(source), PYTHONUNBUFFERED='1',
               VLLM_DISABLE_COMPILE_CACHE='1', TRITON_CACHE_DIR=str(out / 'triton-cache'),
               TORCHINDUCTOR_CACHE_DIR=str(out / 'inductor-cache'), FLAGGEMS_CACHE_DIR=str(out / 'flaggems-cache'),
               BIV150_NATIVE_W8A8_LIBRARY=str(a.gate_library.resolve()), **{ENABLE: '0' if a.baseline else '1'})
    if a.native_gqa_library is not None:
        env['BIV150_NATIVE_GQA_LIBRARY'] = str(a.native_gqa_library.resolve())
    if a.q32_hoist:
        env[HOIST_ENABLE] = '1'
    env=native_env(env,a.native_k8_library)
    env=mixed_bridge.service_environment(env,a.mixed_native_library)
    env,graph_bootstrap=graph_bootstrap_environment(env,a.graph_coverage_bootstrap_dir,out)
    resolution = m.package_resolution(env, out)
    assert Path(resolution['origin']).resolve() == source / 'vllm_fl/__init__.py'
    off_env=native_env(dict(env,**{ENABLE:'0',HOIST_ENABLE:'0'}))
    abi = m.abi_probe(off_env, out)
    native_identity=None
    if a.native_k8_library is not None:
        native_identity=dict(module_file=str(source/K8_MODULE),module_sha256=sha(source/K8_MODULE),torch=torch_identity_probe(off_env,out))
    expected = m.expected_fingerprint(source, env, abi)
    off = m.expected_fingerprint(source,off_env,abi)
    mixed_bridge.fingerprint(expected)
    if a.native_k8_library is not None:
        config=expected.get('konly_encoded',{}).get('native_decode',{})
        if config.get('enabled') is not True or config.get('library_sha256')!=K8_SHA or config.get('operator')!=K8_OPERATOR:raise RuntimeError('Reviewed Native fingerprint is not ON')
    if off.get('konly_encoded') or off.get('features',{}).get(K8_ENABLE):raise RuntimeError('OFF fingerprint retained native/encoded')
    assert expected.get('native_enabled')
    if a.baseline:
        assert expected == off and not expected.get('konly_encoded')
    else:
        assert expected != off and expected.get('konly_encoded')
    paths = subprocess.check_output(['git', '-C', str(source), 'ls-files', '--', 'vllm_fl', 'csrc'], text=True).splitlines()
    hashes = {name: sha(source / name) for name in paths}
    manifest = dict(source_commit=a.commit, source=str(source), source_hashes=hashes,
                    helper_sha256=HELPER_SHA, controller_sha256=sha(__file__), native_sha256=NATIVE_SHA,
                    tuning_snapshot=tuning, expected_compile_semantics=expected, off_compile_semantics=off,
                    command=m.SERVICE_COMMAND, actual_resolution=resolution, abi=abi,
                    cleared_switch_names=sorted(cleared), private_enabled=not a.baseline,
                    official_service_parameters_unchanged=True, smoke_only=not (a.math105 or a.supplemental_native_math105_64 or a.throughput),
                    throughput_diagnostic_requested=a.throughput,
                    full_workload_warmup=a.full_workload_warmup, throughput_case=a.throughput_case,
                    warmup_plan=a.warmup_plan, warmup_plan_metadata=warm_plan,
                    fresh_q32_hoist_requested=a.q32_hoist,
                    native_gqa_library=None if a.native_gqa_library is None else str(a.native_gqa_library.resolve()),
                    native_gqa_library_sha256=a.native_gqa_sha256,
                    full_acceptance=False,native_decode_requested=a.native_k8_library is not None,
                    native_decode_library=None if a.native_k8_library is None else dict(path=str(a.native_k8_library.resolve()),sha256=K8_SHA),native_identity=native_identity,graph_coverage_bootstrap=graph_bootstrap,
                    official_Math105_parallel=8,supplemental_Math105_parallel64_prepared=True,supplemental_Math105_parallel64_requested=a.supplemental_native_math105_64,native_quality_covered=False)
    manifest['mixed_native']=dict(library=dict(path=str(a.mixed_native_library.resolve()),sha256=mixed_bridge.MIXED),
        source_bridge_sha256=a.mixed_source_bridge_sha256,bridge_tool_sha256=bridge_tool_sha,
        qualification=mixed_bridge.QUALIFICATION,production=bridge['production'],service_verified=False)
    shutil.copy2(a.mixed_source_bridge,out/'root-source-bridge.json')
    shutil.copy2(__file__, out / 'executed-controller.py')
    state = dict(completed=False, smoke_passed=False, model_precision_verified=False,
                 throughput_measured=False, cleanup_complete=False, errors=[],native_quality_covered=False,native_model_precision_verified=False)
    m.atomic(out / 'manifest.json', manifest)
    owned = m.Owned(); server = None; log = None
    def save(phase):
        state.update(phase=phase, updated_utc=m.now()); m.atomic(out / 'state.json', state)
    try:
        m.idle()
        log = (out / 'server.log').open('x')
        server = subprocess.Popen(m.SERVICE_COMMAND, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        state['server_owner'] = owned.add_launcher(server); save('waiting_service')
        deadline = time.monotonic() + 1200
        while True:
            owned.remember()
            if server.poll() is not None:
                raise RuntimeError('Owned server exited before readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:9031/v1/models', timeout=3) as response:
                    json.load(response)
                break
            except Exception:
                if time.monotonic() >= deadline: raise TimeoutError('Service readiness deadline')
                time.sleep(3)
        smoke_start=(out/'server.log').stat().st_size
        save('short_request_smoke')
        results = []
        prompts = ['请计算1+1，简短回答。', '请计算1+1，简短回答。', '数列：' + '1 2 3 4 5 ' * 512 + '。请回答最后一个数。']
        for prompt in prompts:
            payload = dict(model='minicpm', messages=[dict(role='user', content=prompt)],
                           max_tokens=16, temperature=0, top_p=1)
            request = urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=180) as response:
                value = json.load(response)
            if len(value.get('choices', [])) != 1 or value.get('usage', {}).get('completion_tokens', 0) <= 0:
                raise RuntimeError('Short diagnostic request produced no generated tokens')
            results.append(dict(request=payload, response=value))
            m.atomic(out / 'smoke-requests.json', results); owned.remember()
        offset=(out/'server.log').stat().st_size
        burst=concurrent_burst64();m.atomic(out/'native-burst64-requests.json',burst);owned.remember()
        manifest['native_burst64']=dict(completed=burst['completed'],diagnostic_only=True,quality_covered=False)
        manifest['native_burst64_window']=runtime_window(out/'server.log',offset,out,'native-burst64',a.native_k8_library)
        if not (a.math105 or a.supplemental_native_math105_64 or a.throughput):
            save('mixed_overlap_smoke')
            mixed_smoke=mixed_bridge.mixed_smoke_requests();m.atomic(out/'mixed-overlap-smoke-requests.json',mixed_smoke);owned.remember()
        manifest['mixed_smoke_window']=mixed_bridge.smoke_window(out/'server.log',smoke_start,(out/'server.log').stat().st_size)
        manifest['actual_runtime'] = evidence((out / 'server.log').read_text(), expected, native_gqa_sha=a.native_gqa_sha256,native_library=a.native_k8_library,native_identity=native_identity)
        m.atomic(out / 'manifest.json', manifest)
        state['smoke_passed'] = True; save('smoke_complete')
        if a.math105:
            client = native_env(dict(env,CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS=''));client.pop(ENABLE,None)
            save('Math105')
            m.official_accuracy(source, out, m.EvalAdapter(out, client, owned))
            verdict = json.loads((out / 'accuracy-verdict.json').read_text()); m.validate_verdict(verdict)
            inventory = m.prediction_inventory(out); m.atomic(out / 'prediction-inventory.json', inventory)
            assert inventory['count'] == inventory['unique_question_ids'] == inventory['unique_indices'] == 105
            assert not inventory['failed_rows'] and not inventory['parse_errors']
            state.update(accuracy=verdict, model_precision_verified=True)
            manifest['actual_runtime_after_math'] = evidence((out / 'server.log').read_text(), expected, native_gqa_sha=a.native_gqa_sha256,native_library=a.native_k8_library,native_identity=native_identity)
        if a.supplemental_native_math105_64:
            supplemental=out/'supplemental-native-math105-parallel64';supplemental.mkdir()
            client=native_env(dict(env,CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS=''));client.pop(ENABLE,None)
            offset=(out/'server.log').stat().st_size;save('supplemental_native_Math105_parallel64')
            try:
                m.official_accuracy(source,supplemental,SupplementalMath64Adapter(m,out,supplemental,client,owned))
                verdict=json.loads((supplemental/'accuracy-verdict.json').read_text());m.validate_verdict(verdict)
                inventory=m.prediction_inventory(supplemental);m.atomic(supplemental/'prediction-inventory.json',inventory)
                if inventory['count']!=105 or inventory['unique_question_ids']!=105 or inventory['unique_indices']!=105 or inventory['failed_rows'] or inventory['parse_errors']:
                    raise RuntimeError('Supplemental105 identity/error closure failed')
                state['supplemental_native_math105_parallel64']=dict(metric_gate_passed=True,verdict=verdict,diagnostic_only=True,quality_covered=False,not_official_Math105=True)
                manifest['actual_runtime_after_supplemental64']=evidence((out/'server.log').read_text(),expected,native_gqa_sha=a.native_gqa_sha256,native_library=a.native_k8_library,native_identity=native_identity)
            finally:
                manifest['supplemental64_request_window']=runtime_window(out/'server.log',offset,out,'supplemental-native-math105-parallel64',a.native_k8_library)
                if (supplemental/'accuracy-verdict.json').exists():state['supplemental64_raw_verdict']=json.loads((supplemental/'accuracy-verdict.json').read_text())
                m.atomic(out/'manifest.json',manifest)
        if a.throughput:
            # Reuse the reviewed, shipping benchmark functions byte-for-byte:
            # concurrency64, 4096/16384 input, output1024, prompts256/128, seed0;
            # Default: short output128 warmup until stable TPOT. Explicit full mode
            # uses three short warmups + one full geometry warmup + one score.
            text = (source / 'tools/run_biv150_normalization_accuracy.py').read_text()
            tree = ast.parse(text)
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('benchmark', 'throughput')]
            assert [n.name for n in functions] == ['benchmark', 'throughput']
            exact = '\n'.join(''.join(text.splitlines(keepends=True)[n.lineno - 1:n.end_lineno]) for n in functions)
            if hashlib.sha256(exact.encode()).hexdigest() != THROUGHPUT_FUNCTION_SHA:
                raise RuntimeError('Reviewed shipping throughput/warmup functions changed')
            manifest['throughput_protocol_function_sha256'] = hashlib.sha256(exact.encode()).hexdigest()
            manifest['throughput_protocol_source'] = exact
            manifest['throughput_protocol_execution'] = (
                'pinned benchmark function with explicit complete full warm + 1 score; no optional short warm calls'
                if a.warmup_plan == 'full-only' else (
                'pinned benchmark function with explicit 3 short + 1 full warm + 1 score control flow'
                if a.full_workload_warmup else 'pinned original throughput function'))
            m.atomic(out / 'manifest.json', manifest)
            client = native_env(dict(env,CUDA_VISIBLE_DEVICES='',VLLM_PLUGINS=''));client.pop(ENABLE,None)
            namespace = dict(json=json, subprocess=OwnedBenchmarkAdapter(out, client, owned))
            exec(compile(exact, '<pinned-shipping-warm-one-round-throughput>', 'exec'), namespace)
            save('warmup_and_throughput_diagnostic')
            if a.full_workload_warmup:
                warm_function=full_workload_only_throughput if a.warmup_plan=='full-only' else full_geometry_throughput
                scores = warm_function(out, namespace['benchmark'], a.throughput_case)
            else:
                scores = namespace['throughput'](out)
            manifest['independent_score_validation'] = {
                str(length): validate_score(scores[str(length)], length, 1024, count)
                for length, count in ((4096, 256), (16384, 128)) if str(length) in scores}
            manifest['official_score_commands']={key:json.loads((out/('score-'+key)/'command.json').read_text()) for key in scores}
            manifest['official_score_command_validation']='unchanged e236 benchmark and OwnedBenchmarkAdapter exact command guard'
            state.update(throughput_measured=True, throughput=scores)
            manifest['actual_runtime_after_throughput'] = evidence((out / 'server.log').read_text(), expected, native_gqa_sha=a.native_gqa_sha256,native_library=a.native_k8_library,native_identity=native_identity)
            if a.baseline and a.native_gqa_sha256 is not None:
                events = manifest['actual_runtime_after_throughput']['native_gqa']['events']
                observed = {(x.get('rows'), x.get('segments')) for x in events}
                required = {(64, 2) if key == '4096' else (32, 4) for key in scores}
                if not required.issubset(observed):
                    raise RuntimeError('Requested new-main native shapes were never selected/captured')
                manifest['native_shape_scope'] = dict(required=sorted(required), observed=sorted(observed),
                    request_frequency_proven=False)

        if a.native_gqa_library is not None and sha(a.native_gqa_library) != a.native_gqa_sha256:
            raise RuntimeError('Original new NativeGQA library changed')
        if a.native_k8_library is not None and (sha(a.native_k8_library)!=K8_SHA or sha(native_identity['torch']['file'])!=native_identity['torch']['sha256']):raise RuntimeError('Native K8 library/Torch source changed')
        mixed_bridge.validate(source,a.commit,a.mixed_source_bridge,a.mixed_source_bridge_sha256,a.mixed_native_library)
        if sha(tool)!=bridge_tool_sha or sha(out/'root-source-bridge.json')!=a.mixed_source_bridge_sha256:raise RuntimeError('Mixed bridge/tool changed')
        if a.graph_coverage_bootstrap_dir is not None:graph_bootstrap_environment(env,a.graph_coverage_bootstrap_dir,out)
        if sha(a.gate_library) != NATIVE_SHA or m.snapshot_inventory(snapshot) != tuning:
            raise RuntimeError('Original library/snapshot changed')
        if any(sha(source / name) != value for name, value in hashes.items()):
            raise RuntimeError('Frozen source changed')
        if sha(out / 'executed-controller.py') != manifest['controller_sha256']:
            raise RuntimeError('Executed controller snapshot changed')
        m.atomic(out / 'manifest.json', manifest)
    except BaseException as error:
        state['errors'].append(dict(type=type(error).__name__, message=str(error)))
        (out / 'error-traceback.txt').write_text(traceback.format_exc())
    finally:
        try:
            graceful = graceful_api_shutdown(server, state.get('server_owner', {}))
        except BaseException as error:
            graceful = dict(attempted=False, error=type(error).__name__ + ': ' + str(error))
            state['errors'].append(dict(type=type(error).__name__, message=str(error)))
        m.atomic(out / 'graceful-api-shutdown.json', graceful)
        cleanup = owned.cleanup()
        if server is not None:
            try: server.wait(timeout=5)
            except subprocess.TimeoutExpired: cleanup['complete'] = False
        if log is not None: log.close()
        m.atomic(out / 'owned-cleanup.json', cleanup)
        try:
            if 'mixed_smoke_window' in manifest:
                observed=mixed_bridge.final_evidence((out/'server.log').read_text(),out,
                    dict(expected,_mixed_library_path=str(a.mixed_native_library.resolve())),owned,state['server_owner'],
                    manifest['mixed_smoke_window'],a.native_k8_library,native_events)
                manifest['mixed_final_evidence']=observed
                state['mixed_same_worker_smoke_verified']=observed['mixed_same_worker_smoke_verified']
                m.atomic(out/'manifest.json',manifest)
        except BaseException as error:
            state['errors'].append(dict(type=type(error).__name__,message=str(error)))
        state['cleanup_complete'] = cleanup['complete']
        state['completed'] = not state['errors'] and state['smoke_passed'] and cleanup['complete']
        save('completed' if state['completed'] else 'failed')
    if not state['completed']: raise RuntimeError('Encoded service diagnostic failed; preserve raw evidence')
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-service', action='store_true'); p.add_argument('--math105', action='store_true')
    p.add_argument('--throughput', action='store_true'); p.add_argument('--baseline', action='store_true')
    p.add_argument('--q32-hoist', action='store_true')
    p.add_argument('--supplemental-native-math105-64',action='store_true')
    p.add_argument('--graph-coverage-bootstrap-dir',type=Path)
    p.add_argument('--full-workload-warmup', action='store_true')
    p.add_argument('--warmup-plan', choices=('legacy','full-only'), default='legacy')
    p.add_argument('--throughput-case', choices=('both', '4096', '16384'), default='both')
    for name in ('source', 'out', 'helper', 'gate-library', 'tuning-snapshot', 'native-gqa-library'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--native-gqa-sha256')
    p.add_argument('--native-k8-library',type=Path);p.add_argument('--native-k8-sha256')
    p.add_argument('--mixed-native-library',type=Path);p.add_argument('--mixed-native-sha256')
    p.add_argument('--mixed-source-bridge',type=Path);p.add_argument('--mixed-source-bridge-sha256')
    p.add_argument('--commit'); p.add_argument('--base', default='4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b')
    a = p.parse_args()
    try: warmup_plan_metadata(a)
    except ValueError as error: p.error(str(error))
    if a.graph_coverage_bootstrap_dir is not None and (not a.graph_coverage_bootstrap_dir.is_absolute() or a.native_k8_library is None or a.throughput):p.error('Graph coverage is explicit Native candidate diagnosis only, never throughput')
    if (a.native_k8_library is None)!=(a.native_k8_sha256 is None):p.error('Native K8 path/SHA must be pinned together')
    if a.native_k8_library is not None and (not a.native_k8_library.is_absolute() or a.native_k8_sha256!=K8_SHA or a.baseline):p.error('Native K8 requires candidate mode and absolute reviewed5898 K-lookahead path/SHA')
    if (a.native_gqa_library is None) != (a.native_gqa_sha256 is None):
        p.error('NativeGQA library path and SHA must be pinned together')
    if a.native_gqa_sha256 is not None and re.fullmatch(r'[0-9a-f]{64}', a.native_gqa_sha256) is None:
        p.error('NativeGQA SHA must be a full SHA256')
    if a.baseline and a.q32_hoist: p.error('Fresh private Q32 hoist requires encoded candidate mode')
    if (a.math105 or a.supplemental_native_math105_64) and a.throughput:p.error('Accuracy and throughput stages must stay separate')
    if a.supplemental_native_math105_64 and a.native_k8_library is None:p.error('Supplemental64 requires explicit Native ON candidate')
    if a.full_workload_warmup and not a.throughput: p.error('Full workload warmup needs throughput mode')
    if a.throughput_case != 'both' and not a.full_workload_warmup:
        p.error('A single-case run needs the explicit full-workload diagnostic mode')
    if not a.run_service:
        print(json.dumps(dict(cpu_plan=True, gpu_executed=False, fixed_service_CLI=True,
              warmup_plan=a.warmup_plan, warmup_plan_metadata=warmup_plan_metadata(a),
              private_smoke=True,native_decode_opt_in=True,graph_coverage_bootstrap_opt_in=True,graph_bootstrap_manifest_sha256=GRAPH_MANIFEST_SHA,native_library_sha256=K8_SHA,burst64=True,native_quality_covered=False,official_Math105_parallel8_unchanged=True,supplemental_Math105_parallel64_prepared=True,optional_original_math105=True,formal_throughput=False)))
    else:
        if any(getattr(a, name) is None for name in ('source', 'out', 'helper', 'gate_library', 'tuning_snapshot', 'commit')):
            p.error('All pinned source/helper/native/snapshot/output arguments are required')
        try:mixed_bridge.options(a)
        except ValueError as error:p.error(str(error))
        print(json.dumps(run(a), indent=2))
