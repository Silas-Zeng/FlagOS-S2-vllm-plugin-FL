#!/usr/bin/env python3
"""Read-only, stdlib audit of the saved PR109 smoke, not a device/model test."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import posixpath
import re
import subprocess

COMMIT = 'fc25c86a13fc273bb7368b83af4ca264524b4e12'
CONTROLLER = 'd8a5bd435845bdb3dd1e4d3741f4ca38fe85b4cc152036517cbcf034f0eb4495'
HELPER = '94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
GATE = '5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
STOCK = '0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8'
COMPACT = '40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453'
FORMAT = 'cjj-konly-compact12544-rn32-rne-v1'
OPERATOR = 'cjj_native_k8_compact_probe::produce'
SCHEMA = OPERATOR + '(Tensor query, Tensor key_code, Tensor scale_full, Tensor value_offset, Tensor lengths, Tensor table, Tensor(a!) partial, Tensor(b!) maximum, Tensor(c!) summation, float scale_log2e, int segments) -> ()'
SERVICE = ['/usr/local/bin/vllm', 'serve', '/workspace/MiniCPM5-2B', '--port', '9031',
           '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
           '--served-model-name', 'minicpm', '--gpu-memory-utilization', '0.85', '--max-model-len', '131072']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def rows(log, marker):
    result = []
    for line in log.splitlines():
        if marker + ' ' in line:
            value = line.split(marker + ' ', 1)[1]
            try:
                result.append(json.loads(value))
            except json.JSONDecodeError:
                result.append(ast.literal_eval(value))
    return result


def assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing literal assignment ' + name)


class GitSource:
    def __init__(self, repo):
        self.repo, self.cache, self.symlinks = repo, {}, {}
        tree = self.git('ls-tree', '-rz', COMMIT)
        self.modes = {}
        for record in tree.split(b'\0'):
            if record:
                meta, name = record.split(b'\t', 1)
                self.modes[name.decode()] = meta.split()[0].decode()

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    def read(self, relative, depth=0):
        need(depth < 8 and not relative.startswith('../'), 'Unsafe source symlink')
        if relative not in self.cache:
            value = self.git('show', COMMIT + ':' + relative)
            if self.modes[relative] == '120000':
                target = posixpath.normpath(posixpath.join(posixpath.dirname(relative), value.decode()))
                self.symlinks[relative] = target
                value = self.read(target, depth + 1)
            self.cache[relative] = value
        return self.cache[relative]


def audit(a):
    r = a.raw
    download = load(r / 'download-manifest.json')
    entries = download['files']
    need(download['raw_files'] == len(entries) == 18, 'Raw file count')
    need(download['raw_bytes'] == sum(e['bytes'] for e in entries) == 1681499, 'Raw bytes')
    need({str(p.relative_to(r)) for p in r.rglob('*') if p.is_file()} == {e['relative'] for e in entries} | {'download-manifest.json'}, 'Raw file set')
    for e in entries:
        b = (r / e['relative']).read_bytes()
        need(len(b) == e['bytes'] and sha(b) == e['sha256'], 'Raw hash ' + e['relative'])
    need(download['original_owner'] == dict(pid=39057, start_ticks=354896408, same_start=True, state='Z'), 'Terminal original owner')
    owner = load(r / 'exports/cjj-agent-native109-service-smoke-v1-20261028-owner.json')
    state, m = load(r / 'out/state.json'), load(r / 'out/manifest.json')
    need((owner['pid'], owner['start_ticks'], owner['source_commit']) == (39057, 354896408, COMMIT), 'Owner source/start')
    need(state['completed'] and state['smoke_passed'] and state['cleanup_complete'] and state['errors'] == [], 'Terminal smoke state')
    for key in ('model_precision_verified', 'throughput_measured', 'native_quality_covered', 'native_model_precision_verified'):
        need(state[key] is False, 'Unexpected qualification ' + key)
    need(m['source_commit'] == COMMIT and m['controller_sha256'] == CONTROLLER and sha((r / 'out/executed-controller.py').read_bytes()) == CONTROLLER, 'Controller/source binding')
    need(m['helper_sha256'] == HELPER and sha(a.helper.read_bytes()) == HELPER, 'Actual saved helper source')
    need(m['native_sha256'] == GATE and m['native_gqa_library_sha256'] == STOCK and m['native_decode_library']['sha256'] == COMPACT, 'Three distinct library roles')
    need(m['command'] == SERVICE and m['official_service_parameters_unchanged'] is True, 'Fixed service command')
    need(m['smoke_only'] and not m['throughput_diagnostic_requested'] and not m['full_acceptance'] and not m['graph_coverage_bootstrap']['enabled'], 'Smoke scope')
    need(m['actual_resolution']['origin'] == m['source'] + '/vllm_fl/__init__.py', 'Resolved package root')
    need([x['sha256'] for x in m['tuning_snapshot']] == ['5764f125233211d8ed8369f0bcdc6a08a58a00c9a3470bfaa36be0af7665ca55', '2c5aa448065453188f1ff1794d8e929bacb63dcad84a600e086ae05165821314'], 'v6 snapshot identities')
    source = GitSource(a.repo)
    for relative, pin in m['source_hashes'].items():
        need(sha(source.read(relative)) == pin, 'Immutable Git source mismatch ' + relative)
    package = 'vllm_fl/patches/'
    compact_sources = {Path(k).name: pin for k, pin in m['source_hashes'].items() if k.startswith(package + 'biv150_konly_compact_') and (k.endswith('.py') or k.endswith('_sources.json'))}
    implementation = sha(canonical(compact_sources).encode())
    expected = m['expected_compile_semantics']
    need(implementation == expected['konly_compact']['implementation_sha256'], 'Recomputed compact implementation digest')
    native_source = package + 'biv150_konly_compact_native.py'
    need(m['native_identity']['module_file'] == m['source'] + '/' + native_source and m['native_identity']['module_sha256'] == m['source_hashes'][native_source], 'Native module source binding')
    log_bytes = (r / 'out/server.log').read_bytes()
    log = log_bytes.decode()
    runtime = m['actual_runtime']
    fingerprints = rows(log, 'BIV150_COMPILE_SEMANTICS')
    need(len(fingerprints) == 2 and all(fp == expected for fp in fingerprints) and fingerprints == runtime['actual_compile_semantics'], 'Actual worker C3 fingerprints')
    need(not any(x in log for x in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED', 'BIV150_PREFILL_MULTI2_FALLBACK')), 'Existing production C3 scope gate')
    for marker, observed in runtime['preserved_parent_installations'].items():
        actual = rows(log, marker)
        if marker == 'BIV150_NATIVE_W8A8_INSTALL':
            # This Python-repr log has ('gate_up_proj',); JSON mirrors a list.
            # Normalize only this checked representation, preserving all fields.
            need(all(x['layers'] == ('gate_up_proj',) for x in actual), 'Actual W8A8 layers tuple')
            actual = [dict(x, layers=list(x['layers'])) for x in actual]
        need(actual == observed and all(x['installed'] for x in actual), 'Preserved parent installer ' + marker)
    stock = rows(log, 'BIV150_NATIVE_GQA_INSTALL')
    need(stock == runtime['native_gqa']['installs'] and all(x['installed'] and x['overlap_v'] and x['library_sha256'] == STOCK for x in stock), 'Actual main stock native overlap library')
    need(rows(log, 'BIV150_KONLY_COMPACT_INSTALL') == runtime['actual_installs'], 'Compact installer mirror')
    selected = rows(log, 'BIV150_KONLY_ENCODED_SELECTED')
    need(selected == runtime['selected'] and all(x['implementation_sha256'] == implementation and x['format'] == FORMAT and x['hash'] for x in selected), 'Actual format-aware private selections')
    need('writer83' in {x['profile'] for x in selected} and any(x['profile'] != 'writer83' for x in selected), 'Writer and reader selection')
    allocations = rows(log, 'BIV150_KONLY_COMPACT_ALLOCATION')
    cap = runtime['compact_allocation_and_fallback']
    need(len(allocations) == 1 and allocations == cap['allocations'], 'Allocation log/manifest mirror')
    allocation = allocations[0]
    core_pins = assignment(source.read(package + 'biv150_konly_compact_identity.py'), 'CORE_PINS')
    need(allocation['verified_core_source_sha256'] == core_pins, 'Actual checked SDK/core-source pins')
    pages, layers, page_bytes = 44769, 42, 12544
    need(allocation['configured_num_blocks'] == pages and allocation['recorded_layers'] == layers and allocation['initialized'] and allocation['format'] == FORMAT and allocation['page_bytes'] == page_bytes, 'Actual block plan')
    need(len(allocation['records']) == layers and allocation['raw_storage_bytes'] == layers * pages * page_bytes == 23586458112, '42-layer storage total')
    expected_record = dict(storage_bytes=pages*page_bytes, physical_pages=pages, carrier_shape=[pages,2,16,2,128], carrier_strides=[6272,2176,256,128,1], pointer_offset_bytes=0)
    need(all(x == expected_record for x in allocation['records']), 'Every actual layer storage extent/layout')
    tokens = [int(x.replace(',', '')) for x in re.findall(r'GPU KV cache size: ([\d,]+) tokens', log)]
    need(tokens == [pages*16] == [716304] and cap['physical_tokens_per_layer'] == tokens, 'SDK token count agrees with actual physical plan')
    fallback = rows(log, 'BIV150_KONLY_COMPACT_NATIVE_FALLBACK')
    need(fallback == cap['fallback_events'] and all(x['format'] == FORMAT and x['selected_reader'] == 'private_compact_Triton' for x in fallback), 'Fallback preserves encoded format')
    library = dict(m['native_decode_library'], qualification=expected['konly_compact']['native_decode']['qualification'])
    installs = rows(log, 'BIV150_KONLY_NATIVE_DECODE_INSTALL')
    need(installs == runtime['native_decode']['installations'] and len(installs) == 1, 'Native install mirror')
    for x in installs:
        need(x['library'] == library and x['operator'] == OPERATOR and x['schema'] == SCHEMA and x['module_sha256'] == m['source_hashes'][native_source] and x['module_file'] == m['native_identity']['module_file'] and x['torch_file'] == m['native_identity']['torch']['file'], 'Loaded native source/library/schema/import receipt')
        need(x['installed'] and x['enabled'] and x['loaded_library_confirmed'] and not x['runtime_compile'] and x['physical_page_bytes'] == page_bytes, 'Native install gates')
    events = [dict(marker=marker, event=e) for marker in ('BIV150_KONLY_NATIVE_DECODE_SELECTED', 'BIV150_KONLY_NATIVE_DECODE_CAPTURED') for e in rows(log, marker)]
    need(events == runtime['native_decode']['events'] and len(events) == 168, 'All native host/capture records mirror')
    event_counts = Counter()
    for x in events:
        e = x['event']; n, segments = e['shape'], e['segments']
        need((n,segments) in ((64,2),(32,4)) and e['cache_pages'] == pages and e['library'] == library and e['format'] == FORMAT and e['physical_page_bytes'] == page_bytes and e['operator'] == OPERATOR, 'Actual native event format/ABI')
        need(e['native_launch_returned'] and e['host_or_capture_event_not_Graph_replay_count'] and e['service_numeric_verified'] is False and e['capture'] == x['marker'].endswith('CAPTURED'), 'Host/capture qualification boundary')
        need(e['query'] == dict(dtype='torch.bfloat16',shape=[n,16,128],strides=[2048,128,1],offset=0,device='cuda:0'), 'Actual Q metadata')
        need(e['table'] == dict(dtype='torch.int32',shape=[n,8192],strides=[8192,1],offset=0,device='cuda:0'), 'Actual table metadata')
        # The service retains a 64-row scratch even when Q uses the 32 bucket.
        for name, shape, strides in (('partial',[64,16,segments,128],[32768,2048,128,1]),('maximum',[64,16,segments],[256,16,1]),('summation',[64,16,segments],[256,16,1])):
            need(e[name] == dict(dtype='torch.float32',shape=shape,strides=strides,offset=0,device='cuda:0'), 'Actual scratch metadata ' + name)
        need(1 <= e['entry_generation'] <= 42, 'Observed cache owner epoch')
        event_counts[(n,segments,e['capture'])] += 1
    need(set(event_counts.values()) == {42} and len(event_counts) == 4, '42 layers at each observed startup bucket/capture combination')
    requests = load(r / 'out/smoke-requests.json')
    need(len(requests) == 3, 'Three original short requests')
    request_counts, ids, usage, finishes = {}, set(), Counter(), Counter()
    batches = [('short', requests)]
    for count, summary in zip((32,64,40,48), m['native_bursts'], strict=True):
        b = load(r / f'out/native-burst{count}-requests.json')
        need(b['completed'] == b['concurrency_requested'] == summary['completed'] == summary['concurrency_requested'] == count and len(b['records']) == count and {x['index'] for x in b['records']} == set(range(count)), 'Burst response count/index')
        window = summary['runtime_window']; data = (r / f'out/native-burst{count}-runtime-window.log').read_bytes()
        need(len(data) == window['bytes'] and sha(data) == window['sha256'] and data == log_bytes[window['log_byte_offset']:window['log_byte_offset']+len(data)], 'Original request-window log bytes')
        need(window['native_events'] == [] and window['Graph_replays_counted'] is False and b['quality_covered'] is False, 'No fabricated real-request native coverage')
        batches.append((f'burst{count}', b['records']))
    for label, batch in batches:
        request_counts[label] = len(batch)
        for x in batch:
            response, request = x['response'], x['request']
            need(not response.get('error') and response['model'] == 'minicpm' and len(response['choices']) == 1 and response['id'] not in ids, 'Successful unique saved API response')
            ids.add(response['id']); u = response['usage']
            need(u['completion_tokens'] == request['max_tokens'] == 16 and u['total_tokens'] == u['prompt_tokens'] + u['completion_tokens'], 'Existing smoke response token/count arithmetic')
            usage.update({k:u[k] for k in ('prompt_tokens','completion_tokens','total_tokens')})
            finishes[response['choices'][0]['finish_reason']] += 1
    http200 = len(re.findall(r'"POST /v1/chat/completions HTTP/1\.1" 200 OK', log))
    need(len(ids) == sum(request_counts.values()) == http200 == 187, 'Original responses and HTTP200 request count')
    need(m['compact_final_exports'] == [] and m['compact_final_export_found'] is False, 'Final stats absence retained')
    cleanup, graceful = load(r / 'out/owned-cleanup.json'), load(r / 'out/graceful-api-shutdown.json')
    need(cleanup['complete'] and cleanup['remaining_owned'] == [] and cleanup['unobserved_same_session'] == [] and cleanup['signal_errors'] == [] and cleanup['pid_reuse_never_signalled'] and not cleanup['group_signal_used'], 'Owned cleanup')
    need(graceful['attempted'] and graceful['identity_verified'] and graceful['API_exited'] and graceful['returncode'] == 0 and graceful['signal'] == 'SIGTERM', 'Graceful API owner exit')
    need('EngineDeadError' not in log, 'Unexpected EngineDeadError')
    baseline_manifest = load(a.baseline_raw / 'download-manifest.json')
    baseline_record = next(e for e in baseline_manifest['files'] if e['relative'] == 'out/server.log')
    old_bytes = (a.baseline_raw / baseline_record['relative']).read_bytes()
    need(sha(old_bytes) == baseline_record['sha256'] and len(old_bytes) == baseline_record['bytes'], 'Saved PR107 baseline log integrity')
    old_tokens = [int(x.replace(',','')) for x in re.findall(r'GPU KV cache size: ([\d,]+) tokens', old_bytes.decode())]
    need(old_tokens == [548320], 'Actual PR107 token capacity log')
    return dict(schema=1, audit_passed=True, scope='saved source/library/C3/allocation/response/cleanup smoke evidence only', raw=str(r), raw_files=18, raw_bytes=1681499,
        download_manifest_sha256=sha((r/'download-manifest.json').read_bytes()), source_commit=COMMIT, controller_sha256=CONTROLLER, source_files_verified=len(m['source_hashes']), source_symlinks_followed=source.symlinks,
        identities=dict(helper=HELPER,w8a8_gate=GATE,stock_native=STOCK,compact_native=COMPACT,compact_implementation=implementation,c3_sha256=expected['sha256'],c3_logged_equal_copies=len(fingerprints),sdk_core_pins=core_pins, library_recheck_scope='saved actual load receipts; SO bytes excluded from this smoke archive'),
        allocation=dict(layers=layers,physical_pages_per_layer=pages,page_bytes=page_bytes,bytes_per_layer=pages*page_bytes,total_bytes=layers*pages*page_bytes,sdk_tokens=716304,stock_pr107_tokens=548320,token_capacity_delta=716304/548320-1,baseline_log=str(a.baseline_raw/'out/server.log'),baseline_log_sha256=baseline_record['sha256'],distinct_storage_evidence='source-pinned Host._reshape storage._cdata duplicate rejection and initialize plan/42 bound-cache checks; no storage IDs or cache data body saved'),
        requests=dict(counts=request_counts,total=len(ids),http200=http200,usage=dict(usage),finish_reasons=dict(finishes),content_quality_evaluated=False),
        native=dict(host_selected=84,captured=84,actual_request_frequency_proven=False,graph_replay_verified=False,final_stats_saved=False,final_stats_error_or_overflow_known=False,writer_source_stride_2560_observed=False,writer_stride_limitation='writer83 selection saves hash/profile but no K/V source strides; do not infer pitch2560 from source or requests',observed_fallback_shapes=sorted({e['shape'] for e in fallback}),fallback_reader='private_compact_Triton',observed_query_row_stride=2048),
        cleanup=dict(complete=True,original_owner=download['original_owner'],api_returncode=0,EngineDeadError=False,post_cleanup_gpu_idle_not_independently_observed=True),
        qualifications=dict(component_numerics_repeated=False,model_precision_verified=False,native_quality_covered=False,throughput_verified=False,capacity_comparison_is_not_throughput=True,final_stats_counters_not_inferred=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('/private/tmp/cjj-native109-service-smoke-v1-raw-20261028'))
    p.add_argument('--repo', type=Path, default=Path('/Users/chenjunjie/Documents/flagos/repos/FlagOS-S2-vllm-plugin-FL'))
    p.add_argument('--helper', type=Path, default=Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py'))
    p.add_argument('--baseline-raw', type=Path, default=Path('/private/tmp/cjj-native107-main105-throughput-v1-raw-20261028'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    need(not a.out.exists(), 'Refuse to overwrite audit output')
    report = audit(a)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    print(json.dumps(dict(audit_passed=True,report=str(a.out),report_sha256=sha(a.out.read_bytes()),script_sha256=sha(Path(__file__).read_bytes())), sort_keys=True))


if __name__ == '__main__':
    main()
