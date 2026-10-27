"""Read-only replay of the DAC1 first-full-geometry throughput comparison."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def metric(value, length, count):
    expected = dict(completed=count, failed=0, num_prompts=count,
                    max_concurrency=64, total_input_tokens=length * count,
                    total_output_tokens=1024 * count)
    assert all(value.get(k) == v for k, v in expected.items()), expected
    for key in ('duration', 'total_token_throughput', 'output_throughput',
                'mean_ttft_ms', 'mean_tpot_ms'):
        assert type(value.get(key)) in (int, float) and math.isfinite(value[key]) and value[key] > 0
    for key, tokens in [('total_token_throughput', (length + 1024) * count),
                        ('output_throughput', 1024 * count)]:
        assert math.isclose(value[key], tokens / value['duration'], rel_tol=1e-12)
    return {key: value[key] for key in (*expected, 'duration', 'total_token_throughput',
                                      'output_throughput', 'mean_ttft_ms', 'mean_tpot_ms')}


def normalized_command(path):
    argv = read(path)
    for flag in ('--result-dir', '--label'):
        index = argv.index(flag)
        del argv[index:index + 2]
    return argv


def review(new, old):
    files = []

    def evidence(path):
        files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))

    download = new / 'download-manifest.json'
    assert sha(download) == '912f8a26640d77adb4f9539461d0e043edd40dc13b09c69d35ce4eee1170eaf3'
    inventory = read(download)
    for item in inventory['files']:
        path = new / item['relative']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        evidence(path)
    assert inventory['raw_files'] == 34 and inventory['raw_bytes'] == 354614
    assert inventory['original_owner'] == dict(pid=643740, start_ticks=345932200,
                                              same_start=True, state='Z')
    evidence(download)
    n = new / 'out'
    manifest = read(n / 'manifest.json')
    assert manifest['source_commit'] == 'dac1b8fb48e7983e48283032efeee9b592c5fbe5'
    state = read(n / 'state.json')
    assert state['completed'] is True and not state['errors']
    logs = (n / 'server.log').read_text()
    rejected = {tag: logs.count(tag) for tag in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED',
                'BIV150_PREFILL_MULTI2_FALLBACK', 'BIV150_KONLY_ENCODED_SELECTED ')}
    assert not any(rejected.values())
    markers = ('NATIVE_W8A8_INSTALL', 'STABLE_SMALL_O_INSTALL', 'SMALL_OUT_LOWERING_INSTALL',
               'Q_FRAGMENT_INSTALL', 'PREFILL_NORMALIZATION_INSTALL', 'GQA_DECODE_INSTALL',
               'MIXED_DECODE_INSTALLED', 'PREFILL_Q32_INSTALL', 'PREFILL_MULTI2_INSTALL')
    for marker in markers:
        tag = 'BIV150_' + marker + ' '
        values = [ast.literal_eval(line.split(tag, 1)[1]) for line in logs.splitlines() if tag in line]
        assert values and all(value.get('installed') is True for value in values)
    tag = 'BIV150_COMPILE_SEMANTICS '
    fingerprints = [json.loads(line.split(tag, 1)[1]) for line in logs.splitlines() if tag in line]
    assert len(fingerprints) == 2 and all(value == manifest['expected_compile_semantics'] for value in fingerprints)
    required = ('GPU KV cache size: 548,416 tokens', 'BIV150_GQA_DECODE_FIRST_WIDTH seqs=64',
                'BIV150_PREFILL_Q32_SELECTED ', 'BIV150_PREFILL_MULTI2_SELECTED ',
                'BIV150_HEAD_TEMPERATURE_READY ')
    assert all(tag in logs for tag in required)
    comparisons = {}
    for arm in ('baseline', 'candidate'):
        root = old / arm
        previous = read(root / 'manifest.json')
        assert previous['command'] == manifest['command']
        assert previous['gate_library_sha256'] == manifest['native_sha256']
        assert [(x['name'], x['sha256']) for x in previous['tuning_snapshot']] == [
            (x['name'], x['sha256']) for x in manifest['tuning_snapshot']]
        assert previous['full_workload_warmup'] is True and previous['run_accuracy'] is True
        old_logs = (root / 'server.log').read_text()
        assert required[0] in old_logs
        for name in ('manifest.json', 'scores.json', 'warmups.json', 'server.log',
                     'score-16384-selected-config.json', 'score-16384-new-kernels.json'):
            evidence(root / name)
        warmups = read(root / 'warmups.json')
        rows = {}
        for length, count in ((4096, 256), (16384, 128)):
            key = str(length)
            warm = warmups[key]
            assert len(warm) == 4 and warm[-1]['full_workload'] is True
            for name in (f'full-warm-{length}', f'score-{length}'):
                command = root / name / 'command.json'
                assert normalized_command(command) == normalized_command(n / f'score-{length}' / 'command.json')
                evidence(command)
            current = metric(read(n / 'scores.json')[key], length, count)
            first = metric(warm[-1], length, count)
            later = metric(read(root / 'scores.json')[key], length, count)
            rows[key] = dict(new_first_score=current, old_first_full_warm=first,
                             old_subsequent_score=later,
                             new_over_old_first=current['total_token_throughput'] / first['total_token_throughput'],
                             new_over_old_subsequent=current['total_token_throughput'] / later['total_token_throughput'])
        comparisons[arm] = dict(commit=previous['commit'], measurements=rows,
                               old_16k_new_kernels=read(root / 'score-16384-new-kernels.json'),
                               old_16k_selected_config=read(root / 'score-16384-selected-config.json'))
    return dict(schema=1, scope='CPU-only original-evidence review; no device execution',
                current_first_full_geometry_diagnostic_valid=True,
                same_warm_state_source_regression_proven=False,
                formal_acceptance=False, source_commit=manifest['source_commit'],
                native_sha256=manifest['native_sha256'], tuning_snapshot=manifest['tuning_snapshot'],
                original_owner=inventory['original_owner'], actual_compile_fingerprints=fingerprints,
                absent_reject_markers=rejected, kv_tokens=548416, kv_page_tokens=16, kv_blocks=34276,
                comparisons=comparisons,
                caveats=['Current run has short 128-output warmups but no preceding full 1024-output run.',
                         'Old run also performed mixed diagnostics and Math105 before throughput.',
                         'Equal initial tuning snapshots do not prove equal live tuned configurations.',
                         'No current per-phase kernel/config inventory or per-request lengths was saved.',
                         'Warm-state difference is demonstrated; its exact causal cost is not attributed.'],
                files=files)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-root', type=Path, default=Path('/private/tmp/cjj-konly-main-dac-throughput-records-raw-v1-20261027'))
    p.add_argument('--old-pair-root', type=Path, default=Path('/private/tmp/cjj-multi2-final-raw-snapshot-v1-20261027/pair'))
    a = p.parse_args()
    print(json.dumps(review(a.new_root.resolve(), a.old_pair_root.resolve()), indent=2) + '\n', end='')
