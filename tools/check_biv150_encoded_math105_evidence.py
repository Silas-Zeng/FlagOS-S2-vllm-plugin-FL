#!/usr/bin/env python3
"""CPU replay of the existing d8ef encoded-cache Math105 run; no model execution."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import tarfile
import types

from check_biv150_encoded_smoke_evidence import events

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'd8ef055422d9d5c45049f85468b486b47b993406'
MAIN = '4229b48b64c7bd21f9f8fc3c5ff8775500f8d02b'
IMPL = '1cfc33340e80030660e15b4475b777f6c0975a23bd256dfe7e10e151aa9f9fd5'
ROOT_REVIEW = '9cc6107eeb13356c687439a24c47472171708cf2c2a96d6a6f59c305c0664caa'
FOLDER = ROOT / 'docs/evidence/konly-encoded-service-20261027/math105-d8e-records-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def describe(data):
    return dict(bytes=len(data), sha256=sha(data))


def checked(data, item):
    assert describe(data) == {k: item[k] for k in ('bytes', 'sha256')}
    return data


def git_blob(path, source=SOURCE, seen=()):
    assert path not in seen and not path.startswith(('/', '../'))
    data = subprocess.check_output(['git', '-C', str(ROOT), 'show', source + ':' + path])
    entry = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', source, '--', path]).decode().split()
    if entry[0] == '120000':
        target = posixpath.normpath(posixpath.join(posixpath.dirname(path), data.decode()))
        return git_blob(target, source, (*seen, path))
    return data


def build(raw, root_review):
    manifest = (raw / 'download-manifest.json').read_bytes()
    download = json.loads(manifest)
    assert download['raw_files'] == 38 and download['raw_bytes'] == 6262576
    files = {'raw/download-manifest.json': manifest}
    for item in download['files']:
        name = item['relative']
        assert '__pycache__' not in name and not name.endswith(('.db', '.pt', '.pth', '.safetensors'))
        files['raw/' + name] = checked((raw / name).read_bytes(), item)
    files['root/independent-review.json'] = root_review.read_bytes()
    assert sha(files['root/independent-review.json']) == ROOT_REVIEW
    helper = 'tools/run_biv150_stock_decode_105.py'
    files['protocol/original-stdlib-helper-a48.py'] = git_blob(helper, 'a48cf1c82f8665b7e70e3e70fea5e0845eeee504')
    # These are public benchmark responses, own code/compiled artifacts and logs.
    # No model weights, databases, generated cache trees or private keys enter.
    for name, data in files.items():
        assert not re.search(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----', data), name
        assert not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})', data), name
    stream = io.BytesIO()
    with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as z:
        with tarfile.open(fileobj=z, mode='w') as archive:
            for name, data in sorted(files.items()):
                item = tarfile.TarInfo(name)
                item.size, item.mode, item.mtime = len(data), 0o644, 0
                archive.addfile(item, io.BytesIO(data))
    data = stream.getvalue()
    assert len(data) <= 2 * 1024 * 1024
    FOLDER.mkdir(parents=True, exist_ok=False)
    name = 'encoded-d8e-math105-records-v1.tar.gz'
    (FOLDER / name).write_bytes(data)
    index = dict(schema='cjj-encoded-d8e-math105-records-v1', source_commit=SOURCE,
                 archive=dict(file=name, **describe(data)), original_raw_files=38,
                 original_raw_bytes=6262576, original_owner=download['original_owner'],
                 members=[dict(path=p, **describe(b)) for p, b in sorted(files.items())],
                 omitted_original_record_files=[], generated_caches_excluded=download['excluded_cache_prefixes'],
                 local_unmanifested_pycache_excluded=True, model_rerun=False)
    (FOLDER / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


def run():
    index = json.loads((FOLDER / 'index.json').read_text())
    data = checked((FOLDER / index['archive']['file']).read_bytes(), index['archive'])
    assert len(data) <= 2 * 1024 * 1024
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for item in archive:
            name = PurePosixPath(item.name)
            assert item.isfile() and not name.is_absolute() and '..' not in name.parts and item.name not in files
            files[item.name] = archive.extractfile(item).read()
    assert set(files) == {v['path'] for v in index['members']}
    for item in index['members']:
        checked(files[item['path']], item)
    download = json.loads(files['raw/download-manifest.json'])
    assert download['raw_files'] == len(download['files']) == 38
    assert download['raw_bytes'] == sum(v['bytes'] for v in download['files']) == 6262576
    assert download['original_owner'] == index['original_owner'] == dict(pid=688948, start_ticks=347274628, same_start=True, state='Z')
    for item in download['files']:
        checked(files['raw/' + item['relative']], item)
    assert sha(files['root/independent-review.json']) == ROOT_REVIEW

    def read(name):
        return json.loads(files['raw/out/' + name])

    def only(suffix):
        names = [p for p in files if p.endswith(suffix)]
        assert len(names) == 1, names
        return names[0]

    actual, state = read('manifest.json'), read('state.json')
    verdict, cleanup = read('accuracy-verdict.json'), read('owned-cleanup.json')
    assert actual['source_commit'] == index['source_commit'] == SOURCE
    assert state['completed'] and state['model_precision_verified'] and state['smoke_passed']
    assert state['errors'] == [] and state['cleanup_complete'] and not state['throughput_measured']
    assert cleanup['complete'] and cleanup['remaining_owned'] == cleanup['unobserved_same_session'] == cleanup['signal_errors'] == []
    assert cleanup['pid_reuse_never_signalled'] and not cleanup['group_signal_used']
    for name, digest in actual['source_hashes'].items():
        assert sha(git_blob(name)) == digest, name
    assert sha(files['raw/out/executed-controller.py']) == actual['controller_sha256']
    assert sha(files['protocol/original-stdlib-helper-a48.py']) == actual['helper_sha256'] == '94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
    assert actual['private_enabled'] and actual['fresh_q32_hoist_requested'] and not actual['full_acceptance']
    assert actual['native_sha256'] == '5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
    assert actual['native_gqa_library_sha256'] == 'aace693bdb302d2685ec1b5e11e859c8a4a77d7f45c0f4dacddc6b5e49b015b8'
    assert [(v['bytes'], v['sha256']) for v in actual['tuning_snapshot']] == [
        (1605632, '5764f125233211d8ed8369f0bcdc6a08a58a00c9a3470bfaa36be0af7665ca55'),
        (15712256, '2c5aa448065453188f1ff1794d8e929bacb63dcad84a600e086ae05165821314')]
    resolution = actual['actual_resolution']
    assert resolution['PYTHONPATH'] == actual['source'] and resolution['origin'] == actual['source'] + '/vllm_fl/__init__.py'
    cmd = actual['command']
    for option, value in [('--port', '9031'), ('--gpu-memory-utilization', '0.85'), ('--max-model-len', '131072')]:
        assert cmd[cmd.index(option) + 1] == value
    assert json.loads(cmd[cmd.index('--compilation-config') + 1]) == dict(cudagraph_mode='FULL_DECODE_ONLY')
    cmd = read('math500-level3/command.json')
    for option, value in [('--datasets', 'math_500'), ('--eval-batch-size', '8'), ('--eval-type', 'openai_api'), ('--model', 'minicpm')]:
        assert cmd[cmd.index(option) + 1] == value
    assert json.loads(cmd[cmd.index('--dataset-args') + 1])['math_500']['subset_list'] == ['Level 3']
    assert json.loads(cmd[cmd.index('--generation-config') + 1]) == dict(temperature=1.0, top_p=0.95, max_tokens=32768)

    log = files['raw/out/server.log'].decode()
    runtime = actual['actual_runtime_after_math']
    fp = events(log, 'BIV150_COMPILE_SEMANTICS')
    assert fp == runtime['actual_compile_semantics'] and len(fp) == 2
    assert all(v == actual['expected_compile_semantics'] for v in fp)
    assert fp[0] != actual['off_compile_semantics'] and fp[0]['native_enabled']
    enabled = fp[0]['konly_encoded']
    assert enabled['implementation_sha256'] == IMPL and enabled['q32_hoist_enabled']
    assert enabled['producer_boundary'] == 'owned-encoded-before-stock-native'
    installs = events(log, 'BIV150_KONLY_ENCODED_INSTALL')
    assert installs == runtime['actual_installs'] and len(installs) == 1
    assert installs[0]['installed'] and installs[0]['implementation_sha256'] == IMPL
    assert installs[0]['stock_native_gqa_parent_preserved'] and not installs[0]['encoded_GQA_uses_stock_native']
    parents = runtime['preserved_parent_installations']
    assert len(parents) == 9
    assert all(events(log, k) == v and all(row['installed'] for row in v) for k, v in parents.items())
    assert parents['BIV150_SMALL_OUT_LOWERING_INSTALL'][0]['large_linear']
    native_parent = runtime['native_gqa']
    assert events(log, 'BIV150_NATIVE_GQA_INSTALL') == native_parent['installs']
    assert len(native_parent['installs']) == 1 and native_parent['installs'][0]['installed']
    assert native_parent['installs'][0]['library_sha256'] == actual['native_gqa_library_sha256']
    assert native_parent['events'] == [] and not native_parent['request_frequency_proven']
    assert events(log, 'BIV150_NATIVE_GQA_SELECTED') == events(log, 'BIV150_NATIVE_GQA_CAPTURED') == []
    selected = events(log, 'BIV150_KONLY_ENCODED_SELECTED')
    assert selected == runtime['selected'] and len(selected) == 11
    assert all(v['implementation_sha256'] == IMPL and v['host_event_not_Graph_replay_count'] for v in selected)
    assert runtime['host_capture_events_only'] and not runtime['graph_replays_counted']

    binding_path = only('/binding.json')
    directory = posixpath.dirname(binding_path)
    binding = json.loads(files[binding_path])
    assert binding['implementation_sha256'] == IMPL
    meta, baseline = binding['actual_metadata'], binding['unhoisted']
    assert meta['kernel_hash'] == baseline['hash']
    assert meta['source_digest'] == baseline['source_digest']
    assert meta['function_name'] == baseline['function_name']
    assert meta['ttgir_sha256'] == baseline['artifacts']['ttgir']
    assert binding['hoisted']['source_digest'] == baseline['source_digest']
    assert binding['hoisted']['function_name'] == baseline['function_name']
    for role in ('unhoisted', 'hoisted'):
        snapshot = binding[role]
        assert json.loads(files[directory + '/' + role + '/record.json']) == snapshot
        for suffix, digest in snapshot['artifacts'].items():
            assert sha(files[directory + '/' + role + '/actual.' + suffix]) == digest
        assert sha(files[directory + '/' + role + '/actual.jit_python_source.py']) == snapshot['source_digest']
    assert binding['unhoisted']['ABI'] == binding['hoisted']['ABI']
    signature = binding['hoisted']['ABI']['signature']
    assert signature[0] == 'dict'
    sig = {k[1]: v[1] for k, v in signature[1]}
    assert [sig[k] for k in ('output_ptr', 'query_ptr', 'key_cache_ptr', 'key_code_scale_ptr', 'value_cache_ptr')] == ['*bf16', '*bf16', '*i8', '*fp32', '*bf16']
    modules = {}
    for kind in ('proposal', 'native'):
        path = 'vllm_fl/patches/biv150_konly_encoded_hoist_' + kind + '.py'
        module = types.ModuleType('historical_' + kind)
        exec(compile(git_blob(path), path, 'exec'), module.__dict__)
        modules[kind] = module
    ttgir = files[directory + '/unhoisted/actual.ttgir'].decode()
    override, proposal = modules['proposal'].propose(ttgir, binding['actual_metadata'])
    assert override.encode() == files[directory + '/override.ttgir'] and proposal == binding['proposal']
    assert sha(override.encode()) == binding['override_sha256']
    native = modules['native'].audit(override, files[directory + '/hoisted/actual.ttgir'].decode(),
                                   files[directory + '/hoisted/actual.llir'].decode(), files[directory + '/hoisted/actual.asm'].decode())
    assert native == binding['native'] and native['actual_I8_operand_hoist_verified']
    assert [v['hash'] for v in selected if v['I8_fragment_hoist_verified']] == [binding['hoisted']['hash']]
    for role, regs, shared in [('unhoisted', 135, 57344), ('hoisted', 122, 33792)]:
        resource = binding['resources'][role]
        assert resource['known'] and (resource['n_regs'], resource['n_spills'], resource['shared']) == (regs, 0, shared)

    predictions = [json.loads(line) for line in files[only('/predictions/minicpm/math_500_Level 3.jsonl')].splitlines()]
    reviews = [json.loads(line) for line in files[only('/reviews/minicpm/math_500_Level 3.jsonl')].splitlines()]
    assert len(predictions) == len(reviews) == 105
    assert {p['index'] for p in predictions} == {p['index'] for p in reviews} == set(range(105))
    assert len({p['metadata']['question_id'] for p in predictions}) == 105
    scores = {r['index']: r['sample_score'] for r in reviews}
    rows = []
    for p in sorted(predictions, key=lambda p: p['index']):
        output = p['model_output']
        assert not output['error'] and len(output['choices']) == 1
        choice = output['choices'][0]
        assert choice['stop_reason'] == 'stop'
        content = choice['message']['content']
        final = content if isinstance(content, str) else ''.join(v['text'] for v in content if v['type'] == 'text')
        assert final.strip()
        s = scores[p['index']]
        assert s['sample_id'] == p['index'] and s['sample_metadata']['question_id'] == p['metadata']['question_id']
        score = s['score']
        assert score['status'] == 'success' and score['value']['accuracy'] in (0, 1)
        assert score['prediction'] == final and str(score['extracted_prediction']).strip() and score['extracted_prediction'] is not None
        rows.append(dict(index=p['index'], question_id=p['metadata']['question_id'], correct=score['value']['accuracy'],
                         stop_reason=choice['stop_reason'], final_sha256=sha(final.encode()),
                         extracted_prediction=score['extracted_prediction'], output_tokens=output['usage']['output_tokens']))
    correct = int(sum(v['correct'] for v in rows))
    assert correct == 102
    report = json.loads(files[only('/reports/minicpm/math_500.json')])
    metric = report['metrics'][0]
    assert report['num'] == metric['num'] == verdict['metric_count'] == 105
    assert metric['identity']['name'] == 'accuracy' and metric['score'] == verdict['score'] == round(correct / 105, 4) == .9714
    assert report['execution_summary']['errored'] == 0 and report['execution_summary']['succeeded'] == 105
    assert verdict['predictions'] == verdict['unique_questions'] == 105 and verdict['failed_predictions'] == []
    assert read('graceful-api-shutdown.json')['API_exited']
    # These commands compare Git blobs only; no source/helper/model is executed.
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'b3b87d2a1c13448bd8c6bc24e8045c0458b8b9dd', MAIN, '--', 'vllm_fl', 'csrc'])
    return dict(passed=True, CPU_only=True, model_rerun=False, tested_source_commit=SOURCE,
                merged_main=MAIN, merged_main_production_same_as_b3=True, raw_files=38, raw_bytes=6262576,
                source_hashes_verified=len(actual['source_hashes']), implementation_sha256=IMPL,
                original_scorer_recount_only=True, independent_model_rejudge=False,
                correct=correct, total=105, accuracy=correct / 105, rounded_report_accuracy=.9714,
                original_protocol_math105_precision_gate=True, stop_reasons=dict(Counter(v['stop_reason'] for v in rows)),
                empty_final_or_extraction=[], request_errors=[], rows=rows,
                worker_fingerprints=fp, preserved_parent_installers=9, selected_host_capture_events=selected,
                stock_BF16_native_parent_installed=True, encoded_GQA_uses_stock_BF16_native=False,
                actual_q32_typed_ABI_and_native_audit_replayed=True, q32_resources=binding['resources'],
                graph_replays_counted=False, all_service_shape_coverage_proven=False,
                initial_native_and_tuning_descriptors_checked=True, remote_libraries_or_snapshot_rehashed_by_this_CPU_replay=False,
                throughput_verified_on_this_commit=False, TTFT_verified=False, keep_Draft=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-raw', type=Path)
    parser.add_argument('--root-review', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.archive_raw:
        assert args.root_review
        build(args.archive_raw, args.root_review)
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'selected_host_capture_events', 'worker_fingerprints')}, indent=2))
