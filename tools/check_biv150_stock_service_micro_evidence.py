#!/usr/bin/env python3
"""Read immutable micro archives/IR on CPU; optional arrays replay, never a device run."""
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import tarfile

import biv150_konly_decode_ir as ir
import biv150_stock_service_abi as abi

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/stock-qk8-service-diagnostic-20261027/micro-device-v1'
SOURCE = '867b725c08765fd8775d77de59a466da20bfe155'
LIMIT = 2 * 1024 * 1024


def checked(data, item):
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    return data


def read_archives(folder):
    index = json.loads((folder / 'index.json').read_text()); files = {}
    assert index['source_commit'] == SOURCE and index['omitted_files'] == []
    for part in index['archives']:
        data = checked((folder / part['file']).read_bytes(), part); assert len(data) <= LIMIT
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            members = tar.getmembers(); assert len(members) == part['members']
            for member in members:
                path = PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                assert 0 <= member.size <= LIMIT and member.name not in files
                files[member.name] = tar.extractfile(member).read()
    assert len(files) == index['payload_files']
    manifest = json.loads(checked(files['raw/download-manifest.json'], index['original_download_manifest']))
    assert manifest['original_owner'] == index['original_owner'] == dict(pid=627804, start_ticks=344295833, same_start=True, state='Z')
    assert manifest['raw_files'] == index['original_raw_files'] == index['archived_raw_files'] == len(manifest['files']) == 276
    assert manifest['raw_bytes'] == index['original_raw_bytes'] == sum(v['bytes'] for v in manifest['files']) == 27171679
    expected = {'raw/download-manifest.json'}
    for item in manifest['files']:
        expected.add('raw/' + item['relative']); checked(files['raw/' + item['relative']], item)
    for item in index['source_files']:
        expected.add('source/' + item['path']); checked(files['source/' + item['path']], item)
    assert set(files) == expected
    # Auditors are source tools, not opaque device modules; verify exact pinned bytes before replay.
    for loaded in (ir, abi):
        name = 'tools/' + Path(loaded.__file__).name
        assert Path(loaded.__file__).read_bytes() == files['source/' + name]
    return index, files


def metrics(actual, reference):
    assert actual and len(actual) == len(reference) and all(math.isfinite(v) for v in actual + reference)
    errors = [a - b for a, b in zip(actual, reference)]
    rms = math.hypot(*errors) / math.sqrt(len(errors)); reference_rms = math.hypot(*reference) / math.sqrt(len(reference))
    maximum = max(map(abs, errors)); assert rms != 0 or maximum == 0
    violations = sum(abs(a - b) > .005 + .005 * abs(b) for a, b in zip(actual, reference))
    return dict(finite=True, pointwise_passed=violations == 0, pointwise_violations=violations,
        relative_rmse=rms / reference_rms if reference_rms else (math.inf if maximum else 0.), max_abs=maximum, rms=rms)


def replay_startup_failures(folder):
    index = json.loads((folder / 'controller-startup-failures-v1-index.json').read_text())
    part = index['archive']; data = checked((folder / part['file']).read_bytes(), part)
    assert len(data) <= LIMIT
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        members = tar.getmembers(); assert len(members) == part['members'] == 17
        files = {}
        for member in members:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts and member.name not in files
            files[member.name] = tar.extractfile(member).read()
    expected = set(); summaries = []
    for recording in index['recordings']:
        prefix = recording['attempt'] + '/'; name = prefix + 'download-manifest.json'; expected.add(name)
        original = json.loads(checked(files[name], recording['original_download_manifest']))
        assert original['raw_files'] == recording['original_raw_files'] == len(original['files'])
        assert original['raw_bytes'] == recording['original_raw_bytes'] == sum(v['bytes'] for v in original['files'])
        kept = {v['relative']: v for v in recording['retained_non_database_files']}
        omitted = {v['relative']: v for v in recording['omitted_private_databases']}
        assert len(omitted) == 2 and all(n.endswith('.db') for n in omitted)
        assert not set(omitted).intersection(kept) and set(kept) | set(omitted) == {v['relative'] for v in original['files']}
        for item in original['files']:
            descriptor = {k: item[k] for k in ('relative', 'bytes', 'sha256')}
            if item['relative'] in omitted:
                assert omitted[item['relative']] == descriptor
            else:
                assert kept[item['relative']] == descriptor
                expected.add(prefix + item['relative']); checked(files[prefix + item['relative']], item)
        ownername = next(n for n in kept if n.endswith('-owner.json'))
        owner = json.loads(files[prefix + ownername])
        assert (owner['pid'], owner['start_ticks'], owner['source_commit']) == (
            recording['owner_pid'], recording['owner_start_ticks'], recording['source_commit'])
        if recording['attempt'] == 'v1':
            assert prefix + 'out/state.json' not in files and recording['actual_state'] is None
            assert not recording['service_started']
        else:
            state = json.loads(files[prefix + 'out/state.json']); assert state == recording['actual_state']
            assert not state['completed'] and not state['valid_pure_decode_diagnostic'] and state['cleanup_complete']
            assert json.loads(files[prefix + 'out/prediction-inventory.json'])['count'] == 0
        assert recording['prediction_rows'] == 0 and not recording['math105_executed']
        summaries.append({k: recording[k] for k in ('attempt', 'source_commit', 'owner_pid', 'owner_start_ticks', 'stage', 'prediction_rows')})
    assert set(files) == expected
    return summaries


def replay_arrays(files, result):
    import numpy as np
    import biv150_konly_decode_numeric_oracle as oracle
    import biv150_konly_decode_quant_reference as codec
    for loaded in (oracle, codec):
        assert Path(loaded.__file__).read_bytes() == files['source/tools/' + Path(loaded.__file__).name]
    with np.load(io.BytesIO(files['raw/out/codec-M16-witness.npz']), allow_pickle=False) as w:
        assert w['source_bits'].shape == (141, 128)
        for row in range(141):
            code, scale = codec.quantize_v2([oracle.bf16_from_bits(int(v)) for v in w['source_bits'][row]])
            assert list(code) == w['actual_codes'][row].tolist()
            assert struct.unpack('<I', struct.pack('<f', scale))[0] == int(w['actual_scale_bits'][row])
    numeric = []
    for record in result['records']:
        for control in record['controls']:
            name = 'raw/out/b%d-k%d-control%d.npz' % (record['batch'], record['kv'], control['iteration'])
            with np.load(io.BytesIO(files[name]), allow_pickle=False) as w:
                selected = w['selected_rows'].tolist()
                assert selected == [v['row'] for v in control['checks']]
                assert w['baseline_output_bits'].shape == w['candidate_output_bits'].shape == (record['batch'], 16, 128)
                assert w['original_reference'].dtype == w['quantized_reference'].dtype == np.float64
                for i, row in enumerate(selected):
                    def output(label):
                        return [oracle.bf16_from_bits(int(v)) for v in w[label + '_output_bits'][row].reshape(-1)]
                    original, quantized = [w[k][i].reshape(-1).tolist() for k in ('original_reference', 'quantized_reference')]
                    values = dict(baseline=metrics(output('baseline'), original),
                        quantized_operand=metrics(output('candidate'), quantized), original_bf16=metrics(output('candidate'), original))
                    saved = control['checks'][i]
                    assert all(saved[k] == value for k, value in values.items())
                    limit = max(.005, 1.2 * values['baseline']['relative_rmse'])
                    passed = values['baseline']['pointwise_passed'] and values['quantized_operand']['pointwise_passed'] and values['original_bf16']['relative_rmse'] <= limit
                    assert saved['length'] == int(w['lengths'][row]) and saved['thresholds_relaxed'] is False
                    assert saved['original_relative_rmse_limit'] == limit and saved['candidate_numeric_passed'] == passed
                    numeric.append(dict(batch=record['batch'], kv=record['kv'], iteration=control['iteration'], row=row,
                        length=saved['length'], **values, quantized_reference_vs_original=metrics(quantized, original)))
    return dict(m16_vectors=141, m16_codes=18048, m16_scales=141, m16_code_scale_bits_exact=True,
        saved_fp64_metric_rows=len(numeric), metrics=numeric,
        references_regenerated=False, reference_scope='SHA-verified saved FP64 arrays; cold K/V cache absent',
        graph_side_canary_scope='Recorded actual device hard gates; scratch/replay buffers not in NPZ')


def run(folder, arrays=False):
    index, files = read_archives(folder)
    result = json.loads(files['raw/out/result.json']); certificate = json.loads(files['raw/out/micro-certificate.json'])
    assert result['source_commit'] == SOURCE and result['completed'] and result['gpu_executed'] and not result['errors']
    assert result['implementation_hard_gates_passed'] and not result['passed'] and not result['original_bf16_proxy_passed']
    assert not any(result[k] for k in ('precision_gate', 'model105_passed', 'full_prefill_quantization', 'timing_attempted', 'service_installed'))
    for name in ('kernel', 'codec'):
        source = files['source/vllm_fl/patches/biv150_stock_qk8_' + name + '.py']
        assert source == files['raw/out/packaged-' + name + '.py.txt']
        assert hashlib.sha256(source).hexdigest() == result[name + '_sha256']
    # Import only our unchanged stdlib certificate contract, never packaged GPU source.
    path = ROOT / 'vllm_fl/patches/biv150_stock_qk8_diagnostic.py'
    assert path.read_bytes() == files['source/vllm_fl/patches/' + path.name]
    spec = importlib.util.spec_from_file_location('stock_micro_evidence_contract', path)
    hook = importlib.util.module_from_spec(spec); spec.loader.exec_module(hook)
    hook.certificate_contract(certificate)
    assert certificate['source_commit'] == SOURCE and certificate['cases'] == [r['implementation_hard_gates'] for r in result['records']]
    assert [(r['batch'], r['kv']) for r in result['records']] == [(b, 129) for b in (1, 8, 16, 32, 40, 64)] + [(1, 4096)]
    records = []
    resources = [result['codec']['resources']]
    assert result['codec']['passed'] and result['codec']['codes_and_scale_bits_exact'] and result['codec']['vectors'] == 141
    for record in result['records']:
        resource = record['resources']['candidate']
        arts = {k: files['raw/out/' + resource['files'][k]['path']].decode() for k in ('ttgir', 'llir', 'asm')}
        actual_ir = ir.audit(arts, files['raw/out/packaged-kernel.py.txt'].decode())
        assert actual_ir['passed'] and actual_ir == resource['actual_ir']
        headers = {label: {k: files['raw/out/' + record['resources'][label]['files'][k]['path']].decode()
                          for k in ('ttir', 'llir')} for label in ('baseline', 'candidate')}
        # re.findall tuples become lists in the original JSON serialization.
        actual_abi = json.loads(json.dumps(abi.compare(headers['baseline'], headers['candidate'])))
        assert actual_abi == record['actual_typed_abi'] and actual_abi['typed_abi_match']
        assert record['passed'] and all(record['implementation_hard_gates'][k] for k in (
            'quantized_operand_gate_passed', 'baseline_gate_passed', 'source_cache_unchanged', 'canaries',
            'actual_ir_passed', 'typed_abi_match', 'eager_warm_passed', 'graph_changed_q_table_lengths_exact', 'side_exact'))
        assert [c['iteration'] for c in record['controls']] == list(range(4))
        resources.extend(record['resources'].values())
        records.append(dict(batch=record['batch'], kv=record['kv'], segments=record['segments'],
            actual_ir=actual_ir, actual_typed_abi=actual_abi,
            loaded={k: {f: v[f] for f in ('n_regs', 'n_spills', 'metadata')} for k, v in record['resources'].items()}))
    for resource in resources:
        assert resource['phase'] == 'post_launch' and resource['loaded_resources_known']
        assert all(type(resource[k]) is int for k in ('n_regs', 'n_spills'))
        for item in resource['files'].values():
            checked(files['raw/out/' + item['path']], item)
    report = dict(cpu_replay_passed=True, arrays=arrays, source_commit=SOURCE, raw_files=276, raw_bytes=27171679,
        archive_parts=len(index['archives']), omitted_files=[], actual_ir_and_abi_replayed=records,
        controller_pre_question_failures=replay_startup_failures(folder.parent),
        original_bf16_proxy_passed=False, runtime_changed=False, gpu_executed_by_replayer=False,
        model105_passed=False, service_verified=False, timing_attempted=False)
    if arrays:
        report['array_replay'] = replay_arrays(files, result)
    assert not {'torch', 'triton', 'vllm'}.intersection(sys.modules)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--arrays', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(); report = run(args.evidence, args.arrays)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cpu_replay_passed=True, arrays=args.arrays, raw_files=276,
        cases=len(report['actual_ir_and_abi_replayed']), numeric_rows=report.get('array_replay', {}).get('saved_fp64_metric_rows', 0),
        gpu_executed=False, runtime_changed=False)))
