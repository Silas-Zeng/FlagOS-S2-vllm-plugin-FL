#!/usr/bin/env python3
"""Replay immutable recorded codec/full4K/full16K evidence on CPU; never launch a kernel."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import tarfile

from biv150_konly_decode_ir import audit

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/konly-i8-qk-20261027'
LIMIT = 2 * 1024 * 1024


def checked(data, descriptor):
    assert len(data) == descriptor['bytes']
    assert hashlib.sha256(data).hexdigest() == descriptor['sha256']
    return data


def read_parts(parts):
    files = {}
    for part in parts:
        data = checked((EVIDENCE / part['file']).read_bytes(), part)
        assert len(data) <= LIMIT
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            members = tar.getmembers()
            assert len(members) == part['members']
            for member in members:
                path = PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                assert '__pycache__' not in path.parts and path.suffix != '.pyc'
                assert member.name not in files and 0 <= member.size <= LIMIT
                files[member.name] = tar.extractfile(member).read()
    return files


def original_files(entry):
    files = read_parts(entry['archives'])
    manifest = json.loads(checked(files['download-manifest.json'], entry['original_download_manifest']))
    assert manifest['original_owner'] == entry['original_owner']
    assert manifest['raw_files'] == entry['original_raw_files'] == len(manifest['files'])
    assert manifest['raw_bytes'] == entry['original_raw_bytes'] == sum(f['bytes'] for f in manifest['files'])
    omitted = {f['relative']: f for f in entry['omitted_pycache']}
    assert len(omitted) == 2
    for name in omitted:
        assert '__pycache__' in PurePosixPath(name).parts and name.endswith('.pyc')
    expected = {'download-manifest.json'}
    for descriptor in manifest['files']:
        name = descriptor['relative']
        if name in omitted:
            assert omitted[name] == {k: descriptor[k] for k in ('relative', 'bytes', 'sha256')}
            continue
        checked(files[name], descriptor)
        expected.add(name)
    assert set(files) == expected and len(files) - 1 == entry['archived_raw_files']
    assert sum(len(data) for name, data in files.items() if name != 'download-manifest.json') == entry['archived_raw_bytes']
    return files


def independent_metrics(actual, reference):
    # Deliberately recompute from saved bits/FP64 operands, independently of the runner's metrics helper.
    assert actual and len(actual) == len(reference)
    assert all(math.isfinite(v) for v in actual + reference)
    errors = [a - b for a, b in zip(actual, reference)]
    rms = math.hypot(*errors) / math.sqrt(len(actual))
    reference_rms = math.hypot(*reference) / math.sqrt(len(actual))
    maximum = max(map(abs, errors))
    assert rms != 0 or maximum == 0
    count = sum(abs(a - b) > .005 + .005 * abs(b) for a, b in zip(actual, reference))
    return dict(finite=True, pointwise_passed=count == 0, pointwise_violations=count,
                relative_rmse=rms / reference_rms if reference_rms else (math.inf if maximum else 0.),
                max_abs=maximum, rms=rms)


def replay_arrays(files, result):
    import numpy as np
    from biv150_konly_decode_numeric_oracle import bf16_from_bits
    from biv150_konly_decode_quant_reference import quantize_v2

    codec = []
    for m in (1, 8):
        with np.load(io.BytesIO(files['out/codec-M' + str(m) + '-witness.npz']), allow_pickle=False) as witness:
            assert witness['source_bits'].shape == (141, 128)
            for row in range(141):
                values = [bf16_from_bits(int(v)) for v in witness['source_bits'][row]]
                codes, scale = quantize_v2(values)
                assert list(codes) == witness['actual_codes'][row].tolist()
                bits = struct.unpack('<I', struct.pack('<f', scale))[0]
                assert bits == int(witness['actual_scale_bits'][row])
            codec.append(dict(m=m, vectors=141, codes=18048, scales=141, bits_exact=True))
    numeric = []
    for record in result.get('records', []):
        for control in record['numeric_checks']:
            data = checked(files['out/' + control['witness']['path']], control['witness'])
            with np.load(io.BytesIO(data), allow_pickle=False) as witness:
                assert np.array_equal(witness['inline_output_bits'], witness['preq_output_bits'])
                selected = witness['selected_rows'].tolist()
                assert selected == [v['row'] for v in control['checks']['inline']]
                for i, row in enumerate(selected):
                    def decode(name):
                        return [bf16_from_bits(int(v)) for v in witness[name][row].reshape(-1)]
                    original = witness['original_reference'][i].reshape(-1).tolist()
                    quantized = witness['quantized_reference'][i].reshape(-1).tolist()
                    baseline, candidate = decode('baseline_output_bits'), decode('inline_output_bits')
                    metrics = dict(baseline=independent_metrics(baseline, original),
                                   quantized_operand=independent_metrics(candidate, quantized),
                                   original_bf16=independent_metrics(candidate, original))
                    limit = max(.005, 1.2 * metrics['baseline']['relative_rmse'])
                    passed = (metrics['baseline']['pointwise_passed'] and metrics['quantized_operand']['pointwise_passed']
                              and metrics['original_bf16']['relative_rmse'] <= limit)
                    for mode in ('inline', 'preq'):
                        saved = control['checks'][mode][i]
                        for key, value in metrics.items():
                            assert saved[key] == value
                        assert saved['original_relative_rmse_limit'] == limit
                        assert saved['thresholds_relaxed'] is False and saved['candidate_numeric_passed'] == passed
                    numeric.append(dict(iteration=control['iteration'], row=row, length=int(witness['lengths'][row]),
                                        **metrics, quantized_reference_vs_original=independent_metrics(quantized, original)))
    return dict(codec=codec, numeric_selected_rows=len(numeric), numeric_checks=numeric,
                output_bits_inline_preq_equal=True,
                reference_scope='Saved SHA-verified FP64 references; cold BF16 cache not archived, so not regenerated here',
                scratch_maxima_sums_equality_scope='Recorded device checks; those buffers are not present in NPZ')


def run(arrays=False):
    provenance = json.loads((EVIDENCE / 'numeric-device-v1-provenance.json').read_text())
    assert provenance['source_commit'] == 'f28783e15eaf8593305c3e6a057b76bb2f7189cc'
    source_files = read_parts(provenance['runtime_source']['archives'])
    source_manifest = json.loads(checked(source_files['source-manifest.json'], provenance['runtime_source']['manifest']))
    assert source_manifest['source_commit'] == provenance['source_commit']
    assert len(source_manifest['files']) == provenance['runtime_source']['file_count'] == len(source_files) - 1
    for descriptor in source_manifest['files']:
        name = descriptor['relative']
        checked(source_files[name], descriptor)
        assert (ROOT / name).read_bytes() == source_files[name], 'Frozen f287 runtime changed: ' + name
    for review in provenance['independent_cpu_reviews']:
        checked((EVIDENCE / review['file']).read_bytes(), review)
    for report in provenance['archived_replay_reports']:
        assert json.loads(checked((EVIDENCE / report['file']).read_bytes(), report))['cpu_replay_passed']
    recordings = []
    for entry in provenance['recordings']:
        files = original_files(entry)
        result = json.loads(files['out/result.json'])
        assert result['source_commit'] == provenance['source_commit'] and result['completed'] and not result['errors']
        assert not result.get('service_verified') and not result.get('capacity_reclaimed')
        assert result['codec']['passed'] and result['writer_edges']['passed']
        prefix = entry['label'] == 'codec-prefix-v1'
        replay = dict(label=entry['label'], original_raw_files=entry['original_raw_files'],
                      archived_raw_files=entry['archived_raw_files'], original_raw_bytes=entry['original_raw_bytes'],
                      omitted_pycache_files=len(entry['omitted_pycache']), recorded_passed=result['passed'])
        if prefix:
            assert result['passed'] and result['status'] == 'codec_and_writer_only'
            assert not result['full_decoder_numeric_verified'] and not result['micro_cost_measured']
        else:
            assert not result['passed'] and not result['precision_gate'] and result['numeric_diagnostic']
            record, = result['records']
            expected_geometry = {'inline-full4k-v1': (64, 4096, 2), 'inline-full16k-v1': (32, 16384, 4)}
            assert (record['batch'], record['kv'], record['segments']) == expected_geometry[entry['label']]
            assert len(record['hard_checks']) == 9 and all(v is True for v in record['hard_checks'].values())
            assert record['numeric_passed'] == {'inline': False, 'preq': False}
            assert record['timing_policy'] == dict(allowed=True, numeric_passed=False, diagnostic_only=True, timing_rounds=1, precision_gate=False)
            assert record['post_timing_poisoned_step_exact'] and len(record['timings']) == 1
            ir_replay = {}
            resources = {}
            for mode in ('baseline', 'inline', 'preq'):
                resource = record['resources'][mode + '-producer']
                assert resource['phase'] == 'post_launch' and resource['loaded_resources_known']
                assert all(type(resource[k]) is int for k in ('n_regs', 'n_spills'))
                for artifact in resource['files'].values():
                    checked(files['out/' + artifact['path']], artifact)
                resources[mode] = dict(n_regs=resource['n_regs'], n_spills=resource['n_spills'], shared=resource['metadata']['shared'])
                if mode != 'baseline':
                    artifacts = {k: files['out/' + resource['files'][k]['path']].decode() for k in ('ttgir', 'llir', 'asm')}
                    actual = audit(artifacts, files['out/generated-' + mode + '.py'].decode())
                    assert actual['passed'] and actual == resource['mechanism']
                    ir_replay[mode] = actual
            timing, = record['timings']
            ratio = timing['baseline']['complete_step_ms'] / timing['inline']['complete_step_ms']
            assert ratio == timing['baseline_over_candidate']
            replay.update(hard_checks=record['hard_checks'], actual_ir=ir_replay, resources=resources,
                          diagnostic_timing=record['timings'], stage_ms=record['stage_ms'], precision_gate=False)
        if arrays:
            replay['npz_replay'] = replay_arrays(files, result)
        recordings.append(replay)
    assert not {'torch', 'triton'}.intersection(sys.modules)
    return dict(cpu_replay_passed=True, executed_gpu=False, source_commit=provenance['source_commit'],
                frozen_runtime_changed=False, numpy_opt_in=arrays, recordings=recordings,
                full_decoder_precision_passed=False, diagnostic_only=True, service_verified=False, capacity_reclaimed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', action='store_true', help='Use existing NumPy for independent NPZ metric/codec replay on CPU')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = run(args.arrays)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cpu_replay_passed=result['cpu_replay_passed'], numpy_opt_in=args.arrays,
                          recordings=[dict(label=r['label'], recorded_passed=r['recorded_passed'],
                                           numeric_rows=r.get('npz_replay', {}).get('numeric_selected_rows')) for r in result['recordings']],
                          frozen_runtime_changed=False, executed_gpu=False, service_verified=False)))
