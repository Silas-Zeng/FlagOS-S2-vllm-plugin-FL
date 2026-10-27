#!/usr/bin/env python3
"""Replay immutable mixed-reader files/IR on CPU; --arrays reads saved NPZ only."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

from biv150_konly_mixed_cfg_audit import audit as audit_cfg
from biv150_konly_mixed_factory import build, NAMES
from biv150_konly_mixed_ir import expected_abi, mechanism, typed_abi

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '4a3fc7773b37e745082d0d98f8d53713130384ea'
LIMIT = 2 * 1024 * 1024


def checked(data, descriptor):
    assert len(data) == descriptor['bytes'] and hashlib.sha256(data).hexdigest() == descriptor['sha256']
    return data


def read_archives(folder):
    index = json.loads((folder / 'index.json').read_text()); physical = {}; logical = {}
    assert index['source_commit'] == SOURCE and len(index['omitted_files']) == 2
    assert all('__pycache__' in PurePosixPath(v['relative']).parts for v in index['omitted_files'])
    for part in index['archives']:
        data = checked((folder / part['file']).read_bytes(), part); assert len(data) <= LIMIT
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            members = tar.getmembers(); assert len(members) == part['members']
            for member in members:
                path = PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                assert 0 <= member.size <= 1024 * 1024 and member.name not in physical
                physical[member.name] = tar.extractfile(member).read()
    assert len(physical) == index['physical_members']; used = set()
    for item in index['logical_files']:
        name = item['path']; path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and name not in logical
        assert all(v['member'] not in used for v in item['chunks'])
        data = b''.join(checked(physical[v['member']], v) for v in item['chunks'])
        used.update(v['member'] for v in item['chunks']); logical[name] = checked(data, item)
    assert used == set(physical)
    manifest = json.loads(checked(logical['raw/download-manifest.json'], index['original_download_manifest']))
    assert manifest['raw_files'] == index['original_raw_files'] == len(manifest['files']) == 86
    assert manifest['raw_bytes'] == index['original_raw_bytes'] == sum(v['bytes'] for v in manifest['files'])
    omitted = {v['relative']: v for v in index['omitted_files']}
    for item in manifest['files']:
        if item['relative'] in omitted:
            assert all(omitted[item['relative']][k] == item[k] for k in ('bytes', 'sha256'))
        else: checked(logical['raw/' + item['relative']], item)
    assert index['archived_raw_files'] == 84
    for item in index['source_files']: checked(logical['source/' + item['path']], item)
    expected = {'raw/download-manifest.json'} | {'raw/' + v['relative'] for v in manifest['files'] if v['relative'] not in omitted}
    expected |= {'source/' + v['path'] for v in index['source_files']}
    assert set(logical) == expected
    return index, logical


def cfg_mutants(text, names, role, n, capacity, report):
    """Reject exact guard/address failures in copies of real LLVM, never compile."""
    d = report['ssa']; tests = {}
    tests['wrong_sequence_bound'] = text.replace(', ' + str(n - 1) + ', !dbg', ', ' + str(n - 2) + ', !dbg', 1)
    tests['wrong_decode_length'] = text.replace('icmp ne i64 ' + d['length_difference'] + ', 1', 'icmp ne i64 ' + d['length_difference'] + ', 2', 1)
    tests['unsigned_metadata'] = text.replace(d['q_start_i64'] + ' = sext i32', d['q_start_i64'] + ' = zext i32', 1)
    row = text.splitlines()[report['decode_guard_branch_line'] - 1]
    tests['inverted_decode_edges'] = text.replace(row, row.replace('label %' + report['early_return_block'], 'label %TEMP').replace('label %' + report['valid_decode_block'], 'label %' + report['early_return_block']).replace('label %TEMP', 'label %' + report['valid_decode_block']), 1)
    tests['unconditional_decode_edge'] = text.replace(row, '  br label %' + report['valid_decode_block'], 1)
    query_arg = '%' + str(names.index('query_ptr' if role == 'producer' else 'segm_output_ptr'))
    early = '  %audit_early = load i32, ptr addrspace(1) ' + query_arg + ', align 4\n'
    tests['ordinary_load_before_decode_guard'] = text.replace(row, early + row, 1)
    intrinsic = '  %audit_early = tail call <1 x i32> @llvm.bi.load.kop.v1i32(ptr addrspace(1) ' + query_arg + ', i32 0, i1 false)\n'
    tests['intrinsic_load_before_decode_guard'] = text.replace(row, intrinsic + row, 1)
    if role == 'reducer':
        old = 'shl nuw nsw i64 ' + d['q_start_i64'] + ', 12'
        assert text.count(old) == 1
        tests['output_uses_sequence_instead_of_qstart'] = text.replace(old, 'shl nuw nsw i64 ' + d['pid64'] + ', 12', 1)
    else:
        arg = '%' + str(names.index('query_stride_token'))
        old = 'mul i64 ' + arg + ', ' + d['q_start_i64']
        assert text.count(old) == 1
        tests['query_uses_sequence_instead_of_qstart'] = text.replace(old, 'mul i64 ' + arg + ', ' + d['pid64'], 1)
    rejected = []
    for name, mutant in tests.items():
        assert mutant != text, name
        try: audit_cfg(mutant, names, role, n, capacity)
        except (AssertionError, KeyError, StopIteration): rejected.append(name)
        else: raise AssertionError('Actual-LLVM mutant accepted: ' + name)
    return rejected


def metrics(actual, reference):
    assert actual and len(actual) == len(reference) and all(math.isfinite(v) for v in actual + reference)
    errors = [a - b for a, b in zip(actual, reference)]
    rms = math.hypot(*errors) / math.sqrt(len(errors)); ref_rms = math.hypot(*reference) / math.sqrt(len(reference))
    maximum = max(map(abs, errors)); assert rms != 0 or maximum == 0
    violations = sum(abs(a - b) > .005 + .005 * abs(b) for a, b in zip(actual, reference))
    return dict(finite=True, pointwise_passed=violations == 0, pointwise_violations=violations,
        relative_rmse=rms / ref_rms if ref_rms else (math.inf if maximum else 0.), max_abs=maximum, rms=rms)


def replay_arrays(files, result):
    import numpy as np
    rows = []; controls = []
    for control in result['controls']:
        iteration = control['iteration']
        with np.load(io.BytesIO(files['raw/out/numeric-control%d.npz' % iteration]), allow_pickle=False) as w:
            assert set(w.files) == {'q_bits', 'cu_q', 'lengths', 'table', 'selected_sequences', 'original_reference', 'quantized_reference', 'bf16_output_bits', 'k8_output_bits'}
            cu = w['cu_q'].tolist(); lengths = w['lengths'].tolist(); qlens = [b - a for a, b in zip(cu, cu[1:])]
            assert qlens == control['query_lengths'] and lengths == control['lengths']
            selected = w['selected_sequences'].tolist(); assert selected == [v['sequence'] for v in control['checks']]
            assert w['q_bits'].shape == (result['query_capacity'], 16, 128) and w['q_bits'].dtype == np.uint16
            assert w['original_reference'].dtype == w['quantized_reference'].dtype == np.float64
            decoded = {cu[s] for s, qlen in enumerate(qlens) if qlen == 1}
            untouched = [t for t in range(result['query_capacity']) if t not in decoded]
            for mode in ('bf16', 'k8'):
                data = w[mode + '_output_bits']; assert data.shape == (result['query_capacity'], 16, 128) and data.dtype == np.uint16
                assert bool((data[untouched] == 0x422a).all()), 'Prefill/padding output changed'
                for seq, qlen in enumerate(qlens):
                    if qlen != 1: continue
                    values = (data[cu[seq]].astype(np.uint32) << 16).view(np.float32)
                    assert bool(np.isfinite(values).all())
                    if lengths[seq] == 0: assert bool((values == 0).all())
            for i, seq in enumerate(selected):
                token = cu[seq]; saved = control['checks'][i]
                outputs = {mode: (w[mode + '_output_bits'][token].astype(np.uint32) << 16).view(np.float32).reshape(-1).tolist() for mode in ('bf16', 'k8')}
                original, quantized = [w[k][i].reshape(-1).tolist() for k in ('original_reference', 'quantized_reference')]
                values = dict(baseline=metrics(outputs['bf16'], original), quantized_operand=metrics(outputs['k8'], quantized), original_bf16=metrics(outputs['k8'], original))
                assert all(saved[k] == value for k, value in values.items())
                limit = max(.005, 1.2 * values['baseline']['relative_rmse'])
                passed = values['baseline']['pointwise_passed'] and values['quantized_operand']['pointwise_passed'] and values['original_bf16']['relative_rmse'] <= limit
                assert saved['length'] == lengths[seq] and saved['q_start'] == token and not saved['thresholds_relaxed']
                assert saved['candidate_numeric_passed'] == passed and saved['original_relative_rmse_limit'] == limit
                rows.append(dict(iteration=iteration, sequence=seq, q_start=token, length=lengths[seq], **values,
                    quantized_reference_vs_original=metrics(quantized, original)))
            controls.append(dict(iteration=iteration, decoded_output_rows=len(decoded), untouched_prefill_padding_rows=len(untouched),
                full_output_bit_equality_between_bf16_and_k8=bool(np.array_equal(w['bf16_output_bits'], w['k8_output_bits']))))
    assert len(rows) == 15
    return dict(saved_fp64_rows=len(rows), metrics=rows, full_output_controls=controls,
        original_gate_failures=sum(not c['candidate_numeric_passed'] for v in result['controls'] for c in v['checks']),
        references_regenerated=False, reference_scope='SHA-checked saved FP64 arrays only; cold K/V absent',
        independent_partial_graph_array_replay=False, reason='Scratch/eager/Graph/side snapshots were not saved in original NPZ')


def run(folder, arrays=False):
    index, files = read_archives(folder); result = json.loads(files['raw/out/result.json'])
    assert result['source_commit'] == SOURCE and result['case'] == index['case']
    assert result['completed'] and result['compile_passed'] and result['gpu_executed'] and not result['errors']
    assert result['gpu_implementation_numeric_verified'] and result['implementation_hard_gates_passed']
    assert result['stock_input_bits_exact'] and result['encoder_bytes_exact'] and result['graph_side_exact']
    assert all(not result[k] for k in ('passed', 'gpu_numeric_verified', 'original_bf16_proxy_passed', 'precision_gate', 'service_installed', 'full_service_verified', 'timing_attempted', 'capacity_reclaimed'))
    assert result['query_stride'] == [2560, 128, 1] and result['output_stride'] == [2048, 128, 1]
    assert result['prefill_table_rows_poisoned_in_this_isolated_decode_reader']
    assert result['segments'] == (2 if result['case'] == 'mixed4k' else 4)
    for name in ('biv150_konly_mixed_factory.py', 'biv150_konly_mixed_ir.py', 'biv150_konly_decode_ir.py', 'biv150_konly_decode_factory.py'):
        assert (ROOT / 'tools' / name).read_bytes() == files['source/tools/' + name]
    source = files['source/vllm_fl/patches/biv150_gqa_decode.py'].decode(); generated, proof = build(source)
    assert proof == result['source_proof']
    records = []
    for mode in ('bf16', 'k8'):
        assert generated[mode].encode() == files['raw/out/generated-' + mode + '.py']
        for role, function in (('producer', NAMES[mode]), ('reducer', '_reduce_mixed_partials')):
            name = mode + '-' + role; actual = result['resources'][name]; pre = result['resources_pre_launch'][name]
            assert actual['phase'] == 'post_launch' and actual['loaded_resources_known']
            assert pre['phase'] == 'pre_launch' and pre['n_regs'] is None and not pre['loaded_resources_known']
            artifacts = {kind: checked(files['raw/out/' + v['path']], v).decode() for kind, v in actual['files'].items() if kind != 'cubin'}
            for v in actual['files'].values(): checked(files['raw/out/' + v['path']], v)
            abi = typed_abi(artifacts, generated[mode], function, mode)
            assert json.loads(json.dumps(abi)) == actual['actual_typed_abi']
            cfg = audit_cfg(artifacts['llir'], [v[0] for v in expected_abi(generated[mode], function, mode)], role, result['num_seqs'], result['query_capacity'])
            mutants = cfg_mutants(artifacts['llir'], [v[0] for v in expected_abi(generated[mode], function, mode)], role, result['num_seqs'], result['query_capacity'], cfg)
            mechanism_report = mechanism(artifacts, generated[mode], mode) if role == 'producer' else None
            if mechanism_report is not None:
                assert mechanism_report['passed'] and mechanism_report == actual['actual_mechanism']
            records.append(dict(role=name, hash=actual['hash'], actual_typed_abi=abi,
                loaded=dict(n_regs=actual['n_regs'], n_spills=actual['n_spills'], shared=actual['metadata']['shared']),
                actual_mechanism=mechanism_report, new_cpu_llvm_cfg_audit=cfg, actual_llvm_mutants_rejected=mutants))
    for control in result['controls']:
        assert control['prefill_and_padding_untouched'] and control['compact_partials_active_and_poison_verified'] and control['cache_readonly']
        assert all(v['baseline_hard_passed'] and v['quantized_operand']['pointwise_passed'] and not v['thresholds_relaxed'] for v in control['checks'])
    frozen_runner = files['source/tools/bench_biv150_konly_mixed_numeric.py'].decode()
    for exact in ("snapshots[name]=bits(view,torch).clone()", "assert all(torch.equal(got[k],eager[mode][index][k]) for k in got)", "assert all(torch.equal(got[k],eager[mode][0][k]) for k in got)"):
        assert exact in frozen_runner
    assert not {'torch', 'triton', 'vllm'}.intersection(sys.modules)
    return dict(passed=True, cpu_evidence_replay_only=True, case=result['case'], source_commit=SOURCE,
        original_raw_files=86, archived_raw_files=84, explicitly_omitted_pycache=2, parts=len(index['archives']),
        raw_top_level_passed=False, original_bf16_proxy_passed=False, numerical_gates_unchanged=True,
        gpu_executed_for_replay=False, actual_ir_records=records,
        graph_side_partial_scope='Original frozen runner asserts bitwise equality of all scratch/max/sum/out; original result flags true. Snapshots absent, no independent CPU partial/Graph replay.',
        encoder_cache_scope='Recorded device CPU-byte-oracle gate true; complete cold/encoded caches not archived',
        arrays=replay_arrays(files, result) if arrays else dict(replayed=False, optional_flag='--arrays'),
        precision_gate=False, service_verified=False, performance_measured=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--case', choices=('mixed4k', 'mixed16k'), default='mixed4k'); p.add_argument('--evidence', type=Path); p.add_argument('--arrays', action='store_true')
    a = p.parse_args(); folder = a.evidence or ROOT / 'docs/evidence/konly-mixed-decode-20261027' / (a.case + '-device-v1')
    print(json.dumps(run(folder.resolve(), a.arrays), indent=2))
