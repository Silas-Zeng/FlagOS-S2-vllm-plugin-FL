#!/usr/bin/env python3
"""Replay actual stock diagnostic archives on CPU, never extract/import device artifacts."""
import argparse
import io
import json
from pathlib import Path
import struct
import sys

from check_biv150_konly_decode_numeric_evidence import EVIDENCE, independent_metrics, original_files
from biv150_konly_decode_ir import audit


def run(arrays=False):
    provenance = json.loads((EVIDENCE / 'stock-device-v1-provenance.json').read_text())
    records = []
    for entry in provenance['recordings']:
        files = original_files(entry)
        result = json.loads(files['out/result.json'])
        assert result['source_commit'] == provenance['source_commit'] and result['completed'] and not result['errors']
        assert result['compile_passed'] and not result['passed'] and not result['candidate_original_precision_passed']
        assert not any(result[k] for k in ('precision_gate', 'timing_attempted', 'service_installed', 'throughput_candidate', 'full_prefill_quantization_represented'))
        actual = {}
        for label in ('stock', 'packed'):
            resource = result['resources'][label + '-producer']
            artifacts = {k: files['out/' + resource['files'][k]['path']].decode() for k in ('ttgir', 'llir', 'asm')}
            actual[label] = audit(artifacts, files['out/generated-' + label + '.py'].decode())
            assert actual[label]['passed'] and actual[label] == resource['actual_ir']
        numeric = not result['compile_only']
        if numeric:
            assert result['quantized_operand_device_gate_verified'] and result['codec']['passed']
            assert result['graph_changed_inputs_poison_exact'] and result['side_eager_exact']
            assert result['graph_changed_fields'] == ['Q', 'block_table', 'lengths']
            assert not result['kv_append_or_changed_cache_writer_tested']
            assert len(result['records']) == 3
            assert all(all(r['stock_packed_equal'].values()) for r in result['records'])
        for resource in result['resources'].values():
            if numeric:
                assert resource['phase'] == 'post_launch' and resource['loaded_resources_known']
            else:
                assert resource['phase'] == 'pre_launch' and not resource['loaded_resources_known']
        replay = dict(label=entry['label'], original_raw_files=entry['original_raw_files'], actual_ir=actual,
                      recorded_numeric=numeric, original_precision_passed=False, timing_attempted=False, npz_metrics=[])
        if arrays and numeric:
            import numpy as np
            from biv150_konly_decode_numeric_oracle import bf16_from_bits
            from biv150_konly_decode_quant_reference import quantize_v2
            with np.load(io.BytesIO(files['out/codec-M16-witness.npz']), allow_pickle=False) as w:
                assert w['source_bits'].shape == (141, 128)
                for row in range(141):
                    codes, scale = quantize_v2([bf16_from_bits(int(x)) for x in w['source_bits'][row]])
                    assert list(codes) == w['actual_codes'][row].tolist()
                    assert struct.unpack('<I', struct.pack('<f', scale))[0] == int(w['actual_scale_bits'][row])
            for item in result['records']:
                with np.load(io.BytesIO(files['out/numeric-witness-' + str(item['iteration']) + '.npz']), allow_pickle=False) as w:
                    assert np.array_equal(w['stock_output_bits'], w['packed_output_bits'])
                    for index, row in enumerate(w['selected_rows'].tolist()):
                        def decode(name):
                            return [bf16_from_bits(int(x)) for x in w[name][row].reshape(-1)]
                        original, quantized = [w[k][index].reshape(-1).tolist() for k in ('original_reference', 'quantized_reference')]
                        values = dict(baseline=independent_metrics(decode('baseline_output_bits'), original),
                                      quantized_operand=independent_metrics(decode('stock_output_bits'), quantized),
                                      original_bf16=independent_metrics(decode('stock_output_bits'), original))
                        for label in ('stock', 'packed'):
                            saved = item['checks'][label][index]
                            assert saved['row'] == row and saved['thresholds_relaxed'] is False
                            assert all(saved[k] == v for k, v in values.items())
                            limit = max(.005, 1.2 * values['baseline']['relative_rmse'])
                            assert saved['original_relative_rmse_limit'] == limit
                            assert saved['candidate_numeric_passed'] == (values['baseline']['pointwise_passed'] and values['quantized_operand']['pointwise_passed'] and values['original_bf16']['relative_rmse'] <= limit)
                        replay['npz_metrics'].append(dict(iteration=item['iteration'], row=row, **values,
                            quantized_reference_vs_original=independent_metrics(quantized, original)))
            replay['m16_codec_bits_recomputed'] = True
            replay['output_bits_stock_packed_equal'] = True
            replay['reference_scope'] = 'Saved SHA-verified FP64 reference arrays; absent cold cache not regenerated'
        records.append(replay)
    assert not {'torch', 'triton'}.intersection(sys.modules)
    return dict(cpu_replay_passed=True, arrays=arrays, source_commit=provenance['source_commit'], records=records,
                gpu_executed=False, runtime_changed=False, service_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arrays', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = run(args.arrays)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cpu_replay_passed=True, arrays=args.arrays, records=[dict(label=r['label'], numeric_rows=len(r['npz_metrics'])) for r in result['records']], gpu_executed=False)))
