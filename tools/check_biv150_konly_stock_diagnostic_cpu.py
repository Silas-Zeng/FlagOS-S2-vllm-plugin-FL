#!/usr/bin/env python3
"""CPU source/ABI/address/quantized-score targets for the stock-cache diagnostic only."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from biv150_konly_decode_factory import original_kernel
from biv150_konly_decode_quant_reference import quantize_v2
from biv150_konly_decode_reference import attention_reference, encode_reference
from biv150_konly_i8_qk_reference import decode_key, references
from biv150_konly_stock_diagnostic_factory import build, restore, validate_stock_contract
from biv150_konly_stock_diagnostic_reference import tile_reference
from check_biv150_konly_i8_qk_cpu import fixture

ROOT = Path(__file__).resolve().parents[1]


def reject(call):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError('Expected fail-closed rejection')


def calls(source, function):
    return [ast.dump(n) for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and ast.unparse(n.func) == function]


def run(emit_source=None):
    source = (ROOT / 'vllm_fl/patches/biv150_gqa_decode.py').read_text()
    candidate, proof = build(source)
    original = original_kernel(source)
    assert restore(candidate) == original
    for name in ('tl.load', 'tl.store'):
        assert calls(candidate, name) == calls(original, name)
    tree = ast.parse(candidate)
    codec = [ast.unparse(n) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and ast.unparse(n.func) == 'quantize_bf16_rne']
    assert codec.count('quantize_bf16_rne(Q)') == 1
    assert codec.count('quantize_bf16_rne(tl.trans(K))') == 2
    assert len(codec) == 3
    dots = [ast.unparse(n) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == 'tl.dot']
    assert dots.count('tl.dot(Q, K, tl.zeros([ROWS, PAGE_SIZE], dtype=tl.int32), out_dtype=tl.int32)') == 2
    assert dots.count('tl.dot(probability.to(V.dtype), V)') == 2
    mutants = [candidate.replace('out_dtype=tl.int32', 'out_dtype=tl.float32', 1),
               candidate.replace('tl.trans(K)', 'K', 1),
               candidate.replace('K = tl.trans(key_code_rows)', 'K = key_code_rows', 1),
               candidate.replace('other=0.0)', 'other=float("nan"))', 1),
               candidate.replace('mask=column_valid[None, :]', 'mask=row_valid[None, :]', 1),
               candidate.replace('page * k_stride_block', 'page * v_stride_block', 1),
               candidate.replace('key_cache_ptr.dtype.element_ty == tl.bfloat16', 'key_cache_ptr.dtype.element_ty == tl.int8', 1),
               candidate.replace('probability.to(V.dtype), V', 'probability.to(tl.int8), V', 1),
               candidate.replace('running_sum * rescale', 'running_sum + rescale', 1),
               candidate.replace('key_scale[None, :]', 'query_scale[None, :]', 1)]
    for mutant in mutants:
        assert mutant != candidate
        reject(lambda mutant=mutant: restore(mutant))
    descriptor = dict(payload_format='stock-bf16', query_dtype='bfloat16', key_dtype='bfloat16',
                      value_dtype='bfloat16', dimension_stride=1)
    assert validate_stock_contract(**descriptor)
    for change in (dict(payload_format='cjj-konly-rn32-rne-v2'), dict(payload_format=None),
                   dict(key_dtype='int8'), dict(value_dtype='int8'), dict(dimension_stride=2)):
        reject(lambda change=change: validate_stock_contract(**(descriptor | change)))
    cases = 0
    score_cells = 0
    for pattern in (0, 1):
        keys, values, query = fixture(pattern)
        before = repr((keys, values, query))
        packed = encode_reference(keys, values)
        query_codes, query_scales = zip(*(quantize_v2(row) for row in query))
        for page in range(3):
            for head in range(2):
                for valid in range(17):
                    # Model exactly what a masked stock load may observe. Poison
                    # values in invalid memory must never reach quantization.
                    rows = [keys[page][slot][head] if slot < valid else [math.nan] * 128 for slot in range(16)]
                    got = tile_reference(query, rows, valid)
                    expected = references(query_codes, query_scales, packed, page, head, valid)
                    for row in range(8):
                        assert got['integer'][row][:valid] == expected['integer'][row][:valid]
                        assert got['fp32'][row][:valid] == expected['fp32'][row][:valid]
                        assert got['quantized_operand_fp64'][row][:valid] == expected['quantized_operand_fp64'][row][:valid]
                        assert all(v == -math.inf for v in got['fp32'][row][valid:])
                        assert all(v == 0 for v in got['integer'][row][valid:])
                    for slot in range(valid):
                        assert (got['key_codes'][slot], got['key_scales'][slot]) == decode_key(packed, page, slot, head)
                    for slot in range(valid, 16):
                        assert got['key_codes'][slot] == (0,) * 128 and got['key_scales'][slot] == 1.
                    cases += 1
                    score_cells += 8 * valid
        assert repr((keys, values, query)) == before
    # Same independent closed attention oracle: on-load and stored K8 represent
    # identical quantized operands in CPU math. Do not simulate GPU recurrence.
    keys, values, query = fixture(0)
    closed = []
    for length in (0, 1, 15, 16, 17, 31, 33):
        kt = [row for page in keys for row in page][:length]
        vt = [row for page in values for row in page][:length]
        original_output = attention_reference(query + query, kt, vt, False)
        quantized_output = attention_reference(query + query, kt, vt, True)
        assert all(math.isfinite(x) for output in (original_output, quantized_output) for row in output for x in row)
        closed.append(dict(length=length, original_and_quantized_fp64_finite=True))
    boundary = [240., 120., -120.] + [0.] * 125
    assert quantize_v2(boundary)[0][:3] == (127, 64, -64)
    tiny = [2. ** -133, -2. ** -133, -0.] + [0.] * 125
    t = tile_reference([tiny] * 8, [tiny] * 16, 16)
    assert all(v == 0 for row in t['fp32'] for v in row)
    default_entry = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'tools/bench_biv150_konly_stock_diagnostic.py')], text=True))
    assert default_entry['cpu_only'] and default_entry['compile_entry_prepared'] and default_entry['numeric_entry_prepared']
    assert not any(default_entry[k] for k in ('torch_imported', 'triton_imported', 'numpy_imported', 'gpu_executed', 'timing_entry_prepared'))
    probe = ast.parse((ROOT / 'tools/kernels/biv150_konly_stock_codec_falsifier.py').read_text())
    assert any(isinstance(n, ast.Call) and ast.unparse(n) == 'tl.arange(0, 16)' for n in ast.walk(probe))
    assert len([n for n in ast.walk(probe) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'quantize_bf16_rne']) == 1
    # Existing f287 runtime and production are immutable throughout this bridge.
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', 'f28783e', '--',
                    'vllm_fl', 'csrc', 'tools/bench_biv150_konly_decode_numeric.py',
                    'tools/bench_biv150_konly_decode.py', 'tools/biv150_konly_decode_factory.py',
                    'tools/biv150_konly_decode_numeric_contract.py', 'tools/biv150_konly_decode_numeric_oracle.py',
                    'tools/biv150_konly_decode_quant_reference.py', 'tools/kernels/biv150_konly_decode_support.py'], check=True)
    assert not {'torch', 'triton', 'numpy'}.intersection(sys.modules)
    if emit_source:
        emit_source.write_text(candidate)
    return dict(passed=True, cpu_only=True, factory=proof, source_load_store_ast_calls_identical=True,
                same_raw_parameter_schema=True, score_tile_cases=cases, valid_integer_score_cells=score_cells,
                invalid_tail_poison_not_observed=True, packed_v2_score_targets_exact=True,
                source_cache_unchanged=True, closed_attention_targets=closed, rejected_mutants=len(mutants) + 5,
                v2_half_boundary_code=64, tiny_cpu_fp32_target_zero=True,
                cpu_is_not_gpu_recurrence_or_FTZ_simulation=True, f287_runtime_and_production_unchanged=True,
                default_entry_cpu_only=True, compile_only_and_numeric_entry_prepared=True, timing_entry_prepared=False,
                gpu_executed=False, device_numeric_verified=False, actual_m16_codec_verified=False,
                full_prefill_quantization_represented=False, throughput_candidate=False, service_installed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--emit-source', type=Path, help='Write generated Python text only; never import or JIT it')
    args = parser.parse_args()
    result = run(args.emit_source)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
