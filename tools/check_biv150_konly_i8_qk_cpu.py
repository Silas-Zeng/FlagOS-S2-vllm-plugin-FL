#!/usr/bin/env python3
"""Standard-library evidence and representation checks; no framework imports."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys

from biv150_konly_i8_qk_reference import (PAGE_BYTES, bf16, bf16_bits, decode_key, encode_pages,
                                        indices, quantize, references, round_ratio_even)
from biv150_konly_i8_qk_ir import audit_artifacts


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / 'tools/kernels/biv150_konly_i8_qk.py'
CASES = 36


def fixture(pattern=0):
    rng = random.Random(20261027 + pattern)
    keys = [[[[bf16(rng.uniform(-4, 4) * (1 + pattern)) for _ in range(128)]
              for _ in range(2)] for _ in range(16)] for _ in range(3)]
    values = [[[[bf16((p + 1) * 2 + h + s / 32 + d / 256)
                for d in range(128)] for h in range(2)] for s in range(16)] for p in range(3)]
    queries = [[bf16(rng.uniform(-3, 3) * (pattern + 1)) for _ in range(128)] for _ in range(8)]
    if pattern == 1:
        queries[0] = [0.0] * 128
        keys[1][15][1] = [bf16((-1)**d * 127) for d in range(128)]
    elif pattern == 2:
        queries[0] = [127.0] * 128
        queries[1] = [-127.0] * 128
        keys[1][15][1] = [127.0] * 128
        keys[0][0][0] = [0.0] * 128
        values[0][0][0][:3] = [-0.0, 2.0**-133, struct.unpack('<f',struct.pack('<I',0x7f7f0000))[0]]
    return keys, values, queries


def host_case(pattern):
    keys, values, query = fixture(pattern)
    packed = encode_pages(keys, values)
    qcodes, qscales, pages, heads, tails, expected = [], [], [], [], [], []
    for case in range(CASES):
        page = (case + pattern) % 3
        head = (case % 2 + pattern) % 2
        tail = (case // 2 + 3 * pattern) % 17
        if case == 34:
            page, head, tail = 1, 1, 16
        elif case == 35:
            page, head, tail = 2, 0, 0
        rows = query[case % 8:] + query[:case % 8]
        encoded = [quantize(row) for row in rows]
        qc, qs = [r[0] for r in encoded], [r[1] for r in encoded]
        qcodes.extend(v for row in qc for v in row)
        qscales.extend(qs)
        pages.append(page); heads.append(head); tails.append(tail)
        expected.append(references(qc, qs, packed, page, head, tail))
    return dict(packed=packed, qcodes=qcodes, qscales=qscales, pages=pages, heads=heads,
                tails=tails, expected=expected)


def validate_dot_contract(source):
    """I8 dot with an I32 acc must explicitly override tl.dot's F32 default."""
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or funcs[0].name != 'konly_i8_qk_probe':
        raise ValueError('unexpected primitive function')
    calls = [n for n in ast.walk(funcs[0]) if isinstance(n, ast.Call)]
    dots = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'dot']
    if len(dots) != 1 or ast.unparse(dots[0]) != 'tl.dot(Q, K, zero, out_dtype=tl.int32)':
        raise ValueError('explicit signed-I8/I32 tl.dot output dtype required')
    zero = [n.value for n in ast.walk(funcs[0]) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id == 'zero' for t in n.targets)]
    if len(zero) != 1 or ast.unparse(zero[0]) != 'tl.zeros((8, 16), dtype=tl.int32)':
        raise ValueError('I32 accumulator required')


def run_checks():
    source = KERNEL.read_text()
    validate_dot_contract(source)
    dtype_negatives = [source.replace(', out_dtype=tl.int32',''),
                       source.replace('out_dtype=tl.int32','out_dtype=tl.float32'),
                       source.replace('zero = tl.zeros((8, 16), dtype=tl.int32)',
                                      'zero = tl.zeros((8, 16), dtype=tl.float32)')]
    for bad_source in dtype_negatives:
        assert bad_source != source
        try: validate_dot_contract(bad_source)
        except ValueError: pass
        else: raise AssertionError('acc/output dtype mismatch accepted')
    assert 'page = tl.load(physical_pages + case).to(tl.int64)' in source
    assert 'head = tl.load(kv_heads + case).to(tl.int64)' in source
    assert 'tl.zeros((8, 16), dtype=tl.int32)' in source
    assert 'V =' not in source and 'bfloat16' not in source and 'softmax' not in source
    rounding = 0
    for scale in (0.5, 1.0, 2.0, 4.0):
        for integer in range(-126, 127):
            x = (integer + .5) * scale
            assert round_ratio_even(x, scale) == round(x / scale)
            rounding += 1
    vectors = [[0.0] * 128, [127.0] * 128, [-127.0] * 128,
               [bf16((-1)**d * 2.0**-133) for d in range(128)],
               [bf16((-1)**d * 2.0**120) for d in range(128)]]
    for vector in vectors:
        codes, scale = quantize(vector)
        assert all(abs(c) <= 127 for c in codes) and math.isfinite(scale) and scale > 0
    cases = 0
    for pattern in range(3):
        keys, values, query = fixture(pattern)
        before = repr((keys, values))
        packed = encode_pages(keys, values)
        assert repr((keys, values)) == before
        assert len(packed) == 3 * PAGE_BYTES
        for page in range(3):
            assert packed[page*PAGE_BYTES+4224:page*PAGE_BYTES+8192] == bytes([0xa5]) * 3968
            for slot in range(16):
                for head in range(2):
                    assert decode_key(packed, page, slot, head) == quantize(keys[page][slot][head])
                    for dim in range(128):
                        _, _, vo = indices(page, slot, head, dim)
                        bits = struct.unpack_from('<H', packed, vo)[0]
                        assert bits == bf16_bits(values[page][slot][head][dim])
        qs = [quantize(row) for row in query]
        for page in (2, 0, 1):
            for head in (1, 0):
                for valid in range(17):
                    ref = references([v[0] for v in qs], [v[1] for v in qs], packed, page, head, valid)
                    for row in range(8):
                        assert ref['integer'][row][valid:] == [0] * (16 - valid)
                        assert ref['fp32'][row][valid:] == [-math.inf] * (16 - valid)
                    cases += 1
    negatives = 0
    for bad in ([0.0]*127, [math.nan]*128, [math.inf]*128, [0.1]*128):
        try: quantize(bad)
        except ValueError: negatives += 1
        else: raise AssertionError('bad vector accepted')
    for args in ((-1,0,0), (2**31,0,0), (0,16,0), (0,0,2), (True,0,0)):
        try: indices(*args)
        except ValueError: negatives += 1
        else: raise AssertionError('bad address accepted')
    good_codes = [[0]*128 for _ in range(8)]
    good_scales = [1.0]*8
    invalid = [dict(valid=1.5), dict(valid=True), dict(valid=-1), dict(valid=17),
               dict(page=3), dict(head=2), dict(query_scales=[0.0]*8),
               dict(query_scales=[math.nan]*8), dict(query_codes=[[-128]*128]*8)]
    for change in invalid:
        call = dict(query_codes=good_codes,query_scales=good_scales,packed=packed,page=0,head=0,valid=16)
        call.update(change)
        try: references(**call)
        except ValueError: negatives += 1
        else: raise AssertionError('bad primitive metadata accepted')
    for payload in ('zero_scale','nan_scale','bad_code','poison'):
        damaged = bytearray(packed)
        ko,so,_ = indices(0,0,0)
        if payload == 'bad_code': damaged[ko] = 128
        elif payload == 'poison': damaged[:] = bytes([0xa5])*len(damaged)
        else: struct.pack_into('<f',damaged,so,0.0 if payload == 'zero_scale' else math.nan)
        try: references(good_codes,good_scales,damaged,0,0,1)
        except ValueError: negatives += 1
        else: raise AssertionError('invalid/poisoned cache payload accepted')
    overflow = bytearray(packed)
    ko,so,_ = indices(0,0,0)
    overflow[ko:ko+128] = bytes([127])*128
    struct.pack_into('<f',overflow,so,1.0)
    try: references([[127]*128]*8,[2.0**127]*8,overflow,0,0,1)
    except ValueError: negatives += 1
    else: raise AssertionError('FP32 score overflow was accepted')
    gpu_fixtures = [host_case(p) for p in range(3)]
    assert all(gpu_fixtures[0][key] != gpu_fixtures[1][key] for key in
               ('packed','qcodes','qscales','pages','heads','tails'))
    raw_extrema = [v for e in gpu_fixtures[2]['expected'] for row in e['integer'] for v in row]
    assert max(raw_extrema) == 128*127**2 and min(raw_extrema) == -128*127**2
    assert not audit_artifacts({})['passed']
    evidence = ROOT / 'docs/evidence/konly-i8-qk-20261027'
    provenance = json.loads((evidence/'reference-provenance.json').read_text())
    for item in provenance['files']:
        raw = (evidence/item['file']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
    old_ir = {kind:(evidence/('pr55-reference-primitive.'+kind+'.txt')).read_text()
              for kind in ('ttgir','llir','asm')}
    rejected = audit_artifacts(old_ir)
    assert not rejected['passed']
    assert not rejected['flags']['logical_m8_k128_n16_signed_i8_i32']
    assert not rejected['flags']['exactly_four_i8_machine_mmas']
    ko, so, vo = indices(2**30+3, 15, 1, 127)
    assert max(ko, so, vo) > 2**32 and max(ko, so, vo) < 2**63
    assert not any(name == 'torch' or name == 'triton' or name.startswith(('torch.', 'triton.')) for name in sys.modules)
    return dict(passed=True, cpu_only=True, gpu_executed=False, kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
                quantization_rounding_cases=rounding, page_head_tail_reference_cases=cases, rejection_cases=negatives,
                prepared_gpu_fixture_cases=3*CASES, changed_input_and_metadata_proof=True,
                exact_integer_extrema=[min(raw_extrema),max(raw_extrema)],
                missing_actual_ir_rejected=True, positive_actual_ir_not_available=True,
                actual_pr55_wrong_geometry_ir_rejected=True,
                explicit_i32_dot_dtype_contract=True, dtype_contract_negative_cases=len(dtype_negatives),
                source_destination_separate=True, active_page_payload_bytes=12416, allocated_page_bytes=16384,
                capacity_reclaimed=False, device_native_path_verified=False, performance_verified=False, service_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = run_checks()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
