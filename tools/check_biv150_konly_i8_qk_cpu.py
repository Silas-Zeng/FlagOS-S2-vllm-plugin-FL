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

from biv150_konly_i8_qk_reference import (PAGE_BYTES, bf16, decode_key, encode_pages,
                                        f32, indices, quantize, references, round_ratio_even)


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / 'tools/kernels/biv150_konly_i8_qk.py'


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
        queries[0] = [bf16((-1)**d * 127) for d in range(128)]
        keys[0][0][0] = [0.0] * 128
    return keys, values, queries


def run_checks():
    source = KERNEL.read_text()
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert len(funcs) == 1 and funcs[0].name == 'konly_i8_qk_probe'
    calls = [n for n in ast.walk(funcs[0]) if isinstance(n, ast.Call)]
    dots = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'dot']
    assert len(dots) == 1 and ast.unparse(dots[0]) == 'tl.dot(Q, K, zero)'
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
                        actual = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
                        assert actual == values[page][slot][head][dim]
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
    ko, so, vo = indices(2**30+3, 15, 1, 127)
    assert max(ko, so, vo) > 2**32 and max(ko, so, vo) < 2**63
    assert not any(name == 'torch' or name == 'triton' or name.startswith(('torch.', 'triton.')) for name in sys.modules)
    return dict(passed=True, cpu_only=True, gpu_executed=False, kernel_sha256=hashlib.sha256(source.encode()).hexdigest(),
                quantization_rounding_cases=rounding, page_head_tail_reference_cases=cases, rejection_cases=negatives,
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
