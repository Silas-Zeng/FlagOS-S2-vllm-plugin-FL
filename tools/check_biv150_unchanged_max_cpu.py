#!/usr/bin/env python3
"""Source/IEEE targets and conservative guard tests, never a GPU simulator."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys

from biv150_unchanged_max_factory import MAIN, NEW, SOURCE_SHA, build, restore, skip_contract
from biv150_unchanged_max_ir import audit_artifacts, audit_ttgir


def reject(function):
    try:
        function()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Expected conservative rejection')


def structural_fixture(proof):
    """Synthetic parser tests only; never claim these are compiled artifacts."""
    tt = []
    ll = ['define void @fixture() {', 'entry:']
    asm = []
    metadata = []
    number = 0

    def debug(source):
        nonlocal number
        number += 1
        metadata.append(f'!{number} = !DILocation(line: {source}, column: 1, scope: !0)')
        return f'!dbg !{number}'

    for i in range(2):
        tt.extend([
            f'%qk{i} = tt.dot %q{i}, %k{i}, %zero{i}, inputPrecision = tf32 : x -> tensor<8x16xf32>',
            f'%score{i} = arith.mulf %qk{i}, %scale{i} : tensor<8x16xf32>',
            f'%max_score{i} = "tt.reduce"(%score{i}) : tensor<8x16xf32>',
            f'%anchor{i} = arith.maxnumf %old_anchor{i}, %max_score{i} : tensor<8xf32>',
            f'%anchor_expand{i} = tt.expand_dims %anchor{i} : tensor<8x1xf32>',
            f'%anchor_broadcast{i} = tt.broadcast %anchor_expand{i} : tensor<8x16xf32>',
            f'%shift{i} = arith.subf %score{i}, %anchor_broadcast{i} : tensor<8x16xf32>',
            f'%prob{i} = math.exp2 %shift{i} : tensor<8x16xf32>',
            f'%p_sum{i} = "tt.reduce"(%prob{i}) : tensor<8x16xf32>',
            f'%b{i}:2 = scf.if %gate{i} -> (tensor<8xf32>, tensor<8x128xf32>) {{',
            f'  %delta{i} = arith.subf %old_anchor{i}, %anchor{i} : tensor<8xf32>',
            f'  %a{i} = math.exp2 %delta{i} : tensor<8xf32>',
            f'  %f{i} = math.fma %l{i}, %a{i}, %p_sum{i} : tensor<8xf32>',
            f'  %expand{i} = tt.expand_dims %a{i} : tensor<8x1xf32>',
            f'  %broadcast{i} = tt.broadcast %expand{i} : tensor<8x128xf32>',
            f'  %c{i} = arith.mulf %old{i}, %broadcast{i} : tensor<8x128xf32>',
            f'  scf.yield %f{i}, %c{i} : tensor<8xf32>, tensor<8x128xf32>',
            '} else {',
            f'  %s{i} = arith.addf %l{i}, %p_sum{i} : tensor<8xf32>',
            f'  scf.yield %s{i}, %old{i} : tensor<8xf32>, tensor<8x128xf32>',
            '}',
            f'%p_bf16{i} = arith.truncf %prob{i} : tensor<8x16xf32> to tensor<8x16xbf16>',
            f'%p_frag{i} = ttg.convert_layout %p_bf16{i} : tensor<8x16xbf16>',
            f'%pv{i} = tt.dot %p_frag{i}, %v{i}, %b{i}#1, inputPrecision = tf32 : x -> tensor<8x128xf32>',
        ])
        ll.extend([f'  br i1 %gate{i}, label %yes{i}, label %no{i}, '+debug(proof['source_lines']['branch'][i]),
                   f'yes{i}:'])
        for j in range(2):
            ll.append(f'  %a{i}_{j} = call float @llvm.nvvm.ex2.approx.ftz.f32(float %x), '+debug(proof['source_lines']['alpha_exp'][i]))
        for j in range(32):
            ll.append(f'  %m{i}_{j} = fmul float %x, %a, '+debug(proof['source_lines']['accumulator_scale'][i]))
        for j in range(2):
            ll.append(f'  %f{i}_{j} = call float @llvm.fma.f32(float %x, float %a, float %p), '+debug(proof['source_lines']['running_sum_fma'][i]))
        ll.extend([f'  br label %no{i}', f'no{i}:'])
        for j in range(8):
            ll.append(f'  %pv{i}_{j} = call <4 x float> @llvm.bi.matrix.mad.f32x4.bf16x4(<4 x bfloat> %p, <4 x bfloat> %v, <4 x float> %c), '+debug(proof['source_lines']['pv'][i]))
        asm.extend([f'.loc 1 {proof["source_lines"]["branch"][i]} 1',
                    f'sl_cbr_tmskaz JOIN{i}', f'CORRECTION{i}:'])
        for key, count, op in [('alpha_exp', 2, 'ml_exp_f32'), ('accumulator_scale', 32, 'ml_mul_f32'),
                               ('running_sum_fma', 2, 'ml_fma_f32')]:
            asm.append(f'.loc 1 {proof["source_lines"][key][i]} 1')
            asm.extend([op+' v0, v1, v2']*count)
        asm.extend([f'sl_jump JOIN{i}', f'JOIN{i}:',
                    f'.loc 1 {proof["source_lines"]["pv"][i]} 1'])
        asm.extend(['ml_matrix_mad_f32_bf16_m16n16k16 v0, v1, v2']*8)
    ll.extend(['  ret void', '}', *metadata])
    return dict(ttgir='\n'.join(tt), llir='\n'.join(ll), asm='\n'.join(asm))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path)
    parser.add_argument('--existing-ir', type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    original = (root/'vllm_fl/patches/biv150_gqa_decode.py').read_text()
    source, proof = build(original)
    assert hashlib.sha256(original.encode()).hexdigest() == SOURCE_SHA
    compile(source, '<generated-python-syntax-only>', 'exec')
    assert restore(source)
    predicate_cases = 0
    for old, new, expected in [([-float('inf')]*8, [0.]*8, False),
                               ([1.]*8, [1.]*8, True), ([0.]*8, [-0.]*8, True),
                               ([float('inf')]*8, [float('inf')]*8, False),
                               ([-float('inf')]*8, [-float('inf')]*8, False),
                               ([2.**-149]*8, [2.**-149]*8, True)]:
        assert skip_contract(old, new) == expected
        predicate_cases += 1
    for head in range(8):
        for value in (2., float('nan'), float('inf'), -float('inf')):
            old, new = [1.]*8, [1.]*8
            new[head] = value
            assert not skip_contract(old, new)
            predicate_cases += 1
        old = [1.]*8
        old[head] = float('nan')
        assert not skip_contract(old, [1.]*8)
        predicate_cases += 1
    # Host IEEE illustration only; CPU multiplication is not a GPU FTZ model.
    rng = random.Random(20261027)
    patterns = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                0x7f7fffff, 0xff7fffff]
    patterns.extend(rng.getrandbits(32) for _ in range(4096))
    ieee_checked = 0
    for bits in patterns:
        if bits & 0x7f800000 == 0x7f800000:
            continue
        value = struct.unpack('<f', struct.pack('<I', bits))[0]
        got = struct.unpack('<I', struct.pack('<f', value*1.0))[0]
        assert got == bits
        ieee_checked += 1
    rejects = []
    for label, action in [
            ('module-sha', lambda: build(original+'\n')),
            ('nonrecurrence-edit', lambda: restore(source.replace('scores = tl.dot(Q, K)', 'scores = 2 * tl.dot(Q, K)'))),
            ('fma-removed', lambda: restore(source.replace('tl.fma', 'tl.maximum'))),
            ('one-transform-missing', lambda: restore(source.replace(NEW, '', 1))),
            ('head-count', lambda: skip_contract([0.]*7, [0.]*7))]:
        reject(action); rejects.append(label)
    fixture = structural_fixture(proof)
    audit_artifacts(fixture, proof)
    diamond = dict(fixture, asm=fixture['asm'].replace('sl_cbr_tmskaz JOIN0', 'sl_cbr_tmskaz SKIP0').replace(
        'sl_jump JOIN0\nJOIN0:', 'sl_jump JOIN0\nSKIP0:\nml_add_f32 v0, v1, v2\nJOIN0:'))
    audit_artifacts(diamond, proof)
    for label, changed in [
            ('missing-actual-ir', {}),
            ('pv-zero-c', dict(fixture, llir=fixture['llir'].replace('<4 x float> %c', '<4 x float> zeroinitializer'))),
            ('wrong-fma-opcode', dict(fixture, asm=fixture['asm'].replace('ml_fma_f32', 'ml_add_f32'))),
            ('asm-no-rescale-branch', dict(fixture, asm=fixture['asm'].replace('sl_cbr_tmskaz', 'sl_nop'))),
            ('pv-c-not-phi', dict(fixture, ttgir=fixture['ttgir'].replace('%b0#1', '%zero'))),
            ('changed-p-anchor', dict(fixture, ttgir=fixture['ttgir'].replace('arith.maxnumf %old_anchor0, %max_score0', 'arith.select %gate0, %old_anchor0, %max_score0'))),
            ('unconditional-alpha', dict(fixture, llir=fixture['llir'].replace('yes0:\n', '').replace('no0:\n', 'yes0:\nno0:\n')))]:
        reject(lambda changed=changed: audit_artifacts(changed, proof)); rejects.append(label)
    actual_negative = []
    if args.existing_ir:
        for name in ('b64-baseline.ttgir', 'b64-candidate.ttgir'):
            p = args.existing_ir/name
            if p.exists():
                reject(lambda p=p: audit_ttgir(p.read_text()))
                actual_negative.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    output = subprocess.check_output([sys.executable, str(root/'tools/bench_biv150_unchanged_max.py'),
                                      '--source', str(root)], text=True)
    default = json.loads(output)
    assert default['cpu_only'] and not default['torch_imported'] and not default['triton_imported']
    assert not {'torch', 'triton'}.intersection(sys.modules)
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', MAIN, '--', 'vllm_fl', 'csrc'], check=True)
    result = dict(cpu_only=True, checks_passed=True, proof=proof, predicate_cases=predicate_cases,
                  host_ieee_bit_patterns=ieee_checked, host_not_a_gpu_simulator=True,
                  firstpage_empty_tail_source_preserved=True, rejected=rejects,
                  structural_ir_parser_fixture_only=True, actual_ir_negative_tests=actual_negative,
                  default_cli_no_gpu_import=True, tiny_device_primitive_pending=True,
                  gpu_executed=False, queued=False, precision_gate=False,
                  limitation='Source, predicate, and IEEE targets only; actual CFG/FMA/PV/FTZ/exactness/resources/performance unverified.')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('cpu_only', 'checks_passed', 'predicate_cases',
                                           'host_ieee_bit_patterns', 'precision_gate')}))


if __name__ == '__main__':
    main()
