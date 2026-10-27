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

from biv150_unchanged_max_factory import MAIN, NEW, SOURCE_SHA, build, pv_call_spans, restore, skip_contract
from biv150_unchanged_max_ir import audit_artifacts, audit_ttgir, pv_debug_inventory
from biv150_unchanged_max_mutants import run_mutants
from biv150_unchanged_max_replay import load_v1


def reject(function):
    try:
        function()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Expected conservative rejection')


def structural_fixture(proof, pv_debug_sites=None):
    """Synthetic parser tests only; never claim these are compiled artifacts."""
    tt = []
    ll = ['define void @fixture() {', 'entry:']
    asm = []
    metadata = []
    number = 0
    pv_debug_sites = pv_debug_sites or proof['source_lines']['pv']

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
            ll.append(f'  %pv{i}_{j} = call <4 x float> @llvm.bi.matrix.mad.f32x4.bf16x4(<4 x bfloat> %p, <4 x bfloat> %v, <4 x float> %c), '+debug(pv_debug_sites[i]))
        asm.extend([f'.loc 1 {proof["source_lines"]["branch"][i]} 1',
                    f'sl_cbr_tmskaz JOIN{i}', f'CORRECTION{i}:'])
        for key, count, op in [('alpha_exp', 2, 'ml_exp_f32'), ('accumulator_scale', 32, 'ml_mul_f32'),
                               ('running_sum_fma', 2, 'ml_fma_f32')]:
            asm.append(f'.loc 1 {proof["source_lines"][key][i]} 1')
            asm.extend([op+' v0, v1, v2']*count)
        asm.extend([f'sl_jump JOIN{i}', f'JOIN{i}:',
                    f'.loc 1 {pv_debug_sites[i]} 1'])
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
    assert proof['generated_sha256'] == '225c9cc1be483f7cee32eb8c61b87f97f238ea740ee1348c4f5a4023adf35313'
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
    audit_ttgir(fixture['ttgir'])
    parameter_fixture = structural_fixture(proof, [s['argument_lines'][0] for s in proof['pv_call_spans']])
    audit_ttgir(parameter_fixture['ttgir'])
    expected = {s['start']:8 for s in proof['pv_call_spans']}
    inventory = pv_debug_inventory(parameter_fixture['llir'],parameter_fixture['asm'],proof)
    assert inventory['llvm_counts'] == inventory['asm_counts'] == expected
    diamond = dict(fixture, asm=fixture['asm'].replace('sl_cbr_tmskaz JOIN0', 'sl_cbr_tmskaz SKIP0').replace(
        'sl_jump JOIN0\nJOIN0:', 'sl_jump JOIN0\nSKIP0:\nml_add_f32 v0, v1, v2\nJOIN0:'))
    # The old synthetic diamond lacks real saved-mask/C-phi data: never use it as a mechanism positive.
    reject(lambda: audit_artifacts(diamond,proof))
    for label, changed in [
            ('missing-actual-ir', {}),
            ('pv-zero-c', dict(fixture, llir=fixture['llir'].replace('<4 x float> %c', '<4 x float> zeroinitializer'))),
            ('wrong-fma-opcode', dict(fixture, asm=fixture['asm'].replace('ml_fma_f32', 'ml_add_f32'))),
            ('asm-no-rescale-branch', dict(fixture, asm=fixture['asm'].replace('sl_cbr_tmskaz', 'sl_nop'))),
            ('pv-c-not-phi', dict(fixture, ttgir=fixture['ttgir'].replace('%b0#1', '%zero'))),
            ('changed-p-anchor', dict(fixture, ttgir=fixture['ttgir'].replace('arith.maxnumf %old_anchor0, %max_score0', 'arith.select %gate0, %old_anchor0, %max_score0'))),
            ('unconditional-alpha', dict(fixture, llir=fixture['llir'].replace('yes0:\n', '').replace('no0:\n', 'yes0:\nno0:\n')))]:
        reject(lambda changed=changed: audit_artifacts(changed, proof)); rejects.append(label)
    wrong_qk = structural_fixture(proof, proof['source_lines']['qk'])
    outside_call = structural_fixture(proof, [s['end']+1 for s in proof['pv_call_spans']])
    for label, changed in [
            ('pv-qk-llvm-site', dict(fixture, llir=wrong_qk['llir'])),
            ('pv-qk-asm-site', dict(fixture, asm=wrong_qk['asm'])),
            ('pv-outside-ast-llvm-site', dict(fixture, llir=outside_call['llir'])),
            ('pv-outside-ast-asm-site', dict(fixture, asm=outside_call['asm']))]:
        reject(lambda changed=changed: audit_artifacts(changed, proof)); rejects.append(label)
    candidate, replay, raw = load_v1(root,proof)
    mutants = run_mutants(candidate,proof)
    actual_negative = []
    primary_debug_regressions = []
    if args.existing_ir:
        for name in ('b64-baseline.ttgir', 'b64-candidate.ttgir'):
            p = args.existing_ir/name
            if p.exists():
                reject(lambda p=p: audit_ttgir(p.read_text()))
                actual_negative.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        llir_path, asm_path = args.existing_ir/'b64-baseline.llir', args.existing_ir/'b64-baseline.asm'
        if llir_path.exists() and asm_path.exists():
            baseline_ast = ast.parse(original)
            spans = pv_call_spans(baseline_ast)
            qk_ast = ast.dump(ast.parse('tl.dot(Q, K)', mode='eval').body, include_attributes=False)
            qk_sites = sorted(n.lineno for n in ast.walk(baseline_ast) if isinstance(n, ast.Call)
                              and ast.dump(n, include_attributes=False) == qk_ast)
            baseline_proof = dict(source_lines=dict(pv=[s['start'] for s in spans], qk=qk_sites), pv_call_spans=spans)
            inventory = pv_debug_inventory(llir_path.read_text(), asm_path.read_text(), baseline_proof)
            expected = {s['start']: 8 for s in spans}
            assert inventory['llvm_counts'] == inventory['asm_counts'] == expected
            assert inventory['accepted_sites'] == sorted({s['argument_lines'][0] for s in spans})
            assert inventory['excluded_sites'] == qk_sites
            primary_debug_regressions.append(dict(
                source_sha256=SOURCE_SHA, pv_call_spans=spans, inventory=inventory,
                llir_sha256=hashlib.sha256(llir_path.read_bytes()).hexdigest(),
                asm_sha256=hashlib.sha256(asm_path.read_bytes()).hexdigest()))
    output = subprocess.check_output([sys.executable, str(root/'tools/bench_biv150_unchanged_max.py'),
                                      '--source', str(root)], text=True)
    default = json.loads(output)
    assert default['cpu_only'] and not default['torch_imported'] and not default['triton_imported']
    assert not {'torch', 'triton'}.intersection(sys.modules)
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', MAIN, '--', 'vllm_fl', 'csrc'], check=True)
    result = dict(cpu_only=True, checks_passed=True, proof=proof, predicate_cases=predicate_cases,
                  host_ieee_bit_patterns=ieee_checked, host_not_a_gpu_simulator=True,
                  firstpage_empty_tail_source_preserved=True, rejected=rejects,
                  structural_fixture_only_for_debug_sites=True, actual_ir_negative_tests=actual_negative,
                  validator_version=2, immutable_v1_replay=replay, actual_ir_mutants=mutants,
                  original_pv_parameter_debug_regressions=primary_debug_regressions,
                  kernel_bytes_unchanged_from_5328aa2=True, kernel_bytes_unchanged_from_8e6855f=True,
                  default_cli_no_gpu_import=True, tiny_device_primitive_pending=True,
                  gpu_executed=False, queued=False, precision_gate=False,
                  limitation='CPU source/IEEE and v1 actual-IR replay only. Old run remains parser-rejected; full GQA bitwise/numeric tests, B32, performance and model acceptance remain unverified.')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('cpu_only', 'checks_passed', 'predicate_cases',
                                           'host_ieee_bit_patterns', 'precision_gate')}))


if __name__ == '__main__':
    main()
