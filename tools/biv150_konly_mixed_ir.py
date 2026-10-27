"""Conservative actual mixed-reader typed ABI/PV checks, never source-only success."""
import ast
from collections import Counter
import re

from biv150_konly_decode_ir import audit as audit_k8


def expected_abi(source, function, mode):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
    pointers = dict(output_ptr='bf16', segm_output_ptr='f32', segm_max_ptr='f32', segm_expsum_ptr='f32',
        query_ptr='bf16', query_start_len_ptr='i32', query_code_ptr='i8', query_scale_ptr='f32',
        key_scale_ptr='f32', key_cache_ptr='bf16' if mode == 'bf16' else 'i8', value_cache_ptr='bf16',
        seq_lens_ptr='i32', block_table_ptr='i32')
    result = []
    for arg in fn.args.args:
        annotation = ast.unparse(arg.annotation) if arg.annotation else None
        if annotation == 'tl.constexpr':
            continue
        dtype = '!tt.ptr<' + pointers[arg.arg] + '>' if arg.arg in pointers else (
            'i64' if annotation == 'tl.int64' else 'f32' if arg.arg == 'softmax_scale_log2e' else 'i32')
        result.append((arg.arg, dtype))
    return result


def typed_abi(artifacts, source, function, mode):
    tt = [s for s in artifacts['ttir'].splitlines() if 'tt.func public @' in s]
    ll = [s for s in artifacts['llir'].splitlines() if s.startswith('define iluvatar_kernel void @')]
    if len(tt) != 1 or len(ll) != 1:
        raise ValueError('One actual public target TTIR/LLVM entry required')
    typed = re.findall(r'%([\w_]+):\s*(!tt.ptr<[^>]+>|f32|i64|i32)(?=\s|,|\))', tt[0])
    actual = re.findall(r'(ptr addrspace\(1\)|float|i64|i32)\s+(?:[^%,]*?\s+)?%(\d+)', ll[0])
    expected = expected_abi(source, function, mode)
    llvm_expected = ['ptr addrspace(1)' if t.startswith('!tt.ptr') else 'float' if t == 'f32' else t for _, t in expected]
    # The two implicit target pointer arguments are observed in the pinned
    # CoreX3.6 producer/reducer ABI; a changed ABI must be reviewed, not accepted.
    llvm_expected += ['ptr addrspace(1)'] * 2
    if typed != expected or [t for t, _ in actual] != llvm_expected or [int(n) for _, n in actual] != list(range(len(llvm_expected))):
        raise ValueError('Actual mixed typed ABI does not match source/call contract')
    return dict(typed_abi_match=True, ttir_parameters=typed, llvm_parameters=actual,
                implicit_vendor_pointers=2, native_accumulator_component_independence_assumed=False)


def audit_bf16(artifacts, source):
    tt, ll, asm = [artifacts.get(k, '') for k in ('ttgir', 'llir', 'asm')]
    dots = [s for s in tt.splitlines() if '= tt.dot ' in s]
    qk = [s for s in dots if re.search(r'tensor<8x128xbf16,.*\* tensor<128x16xbf16,.*-> tensor<8x16xf32,', s)]
    pv = [s for s in dots if re.search(r'tensor<8x16xbf16,.*\* tensor<16x128xbf16,.*-> tensor<8x128xf32,', s)]
    calls = [s for s in ll.splitlines() if 'call <4 x float> @llvm.bi.matrix.mad.f32x4.bf16x4(' in s]
    locations = {int(k): int(v) for k, v in re.findall(r'!(\d+) = !DILocation\(line: (\d+)', ll)}
    tree = ast.parse(source)
    sites = {line for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n) == 'tl.dot(probability.to(V.dtype), V)'
             for line in range(n.lineno, n.end_lineno + 1)}
    def site(line):
        match = re.search(r'!dbg !(\d+)', line)
        return locations.get(int(match[1])) if match else None
    pv_calls = [s for s in calls if site(s) in sites]
    defs = dict(re.findall(r'^\s*(%[\w.$-]+) = (.*)$', ll, re.M)); users = {}
    for value, expression in defs.items():
        for operand in re.findall(r'%[\w.$-]+', expression):
            users.setdefault(operand, []).append((value, expression))
    components, chain, post_add_absent = [], True, True
    for line in pv_calls:
        result = re.search(r'(%[\w.$-]+) = ', line)
        extracts = users.get(result[1], []) if result else []
        post_add_absent &= len(extracts) == 4
        for value, expression in extracts:
            post_add_absent &= expression.startswith('extractelement <4 x float> ') and bool(users.get(value)) and all(e.startswith('phi float ') for _, e in users[value])
        argument = re.search(r', <4 x float> (%[\w.$-]+)\)', line)
        if argument is None:
            chain = False; continue
        value = argument[1]; values = {}
        for _ in range(4):
            node = re.match(r'insertelement <4 x float> (poison|%[\w.$-]+), float (%[\w.$-]+), i\d+ ([0-3])', defs.get(value, ''))
            if node is None:
                chain = False; break
            values[int(node[3])] = node[2]; value = node[1]
        chain &= value == 'poison' and set(values) == set(range(4))
        for item in values.values():
            chain &= defs.get(item, '').startswith('fmul float ')
            components.append(item)
    opcodes = Counter(re.findall(r'^\s*(ml_\w+)\s', asm, re.M))
    flags = dict(required_artifacts=all((tt, ll, asm)), original_qk_and_pv_shapes=len(dots) == 4 and len(qk) == len(pv) == 2,
        native_bf16_calls=len(calls) == 32, pv_calls=len(pv_calls) == 16,
        all_64_scaled_pv_c_components=bool(chain) and len(components) == len(set(components)) == 64,
        pv_post_add_absent=bool(post_add_absent), machine_bf16_mmas=opcodes['ml_matrix_mad_f32_bf16_m16n16k16'] == 32,
        no_i8_mma=opcodes['ml_matrix_mad_i32_i8_m16n16k32'] == 0)
    return dict(passed=all(flags.values()), flags=flags, isa_opcodes=dict(opcodes),
        actual_metadata_cfg_verified=False, native_accumulator_component_independence_assumed=False)


def mechanism(artifacts, source, mode):
    report = audit_bf16(artifacts, source) if mode == 'bf16' else audit_k8(artifacts, source)
    report['metadata_guard_correctness_requires_numeric_and_actual_cfg_review'] = True
    return report
