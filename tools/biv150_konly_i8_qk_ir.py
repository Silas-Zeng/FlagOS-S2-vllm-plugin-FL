"""Standard-library gates over actual compiler artifacts, never synthetic passes."""
import ast
from collections import Counter
from pathlib import Path
import re


def audit_artifacts(artifacts, source=None):
    """Unknown or missing evidence fails the corresponding mechanism gate.

    LLVM cast counts and ISA opcode frequencies are recorded separately. LLVM
    scalar counts are not claimed to be instruction costs or unique output count.
    """
    source = source or (Path(__file__).parent / 'kernels/biv150_konly_i8_qk.py').read_text()
    tree = ast.parse(source)
    cast_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'to'
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == 'raw']
    assert len(cast_nodes) == 1
    cast_span = range(cast_nodes[0].lineno, cast_nodes[0].end_lineno + 1)
    tt, ll, asm = [artifacts.get(k) if isinstance(artifacts.get(k), str) else ''
                   for k in ('ttgir','llir','asm')]
    native = [line for line in ll.splitlines()
              if 'call <4 x i32> @llvm.bi.matrix.mad.i32x4.i8x8(' in line]
    chain = len(native) == 4
    previous = None
    for line in native:
        lhs = re.search(r'(%[\w.]+)\s*=.*call', line)
        c = re.search(r', <4 x i32> (zeroinitializer|%[\w.]+)\)', line)
        chain = chain and lhs is not None and c is not None
        if lhs is None or c is None:
            continue
        chain = chain and c[1] == ('zeroinitializer' if previous is None else previous)
        previous = lhs[1]
    dots = [line for line in tt.splitlines() if re.search(r'\btt\.dot\b', line)]
    shape = (len(dots) == 1 and re.search(
        r'tensor<8x128xi8,.*\* tensor<128x16xi8,.*-> tensor<8x16xi32,', dots[0]) is not None)
    locations = {int(a): int(b) for a, b in re.findall(r'!(\d+) = !DILocation\(line: (\d+)', ll)}
    casts = []
    for number, line in enumerate(ll.splitlines(), 1):
        match = re.search(r'sitofp\s+(i32|<(\d+) x i32>)\s+.*?\s+to\s+(float|<\d+ x float>)', line)
        if match:
            dbg = re.search(r'!dbg !(\d+)', line)
            casts.append(dict(llir_line=number, elements_per_lane=int(match[2] or 1),
                              source_line=locations.get(int(dbg[1])) if dbg else None))
    opcodes = Counter(re.findall(r'^\s*(ml_[a-zA-Z0-9_]+)\s', asm, re.M))
    flags = dict(required_artifacts_present=all(isinstance(artifacts.get(k), str) and artifacts[k]
                                               for k in ('ttgir','llir','asm')),
                 logical_m8_k128_n16_signed_i8_i32=shape,
                 native_k32_encoding=re.search(r'instrShape\s*=\s*\[\s*16\s*,\s*16\s*,\s*32\s*\]',tt) is not None,
                 exactly_four_i8_mma_calls=len(native) == 4,
                 complete_native_c_chain=bool(chain),
                 exactly_four_i8_machine_mmas=opcodes['ml_matrix_mad_i32_i8_m16n16k32'] == 4,
                 no_bf16_mma=not any('matrix_mad_f32_bf16' in k for k in opcodes),
                 actual_score_cast_present=bool(casts),
                 casts_at_exact_raw_to_float_ast_site=bool(casts) and all(c['source_line'] in cast_span for c in casts))
    return dict(passed=all(flags.values()), flags=flags, i32_to_f32_casts=casts,
                llvm_cast_elements_per_lane=sum(c['elements_per_lane'] for c in casts),
                logical_score_elements=128, isa_opcode_frequencies=dict(opcodes),
                isa_cast_cost_inferred=False, native_abi_component_independence_assumed=False,
                caveat='No LLVM count is presented as hardware latency or reclaimed registers.')
