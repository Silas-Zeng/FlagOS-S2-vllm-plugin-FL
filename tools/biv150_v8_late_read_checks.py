"""CPU-only source inverse and conservative actual-lowering checks (no imports of Triton)."""
import ast
import copy
import hashlib
from pathlib import Path
import re

DEPENDENCY = '4a1f7ddd38e4bcade30e702faba2e9a7cdb5441d'
FROZEN_KERNEL_SHA = 'a7fc539dccbd2e2910538665abf42d0dd6204939d39f1ebeb7a6cac1f992c3ca'
FROZEN_BENCH_SHA = '59546f4657d48e05e3e15d5d5c807f4568126f073dd9683d371e3766cba530ad'
ROOT = Path(__file__).resolve().parents[1]
FROZEN_KERNEL = ROOT/'tools/kernels/biv150_v8_bf16p.py'
CANDIDATE_KERNEL = ROOT/'tools/kernels/biv150_v8_late_read.py'
FROZEN_BENCH = ROOT/'tools/bench_biv150_v8_bf16p.py'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def producer(raw):
    tree = ast.parse(raw)
    return copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name == 'gqa_decode_v8_bf16p'))


def assignment(node, name):
    return isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
        node.targets[0], ast.Name) and node.targets[0].id == name


def paths(fn):
    full = next(n for n in fn.body if isinstance(n, ast.While))
    tail = next(n for n in fn.body if isinstance(n, ast.If) and
                ast.unparse(n.test) == 'full_upper < upper')
    return (full.body, tail.body)


def source_inverse(candidate=None):
    frozen = FROZEN_KERNEL.read_bytes()
    assert sha(frozen) == FROZEN_KERNEL_SHA, 'Frozen73 kernel changed'
    assert sha(FROZEN_BENCH.read_bytes()) == FROZEN_BENCH_SHA, 'Frozen4a benchmark changed'
    raw = CANDIDATE_KERNEL.read_bytes() if candidate is None else candidate
    fn, original = producer(raw), producer(frozen)
    movements = []
    for body, old in zip(paths(fn), paths(original)):
        vi = next(i for i, n in enumerate(body) if assignment(n, 'V'))
        si = next(i for i, n in enumerate(body) if assignment(n, 'value_scale'))
        pi = next(i for i, n in enumerate(body) if assignment(n, 'accumulator'))
        total = next(i for i, n in enumerate(body) if assignment(n, 'running_sum'))
        assert total < vi and si == vi+1 and pi == si+1, 'Loads must immediately precede PV'
        moved = body[vi:si+1]
        assert all(isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'tl.load'
                   for n in moved), 'Only two readonly loads may move'
        assert ast.dump(moved[0]) == ast.dump(next(n for n in old if assignment(n, 'V')))
        assert ast.dump(moved[1]) == ast.dump(next(n for n in old if assignment(n, 'value_scale')))
        dependencies = {n.id for statement in moved for n in ast.walk(statement)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        writes_between = {n.id for statement in body[total:vi] for n in ast.walk(statement)
                          if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        # The full AST inverse below also proves no cache writer or new math is
        # introduced anywhere. Inputs to the moved loads cannot be redefined.
        crossed = body[next(i for i,n in enumerate(body) if assignment(n, 'K'))+1:vi]
        writes_between |= {n.id for statement in crossed for n in ast.walk(statement)
                           if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        assert not (dependencies & writes_between), 'Moved load has a crossed data dependence'
        del body[vi:si+1]
        insert = next(i for i,n in enumerate(body) if assignment(n, 'K'))+1
        body[insert:insert] = moved
        movements.append(dict(loads=['V', 'value_scale'], from_after='running_sum',
                              restored_after='K', independent_of_crossed_writes=True))
    assert ast.dump(fn) == ast.dump(original), 'Inverse is not exact frozen producer AST'
    tree = ast.parse(raw)
    assert [n.name for n in tree.body if isinstance(n, ast.FunctionDef)] == ['gqa_decode_v8_bf16p']
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert len(imports) == 1 and imports[0].module == 'kernels.biv150_v8_bf16p'
    assert [(n.name, n.asname) for n in imports[0].names] == [('encode_cache_v', None), ('append_token_kv', None)]
    # No executable module-level additions beyond the pinned imports, LN2,
    # producer and docstring are allowed.
    expected = ast.parse(frozen)
    permitted = [n for n in expected.body if not isinstance(n, ast.FunctionDef)]
    actual = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ImportFrom))]
    assert ast.dump(ast.Module(body=actual[1:], type_ignores=[])) == ast.dump(
        ast.Module(body=permitted[1:], type_ignores=[]))
    return dict(dependency=DEPENDENCY, frozen_sha256=sha(frozen), candidate_sha256=sha(raw),
                exact_ast_inverse=True, movements=movements, cache_writes_unchanged=True,
                numeric_expression_unchanged=True, scheduling_proven=False)


def adapted_benchmark_tree():
    """Rebind only the experimental filename and inject a pre-launch hash gate.

    The original numeric/codec/writer/Graph/timing implementation stays frozen.
    This factory is CPU Python compilation, not Triton JIT.
    """
    raw = FROZEN_BENCH.read_bytes()
    assert sha(raw) == FROZEN_BENCH_SHA
    tree = ast.parse(raw)
    replacements = 0
    insertion = ast.parse('own.gqa_decode_v8_bf16p = GuardedKernel(own.gqa_decode_v8_bf16p, args.ir_proof)').body[0]

    class Rebind(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal replacements
            if node.value == 'tools/kernels/biv150_v8_bf16p.py':
                replacements += 1
                return ast.copy_location(ast.Constant('tools/kernels/biv150_v8_late_read.py'), node)
            return node

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if assignment(node, 'own'):
                return [node, copy.deepcopy(insertion)]
            return node

    changed = Rebind().visit(copy.deepcopy(tree))
    assert replacements == 1
    # Independently reverse the factory adaptation. Any guard/tolerance/timing
    # change, including a forgotten baseline gate, fails this equality.
    class Restore(ast.NodeTransformer):
        removed = 0

        def visit_Constant(self, node):
            if node.value == 'tools/kernels/biv150_v8_late_read.py':
                return ast.copy_location(ast.Constant('tools/kernels/biv150_v8_bf16p.py'), node)
            return node

        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(insertion):
                self.removed += 1
                return None
            return self.generic_visit(node)

    restore = Restore()
    inverse = restore.visit(copy.deepcopy(changed))
    assert restore.removed == 1 and ast.dump(inverse) == ast.dump(tree)
    return ast.fix_missing_locations(changed)


def matrix_groups(lines, token):
    indices = [i for i, line in enumerate(lines) if token in line and
               not line.lstrip().startswith('declare')]
    assert len(indices) == 32, 'Unrecognized w1 PAGE16 MMA lowering (need full/tail x QK8/PV8)'
    return [(indices[i:i+8], indices[i+8:i+16]) for i in (0, 16)]


def asm_records(text):
    records, files = [], {}
    source_file, source_line = None, None
    for line_number, line in enumerate(text.splitlines(), 1):
        file = re.match(r'\s*\.file\s+(\d+)\s+"[^"]*"\s+"([^"]+)"', line)
        if file:
            files[int(file[1])] = file[2]
        location = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)\b', line)
        if location:
            source_file, source_line = files.get(int(location[1])), int(location[2])
        match = re.match(r'\s*((?:ml|sl)_\w+)\b', line)
        if match:
            records.append(dict(line=line_number, opcode=match[1], text=line.strip(),
                                source_file=source_file, source_line=source_line))
    return records


def lowering_check(ttgir, llir, assembly, *, previous=None, source=None, source_name=None):
    """Fail closed on unfamiliar lowering; line-order evidence is not latency.

    Check both full and masked-tail paths, typed BF16 MMA, actual packed byte
    extraction and conversions after exp2, and unchanged staging instruction
    counts. No register-count threshold substitutes for this mechanism check.
    """
    reasons, evidence = [], {}
    def require(condition, reason):
        if not condition:
            reasons.append(reason)
    try:
        source = CANDIDATE_KERNEL.read_bytes() if source is None else source
        source_name = CANDIDATE_KERNEL.name if source_name is None else source_name
        load_ranges = []
        scale_ranges = []
        for body in paths(producer(source)):
            load = next(n for n in body if assignment(n, 'V'))
            load_ranges.append(range(load.lineno, load.end_lineno+1))
            scale = next(n for n in body if assignment(n, 'value_scale'))
            scale_ranges.append(range(scale.lineno, scale.end_lineno+1))
        dots = [line for line in ttgir.splitlines() if ' = tt.dot ' in line]
        require(len(dots) == 4 and all(line.count('xbf16') == 2 and 'xf32' in line for line in dots),
                'TTGIR must retain four BF16xBF16 -> FP32 QK/PV dots')
        require(sum('arith.sitofp ' in line and 'xi8' in line and 'xbf16' in line
                    for line in ttgir.splitlines()) == 2, 'TTGIR V conversion differs')
        require('!tt.ptr<i8>' in ttgir and '!tt.ptr<f32>' in ttgir, 'V/scale pointer dtypes missing')
        llvm = llir.splitlines()
        llvm_groups = matrix_groups(llvm, '@llvm.bi.matrix.mad.f32x4.bf16x4(')
        previous_end = -1
        llvm_paths = []
        for qk, pv in llvm_groups:
            exps = [i for i in range(qk[-1]+1, pv[0]) if '@llvm.nvvm.ex2.' in llvm[i]]
            extracts = [i for i in range(previous_end+1, pv[0]) if 'extractelement <4 x i8>' in llvm[i]]
            casts = [i for i in range(previous_end+1, pv[0]) if 'sitofp i8 ' in llvm[i] and ' to bfloat' in llvm[i]]
            packed = [i for i in range(previous_end+1, pv[0]) if '@llvm.bi.load.kop.v4i8(' in llvm[i]]
            scales = [i for i in range(previous_end+1, pv[0]) if '@llvm.bi.load.kop.v1f32(' in llvm[i]]
            require(len(exps) == 4 and len(extracts) == 32 and len(casts) == 32 and len(packed) == 8,
                    'LLVM path schema differs (exp2=4, packed V loads=8, V extract/cast=32 each required)')
            late = bool(exps and extracts and casts and min(extracts) > max(exps) and min(casts) > max(exps))
            require(late, 'LLVM still has V byte extraction/conversion before softmax finishes')
            llvm_paths.append(dict(qk=[i+1 for i in qk], pv=[i+1 for i in pv],
                exp2=[i+1 for i in exps], v_extracts=[i+1 for i in extracts],
                v_casts=[i+1 for i in casts], v_packed_loads=[i+1 for i in packed],
                scale_loads=[i+1 for i in scales], late=late))
            previous_end = pv[-1]
        evidence['llvm_paths'] = llvm_paths
        asm = assembly.splitlines()
        asm_groups = matrix_groups(asm, 'ml_matrix_mad_f32_bf16_m16n16k16 ')
        instructions = asm_records(assembly)
        require(not any('matrix_mad' in r['opcode'] and 'bf16' not in r['opcode'] for r in instructions),
                'Non-BF16 MMA introduced')
        # The first full-loop label is before all V loads; tail may have several
        # mask branches, so start its range just after full PV. Never infer cycles.
        labels = [i for i,line in enumerate(asm[:asm_groups[0][0][0]]) if re.match(r'\.?L?BB\w*:', line)]
        assert labels, 'Unknown loop-label syntax'
        start, asm_paths = labels[-1], []
        for path_index, (qk, pv) in enumerate(asm_groups):
            records = [r for r in instructions if start < r['line'] <= pv[-1]+1]
            exps = [r['line'] for r in records if r['opcode'] == 'ml_exp_f32']
            unpack = [r['line'] for r in records if r['opcode'] in ('ml_bfe2_b32', 'ml_and_b32')
                      and r['source_file'] == source_name and r['source_line'] in load_ranges[path_index]]
            casts = [r['line'] for r in records if r['opcode'] in ('ml_cvt_i16_i8', 'ml_cvt_bf16_i16')]
            packed = [r['line'] for r in records if r['opcode'] == 'ml_lsa_load_a64_dword'
                      and r['source_file'] == source_name and r['source_line'] in load_ranges[path_index]]
            scales = [r['line'] for r in records if 'load' in r['opcode']
                      and r['source_file'] == source_name and r['source_line'] in scale_ranges[path_index]]
            require(len(exps) == 4 and len(unpack) == 32 and len(casts) == 64 and len(packed) == 8,
                    'ASM schema differs (exp=4, packed V loads=8, V expand=32, V conversions=64 required)')
            late = bool(exps and unpack and casts and min(unpack) > max(exps) and min(casts) > max(exps))
            require(late, 'ASM still expands/converts packed V before softmax finishes')
            asm_paths.append(dict(qk=[i+1 for i in qk], pv=[i+1 for i in pv], exp=exps,
                                  v_expand=unpack, v_casts=casts, v_packed_loads=packed,
                                  scale_loads=scales, late=late))
            start = pv[-1]
        evidence['asm_paths'] = asm_paths
        counts = {op: sum(r['opcode'] == op for r in instructions) for op in
                  ('ml_slb_store_b16', 'ml_slb_store_b32x2', 'sl_barrier',
                   'ml_cvt_i16_i8', 'ml_cvt_bf16_i16', 'ml_matrix_mad_f32_bf16_m16n16k16')}
        evidence['static_counts_not_time'] = counts
        if previous is not None:
            old_counts = {op: sum(r['opcode'] == op for r in asm_records(previous['asm'])) for op in counts}
            require(counts == old_counts, 'Shared store/barrier/conversion/MMA instruction counts changed')
            # Compare the actual V blocked layout, not numeric #blocked IDs.
            def v_layout(text):
                load = next(line for line in text.splitlines() if ' = tt.load ' in line and '!tt.ptr<i8>' in line)
                name = re.findall(r'#blocked\d*', load)[-1]
                return next(line.split(' = ', 1)[1] for line in text.splitlines() if line.startswith(name+' = '))
            require(v_layout(ttgir) == v_layout(previous['ttgir']), 'V blocked layout changed')
            shared = lambda text: sorted(line.split(' = ', 1)[1] for line in text.splitlines()
                                          if re.match(r'#shared\d* = ', line))
            require(shared(ttgir) == shared(previous['ttgir']), 'Shared layout definitions changed')
    except (AssertionError, StopIteration, IndexError, ValueError) as error:
        reasons.append('Unrecognized lowering: '+str(error))
    return dict(mechanism_passed=not reasons, reasons=reasons, evidence=evidence,
                precision_gate=False, static_instruction_count_is_time=False,
                limitation='Conservative schema check; changed schemas stop for manual review. Resource counts do not prove speed.')
