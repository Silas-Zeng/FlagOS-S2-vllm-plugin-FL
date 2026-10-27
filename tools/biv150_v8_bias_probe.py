"""Private probe contracts/adaptation only; importing this module is CPU-only."""
import ast
import copy
import re

from biv150_v8_decode_factory import ROOT, sha

PINS = {
    'tools/fixtures/biv150_v8_probe_4a.py.txt': '59546f4657d48e05e3e15d5d5c807f4568126f073dd9683d371e3766cba530ad',
    'tools/fixtures/biv150_v8_descriptors_ac6.py.txt': 'e3dc2db7634e31339c30de4b09c46f3fd4ebefde5fbbbf5cfe848922d1b59ac0',
    'tools/biv150_v8_bf16p_reference.py': 'b07f567df00fe77af7dbbca8cb25e7e13c38472489b95ae7b792881a52d51600',
    'tools/biv150_v8_bf16p_validation.py': '60f6e18a163d3ac8d9bc0c330962eb1cef50680f800a4c6afd9e8820f129c084',
    'tools/biv150_benchmark_utils.py': 'ff0cd964aad34e6d19609fa4e1d76e7d46b4f98c98a4d925e33d658efd4e2879',
}
CASES = {'ragged': (8,257,16), 'ragged-long': (8,257,16),
         'steady4k': (64,4096,2), 'steady16k': (32,16384,4)}
OPTION_KEYS = ('enable_fp_fusion','enable_reflect_ftz','default_dot_input_precision',
               'maxnreg','backend_name','arch','sanitize_overflow','debug')


def pinned_sources():
    for path,wanted in PINS.items():
        assert sha((ROOT/path).read_bytes()) == wanted, ('Changed frozen helper',path)
    return dict(PINS)


def descriptor_functions():
    pinned_sources()
    namespace = {}
    # Exact ac6 function slices; definitions only. compile_tensors receives the
    # Torch object only from an explicit device mode, never from CPU preview.
    exec(compile((ROOT/'tools/fixtures/biv150_v8_descriptors_ac6.py.txt').read_text(),
                 'frozen_ac6_descriptors','exec'),namespace)
    return namespace['runtime_descriptors'],namespace['compile_tensors']


def mechanism_check(old, new):
    """Conservative w1 full/tail grammar. Unknown lowering cannot license launch."""
    reasons = []
    evidence = {}
    try:
        for label,ir in [('old',old),('A',new)]:
            dots = [line for line in ir['ttgir'].splitlines() if ' = tt.dot ' in line]
            assert len(dots)==4 and all(line.count('xbf16')==2 and 'xf32' in line for line in dots), 'BF16 QK/PV schema'
            evidence[label+'_numeric_i2fp'] = sum(bool(re.search(r'=\s*(?:s|u)itofp\b',line)) for line in ir['llir'].splitlines())
            if label=='A' and re.search(r'arith\.(?:sitofp|uitofp)\b',ir['ttgir']):
                reasons.append('A still has numeric integer-to-float conversion in TTGIR')
            ops = re.findall(r'^\s*((?:ml|sl)_\w+)\b',ir['asm'],re.M)
            evidence[label+'_ops'] = {name:ops.count(name) for name in sorted(set(ops))}
            evidence[label+'_mma'] = ops.count('ml_matrix_mad_f32_bf16_m16n16k16')
            assert evidence[label+'_mma']==32, 'w1 full/tail MMA count'
        assert evidence['old_numeric_i2fp']==64, 'Old V8 LLVM conversion schema'
        assert evidence['old_ops'].get('ml_cvt_i16_i8')==64 and evidence['old_ops'].get('ml_cvt_bf16_i16')==64, 'Old V8 ASM conversion schema'
        if evidence['A_numeric_i2fp']:
            reasons.append('A still numerically converts integer V to floating point in LLVM')
        forbidden = [name for name in evidence['A_ops'] if name in ('ml_cvt_i16_i8','sl_cvt_i16_i8') or
                     re.match(r'(?:ml|sl)_cvt_(?:f16|f32|f64|bf16)_[iu](?:8|16|32|64)',name)]
        if forbidden:
            reasons.append('A retains numeric integer-to-float CVT / old two-stage decode: '+','.join(forbidden))
        # Require actual dataflow through FP32 bitcasts and the requested bias,
        # not merely absence of a named instruction. Alternate vector schemas
        # stop for review rather than being guessed as a successful mechanism.
        lines = new['llir'].splitlines()
        mma = [i for i,line in enumerate(lines) if '@llvm.bi.matrix.mad.f32x4.bf16x4(' in line and not line.startswith('declare')]
        assert len(mma)==32
        previous = -1
        paths = []
        for offset in (0,16):
            end = mma[offset+8]
            region = lines[previous+1:end]
            bitcasts = {match[1] for line in region if (match:=re.search(r'(%[\w.]+) = bitcast i32 .* to float',line))}
            bias = [line for line in region if re.search(r'= f(?:sub|add)\b.*float',line)
                    and any(re.search(re.escape(token)+r'\b',line) for token in bitcasts)
                    and re.search(r'8388736|8\.388736[eE]\+0?6|0x[Cc4]160001000000000',line)]
            loads = sum('@llvm.bi.load.kop.v4i8(' in line for line in region)
            if len(bias)!=32 or loads!=8:
                reasons.append('Unknown A bias/packed-load schema: need32 scalar bias operations and8 packed loads per path')
            paths.append(dict(bias_operations=len(bias),packed_v_loads=loads))
            previous = mma[offset+15]
        evidence['A_paths'] = paths
        for name in ('ml_slb_store_b16','ml_slb_store_b32x2','sl_barrier'):
            if evidence['old_ops'].get(name,0)!=evidence['A_ops'].get(name,0):
                reasons.append('Shared staging changed: '+name)
        assert evidence['old_ops'].get('ml_slb_store_b16')==64, 'Old V8 scalar store schema'
        layouts = lambda text: sorted(line.split(' = ',1)[1] for line in text.splitlines() if re.match(r'#shared\d* = ',line))
        if layouts(old['ttgir'])!=layouts(new['ttgir']):
            reasons.append('Shared layout changed; manual review required')
        def staging(text):
            lines=[]
            for line in text.splitlines():
                if re.match(r'\s*(?:ml_slb_store_\w+|sl_barrier)\b',line):
                    line=re.sub(r'\b([vs])\[(\d+):(\d+)\]',lambda m:m[1]+'['+str(int(m[3])-int(m[2])+1)+']',line)
                    lines.append(re.sub(r'\b([vs])\d+\b',r'\1',line.strip()))
            return sorted(lines)
        if staging(old['asm'])!=staging(new['asm']):
            reasons.append('Shared store immediate/mode or barrier signature changed')
    except (AssertionError,KeyError,IndexError,ValueError) as error:
        reasons.append('Unknown lowering: '+str(error))
    return dict(mechanism_passed=not reasons,reasons=reasons,evidence=evidence,
                static_instruction_counts_are_time=False,precision_gate=False,official_score=False)


def adapted_probe():
    """Reversible AST edits around the complete pinned4a consumer and gates."""
    pinned_sources()
    original = ast.parse((ROOT/'tools/fixtures/biv150_v8_probe_4a.py.txt').read_bytes())
    tree = copy.deepcopy(original)
    inverse = []
    def replace(target,replacement):
        wanted = ast.dump(target)
        count = 0
        class Swap(ast.NodeTransformer):
            def visit(self,node):
                nonlocal count
                if isinstance(node,ast.AST) and ast.dump(node)==wanted:
                    count += 1
                    return copy.deepcopy(replacement)
                return super().visit(node)
        result = Swap().visit(tree)
        assert count==1, ('Nonunique adaptation',wanted[:80],count)
        assert result is tree
        inverse.append((copy.deepcopy(replacement),copy.deepcopy(target)))
    def after(target, statements):
        # Replace one statement by a list; reverse matches the complete list in
        # a statement-list field. No source gate is edited inside the old nodes.
        replace(target,[copy.deepcopy(target),*ast.parse(statements).body])
    find = lambda name: next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    replace(find('kernel_path'),ast.parse('kernel_path = args.kernel_source').body[0])
    own = find('own')
    new_own = copy.deepcopy(own);new_own.value.args[1] = ast.parse('args.kernel_module',mode='eval').body
    replace(own,new_own)
    after(find('base'),'wrap_base(base, args)')
    after(find('own'),'decode_gate = make_gate(own, args)')
    # make_gate wraps own's writer/encoder as well as its producer in place.
    run = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run')
    modified_run=copy.deepcopy(run)
    modified_run.body[:0]=ast.parse('encode = decode_gate.select(encode)').body
    replace(run,modified_run)
    after(find('replay'),'graph_snapshot = snapshot_fields(c)')
    equality = next(n for n in ast.walk(tree) if isinstance(n,ast.Assert) and
                    isinstance(n.msg,ast.Constant) and n.msg.value=='Changed-input graph/eager mismatch')
    after(equality,'assert_snapshots_equal(graph_snapshot, snapshot_fields(c))')
    artifact = find('artifact');new_artifact = copy.deepcopy(artifact)
    new_artifact.value.keywords.append(ast.keyword(arg='bitwise_fields',value=ast.parse('snapshot_fields(c)',mode='eval').body))
    replace(artifact,new_artifact)
    saved = next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and
                 ast.unparse(n)=='torch.save(artifact, evidence_path)')
    after(saved,'capture_evidence(args, fixture, artifact, frozen_key, frozen_value)')
    # Only the timing block changes: old phase is numeric-only, A phase times
    # current-main BF16, old V8 and A, all through original run(writer=True).
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    case_loop = next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=='name')
    start = next(i for i,n in enumerate(case_loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='orders' for t in n.targets))
    end = next(i for i,n in enumerate(case_loop.body[start:],start) if isinstance(n,ast.Delete))
    old_block = copy.deepcopy(case_loop.body[start:end])
    block = ast.parse('''if args.numeric_only:
    row.update(completed=True, timing_performed=False)
    save()
else:
    require_equivalence(args, name)
    orders = ((False, 'old-v8', True),) if args.numeric_diagnostic else ((False, 'old-v8', True), (True, 'old-v8', False), (False, 'old-v8', True))
    for index, order in enumerate(orders):
        timings = {}
        for token in order:
            timings[timing_label(token)] = bench.graph_time(lambda e=token: run(e))[0]
        row['rounds'].append(dict(index=index, order=[timing_label(x) for x in order], milliseconds=timings,
            speedup=timings['main_bf16']/timings['bias_A'], old_v8_over_A=timings['frozen_v8']/timings['bias_A']))
        save()
    row['speedup_median'] = statistics.median(v['speedup'] for v in row['rounds'])
    row['attention_only_ms'] = None
    row.update(completed=True, timing_performed=True)
    save()
''').body[0]
    case_loop.body[start:end] = [block]
    inverse.append((copy.deepcopy(block),old_block))
    # Reverse each adaptation, including statement-list insertions, then compare
    # the complete original module AST. Original numeric gates/tolerances are
    # therefore retained rather than being rewritten in a new implementation.
    restored = copy.deepcopy(tree)
    for changed,before in reversed(inverse):
        count = 0
        class Undo(ast.NodeTransformer):
            def generic_visit(self,node):
                nonlocal count
                for field,value in ast.iter_fields(node):
                    if isinstance(value,list) and isinstance(changed,list):
                        for i in range(len(value)-len(changed)+1):
                            if all(isinstance(a,ast.AST) and ast.dump(a)==ast.dump(b) for a,b in zip(value[i:i+len(changed)],changed)):
                                value[i:i+len(changed)] = [copy.deepcopy(before)];count+=1;break
                return super().generic_visit(node)
            def visit(self,node):
                nonlocal count
                if not isinstance(changed,list) and isinstance(node,ast.AST) and ast.dump(node)==ast.dump(changed):
                    count += 1
                    return copy.deepcopy(before)
                return super().visit(node)
        restored = Undo().visit(restored)
        assert count==1, ('Inverse adaptation failed',count)
    assert ast.dump(restored)==ast.dump(original), 'Original consumer/gates changed'
    return ast.fix_missing_locations(tree),dict(exact_frozen_module_ast_inverse=True,
        adaptations=len(inverse),full_cost_main_bf16_control=True,old_v8_numeric_phase_untimed=True,
        A_equivalence_required_before_timing=True,precision_gate=False)
