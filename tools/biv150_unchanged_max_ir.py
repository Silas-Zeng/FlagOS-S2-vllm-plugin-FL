"""Conservative actual-artifact gates. Unknown lowering is a rejection, not a pass."""
import re


def need(condition, message):
    if not condition:
        raise ValueError(message)


def audit_ttgir(text):
    lines = text.splitlines()
    definitions = {m[1]: m[2] for line in lines if (m := re.match(r'\s*(%\w+) = (.*)', line))}

    def operands(value, operation):
        expression = definitions.get(value, '')
        need(expression.startswith(operation), 'Unrecognized '+operation+' chain for '+value)
        return re.findall(r'%\w+', expression.split(' : ')[0])

    branches = []
    for i, line in enumerate(lines):
        match = re.search(r'(%[\w]+):2 = scf.if (%[\w]+).*tensor<8x128xf32', line)
        if not match:
            continue
        indent = len(line) - len(line.lstrip())
        split = end = None
        for j in range(i+1, len(lines)):
            if len(lines[j]) - len(lines[j].lstrip()) == indent:
                if lines[j].lstrip().startswith('} else {'):
                    split = j
                elif lines[j].lstrip().startswith('}'):
                    end = j
                    break
        need(split is not None and end is not None, 'Cannot recover rescale SCF branches')
        yes, no = '\n'.join(lines[i+1:split]), '\n'.join(lines[split+1:end])
        alpha = re.findall(r'(%\w+) = math.exp2 (%\w+)', yes)
        scale = re.findall(r'(%\w+) = arith.mulf (%\w+), (%\w+) : tensor<8x128xf32', yes)
        fma = re.findall(r'(%\w+) = math.fma (%\w+), (%\w+), (%\w+)', yes)
        add = re.findall(r'(%\w+) = arith.addf (%\w+), (%\w+) : tensor<8xf32', no)
        yes_yield = re.findall(r'scf.yield (%\w+), (%\w+) :', yes)
        no_yield = re.findall(r'scf.yield (%\w+), (%\w+) :', no)
        need(len(alpha) == len(scale) == len(fma) == len(add) == len(yes_yield) == len(no_yield) == 1,
             'Missing conditional alpha/accumulator scale/explicit FMA/skip state')
        need(fma[0][2] == alpha[0][0], 'Correction FMA does not use alpha')
        need(yes_yield[0] == (fma[0][0], scale[0][0]), 'Correction state order changed')
        need(no_yield[0] == (add[0][0], scale[0][1]), 'Skip path changes accumulator')
        need((fma[0][1], fma[0][3]) == (add[0][1], add[0][2]),
             'Correction and skip use different L/probability sums')
        scale_expand = operands(scale[0][2], 'tt.broadcast')[0]
        need(operands(scale_expand, 'tt.expand_dims') == [alpha[0][0]], 'Accumulator scale is not alpha')
        need('math.exp2' not in no and 'arith.mulf' not in no and 'tt.dot' not in yes+no,
             'Unexpected skip math or duplicated PV')
        branches.append(dict(result=match[1], begin=i+1, end=end+1,
                             alpha_delta=alpha[0][1], probability_sum=fma[0][3]))
    need(len(branches) == 2, 'Expected full and tail conditional rescale regions')
    need(len(re.findall(r'= math.exp2 ', text)) == 4, 'Unexpected unconditional or missing exp path')
    pv = [line for line in lines if '= tt.dot ' in line and '-> tensor<8x128xf32' in line]
    qk = [line for line in lines if '= tt.dot ' in line and '-> tensor<8x16xf32' in line]
    need(len(pv) == len(qk) == 2, 'QK/PV shape/count changed')
    for branch in branches:
        uses = [line for line in pv if re.search(r', '+re.escape(branch['result'])+r'#1,', line)]
        need(len(uses) == 1, 'PV does not consume the conditional accumulator as C')
        p_fragment = re.search(r'= tt.dot (%\w+),', uses[0])[1]
        p_bf16 = operands(p_fragment, 'ttg.convert_layout')[0]
        p_value = operands(p_bf16, 'arith.truncf')[0]
        p_shift = operands(p_value, 'math.exp2')[0]
        score, anchor_broadcast = operands(p_shift, 'arith.subf')
        anchor_expand = operands(anchor_broadcast, 'tt.broadcast')[0]
        anchor = operands(anchor_expand, 'tt.expand_dims')[0]
        old_anchor, max_score = operands(anchor, 'arith.maxnumf')
        need(operands(max_score, '"tt.reduce"') == [score], 'P uses a different maximum/score')
        need(operands(branch['alpha_delta'], 'arith.subf') == [old_anchor, anchor],
             'Alpha does not use the original old/new anchors')
        need(operands(branch['probability_sum'], '"tt.reduce"') == [p_value],
             'Denominator and PV consume different P')
    return dict(passed=True, conditional_regions=branches, pv_c_chain=True,
                correction_explicit_fma=True, original_p_new_max_chain=True)


def llvm_blocks(text):
    body = text[text.index('define '):]
    body = body[:body.index('\n}')]
    blocks = {'entry': []}
    current = 'entry'
    for line in body.splitlines()[1:]:
        match = re.match(r'^([\w.$-]+):', line)
        if match:
            current = match[1]
            blocks[current] = []
        else:
            blocks[current].append(line)
    successors = {key: set() for key in blocks}
    for key, content in blocks.items():
        for line in content:
            if re.search(r'\bbr\b|\bswitch\b', line):
                successors[key].update(re.findall(r'label %([\w.$-]+)', line))
    reachable = {'entry'}
    while True:
        expanded = reachable.union(*(successors[b] for b in reachable))
        if expanded == reachable:
            break
        reachable = expanded
    need(reachable <= blocks.keys(), 'Unknown LLVM CFG target')
    predecessors = {b: {p for p in reachable if b in successors[p]} for b in reachable}
    dom = {b: ({'entry'} if b == 'entry' else set(reachable)) for b in reachable}
    while True:
        changed = False
        for b in reachable - {'entry'}:
            new = {b} | set.intersection(*(dom[p] for p in predecessors[b]))
            if new != dom[b]:
                dom[b] = new
                changed = True
        if not changed:
            break
    return blocks, dom


def source_location(text):
    return {int(k): int(v) for k, v in re.findall(
        r'^!(\d+) = !DILocation\(line: (\d+),', text, re.M)}


def audit_llir(text, proof):
    locations = source_location(text)
    blocks, dom = llvm_blocks(text)
    mapping = proof['source_lines']

    def source_line(line):
        hit = re.search(r'!dbg !(\d+)', line)
        return locations.get(int(hit[1])) if hit else None

    gates = []
    for block, content in blocks.items():
        for line in content:
            match = re.search(r'br i1 %\w+, label %([\w.$-]+), label %([\w.$-]+)', line)
            if match and source_line(line) in mapping['branch']:
                gates.append((block, match[1], match[2], source_line(line)))
    need(len(gates) == 2, 'Missing dynamic full/tail rescale CFG gates')
    need({g[3] for g in gates} == set(mapping['branch']), 'Missing full or tail CFG gate')
    counts = dict(alpha_exp=0, accumulator_mul=0, corrected_fma=0, pv_mma=0)
    by_source = {key: {} for key in counts}
    for block, content in blocks.items():
        for line in content:
            loc = source_line(line)
            kind = None
            if 'call ' in line and 'ex2.approx' in line and loc in mapping['alpha_exp']:
                kind = 'alpha_exp'
            if '= fmul float ' in line and loc in mapping['accumulator_scale']:
                kind = 'accumulator_mul'
            if 'call ' in line and re.search(r'@llvm\.(?:nvvm\.)?fma[.\w]*\(', line) and loc in mapping['running_sum_fma']:
                kind = 'corrected_fma'
            if 'call ' in line and '@llvm.bi.matrix.mad.f32x4.bf16x4' in line and loc in mapping['pv']:
                need('<4 x float> zeroinitializer' not in line, 'PV became a zero-C dot')
                counts['pv_mma'] += 1
                by_source['pv_mma'][loc] = by_source['pv_mma'].get(loc, 0)+1
            if kind:
                need(any(yes in dom[block] and no not in dom[block] for _, yes, no, _ in gates),
                     'Rescale operation hoisted outside the correction branch')
                counts[kind] += 1
                by_source[kind][loc] = by_source[kind].get(loc, 0)+1
    need(counts == dict(alpha_exp=4, accumulator_mul=64, corrected_fma=4, pv_mma=16),
         'Unrecognized LLVM rescale/PV instruction counts: '+repr(counts))
    for key, source_key, expected in [('alpha_exp', 'alpha_exp', 2), ('accumulator_mul', 'accumulator_scale', 32),
                                       ('corrected_fma', 'running_sum_fma', 2), ('pv_mma', 'pv', 8)]:
        need(by_source[key] == {line: expected for line in mapping[source_key]}, 'Full/tail LLVM count mismatch')
    return dict(passed=True, cfg_gates=gates, counts=counts, by_source=by_source)


def audit_asm(text, proof):
    mapping = proof['source_lines']
    counts = dict(alpha_exp=0, accumulator_mul=0, corrected_fma=0, pv_mma=0)
    operations = []
    gates = []
    loc_file = loc_line = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        location = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)', line)
        if location:
            loc_file, loc_line = map(int, location.groups())
        words = line.strip().split()
        if not words or loc_file != 1:
            continue
        op = words[0]
        operations.append((index, loc_line, op))
        if loc_line in mapping['branch'] and op == 'sl_cbr_tmskaz':
            gates.append((index, words[1].rstrip(','), loc_line))
        if loc_line in mapping['alpha_exp'] and op == 'ml_exp_f32':
            counts['alpha_exp'] += 1
        if loc_line in mapping['accumulator_scale'] and op == 'ml_mul_f32':
            counts['accumulator_mul'] += 1
        if loc_line in mapping['running_sum_fma'] and op == 'ml_fma_f32':
            counts['corrected_fma'] += 1
        if loc_line in mapping['pv'] and op == 'ml_matrix_mad_f32_bf16_m16n16k16':
            counts['pv_mma'] += 1
    need(counts == dict(alpha_exp=4, accumulator_mul=64, corrected_fma=4, pv_mma=16),
         'Unrecognized actual ASM type/rounding/count: '+repr(counts))
    need(len(gates) == 2, 'Missing actual full/tail conditional rescale branches')
    need({g[2] for g in gates} == set(mapping['branch']), 'Missing full or tail actual ASM gate')
    regions = []
    for begin, target, branch_source in gates:
        end = next((i for i in range(begin+1, len(lines)) if
                    re.match(r'\s*sl_jump\s+\w+\s*$', lines[i])), None)
        need(end is not None, 'Unrecognized actual rescale reconvergence')
        join = re.match(r'\s*sl_jump\s+(\w+)', lines[end])[1]
        if target != join:
            skip_line = next((i for i, s in enumerate(lines) if re.match(re.escape(target)+':', s)), None)
            join_line = next((i for i, s in enumerate(lines) if re.match(re.escape(join)+':', s)), None)
            need(skip_line is not None and join_line is not None and end < skip_line <= join_line,
                 'Unrecognized actual correction/skip diamond')
            need(not any('sl_cbr_' in s for s in lines[skip_line:join_line]), 'Unknown skip-side CFG')
            need(not any(loc in mapping['alpha_exp']+mapping['accumulator_scale']+mapping['running_sum_fma']
                         and op.startswith(('ml_exp_', 'ml_mul_', 'ml_fma_'))
                         for i, loc, op in operations if skip_line <= i < join_line),
                 'Skip-side still performs rescale')
        need(not any('sl_cbr_' in s for s in lines[begin+1:end]),
             'Nested or unrecognized actual rescale CFG')
        body = [(loc, op) for i, loc, op in operations if begin < i < end]
        body_counts = dict(
            alpha_exp=sum(loc in mapping['alpha_exp'] and op == 'ml_exp_f32' for loc, op in body),
            accumulator_mul=sum(loc in mapping['accumulator_scale'] and op == 'ml_mul_f32' for loc, op in body),
            corrected_fma=sum(loc in mapping['running_sum_fma'] and op == 'ml_fma_f32' for loc, op in body))
        need(body_counts == dict(alpha_exp=2, accumulator_mul=32, corrected_fma=2),
             'Actual rescale ops are unconditional, hoisted, or changed')
        regions.append(dict(branch_line=begin+1, source_line=branch_source, jump_line=end+1,
                            skip_target=target, join_target=join))
    return dict(passed=True, counts=counts, conditional_regions=regions,
                baseline_fma_opcode='ml_fma_f32')


def audit_artifacts(artifacts, proof):
    need(all(isinstance(artifacts.get(k), str) for k in ('ttgir', 'llir', 'asm')),
         'Missing actual TTGIR/LLIR/ASM; mechanism unknown')
    return dict(ttgir=audit_ttgir(artifacts['ttgir']),
                llir=audit_llir(artifacts['llir'], proof),
                asm=audit_asm(artifacts['asm'], proof), passed=True)
