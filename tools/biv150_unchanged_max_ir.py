"""Conservative actual-artifact gates. Unknown lowering is a rejection, not a pass."""
import re


def need(condition, message):
    if not condition:
        raise ValueError(message)


SSA = r'%[\w.$-]+(?:#\d+)?'


class ScopedSSA:
    """Small conservative reader for SSA definitions in printed SCF/reduce regions."""
    def __init__(self, lines):
        self.lines = lines
        self.allow_external = not any(re.search(r'\btt\.func\b', line) for line in lines)
        self.parents, self.defs, self.scopes = {0: None}, {0: {}}, []
        stack = [(0, -1)]
        for number, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            closing = stripped.startswith('}')
            if closing:
                need(len(stack) > 1 and stack[-1][1] == indent, 'Unknown TTGIR region closing')
                stack.pop()
            scope = stack[-1][0]
            self.scopes.append(scope)
            match = re.match(r'\s*(%[\w.$-]+)(?::(\d+))? = (.*)', line)
            if match:
                name, arity, expression = match.groups()
                need(name not in self.defs[scope], 'Duplicate SSA name inside one region')
                self.defs[scope][name] = (number, int(arity or 1), expression)
            if stripped.endswith('{'):
                child = len(self.parents)
                self.parents[child], self.defs[child] = scope, {}
                stack.append((child, indent))
                if 'tt.func ' in line:
                    for name in re.findall(r'(%[\w.$-]+):', line):
                        self.defs[child][name] = (number, 1, None)
                # Bind loop block arguments to this region, not to another same-name region.
                if 'iter_args(' in line:
                    for name in re.findall(r'(%[\w.$-]+)\s*=', line.split('iter_args(', 1)[1]):
                        self.defs[child][name] = (number, 1, None)
            if stripped.startswith('^bb'):
                for name in re.findall(r'(%[\w.$-]+):', line):
                    self.defs[scope][name] = (number, 1, None)
        need(len(stack) == 1, 'Unclosed TTGIR region')

    def resolve(self, token, at):
        base, _, component = token.partition('#')
        selector = int(component or 0)
        scope = self.scopes[at]
        while scope is not None:
            definition = self.defs[scope].get(base)
            if definition and definition[0] <= at:
                need(selector < definition[1], 'SSA result selector out of range')
                return (scope, base, definition[0], selector)
            scope = self.parents[scope]
        # Synthetic fixtures may model external function arguments without declarations.
        need(self.allow_external and not component, 'Unresolved TTGIR SSA value/result selector')
        return (-1, base, -1, 0)

    def operands(self, value, operation):
        scope, name, at, _ = value
        expression = self.defs.get(scope, {}).get(name, (0, 0, None))[2]
        need(expression is not None and expression.startswith(operation),
             'Unrecognized '+operation+' SSA chain for '+str(value))
        return [self.resolve(t, at) for t in re.findall(SSA, expression.split(' : ')[0])]


def audit_ttgir(text):
    lines = text.splitlines()
    ssa = ScopedSSA(lines)
    branches = []
    for i, line in enumerate(lines):
        match = re.search(r'(%[\w.$-]+):2 = scf.if ('+SSA+r').*tensor<8x128xf32', line)
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

        def matches(begin, finish, pattern):
            found = []
            for at in range(begin, finish):
                m = re.search(pattern, lines[at])
                if m:
                    found.append(tuple(ssa.resolve(t, at) for t in m.groups()))
            return found

        alpha = matches(i+1, split, '('+SSA+r') = math.exp2 ('+SSA+')')
        scale = matches(i+1, split, '('+SSA+r') = arith.mulf ('+SSA+'), ('+SSA+r') : tensor<8x128xf32')
        fma = matches(i+1, split, '('+SSA+r') = math.fma ('+SSA+'), ('+SSA+'), ('+SSA+')')
        add = matches(split+1, end, '('+SSA+r') = arith.addf ('+SSA+'), ('+SSA+r') : tensor<8xf32')
        yes_yield = matches(i+1, split, r'scf.yield ('+SSA+'), ('+SSA+r') :')
        no_yield = matches(split+1, end, r'scf.yield ('+SSA+'), ('+SSA+r') :')
        need(len(alpha) == len(scale) == len(fma) == len(add) == len(yes_yield) == len(no_yield) == 1,
             'Missing conditional alpha/accumulator scale/explicit FMA/skip state')
        need(fma[0][2] == alpha[0][0], 'Correction FMA does not use alpha')
        need(yes_yield[0] == (fma[0][0], scale[0][0]), 'Correction state order changed')
        need(no_yield[0] == (add[0][0], scale[0][1]), 'Skip path changes accumulator')
        need((fma[0][1], fma[0][3]) == (add[0][1], add[0][2]),
             'Correction and skip use different L/probability sums')
        scale_expand = ssa.operands(scale[0][2], 'tt.broadcast')[0]
        need(ssa.operands(scale_expand, 'tt.expand_dims') == [alpha[0][0]], 'Accumulator scale is not alpha')
        yes, no = '\n'.join(lines[i+1:split]), '\n'.join(lines[split+1:end])
        need('math.exp2' not in no and 'arith.mulf' not in no and 'tt.dot' not in yes+no,
             'Unexpected skip math or duplicated PV')
        branches.append(dict(result=match[1], result_ref=ssa.resolve(match[1], i), begin=i+1, end=end+1,
                             alpha_delta=alpha[0][1], probability_sum=fma[0][3]))
    need(len(branches) == 2, 'Expected full and tail conditional rescale regions')
    need(len(re.findall(r'= math.exp2 ', text)) == 4, 'Unexpected unconditional or missing exp path')
    pv = [(i,line) for i,line in enumerate(lines) if '= tt.dot ' in line and '-> tensor<8x128xf32' in line]
    qk = [line for line in lines if '= tt.dot ' in line and '-> tensor<8x16xf32' in line]
    need(len(pv) == len(qk) == 2, 'QK/PV shape/count changed')
    for branch in branches:
        result = branch['result_ref']
        conditional_c = result[:3]+(1,)
        uses = []
        for at,line in pv:
            operands = re.findall(SSA, line.split(' = tt.dot ',1)[1].split(' : ')[0])
            if len(operands) == 3 and ssa.resolve(operands[2],at) == conditional_c:
                uses.append((at,operands))
        need(len(uses) == 1, 'PV does not consume the conditional accumulator as C')
        at,operands = uses[0]
        p_fragment = ssa.resolve(operands[0],at)
        p_bf16 = ssa.operands(p_fragment, 'ttg.convert_layout')[0]
        p_value = ssa.operands(p_bf16, 'arith.truncf')[0]
        p_shift = ssa.operands(p_value, 'math.exp2')[0]
        score, anchor_broadcast = ssa.operands(p_shift, 'arith.subf')
        anchor_expand = ssa.operands(anchor_broadcast, 'tt.broadcast')[0]
        anchor = ssa.operands(anchor_expand, 'tt.expand_dims')[0]
        old_anchor, max_score = ssa.operands(anchor, 'arith.maxnumf')
        need(ssa.operands(max_score, '"tt.reduce"') == [score], 'P uses a different maximum/score')
        need(ssa.operands(branch['alpha_delta'], 'arith.subf') == [old_anchor, anchor],
             'Alpha does not use the original old/new anchors')
        need(ssa.operands(branch['probability_sum'], '"tt.reduce"') == [p_value],
             'Denominator and PV consume different P')
    return dict(passed=True, validator_version=2,
                conditional_regions=[{k:v for k,v in b.items() if k != 'result_ref'} for b in branches],
                scoped_multi_result_ssa=True, pv_c_chain=True,
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


def pv_source_sites(proof):
    """Only the factory's exact PV AST call spans, never neighboring/QK sites."""
    spans = proof.get('pv_call_spans', [])
    starts = proof['source_lines']['pv']
    need(len(spans) == 2 and {span['start'] for span in spans} == set(starts), 'Missing exact PV call spans')
    owners = {}
    for span in spans:
        need(span['start'] <= span['end'] and span['argument_lines'] and
             all(span['start'] <= line <= span['end'] for line in span['argument_lines']), 'Invalid PV AST span')
        for line in range(span['start'], span['end']+1):
            need(line not in owners, 'Overlapping PV call spans')
            owners[line] = span['start']
    need(not set(owners).intersection(proof['source_lines']['qk']), 'PV spans include QK')
    return owners


def pv_debug_inventory(llir, assembly, proof):
    """Primary debug-site regression, not a mechanism or numerical acceptance."""
    owners, locations = pv_source_sites(proof), source_location(llir)
    llvm_counts, asm_counts, accepted, excluded = {}, {}, set(), set()
    for line in llir.splitlines():
        if 'call ' not in line or '@llvm.bi.matrix.mad.f32x4.bf16x4' not in line:
            continue
        hit = re.search(r'!dbg !(\d+)', line)
        site = locations.get(int(hit[1])) if hit else None
        if site in owners:
            key = owners[site]
            llvm_counts[key] = llvm_counts.get(key, 0)+1
            accepted.add(site)
        elif site is not None:
            excluded.add(site)
    file = site = None
    for line in assembly.splitlines():
        location = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)', line)
        if location:
            file, site = map(int, location.groups())
        if file == 1 and line.strip().startswith('ml_matrix_mad_f32_bf16_m16n16k16 ') and site in owners:
            key = owners[site]
            asm_counts[key] = asm_counts.get(key, 0)+1
    return dict(llvm_counts=llvm_counts, asm_counts=asm_counts,
                accepted_sites=sorted(accepted), excluded_sites=sorted(excluded))


def llvm_c_chains(blocks, gates, source_line, pv_sites):
    """Trace every native C scalar through its conditional phi, not just nonzero C."""
    definitions, owners, users = {}, {}, {}
    for block, content in blocks.items():
        for line in content:
            match = re.match(r'\s*(%[\w.$-]+) = (.*)', line)
            if not match:
                continue
            value, expression = match.groups()
            definitions[value], owners[value] = expression, block
            for operand in re.findall(r'%[\w.$-]+', expression):
                users.setdefault(operand, []).append((value, expression))

    def fargs(value, opcode, count):
        expression = definitions.get(value, '')
        match = re.fullmatch(opcode+r' float ('+SSA+')(?:, ('+SSA+'))?', expression.split(', !dbg')[0])
        need(match is not None, 'Unrecognized LLVM '+opcode+' scalar')
        result = [x for x in match.groups() if x is not None]
        need(len(result) == count, 'Wrong LLVM scalar operand count')
        return result

    result = []
    for _, yes, no, branch_source in gates:
        def successor(block):
            targets = re.findall(r'^\s*br label %([\w.$-]+)', '\n'.join(blocks[block]), re.M)
            need(len(targets) == 1, 'Correction/skip does not have one unconditional join')
            return targets[0]
        join = successor(yes)
        need(successor(no) == join, 'Correction/skip join differs')
        alphas = [v for v,e in definitions.items() if owners[v] == yes and
                  'call ' in e and 'ex2.approx' in e]
        fmas = [v for v,e in definitions.items() if owners[v] == yes and
                re.search(r'@llvm\.(?:nvvm\.)?fma[.\w]*\(', e)]
        adds = [v for v,e in definitions.items() if owners[v] == no and e.startswith('fadd float ')]
        need(len(alphas) == len(fmas) == len(adds) == 2, 'Missing branch-local alpha/FMA/skip adds')
        denominator_phis = []
        for fma, add in zip(fmas, adds):
            expression = definitions[fma]
            args = re.search(r'@llvm\.(?:nvvm\.)?fma[.\w]*\(float ('+SSA+'), float ('+SSA+'), float ('+SSA+r')\)', expression)
            need(args is not None and args[2] in alphas, 'LLVM corrected denominator is not old L*alpha+Psum')
            need(fargs(add, 'fadd', 2) == [args[1], args[3]], 'LLVM skip denominator changed L/Psum')
            matching = [v for v,e in definitions.items() if owners[v] == join and
                        e.startswith('phi float ') and
                        dict((b,a) for a,b in re.findall(r'\[ ('+SSA+r'), %([\w.$-]+) \]', e)) == {yes:fma,no:add}]
            need(len(matching) == 1, 'Missing conditional denominator phi')
            denominator_phis.extend(matching)
        pv_values = [v for v,e in definitions.items() if owners[v] == join and
                     '@llvm.bi.matrix.mad.f32x4.bf16x4' in e and
                     source_line(e) in pv_sites]
        need(len(pv_values) == 8 and len({pv_sites[source_line(definitions[v])] for v in pv_values}) == 1,
             'PV does not occur eight times at the common join')
        all_phis, all_old = [], []
        alpha_pattern = []
        for pv in pv_values:
            expression = definitions[pv]
            call = re.search(r', <4 x float> ('+SSA+r')\)', expression)
            need(call is not None, 'PV became zero-C or changed native C type')
            packed, components = call[1], {}
            for _ in range(4):
                pack = re.match(r'insertelement <4 x float> (poison|'+SSA+'), float ('+SSA+r'), i\d+ ([0-3])',
                                definitions.get(packed, ''))
                need(pack is not None and int(pack[3]) not in components and owners[packed] == join,
                     'PV C is not four branch-phi scalars packed at the join')
                components[int(pack[3])] = pack[2]
                packed = pack[1]
            need(packed == 'poison' and set(components) == set(range(4)), 'Incomplete native PV C vector')
            for component in range(4):
                phi = components[component]
                phi_expr = definitions.get(phi, '')
                need(owners.get(phi) == join and phi_expr.startswith('phi float '), 'PV C lost conditional phi')
                incoming = dict((b,a) for a,b in re.findall(r'\[ ('+SSA+r'), %([\w.$-]+) \]', phi_expr))
                need(set(incoming) == {yes,no}, 'PV C has a wrong phi predecessor')
                old, corrected = incoming[no], incoming[yes]
                need(owners.get(corrected) == yes, 'PV C scale moved out of correction')
                operands = fargs(corrected, 'fmul', 2)
                need(operands[0] == old and operands[1] in alphas, 'PV C changes original acc or alpha')
                alpha_pattern.append(operands[1])
                all_phis.append(phi); all_old.append(old)
            extracts = users.get(pv, [])
            need(len(extracts) == 4, 'Native PV D result has unexpected users')
            slots = []
            for extracted, expression in extracts:
                match = re.match(r'extractelement <4 x float> '+re.escape(pv)+r', i\d+ ([0-3])', expression)
                need(match is not None, 'PV result decomposed or transformed')
                slots.append(int(match[1]))
                need(users.get(extracted) and all(e.startswith('phi float ') for _,e in users[extracted]),
                     'PV output is post-added/transformed before the original recurrence phi')
            need(sorted(slots) == list(range(4)), 'Native PV D component extraction changed')
        need(len(set(all_phis)) == len(set(all_old)) == 32, 'Not all 32 independent C states reach eight PV groups')
        need(alpha_pattern == alphas*16, 'Conditional C alpha row/component pattern changed')
        result.append(dict(branch_source=branch_source, correction=yes, skip=no, join=join,
                           c_phi_count=32, pv_groups=8, all_c_scalars_consumed_once=True,
                           denominator_phis=denominator_phis, pv_post_add_absent=True))
    return result


def audit_llir(text, proof):
    locations = source_location(text)
    blocks, dom = llvm_blocks(text)
    mapping = proof['source_lines']
    pv_sites = pv_source_sites(proof)

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
            if 'call ' in line and '@llvm.bi.matrix.mad.f32x4.bf16x4' in line and loc in pv_sites:
                need('<4 x float> zeroinitializer' not in line, 'PV became a zero-C dot')
                counts['pv_mma'] += 1
                owner = pv_sites[loc]
                by_source['pv_mma'][owner] = by_source['pv_mma'].get(owner, 0)+1
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
    chains = llvm_c_chains(blocks, gates, source_line, pv_sites)
    return dict(passed=True, validator_version=2, cfg_gates=gates, counts=counts, by_source=by_source,
                conditional_c_chains=chains)


def asm_blocks(text):
    """BI printed labels and explicit/fallthrough edges; .Ltmp debug labels are not blocks."""
    blocks, order, current = {'entry': []}, ['entry'], 'entry'
    location = (None, None)
    for index, line in enumerate(text.splitlines()):
        loc = re.match(r'\s*\.loc\s+(\d+)\s+(\d+)', line)
        if loc:
            location = tuple(map(int, loc.groups()))
        label = re.match(r'^(BB\d+_\d+):', line)
        if label:
            current = label[1]
            need(current not in blocks, 'Duplicate ASM basic block label')
            blocks[current] = []
            order.append(current)
        instruction = re.match(r'\s*((?:sl|ml)_\w+)\s*(.*?)\s*$', line)
        if instruction:
            blocks[current].append(dict(line=index+1, file=location[0], source=location[1],
                                        op=instruction[1], args=instruction[2]))
    successors = {}
    for position, block in enumerate(order):
        content = blocks[block]
        fallthrough = [order[position+1]] if position+1 < len(order) else []
        last = content[-1] if content else None
        if last and last['op'] == 'sl_jump':
            targets = [last['args']]
        elif last and last['op'].startswith('sl_cbr_'):
            targets = [last['args'].split(',')[0]] + fallthrough
        else:
            targets = fallthrough
        successors[block] = targets
    need(all(target in blocks for targets in successors.values() for target in targets), 'Unknown ASM CFG target')
    return blocks, order, successors


def audit_asm(text, proof):
    mapping, pv_sites = proof['source_lines'], pv_source_sites(proof)
    blocks, order, successors = asm_blocks(text)
    all_ops = [(block, op) for block, content in blocks.items() for op in content]
    selectors = dict(alpha_exp=('alpha_exp', 'ml_exp_f32'),
                     accumulator_mul=('accumulator_scale', 'ml_mul_f32'),
                     corrected_fma=('running_sum_fma', 'ml_fma_f32'),
                     pv_mma=('pv', 'ml_matrix_mad_f32_bf16_m16n16k16'))

    def selected(op, kind):
        source_key, opcode = selectors[kind]
        sites = pv_sites if kind == 'pv_mma' else mapping[source_key]
        return op['file'] == 1 and op['source'] in sites and op['op'] == opcode

    counts = {kind: sum(selected(op,kind) for _,op in all_ops) for kind in selectors}
    need(counts == dict(alpha_exp=4, accumulator_mul=64, corrected_fma=4, pv_mma=16),
         'Unrecognized actual ASM type/rounding/count: '+repr(counts))
    gates = [(block,op) for block,op in all_ops if op['op'] == 'sl_cbr_tmskaz']
    need(len(gates) == 2, 'Missing actual full/tail conditional rescale branches')
    regions, gate_sources, covered = [], set(), {key:set() for key in selectors}
    for flow, gate in gates:
        position = order.index(flow)
        need(position >= 2 and position+1 < len(order), 'Missing ASM gate neighbors')
        setup, skip, correction = order[position-2], order[position-1], order[position+1]
        join = gate['args']
        need(join in blocks and len({flow,setup,skip,correction,join}) == 5,
             'Invalid correction/skip/common-join labels')
        need(successors[flow] == [join, correction] and successors[skip] == [flow] and
             successors[setup] == [skip], 'Unknown skip-side ASM CFG')
        need(successors[correction] == [join], 'Correction branch has wrong join')
        # Bind .loc 0 to the actual source-owned saved-mask gate, never to arbitrary .loc 0 branches.
        flow_ops, setup_ops, skip_ops = blocks[flow], blocks[setup][-2:], blocks[skip]
        need(len(flow_ops) == 3 and flow_ops[-1] == gate and gate['source'] == 0 and gate['file'] == 1,
             'Unknown actual mask-flow region')
        need(len(setup_ops) == 2 and setup_ops[0]['op'] == 'sl_and_savetmsk_b64' and
             setup_ops[1]['op'] == 'sl_xor_b64', 'Missing source-owned conditional mask setup')
        source = setup_ops[0]['source']
        need(source in mapping['branch'] and all(o['file'] == 1 and o['source'] == source for o in setup_ops),
             'Mask setup is not the original rescale source gate')
        index = mapping['branch'].index(source)
        need(source not in gate_sources, 'Duplicate full/tail ASM source gate')
        gate_sources.add(source)
        save = re.fullmatch(r'\{(s\[\d+:\d+\]), wcr\}, tcr, tmsk', setup_ops[0]['args'])
        need(save is not None, 'Unrecognized saved skip mask')
        mask = save[1]
        need(setup_ops[1]['args'] == '{'+mask+', wcr}, tmsk, '+mask,
             'Mask setup complement changed')
        need(flow_ops[0]['op'] == 'sl_or_savetmsk_b64' and
             flow_ops[0]['args'] == '{'+mask+', wcr}, '+mask+', tmsk', 'Wrong reconvergence mask union')
        second = re.fullmatch(r'\{(s\[\d+:\d+\]), wcr\}, '+re.escape(mask)+r', tmsk', flow_ops[1]['args'])
        need(flow_ops[1]['op'] == 'sl_xor_savetmsk_b64' and second is not None and second[1] != mask,
             'Wrong correction mask complement')
        need(all(o['file'] == 1 and o['source'] == 0 for o in flow_ops), 'Unexpected flow debug ownership')
        joined = blocks[join]
        need(joined and joined[0]['op'] == 'sl_or_b64' and
             joined[0]['args'] == '{tmsk, wcr}, tmsk, '+mask,
             'Missing matching common-join mask restore')
        need(len(skip_ops) == 2 and all(o['op'] == 'ml_add_f32' for o in skip_ops),
             'Skip side contains rescale or accumulator writes')
        skip_args = [re.fullmatch(r'(v\d+), (v\d+), (v\d+)', o['args']) for o in skip_ops]
        need(all(m and m[1] == m[2] for m in skip_args), 'Skip denominator does not retain original L')
        body = blocks[correction]
        # A backward explicit join and a next-block fallthrough are both valid actual layouts.
        body_math = body[:-1] if body and body[-1]['op'] == 'sl_jump' else body
        need(all(o['op'] in {'ml_sub_f32','ml_exp_f32','ml_mul_f32','ml_fma_f32'} for o in body_math),
             'Nested or unrecognized correction ASM')
        local_counts = {kind: sum(selected(o,kind) for o in body_math) for kind in
                        ('alpha_exp','accumulator_mul','corrected_fma')}
        need(local_counts == dict(alpha_exp=2, accumulator_mul=32, corrected_fma=2),
             'Actual rescale ops are unconditional, hoisted, or changed')
        need(sum(o['op'] == 'ml_sub_f32' for o in body_math) == 2 and len(body_math) == 38,
             'Unexpected correction scalar operations')
        for kind in local_counts:
            expected_site = mapping[selectors[kind][0]][index]
            need(all(o['source'] == expected_site for o in body_math if selected(o,kind)),
                 'Correction owns a different full/tail rescale site')
            covered[kind].update(o['line'] for o in body_math if selected(o,kind))
        alpha_ops = [o for o in body_math if selected(o,'alpha_exp')]
        alpha_args = [re.fullmatch(r'(v\d+), (v\d+)', o['args']) for o in alpha_ops]
        need(all(m and m[1] == m[2] for m in alpha_args), 'Unrecognized actual alpha registers')
        alphas = [m[1] for m in alpha_args]
        fma_ops = [o for o in body_math if selected(o,'corrected_fma')]
        for fma, add, alpha in zip(fma_ops, skip_args, alphas):
            need(fma['args'] == ', '.join((add[1],add[2],alpha,add[3])),
                 'Actual correction FMA/skip L/Psum operands changed')
        pv_ops = [o for o in joined if selected(o,'pv_mma')]
        need(len(pv_ops) == 8 and all(pv_sites[o['source']] == mapping['pv'][index] for o in pv_ops),
             'Common join does not own the original eight PV groups')
        c_registers = []
        for pv in pv_ops:
            args = re.findall(r'v\[(\d+):(\d+)\]', pv['args'])
            need(len(args) == 4 and args[0] == args[3] and int(args[0][1])-int(args[0][0]) == 3,
                 'Native PV changed its actual C/D register group')
            c_registers.extend('v'+str(r) for r in range(int(args[0][0]),int(args[0][1])+1))
        need(len(set(c_registers)) == 32, 'Not all 32 actual C registers reach eight PV groups')
        multiply = {}
        for o in body_math:
            if selected(o,'accumulator_mul'):
                m = re.fullmatch(r'(v\d+), (v\d+), (v\d+)', o['args'])
                need(m and m[1] == m[2] and m[1] not in multiply, 'Actual acc multiply changed original C')
                multiply[m[1]] = m[3]
        need(set(multiply) == set(c_registers) and
             [multiply[r] for r in c_registers] == alphas*16, 'Actual C alpha/component mapping changed')
        covered['pv_mma'].update(o['line'] for o in pv_ops)
        regions.append(dict(branch_line=gate['line'], source_line=source, mask=mask,
                            setup=setup, skip=skip, flow=flow, correction=correction, join=join,
                            correction_to_join='jump' if body[-1]['op'] == 'sl_jump' else 'fallthrough',
                            backward_join=order.index(join) < position, c_register_count=32,
                            pv_groups=8, counts=local_counts))
    need(gate_sources == set(mapping['branch']), 'Missing full or tail actual ASM gate')
    for kind in selectors:
        need(covered[kind] == {op['line'] for _,op in all_ops if selected(op,kind)},
             'Rescale/PV instruction is outside its conditional/common-join region')
    return dict(passed=True, validator_version=2, counts=counts, conditional_regions=regions,
                source_owned_mask_cfg=True, baseline_fma_opcode='ml_fma_f32')


def audit_artifacts(artifacts, proof):
    need(all(isinstance(artifacts.get(k), str) for k in ('ttgir', 'llir', 'asm')),
         'Missing actual TTGIR/LLIR/ASM; mechanism unknown')
    return dict(ttgir=audit_ttgir(artifacts['ttgir']),
                llir=audit_llir(artifacts['llir'], proof),
                asm=audit_asm(artifacts['asm'], proof), passed=True)
