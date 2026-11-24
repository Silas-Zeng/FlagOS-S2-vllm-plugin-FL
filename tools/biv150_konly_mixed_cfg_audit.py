"""CPU audit of the two pinned mixed-reader LLVM CFGs; never executes raw IR."""
import re

SSA = r'%[\w.$-]+'


def audit(text, names, role, num_seqs, capacity):
    """Check actual front guards and all AS1 accesses, plus row-address SSA cuts.

    This accepts the observed CoreX3.6 lowering only. It is not a general LLVM
    verifier or an intrinsic C/D-lane ABI proof. Metadata is immutable per launch.
    """
    lines = text.splitlines()
    begin = [i for i, line in enumerate(lines) if line.startswith('define iluvatar_kernel void @')]
    assert len(begin) == 1, 'One actual kernel required'
    end = next(i for i in range(begin[0] + 1, len(lines)) if lines[i] == '}')
    blocks = {'entry': []}; current = 'entry'; defs = {}; records = []
    for index in range(begin[0] + 1, end):
        line = lines[index].split(';', 1)[0].strip()
        if not line: continue
        if re.fullmatch(r'[\w.$-]+:', line):
            current = line[:-1]; assert current not in blocks; blocks[current] = []; continue
        match = re.match(r'(' + SSA + r') = (.*)', line)
        row = dict(line=index + 1, text=line, block=current)
        if match:
            assert match[1] not in defs
            row.update(value=match[1], expression=match[2].split(', !dbg', 1)[0]); defs[match[1]] = row
        blocks[current].append(row); records.append(row)
    successors = {}
    for block, rows in blocks.items():
        assert rows
        last = rows[-1]['text']
        if last.startswith('ret void'): successors[block] = []
        else:
            assert last.startswith('br '), ('Unexpected terminator', block, last)
            successors[block] = re.findall(r'label %([\w.$-]+)', last)
            assert len(successors[block]) in (1, 2)
    assert all(v in blocks for values in successors.values() for v in values)
    reachable = {'entry'}
    while True:
        expanded = reachable | {v for b in reachable for v in successors[b]}
        if expanded == reachable: break
        reachable = expanded
    assert reachable == set(blocks), 'Unreviewed unreachable kernel block'
    predecessors = {b: {p for p in blocks if b in successors[p]} for b in blocks}
    dom = {b: ({'entry'} if b == 'entry' else set(blocks)) for b in blocks}
    while True:
        new = {b: ({'entry'} if b == 'entry' else {b} | set.intersection(*(dom[p] for p in predecessors[b]))) for b in blocks}
        if new == dom: break
        dom = new

    def expression(value): return defs[value]['expression']
    def unique(predicate):
        values = [v for v, row in defs.items() if predicate(row['expression'])]
        assert len(values) == 1, ('Expected unique SSA site', values)
        return values[0]
    def inputs(value): return re.findall(SSA, expression(value)) if value in defs else []
    def dependency_cut(value, cut):
        leaves = set(); seen = set(); pending = [value]
        while pending:
            node = pending.pop()
            if node in seen: continue
            seen.add(node)
            if node in cut or node not in defs: leaves.add(node)
            else: pending.extend(inputs(node))
        return leaves
    def branch(block):
        row = blocks[block][-1]
        match = re.match(r'br i1 (' + SSA + r'), label %([\w.$-]+), label %([\w.$-]+)', row['text'])
        assert match
        return match[1], match[2], match[3], row['line']
    def scalar_extract(load):
        return unique(lambda e: re.fullmatch(r'extractelement <1 x i32> ' + re.escape(load) + r', i(?:32|64) 0', e) is not None)
    def sign_extend(value):
        return unique(lambda e: e == 'sext i32 ' + value + ' to i64')

    args = {name: '%' + str(i) for i, name in enumerate(names)}
    cuq = args['query_start_len_ptr']
    pointer_names = {name for name in names if name.endswith('_ptr')}
    pointers = {args[name] for name in pointer_names}
    pid = unique(lambda e: '@llvm.nvvm.read.ptx.sreg.ctaid.x()' in e)
    bound = unique(lambda e: re.fullmatch(r'icmp (?:samesign )?ugt i32 ' + re.escape(pid) + ', ' + str(num_seqs - 1), e) is not None)
    condition, bad, metadata_block, sequence_branch_line = branch('entry')
    assert condition == bound
    assert all(row['text'].startswith('ret void') for row in blocks[bad]), 'Bad sequence/decode path must only return'
    pid64 = unique(lambda e: re.fullmatch(r'zext (?:nneg )?i32 ' + re.escape(pid) + ' to i64', e) is not None)
    start_ptr = unique(lambda e: e == 'getelementptr i32, ptr addrspace(1) ' + cuq + ', i64 ' + pid64)
    end_ptr = unique(lambda e: e == 'getelementptr i8, ptr addrspace(1) ' + start_ptr + ', i64 4')
    def metadata_load(ptr):
        return unique(lambda e: re.fullmatch(r'(?:tail )?call <1 x i32> @llvm.bi.load.kop.v1i32\(ptr addrspace\(1\) ' + re.escape(ptr) + r', i32 0, i1 false\)', e) is not None)
    loads = [metadata_load(ptr) for ptr in (start_ptr, end_ptr)]
    start32, end32 = [scalar_extract(v) for v in loads]
    start64, end64 = [sign_extend(v) for v in (start32, end32)]
    difference = unique(lambda e: e == 'sub nsw i64 ' + end64 + ', ' + start64)
    wrong_length = unique(lambda e: e == 'icmp ne i64 ' + difference + ', 1')
    negative_start = unique(lambda e: e == 'icmp slt i32 ' + start32 + ', 0')
    excessive_end = unique(lambda e: e == 'icmp sgt i32 ' + end32 + ', ' + str(capacity))
    decode_cond, decode_bad, decode_good, decode_branch_line = branch(metadata_block)
    assert decode_bad == bad and decode_good not in ('entry', metadata_block, bad)
    def or_leaves(value):
        match = re.fullmatch(r'or i1 (' + SSA + '), (' + SSA + ')', expression(value))
        return or_leaves(match[1]) | or_leaves(match[2]) if match else {value}
    assert or_leaves(decode_cond) == {wrong_length, negative_start, excessive_end}
    assert all(defs[v]['block'] == metadata_block and defs[v]['line'] < decode_branch_line for v in loads)
    assert all(metadata_block in dom[defs[v]['block']] for v in loads)
    # The precise scalar load addresses above contain only the CTA x index and
    # constant+1. The decision cannot depend on tid/lane/data/prefill payload.
    assert all(dependency_cut(v, {pid, cuq}) <= {pid, cuq} for v in (decode_cond, start64, end64))

    # A loaded index is an address offset, not the pointer's storage root.
    loaded_values = {v for v, row in defs.items() if '@llvm.bi.load.kop' in row['expression']}
    accesses = []
    for row in records:
        line = row['text']
        if 'ptr addrspace(1)' not in line: continue
        expr = row.get('expression', line)
        ordinary = expr.startswith('load ') or expr.startswith('store ')
        assert not re.search(r'\b(?:atomicrmw|cmpxchg)\b', expr), 'Unreviewed global atomic'
        if 'call ' not in line and not ordinary: continue
        assert ordinary or re.search(r'@llvm.bi\.(?:load\.kop|store\.kop|stp\.)', line), ('Unknown global pointer call', line)
        addresses = re.findall(r'ptr addrspace\(1\) (' + SSA + ')', line)
        assert len(addresses) == 1
        address = addresses[0]; roots = dependency_cut(address, pointers | loaded_values) & pointers
        assert len(roots) == 1, ('Ambiguous global pointer provenance', roots)
        root = next(iter(roots)); name = next(k for k, v in args.items() if v == root)
        is_load = '@llvm.bi.load.kop' in line or expr.startswith('load ')
        if root == cuq:
            assert is_load and row.get('value') in loads and row['block'] == metadata_block
        else:
            assert decode_good in dom[row['block']], ('Data access before valid decode edge', row)
        cut = dependency_cut(address, {pid, start64})
        if role == 'producer' and name == 'query_ptr':
            assert is_load and start64 in cut and pid not in cut, 'Q must use cu_q token row'
        if role == 'reducer' and name == 'output_ptr':
            assert not is_load and start64 in cut and pid not in cut, 'Output must scatter by cu_q token row'
        if name in ('segm_output_ptr', 'segm_max_ptr', 'segm_expsum_ptr'):
            assert pid in cut and start64 not in cut, 'Partial address must retain compact sequence row'
            assert is_load == (role == 'reducer')
        accesses.append(dict(line=row['line'], block=row['block'], pointer=name, kind='load' if is_load else 'store'))
    assert len([v for v in accesses if v['pointer'] == 'query_start_len_ptr']) == 2
    assert {'segm_output_ptr', 'segm_max_ptr', 'segm_expsum_ptr'} <= {v['pointer'] for v in accesses}
    if role == 'producer':
        assert any(v['pointer'] == 'query_ptr' for v in accesses)
        stride = args['query_stride_token']
        assert unique(lambda e: e == 'mul i64 ' + stride + ', ' + start64)
    else: assert any(v['pointer'] == 'output_ptr' for v in accesses)
    return dict(passed=True, schema='cjj-mixed-actual-llvm-cfg-v1', role=role,
        num_sequences=num_seqs, query_capacity=capacity, kernel_define_line=begin[0] + 1,
        blocks=len(blocks), sequence_bound_branch_line=sequence_branch_line,
        metadata_load_lines=[defs[v]['line'] for v in loads], metadata_difference_line=defs[difference]['line'],
        decode_guard_branch_line=decode_branch_line, metadata_block=metadata_block,
        valid_decode_block=decode_good, early_return_block=bad, ssa=dict(pid=pid, pid64=pid64,
        q_start_i32=start32, q_start_i64=start64, q_end_i64=end64, length_difference=difference),
        qlen_and_bounds_cta_uniform_for_immutable_metadata=True,
        signed_i64_difference=True, all_non_cuq_global_accesses_dominated_by_valid_decode_edge=True,
        partial_rows_use_sequence_index=True, query_output_rows_use_cuq_index=True,
        invalid_or_prefill_path_global_data_accesses=0, accesses=accesses,
        native_accumulator_component_independence_assumed=False, binary_cfg_audited=False)
