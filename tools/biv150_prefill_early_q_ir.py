"""Private diagnostic: shorten Q shared-memory lifetime before K prefetch.

Input is the unchanged production Q-hoist transform's output. This is a textual
SSA/line-permutation proof, not a compiler synchronization or device proof.
"""
import hashlib
from pathlib import Path
import re
import sys
import types

EXPECTED_Q_FRAGMENT = '38e44e6b0272116a1c76561119e9742337caed4dfaf5ab891fc57df4f2c16a82'
SSA = r'%[A-Za-z0-9_]+'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def advance_q_read(raw, block_m=256):
    if block_m != 256:
        raise ValueError('Only the captured BM256/K64/stage2 structure is audited')
    text=raw.decode('utf-8')
    lines=text.splitlines(keepends=True)
    allocation=[]
    for i,line in enumerate(lines):
        match=re.match(rf'^(\s*)({SSA}) = ttg\.local_alloc ({SSA}) : .*tensor<256x128xbf16',line)
        if match:
            allocation.append((i,match))
    if len(allocation)!=1:
        raise ValueError('Ambiguous Q allocation')
    alloc,match=allocation[0]
    indent,descriptor,source=match.groups()
    if 'mutable' in lines[alloc] or alloc==0 or not re.match(
            rf'^\s*{re.escape(source)} = tt\.load\b',lines[alloc-1]):
        raise ValueError('Q allocation must directly consume its captured global load')
    references=[i for i,line in enumerate(lines)
                if re.search(re.escape(descriptor)+r'(?![A-Za-z0-9_])',line)]
    loads=[i for i in references if re.search(r'ttg\.local_load '+re.escape(descriptor)+r'(?=\s| :)',lines[i])]
    loops=[i for i,line in enumerate(lines) if '= scf.for ' in line]
    if len(loads)!=1 or len(loops)!=1 or references!=[alloc,loads[0]]:
        raise ValueError('Q descriptor has unknown uses, writes, or main loops')
    load,loop=loads[0],loops[0]
    if load!=loop-1 or load<=alloc+1:
        raise ValueError('Expected the existing Q read immediately before the main loop')
    line=lines[load]
    if line[:len(line)-len(line.lstrip())]!=indent or 'opIdx = 0' not in line:
        raise ValueError('Q read is not the captured same-block lhs-dot read')
    operands=re.findall(SSA,line)
    if len(operands)!=2 or operands[1]!=descriptor:
        raise ValueError('Q read has extra SSA dependencies')
    span=lines[alloc+1:load]
    if any(re.search(r'\b(?:scf\.|cf\.|tt\.return|func\.return|ttg\.local_store|ttg\.local_dealloc|ttg\.async_wait)|barrier',x)
           for x in span):
        raise ValueError('Unknown control flow, store, or explicit synchronization crossed')
    buffers=[i for i in range(alloc+1,load) if re.search(
        r'ttg\.local_alloc : \(\) -> !ttg\.memdesc<1x128x64xbf16,.*mutable>',lines[i])]
    if len(buffers)!=1 or not any('ttg.async_copy_global_to_local' in x for x in span):
        raise ValueError('Captured independent stage2 K buffer/prefetch missing')
    # All brace-bearing attributes are balanced on their line in this capture.
    # Refuse a region change; the read may cross only the straight-line prologue.
    depths=[];depth=0
    for entry in lines:
        depths.append(depth);depth+=entry.count('{')-entry.count('}')
    if depths[alloc]!=depths[load] or depths[load]!=depths[loop]:
        raise ValueError('Q allocation/read do not share one region')
    changed=lines.copy();moved=changed.pop(load);changed.insert(alloc+1,moved)
    restored=changed.copy();restored.pop(alloc+1);restored.insert(load,moved)
    if restored!=lines or changed[:alloc+1]!=lines[:alloc+1]:
        raise ValueError('Exact inverse/producer prefix proof failed')
    original_other=lines[:load]+lines[load+1:]
    changed_other=changed[:alloc+1]+changed[alloc+2:]
    if original_other!=changed_other:
        raise ValueError('Something besides the single Q read changed')
    output=''.join(changed).encode('utf-8')
    proof=dict(input_sha256=digest(raw),output_sha256=digest(output),
               allocation_line=alloc+1,old_read_line=load+1,new_read_line=alloc+2,
               main_loop_line=loop+1,old_k_allocation_line=buffers[0]+1,
               descriptor=descriptor,q_read_result=operands[0],
               other_lines_sha256=digest(''.join(original_other).encode()),
               exact_inverse=True,other_lines_byte_identical=True,
               explicit_sync_crossed=False,
               limitations='Textual SSA/line proof only; target lowering, synchronization, resources, streams/Graph and numerical validation are pending.')
    return output,proof


def private_manager_module(original):
    """Clone pinned manager globals so only the candidate gets the extra move."""
    source=Path(original.__file__).read_bytes()
    if digest(source)!=EXPECTED_Q_FRAGMENT:
        raise RuntimeError('Production manager changed; re-audit before cloning')
    module=types.ModuleType('_biv150_early_q_private_manager')
    module.__file__=str(Path(original.__file__).resolve())
    sys.modules[module.__name__]=module
    exec(compile(source,'<biv150-early-q-private-manager>','exec'),module.__dict__)
    strict=module.transform_ir
    module.early_q_records=[]
    def transform(raw,block_m=128):
        current=strict(raw,block_m)
        result,proof=advance_q_read(current,block_m)
        if len(module.early_q_records)<16:module.early_q_records.append(proof)
        return result
    module.transform_ir=transform
    return module,digest(source)
