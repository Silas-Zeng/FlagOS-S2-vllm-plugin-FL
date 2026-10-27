"""Shared CFG/SSA observer, independently applied to actual BM128 artifacts.
Copied from PR80 e2563331; this is not inheritance of its Q32 proof.
Function SHA256 f66778e726934d65a4afff725ad3fddbeae700ff2f8d93056f0b4040c6f4e499.
"""
import re
SSA=r'%[-.A-Za-z0-9_]+'

def llvm_observation(text):
    """CFG/SSA audit of physical Q operand reads; unknown grammar never passes."""
    lines=text.splitlines();defs={};blocks={};current=None
    header=next(s for s in lines if s.startswith('define '))
    params=[int(x) for x in re.findall(r'%([0-9]+)(?=[,) ])',header)]
    entry=str(max(params)+1)
    for i,line in enumerate(lines):
        if line.startswith('define '): current=entry;blocks[current]=[]
        elif current is not None:
            label=re.match(r'^([-.A-Za-z0-9_]+):',line)
            if label: current=label.group(1);blocks[current]=[]
            if line=='}': current=None;continue
            if current is not None:
                blocks[current].append((i+1,line))
                match=re.match(r'  ('+SSA+r') = (.*)',line)
                if match: defs[match.group(1)]=(i+1,current,match.group(2))
    succ={b:set(re.findall(r'label %([-.A-Za-z0-9_]+)',' '.join(s for _,s in values if s.strip().startswith('br ')))) for b,values in blocks.items()}
    if any(n not in blocks for values in succ.values() for n in values):
        raise ValueError('Unknown LLVM CFG target')
    pred={b:{p for p in blocks if b in succ[p]} for b in blocks};dom={b:({b} if b==entry else set(blocks)) for b in blocks}
    if any(not pred[b] for b in blocks if b!=entry):
        raise ValueError('Unreachable/unknown LLVM region')
    while True:
        new={b:({b} if b==entry else {b}|set.intersection(*(dom[p] for p in pred[b]))) for b in blocks}
        if new==dom: break
        dom=new
    mma=[(v,*data) for v,data in defs.items() if 'tail call <4 x i32> @llvm.bi.matrix.mad.i32x4.i8x8(' in data[2]]
    pv=[data for data in defs.values() if 'tail call <4 x float> @llvm.bi.matrix.mad.f32x4.bf16x4(' in data[2]]
    if len(mma)!=16 or len(pv)!=32:
        raise ValueError('Actual native I8-QK/BF16-PV counts changed')
    loops=[]
    for back,targets in succ.items():
        for head in targets:
            if head not in dom[back]: continue
            members={head,back};todo=[back]
            while todo:
                node=todo.pop()
                for p in pred[node]-members:
                    members.add(p)
                    if p!=head: todo.append(p)
            if all(data[2] in members for data in mma) and all(data[1] in members for data in pv):
                loops.append((head,members))
    if len(loops)!=1:
        raise ValueError('Cannot identify one natural KV loop containing both matrix paths')
    head,members=loops[0];roots=set();seen=set()
    def walk(value):
        if value in seen: return
        seen.add(value)
        if value not in defs: raise ValueError('Unknown Q operand SSA origin')
        line,block,op=defs[value]
        if op.startswith('load ') and 'ptr addrspace(3)' in op:
            if not re.match(r'load <2 x i8>,',op): raise ValueError('Unknown Q shared-read type')
            roots.add(value);return
        if not op.startswith(('shufflevector ','insertelement ','extractelement ','bitcast ','freeze ')):
            raise ValueError('Unknown Q operand construction')
        args=re.findall(SSA,op)
        if not args: raise ValueError('Constant/unrelated Q operand')
        for arg in args: walk(arg)
    for _,_,_,op in mma:
        a=re.search(r'i8x8\(<8 x i8> ('+SSA+r'),',op)
        if not a: raise ValueError('Unknown I8 matrix operand syntax')
        walk(a.group(1))
    if len(roots)!=16:
        raise ValueError('Only witnessed16 two-byte Q shared reads are admitted')
    def inline_lines(debug):
        found=[];visited=set()
        while debug and debug not in visited:
            visited.add(debug)
            meta=next((s for s in lines if s.startswith('!'+debug+' = !DILocation(')),None)
            if meta is None: break
            line=re.search(r'line: ([0-9]+)',meta)
            if line: found.append(int(line.group(1)))
            parent=re.search(r'inlinedAt: !([0-9]+)',meta);debug=parent.group(1) if parent else None
        return found
    for value in roots:
        debug=re.search(r'!dbg !([0-9]+)',defs[value][2])
        if debug is None or 238 not in inline_lines(debug.group(1)):
            raise ValueError('Q reads not linked to witnessed QCode codec source call238')
    def reaches(value,target,visited=None):
        if value==target: return True
        visited=set() if visited is None else visited
        if value in visited or value not in defs: return False
        visited.add(value)
        return any(reaches(v,target,visited) for v in re.findall(SSA,defs[value][2]))
    q_global=[]
    for value,(line,block,op) in defs.items():
        if '@llvm.bi.load.kop.v2bf16(' in op:
            ptr=re.search(r'ptr addrspace\(1\) ('+SSA+r')',op)
            if ptr and reaches(ptr.group(1),'%1'):
                q_global.append((line,block))
    if len(q_global)!=16 or any(block in members for _,block in q_global):
        raise ValueError('Original16 BF16 Q global reads no longer outside KV loop')
    division=[(line,block) for line,block,op in defs.values() if op.startswith('fdiv float ')]
    pre=[line for line,block in division if block in dom[head]]
    within=[line for line,block in division if block in members]
    if len(pre)!=48 or within:
        raise ValueError('RN32 Q codec division grammar/loop placement changed')
    inside=sorted(defs[v][0] for v in roots if defs[v][1] in members)
    return dict(native_I8_QK_calls=16,BF16_PV_calls=32,KV_header=head,Q_shared_read_lines=sorted(defs[v][0] for v in roots),
                Q_shared_reads_inside_KV=inside,all_Q_operand_reads_outside_KV=not inside,
                BF16_Q_global_read_lines=[line for line,_ in q_global],RN32_divisions_dominating_KV=pre,
                raw_quantization_call_source_line=238,known_grammar=True)

