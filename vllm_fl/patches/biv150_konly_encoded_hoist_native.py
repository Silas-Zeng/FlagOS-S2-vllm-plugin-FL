"""CPU native grammar for fresh Q32 artifacts; old fixtures are tests only."""
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
        raise ValueError('Expected16 two-byte Q shared reads for this Q32 compile')
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


def alpha(text):
    """SSA definition-site normalization, preserving nested region/block scopes.

    This admits MLIR's observed pretty-printer renaming, not operation changes.
    Definition IDs use line/token ordinals so old while-region arg26 and a later
    for-region arg26 remain different values. Non-SSA text is byte preserved.
    """
    scopes=[{}];out=[]
    for lineno,line in enumerate(text.splitlines(),1):
        stripped=line.lstrip()
        if stripped.startswith('}'):
            if len(scopes)==1: raise ValueError('SSA region underflow')
            scopes.pop()
        assignments=list(re.finditer('('+SSA+r')(?::[0-9]+)?\s*=',line))
        blockargs=list(re.finditer('('+SSA+r')\s*:',line)) if stripped.startswith('^bb') else []
        first=bool(re.match(r'\s*'+SSA+r'(?::[0-9]+)?\s*=',line))
        if first: scopes[-1][assignments[0][1]]='pending'
        if line.rstrip().endswith('{'): scopes.append({})
        for match in assignments[1 if first else 0:]+blockargs:
            scopes[-1][match[1]]='pending'
        for ordinal,match in enumerate(re.finditer(SSA,line)):
            if any(match.start()==a.start() for a in assignments+blockargs):
                for scope in reversed(scopes):
                    if match[0] in scope:
                        scope[match[0]]=f'%d{lineno}_{ordinal}';break
        out.append(re.sub(SSA,lambda m:next((s[m[0]] for s in reversed(scopes) if m[0] in s),m[0]),line))
    if len(scopes)!=1: raise ValueError('Unclosed SSA region')
    return '\n'.join(out)


def llvm_sync(text, observation):
    """Known post-hoist bank-reuse contract, with CFG dominance checks."""
    lines=text.splitlines();blocks={};current=None
    header=next(x for x in lines if x.startswith('define '))
    entry=str(max(map(int,re.findall(r'%([0-9]+)(?=[,) ])',header)))+1)
    for i,line in enumerate(lines,1):
        if line.startswith('define '): current=entry;blocks[current]=[]
        label=re.match(r'^([-.A-Za-z0-9_]+):',line)
        if label: current=label[1];blocks[current]=[]
        if line=='}': current=None
        if current is not None: blocks[current].append((i,line))
    succ={b:set(re.findall(r'label %([-.A-Za-z0-9_]+)',' '.join(s for _,s in rows if s.strip().startswith('br ')))) for b,rows in blocks.items()}
    if any(v not in blocks for values in succ.values() for v in values): raise ValueError('Unknown LLVM branch')
    pred={b:{p for p in blocks if b in succ[p]} for b in blocks}
    dom={b:({b} if b==entry else set(blocks)) for b in blocks}
    if any(not pred[b] for b in blocks if b!=entry): raise ValueError('Unreachable LLVM block')
    while True:
        new={b:({b} if b==entry else {b}|set.intersection(*(dom[p] for p in pred[b]))) for b in blocks}
        if new==dom: break
        dom=new
    membership={i:b for b,rows in blocks.items() for i,_ in rows}
    q=observation['Q_shared_read_lines'];qblocks={membership[i] for i in q}
    if len(qblocks)!=1 or not qblocks<=dom[observation['KV_header']]:
        raise ValueError('Q shared reads must dominate KV loop, not just be outside it')
    qb=next(iter(qblocks))
    qstores=[i for b,rows in blocks.items() if b in dom[qb] for i,s in rows if 'store <1 x i8>' in s and 'ptr addrspace(3)' in s]
    barriers=[i for b,rows in blocks.items() if b in dom[qb] for i,s in rows if '@llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)' in s and i<min(q)]
    if len(qstores)!=32 or not barriers or not max(qstores)<max(barriers)<min(q):
        raise ValueError('Q-code shared store/barrier/read sequence changed')
    if any(max(barriers)<i<max(q) and 'store ' in s and 'ptr addrspace(3)' in s for i,s in enumerate(lines,1)):
        raise ValueError('Shared overwrite during Q operand reads')
    kstores=[(i,b) for b,rows in blocks.items() for i,s in rows if 'store <4 x i8>' in s and 'ptr addrspace(3)' in s]
    if len(kstores)!=2 or len({b for _,b in kstores})!=1:
        raise ValueError('Unknown K-code shared stores')
    kb=kstores[0][1];release=[i for i,s in blocks[kb] if '@llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)' in s and i<min(x[0] for x in kstores)]
    if not release or qb not in dom[kb]: raise ValueError('Q-read to K-bank overwrite barrier missing')
    kreads=[(i,b) for b,rows in blocks.items() for i,s in rows if re.search(r'= load <4 x i8>, ptr addrspace\(3\)',s)]
    if len(kreads)!=32 or len({b for _,b in kreads})!=1: raise ValueError('Unknown K shared reads')
    rb=kreads[0][1];acquire=[i for i,s in blocks[rb] if '@llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)' in s and i<min(x[0] for x in kreads)]
    if not acquire or kb not in dom[rb] or not max(x[0] for x in kstores)<max(acquire):
        raise ValueError('K shared stores do not precede synchronized operand reads')
    return dict(Q_store_lines=qstores,Q_publish_barrier=max(barriers),Q_read_block=qb,
                Q_read_block_dominates_KV=True,K_reuse_barrier=max(release),K_store_lines=[i for i,_ in kstores],
                K_acquire_barrier=max(acquire),K_operand_read_lines=[i for i,_ in kreads])



def split_operands(text):
    out=[];start=0;depth=0
    for i,c in enumerate(text):
        if c in '[{(':depth+=1
        elif c in ']})':depth-=1
        elif c==',' and depth==0:out.append(text[start:i].strip());start=i+1
    out.append(text[start:].strip());return out


def registers(text):
    result=set()
    for m in re.finditer(r'\bv(?:\[([0-9]+):([0-9]+)\]|([0-9]+))',text):
        result.update(range(int(m[1]),int(m[2])+1) if m[1] else [int(m[3])])
    return result


def assembly_observation(text):
    lines=text.splitlines();blocks={'entry':[]};current='entry';order=['entry'];membership={};instructions=[]
    labels={};after_branch=False
    for number,line in enumerate(lines,1):
        label=re.match(r'^(BB[0-9]+_[0-9]+):',line)
        if label:
            current=label[1];blocks[current]=[];order.append(current);labels[label[1]]=current;after_branch=False
        m=re.match(r'\s*((?:ml|sl)_[A-Za-z0-9_]+)(?:\s+(.*))?$',line.split(';',1)[0].rstrip())
        if m:
            if after_branch:
                current=f'continuation_{number}';blocks[current]=[];order.append(current);after_branch=False
            record=(number,m[1],split_operands(m[2] or ''));blocks[current].append(record)
            membership[number]=current;instructions.append(record)
            after_branch=m[1] in ('sl_jump','sl_exit','sl_ret') or m[1].startswith('sl_cbr')
    succ={b:set() for b in blocks}
    for i,b in enumerate(order):
        for _,op,args in blocks[b]:
            if op=='sl_jump' or op.startswith('sl_cbr'):
                targets=re.findall(r'BB[0-9]+_[0-9]+',','.join(args))
                if len(targets)!=1 or targets[0] not in labels:raise ValueError('unknown native branch target')
                succ[b].add(labels[targets[0]])
        last=blocks[b][-1][1] if blocks[b] else None
        if last not in ('sl_jump','sl_exit','sl_ret') and i+1<len(order):succ[b].add(order[i+1])
    reachable=set();todo=['entry']
    while todo:
        b=todo.pop()
        if b not in reachable:reachable.add(b);todo.extend(succ[b])
    if any(membership[r[0]] not in reachable for r in instructions if 'matrix_mad' in r[1]):
        raise ValueError('unreachable native matrix operation')
    blocks={b:rows for b,rows in blocks.items() if b in reachable};succ={b:targets&reachable for b,targets in succ.items() if b in reachable}
    pred={b:{p for p in blocks if b in succ[p]} for b in blocks};dom={b:({b} if b=='entry' else set(blocks)) for b in blocks}
    if any(not pred[b] for b in blocks if b!='entry'):raise ValueError('unreachable native block')
    while True:
        new={b:({b} if b=='entry' else {b}|set.intersection(*(dom[p] for p in pred[b]))) for b in blocks}
        if new==dom:break
        dom=new
    qk=[r for r in instructions if r[1]=='ml_matrix_mad_i32_i8_m16n16k32']
    pv=[r for r in instructions if r[1]=='ml_matrix_mad_f32_bf16_m16n16k16']
    if len(qk)!=16 or len(pv)!=32:raise ValueError('native Q32 I8 QK/BF16 PV counts differ')
    loops=[]
    for back,targets in succ.items():
        for head in targets:
            if head not in dom[back]:continue
            members={head,back};todo=[back]
            while todo:
                node=todo.pop()
                for p in pred[node]-members:
                    members.add(p)
                    if p!=head:todo.append(p)
            if all(membership[r[0]] in members for r in qk+pv):loops.append((head,members))
    if len(loops)!=1:raise ValueError('unique native natural KV loop not found')
    head,members=loops[0];first=min(r[0] for b in members for r in blocks[b])
    pairs=[registers(r[2][1]) for r in qk]
    if any(len(p)!=2 for p in pairs) or len({tuple(sorted(p)) for p in pairs})!=4:raise ValueError('unknown native Q operand register partition')
    if any(pairs.count(pair)!=4 for pair in pairs):raise ValueError('unequal native Q partitions')
    qregs=set.union(*pairs)
    def writes(record):
        _,op,args=record
        # BI store operands are source values/addresses, not vector destinations.
        return set() if 'store' in op or not args else registers(args[0])
    if any(writes(r)&qregs for b in members for r in blocks[b]):raise ValueError('KV loop overwrites persistent Q operand registers')
    roots=set();packing=set();visited=set()
    def walk(reg,before,use_block):
        token=(reg,before)
        if token in visited:return
        visited.add(token)
        defs=[r for r in instructions if r[0]<before and reg in writes(r)]
        if not defs:raise ValueError('native Q register origin missing')
        record=defs[-1];number,op,args=record;block=membership[number]
        if block in members or block not in dom[use_block]:raise ValueError('native Q definition does not dominate use outside KV')
        if any((o=='sl_jump' or o.startswith('sl_cbr')) and n<number for n,o,_ in blocks[block]):
            raise ValueError('native Q definition follows a branch within its block')
        if op=='ml_slb_load_b8':roots.add(number);return
        if op!='ml_byte_permute_b32' or len(args)!=4:raise ValueError('native Q not rooted only in shared bytes/packing')
        source=registers(args[1])|registers(args[2])
        if len(source)>2 or not source:raise ValueError('unknown Q byte packing inputs')
        packing.add(number)
        for r in source:walk(r,number,block)
    for reg in qregs:walk(reg,first,head)
    if len(roots)!=32 or len(packing)!=24:raise ValueError('unknown Q32 byte/pack mapping')
    # LLVM below establishes shared-store/publish/read/reuse/acquire dominance.
    # ASM additionally must retain a dominating CTA barrier before byte reads.
    qstores=[r for r in instructions if r[1]=='ml_slb_store_b8' and r[0]<min(roots) and membership[r[0]] not in members]
    if len(qstores)!=32 or len({membership[r[0]] for r in qstores})!=1:raise ValueError('native Q store grammar changed')
    qsb=membership[qstores[0][0]]
    publish=[n for n,op,_ in instructions if op=='sl_barrier' and max(r[0] for r in qstores)<n<min(roots) and membership[n]==qsb and qsb in dom[head]]
    if not publish:raise ValueError('native Q publish barrier absent')
    stores=[r for b in members for r in blocks[b] if r[1]=='ml_slb_store_b32x1']
    kroots=set()
    def kwalk(reg,before,use_block):
        defs=[r for r in instructions if r[0]<before and reg in writes(r)]
        if not defs:raise ValueError('native K operand origin missing')
        record=defs[-1];number,op,args=record;block=membership[number]
        if block not in members or block not in dom[use_block]:raise ValueError('native K operand does not dominate matrix use')
        if op=='ml_slb_load_b32x1':kroots.add(number);return
        source=registers(args[1]) if len(args)>1 else set()
        if op!='ml_mov_b32' or len(source)!=1:raise ValueError('unknown native K operand construction')
        kwalk(next(iter(source)),number,block)
    for line,_,args in qk:
        for reg in registers(args[2]):kwalk(reg,line,membership[line])
    reads=[r for r in instructions if r[0] in kroots]
    sb={membership[r[0]] for r in stores};rb={membership[r[0]] for r in reads};qb={membership[n] for n in roots}
    if len(stores)!=2 or len(reads)!=32 or len(sb)!=1 or len(rb)!=1 or len(qb)!=1:
        raise ValueError('unknown native K shared bank grammar')
    sb=next(iter(sb));rb=next(iter(rb));qb=next(iter(qb))
    release=[n for n,op,_ in blocks[sb] if op=='sl_barrier' and n<min(r[0] for r in stores)]
    acquire=[n for n,op,_ in blocks[rb] if op=='sl_barrier' and n<min(r[0] for r in reads)]
    if not release or not acquire or qb not in dom[sb] or sb not in dom[rb]:
        raise ValueError('native Q/K reuse or acquire synchronization absent')
    return dict(KV_header=head,KV_blocks=sorted(members),Q_registers=sorted(qregs),
        Q_byte_read_lines=sorted(roots),Q_pack_lines=sorted(packing),Q_publish_barrier=max(publish),Q_store_lines=[r[0] for r in qstores],
        Q_operand_definitions_dominate_KV=True,Q_registers_immutable_in_KV=True,
        K_reuse_barrier=max(release),K_store_lines=sorted(r[0] for r in stores),
        K_acquire_barrier=max(acquire),K_read_lines=sorted(r[0] for r in reads),
        I8_QK_lines=[r[0] for r in qk],BF16_PV_lines=[r[0] for r in pv])


def audit(proposed_ttgir,actual_ttgir,llir,assembly):
    if alpha(proposed_ttgir)!=alpha(actual_ttgir):raise ValueError('actual TTGIR differs beyond scoped SSA renaming')
    llvm=llvm_observation(llir)
    if not llvm['all_Q_operand_reads_outside_KV']:raise ValueError('actual LLVM Q operand remains inside KV')
    sync=llvm_sync(llir,llvm);machine=assembly_observation(assembly)
    return dict(actual_I8_operand_hoist_verified=True,TTGIR_scoped_SSA_equivalent=True,
        LLVM=llvm,shared_memory_synchronization=sync,ASM=machine,old_hash_as_admission=False,
        loaded_resources_verified=False,numerical_verified=False,precision_gate=False,service_verified=False)
