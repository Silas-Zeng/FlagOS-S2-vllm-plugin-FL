"""One witnessed Q32 I8-operand move. Stdlib only; not a device launcher/license."""
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'docs/evidence/konly-prefill-reader-20261027/q32-i8-hoist-v1'
ARCHIVE_SHA='f3e4f8e1f4414fc1181818746d568c2e7213211f3924db223a8a1e9d549fe183'
MANIFEST_SHA='ff5011dddc1edaef9d94f52fe5aaf67e01da03147e3ab633bf11e611a234501f'
RAW_TTGIR_SHA='5e87a8fddac1cf254628a409212f5a6fed2be616179bdefb894e101c40b78264'
RAW_LLVM_SHA='250c1e69bd9342b6ba82416a0af60b46c4692c5105ecf77f6132133dccdf7a26'
RAW_ASM_SHA='c5c004c0740ae6d8560a90dc71658c67318603990e364c603d35dfef030ad995'
KERNEL_HASH='c0a523d9d0d4d5dde57b241df9d22c9c38fa71e480878d18e690b4eb002e24f7'
DRIVER='d6a42ba24bd6f48cc73e3116394f0848b2610745'
READER='24c741cc152fc7a53f0b8ce27c1e988898fc59ec'
SSA=r'%[-.A-Za-z0-9_]+'


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def original():
    packed=(DIRECTORY/'original-compile.tar.gz').read_bytes()
    if sha(packed)!=ARCHIVE_SHA:
        raise ValueError('Original compile archive changed')
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        members=archive.getmembers()
        if len(members)!=41 or len({m.name for m in members})!=41:
            raise ValueError('Unexpected original membership')
        if any(not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe original member')
        files={m.name:archive.extractfile(m).read() for m in members}
    manifest=files['download-manifest.json']
    if sha(manifest)!=MANIFEST_SHA:
        raise ValueError('Root original download manifest changed')
    rows=json.loads(manifest)
    if rows['raw_files']!=40 or rows['raw_bytes']!=3047276 or rows['original_owner']!=dict(pid=626170,start_ticks=343488382,same_start=True,state='Z'):
        raise ValueError('Original owner/count changed')
    if set(files)!={x['relative'] for x in rows['files']}|{'download-manifest.json'}:
        raise ValueError('Original file list changed')
    for row in rows['files']:
        value=files[row['relative']]
        if len(value)!=row['bytes'] or sha(value)!=row['sha256']:
            raise ValueError('Original file hash changed')
    summary=json.loads(files['out/compile-summary.json'])
    if (summary['driver_commit'],summary['reader_source_commit'],summary['completed'],summary['errors'],summary['kernel_launches'])!=(DRIVER,READER,True,[],0):
        raise ValueError('Not the original compile-only d6 record')
    record=next(r for r in summary['records'] if r['label']=='candidate')
    if record['hash']!=KERNEL_HASH or summary['actual_I8_operand_hoist_verified'] or record['actual_I8_hoist']['verified']:
        raise ValueError('Original false hoist state/kernel changed')
    if summary['plan']['profile']!='q32' or record['requested_options']!=dict(num_stages=1,num_warps=16):
        raise ValueError('Only this normalized-Q32 case is witnessed')
    for ext,digest in [('ttgir',RAW_TTGIR_SHA),('llir',RAW_LLVM_SHA),('asm',RAW_ASM_SHA)]:
        if sha(files['out/ir/candidate/actual.'+ext])!=digest:
            raise ValueError('Actual candidate IR changed')
    return files,summary,record


def one(rows,predicate,reason):
    matches=[i for i,line in enumerate(rows) if predicate(line)]
    if len(matches)!=1:
        raise ValueError(reason)
    return matches[0]


def grammar(text,hoisted=False):
    """Narrow textual/SSA grammar, independently checked before the hash gate."""
    lines=text.splitlines(keepends=True)
    if '@kernel_cjj_konly_q32(' not in text or text.count(' = scf.for ')!=1:
        raise ValueError('Unknown function/KV-loop shape')
    loop=one(lines,lambda s:' = scf.for ' in s,'Unique KV loop required')
    end=next((i for i in range(loop+1,len(lines)) if lines[i].startswith('    } loc(')),None)
    if end is None or not lines[loop].startswith('    %'):
        raise ValueError('Unknown KV region/parent')
    qglobal=one(lines,lambda s:' = tt.load ' in s and 'tensor<256x128x!tt.ptr<bf16>' in s,'Unique original Q global load required')
    cast=one(lines,lambda s:' = arith.fptosi ' in s and 'to tensor<256x128xi8,' in s,'Unique encoded Q I8 result required')
    alloc=one(lines,lambda s:' = ttg.local_alloc ' in s and '(tensor<256x128xi8,' in s,'Unique QCode allocation required')
    qmatch=re.match(r'    ('+SSA+r') = ttg.local_alloc ('+SSA+r') :',lines[alloc])
    if not qmatch:
        raise ValueError('Unknown allocation SSA')
    memory,code=qmatch.groups()
    if not lines[cast].startswith('    '+code+' = ') or not qglobal<cast<alloc<loop:
        raise ValueError('Q read/quantization/allocation do not dominate KV loop')
    divs=[i for i,s in enumerate(lines) if ' = tt.precise_divf ' in s]
    if len(divs)!=2 or not all(qglobal<i<cast for i in divs):
        raise ValueError('RN32 codec no longer outside KV loop')
    uses=[i for i,s in enumerate(lines) if re.search(re.escape(memory)+r'(?![-.A-Za-z0-9_])',s)]
    load=one(lines,lambda s:' = ttg.local_load '+memory+' :' in s,'Unique immutable QCode read required')
    if sorted(uses)!=sorted([alloc,load]):
        raise ValueError('Q allocation aliases/escapes/is modified')
    loaded=re.match(r'\s+('+SSA+r') =',lines[load])
    if loaded is None: raise ValueError('Unknown Q load result SSA')
    operand=loaded.group(1)
    expected='!ttg.memdesc<256x128xi8, #shared, #smem> -> tensor<256x128xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4, useSme = 0}>>'
    if expected not in lines[load]:
        raise ValueError('Not the witnessed signed-I8 Q operand layout')
    if hoisted:
        if not alloc<load<loop or not lines[load].startswith('    %') or lines[load].startswith('      '):
            raise ValueError('Q operand has not moved to the same parent/preheader')
    elif not loop<load<end or not lines[load].startswith('      %'):
        raise ValueError('Original Q operand is not inside KV loop')
    dots=[i for i,s in enumerate(lines) if ' = tt.dot ' in s]
    if len(dots)!=2 or not all(loop<i<end for i in dots):
        raise ValueError('Original QK/PV dot sites changed')
    qk=one(lines,lambda s:' = tt.dot '+operand+', ' in s,'Q operand no longer directly feeds QK')
    if qk!=dots[0] or '-> tensor<256x64xi32, #mma>' not in lines[qk] or 'tensor<128x64xi8,' not in lines[qk]:
        raise ValueError('I8 QK shape/output changed')
    carried=re.search(r', ('+SSA+r'), inputPrecision',lines[qk])
    if carried is None: raise ValueError('Unknown QK accumulator SSA')
    c=carried.group(1)
    if not any(s.startswith('    '+c+' = arith.constant dense<0> : tensor<256x64xi32, #mma>') for s in lines):
        raise ValueError('I8 QK must retain original zero I32 C')
    if 'tensor<256x64xbf16,' not in lines[dots[1]] or 'tensor<64x128xbf16,' not in lines[dots[1]] or '-> tensor<256x128xf32, #mma1>' not in lines[dots[1]]:
        raise ValueError('Original BF16 PV changed')
    operand_uses=[i for i,s in enumerate(lines) if re.search(re.escape(operand)+r'(?![-.A-Za-z0-9_])',s)]
    if sorted(operand_uses)!=sorted([load,qk]):
        raise ValueError('Unexpected Q operand use')
    return dict(q_global=qglobal+1,q_cast=cast+1,q_allocation=alloc+1,KV_loop=loop+1,KV_end=end+1,
                Q_operand_load=load+1,QK=qk+1,PV=dots[1]+1,memory=memory,operand=operand,
                immutable_Q_allocation=True,codec_and_Q_global_outside_KV=True)


def transform(raw):
    facts=grammar(raw)
    if sha(raw)!=RAW_TTGIR_SHA:
        raise ValueError('Only the exact witnessed d6 Q32 TTGIR can be transformed')
    lines=raw.splitlines(keepends=True);old=lines[facts['Q_operand_load']-1]
    del lines[facts['Q_operand_load']-1]
    lines.insert(facts['KV_loop']-1,old[2:])
    changed=''.join(lines);after=grammar(changed,True)
    restored=lines.copy();del restored[after['Q_operand_load']-1]
    restored.insert(facts['Q_operand_load']-1,old)
    if ''.join(restored)!=raw:
        raise ValueError('Single-operation byte inverse failed')
    return changed,dict(input_sha256=sha(raw),override_sha256=sha(changed),before=facts,after=after,
        moved_operations=1,exact_byte_inverse=True,source_and_ABI_unchanged=True,
        original_stores_math_masks_barriers_unchanged=True,CPU_preconditions_passed=True,
        actual_override_compile_completed=False,actual_I8_hoist_verified=False,
        runtime_binding_qualified=False,precision_gate=False,performance_verified=False)


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


def audit(ttgir,llir,assembly):
    errors=[];facts={};llvm={}
    files,_,_=original();expected,_=transform(files['out/ir/candidate/actual.ttgir'].decode())
    if sha(ttgir)!=sha(expected): errors.append('Actual TTGIR is not the exact CPU-derived override; review required')
    try: facts=grammar(ttgir,True)
    except (ValueError,StopIteration) as error: errors.append(str(error))
    try: llvm=llvm_observation(llir)
    except (ValueError,StopIteration) as error: errors.append(str(error))
    counts=dict(I8_QK=assembly.count('ml_matrix_mad_i32_i8_m16n16k32'),BF16_PV=assembly.count('ml_matrix_mad_f32_bf16_m16n16k16'))
    if counts!=dict(I8_QK=16,BF16_PV=32): errors.append('Unknown ASM native matrix counts')
    if llvm and not llvm['all_Q_operand_reads_outside_KV']: errors.append('LLVM Q operand shared reads remain inside KV loop')
    # A post-hoist ASM positive has not been observed. Do not invent its register grammar.
    return dict(TTGIR=facts,LLVM=llvm,ASM_static_counts=counts,errors=errors,
                ASM_hoist_auditor_supported=False,actual_I8_hoist_verified=False,
                ready_for_numeric_or_timing=False,runtime_binding_qualified=False,precision_gate=False,performance_verified=False)
