"""CPU audit of one actual Q32 I8 hoist. No JIT import, launch or timing license."""
import io
import json
from pathlib import Path
import re
import tarfile

import biv150_konly_q32_i8_hoist as old

DIRECTORY=old.ROOT/'docs/evidence/konly-prefill-reader-20261027/q32-i8-hoist-actual-v1'
ARCHIVE_SHA='341f4edc69299acc644b4da6749595bc88c81faf4cd9df333fdf140498e2a91a'
MANIFEST_SHA='6590624c88d6b6b961a6f629e79ef95a124d35e288482e0ea2a0fa2fc47b1785'
KERNEL_HASH='68f376d4394d07e21221008dd44099e2178ffd8b18a4e11a610bd175088f5ddc'
IR_SHA=dict(ttgir='ffb5820b7e12cfd7db9f6fe28cd388328334e3dde06a726f3685723a8fa1bd2c',
            llir='273529661740c323f7084bfb906140b1d03c14feb9e654bdbbdb8e150fc80605',
            asm='c73c0c100f481241f5686ba9a08f83a8f82d218aba78578eaa42b5b095678d64')


def original():
    packed=(DIRECTORY/'original-compile.tar.gz').read_bytes()
    if old.sha(packed)!=ARCHIVE_SHA: raise ValueError('Actual archive changed')
    with tarfile.open(fileobj=io.BytesIO(packed)) as archive:
        members=archive.getmembers()
        if len(members)!=43 or len({m.name for m in members})!=43 or any(
                not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):
            raise ValueError('Unsafe/unknown actual archive membership')
        files={m.name:archive.extractfile(m).read() for m in members}
    manifest=files['download-manifest.json']
    if old.sha(manifest)!=MANIFEST_SHA: raise ValueError('Download manifest changed')
    rows=json.loads(manifest)
    if (rows['raw_files'],rows['raw_bytes'],rows['original_owner'])!=(42,3151371,
            dict(pid=626927,start_ticks=343979040,same_start=True,state='Z')):
        raise ValueError('Actual owner/count changed')
    if set(files)!={r['relative'] for r in rows['files']}|{'download-manifest.json'}:
        raise ValueError('Actual membership changed')
    for row in rows['files']:
        value=files[row['relative']]
        if (len(value),old.sha(value))!=(row['bytes'],row['sha256']):
            raise ValueError('Raw actual bytes changed')
    summary=json.loads(files['out/compile-summary.json'])
    front=json.loads(files['out/override-frontend.json'])
    if (summary['completed'],summary['errors'],summary['kernel_launches'])!=(True,[],0):
        raise ValueError('Not a successful zero-launch compile')
    if summary['actual_I8_operand_hoist_verified'] or front['actual_I8_hoist_verified'] or front['ASM_hoist_auditor_supported']:
        raise ValueError('Original unsupported/false result changed')
    record=json.loads(files['out/ir/candidate/record.json'])
    if record['hash']!=KERNEL_HASH or record['actual_options']['ir_override']!=front['override_file']['path']:
        raise ValueError('Actual override metadata/hash changed')
    oldfiles,_,oldrecord=old.original()
    if record['typed_ABI']!=oldrecord['typed_ABI'] or record['files']['source']['sha256']!=oldrecord['files']['source']['sha256']:
        raise ValueError('Actual typed ABI/generated JIT source changed')
    if record['requested_options']!={**oldrecord['requested_options'],'ir_override':front['override_file']['path']}:
        raise ValueError('Options changed beyond the single fixed override')
    _,proof=old.transform(oldfiles['out/ir/candidate/actual.ttgir'].decode())
    if old.sha(files['out/fixed-q32-i8-operand-hoist.ttgir'])!=proof['override_sha256']:
        raise ValueError('Actual override input differs from CPU move')
    for ext,digest in IR_SHA.items():
        if old.sha(files['out/ir/candidate/actual.'+ext])!=digest:
            raise ValueError('Actual IR bytes changed')
    return files,summary,record


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
        assignments=list(re.finditer('('+old.SSA+r')(?::[0-9]+)?\s*=',line))
        blockargs=list(re.finditer('('+old.SSA+r')\s*:',line)) if stripped.startswith('^bb') else []
        first=bool(re.match(r'\s*'+old.SSA+r'(?::[0-9]+)?\s*=',line))
        if first: scopes[-1][assignments[0][1]]='pending'
        if line.rstrip().endswith('{'): scopes.append({})
        for match in assignments[1 if first else 0:]+blockargs:
            scopes[-1][match[1]]='pending'
        for ordinal,match in enumerate(re.finditer(old.SSA,line)):
            if any(match.start()==a.start() for a in assignments+blockargs):
                for scope in reversed(scopes):
                    if match[0] in scope:
                        scope[match[0]]=f'%d{lineno}_{ordinal}';break
        out.append(re.sub(old.SSA,lambda m:next((s[m[0]] for s in reversed(scopes) if m[0] in s),m[0]),line))
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


def asm_observation(text):
    lines=text.splitlines();start=old.one(lines,lambda s:s.startswith('BB0_38:'),'Unknown ASM loop body')
    header=old.one(lines,lambda s:s.startswith('BB0_39:'),'Unknown ASM loop header')
    end=old.one(lines,lambda s:s.startswith('BB0_59:'),'Unknown ASM exit')
    if not start<header<end or 'sl_jump BB0_39' not in lines[start-1]: raise ValueError('Unknown loop entry')
    loop=lines[start:end]
    if not any('sl_jump BB0_38' in x for x in loop): raise ValueError('ASM loop backedge missing')
    if any('ml_slb_load_b8 ' in x for x in loop): raise ValueError('Byte Q operand reload remains inside KV loop')
    # Whole textual natural-loop body, including all masked-load subpaths.
    for line in loop:
        m=re.match(r'\s*ml_\w+\s+v(?:([0-9]+)|\[([0-9]+):([0-9]+)\])[, ]',line)
        if m:
            written={int(m[1])} if m[1] else set(range(int(m[2]),int(m[3])+1))
            if written&set(range(32,40)): raise ValueError('Loop overwrites persistent Q registers')
    qk=[(i+1,s) for i,s in enumerate(lines) if 'ml_matrix_mad_i32_i8_m16n16k32' in s]
    pv=[i+1 for i,s in enumerate(lines) if 'ml_matrix_mad_f32_bf16_m16n16k16' in s]
    if len(qk)!=16 or len(pv)!=32 or not all(start<i<=end for i,_ in qk) or not all(start<i<=end for i in pv):
        raise ValueError('Native QK/PV instruction contract changed')
    pairs=[]
    for _,line in qk:
        m=re.match(r'\s*ml_matrix_mad_i32_i8_m16n16k32 v\[[0-9]+:[0-9]+\], v\[([0-9]+):([0-9]+)\],',line)
        if not m or tuple(map(int,m.groups())) not in ((32,33),(34,35),(36,37),(38,39)):
            raise ValueError('I8 MMA A is not the persistent Q operand')
        pairs.append(tuple(map(int,m.groups())))
    if any(pairs.count(pair)!=4 for pair in set(pairs)) or len(set(pairs))!=4:
        raise ValueError('Unknown I8 Q partition')
    roots=set();packing=set()
    def walk(reg,before):
        defs=[(i,s) for i,s in enumerate(lines[:before]) if re.match(r'\s*ml_\w+\s+v'+str(reg)+r',',s)]
        if not defs: raise ValueError('Unknown persistent Q register definition')
        i,line=defs[-1]
        if 'ml_slb_load_b8 ' in line: roots.add(i+1);return
        m=re.match(r'\s*ml_byte_permute_b32 v[0-9]+, v([0-9]+), v([0-9]+), (?:4|s[01])$',line)
        if not m: raise ValueError('Q register does not derive solely from shared bytes/permutations')
        packing.add(i+1)
        for source in map(int,m.groups()): walk(source,i)
    for register in range(32,40): walk(register,start)
    if len(roots)!=32 or len(packing)!=24: raise ValueError('Unknown packed Q byte mapping')
    release=[i+1 for i,s in enumerate(lines[:min(roots)-1]) if s.strip()=='sl_barrier']
    if not release: raise ValueError('ASM Q publish barrier missing')
    first_k_store=next(i+1 for i in range(header,end) if 'ml_slb_store_b32x1' in lines[i])
    reuse=[i+1 for i in range(header,first_k_store-1) if lines[i].strip()=='sl_barrier']
    if not reuse: raise ValueError('ASM Q-read/K-overwrite barrier missing')
    return dict(KV_loop_body=start+1,KV_header=header+1,KV_exit=end+1,Q_registers=list(range(32,40)),
                Q_byte_read_lines=sorted(roots),Q_pack_lines=sorted(packing),Q_publish_barrier=release[-1],
                K_reuse_barrier=reuse[-1],first_K_shared_store=first_k_store,I8_MMA_lines=[i for i,_ in qk],
                BF16_PV_lines=pv,all_Q_byte_reads_outside_KV=True,Q_registers_immutable_in_KV=True)


def audit(ttgir,llir,assembly,*,known_hash=True):
    before=old.original()[0]['out/ir/candidate/actual.ttgir'].decode();expected,_=old.transform(before)
    if alpha(ttgir)!=alpha(expected): raise ValueError('Actual TTGIR differs beyond scoped SSA renaming')
    facts=old.grammar(ttgir,True);llvm=old.llvm_observation(llir)
    if not llvm['all_Q_operand_reads_outside_KV']: raise ValueError('Q operand still in KV loop')
    sync=llvm_sync(llir,llvm);asm=asm_observation(assembly)
    if known_hash and any(old.sha(s)!=IR_SHA[e] for e,s in [('ttgir',ttgir),('llir',llir),('asm',assembly)]):
        raise ValueError('Different actual specialization requires independent review')
    return dict(TTGIR=facts,scoped_SSA_equivalent_to_fixed_move=True,LLVM=llvm,synchronization=sync,ASM=asm,
                actual_I8_operand_hoist_verified=True,witnessed_specialization_only=known_hash,
                runtime_binding_qualified=False,accuracy_verified=False,precision_gate=False,
                performance_verified=False,service_verified=False,kernel_launches=0)
