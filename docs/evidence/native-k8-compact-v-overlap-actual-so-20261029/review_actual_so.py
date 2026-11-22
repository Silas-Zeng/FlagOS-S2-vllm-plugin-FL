#!/usr/bin/env python3
"""CPU review of PR110's final 7892 SO; never import Torch or load machine code.

The small ELF/CFG/IR helpers reuse the previous actual-SO audit algorithms.
This receipt is an independent review recommendation, not Root approval or a GPU result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'tools'))
import audit_biv150_native_k8_compile_cpu as elfutil
import probe_biv150_native_k8_compact_v_overlap_consumer as consumer

SOURCE='4d1166792a5247b774d6b1977e60466574e6a34c'
SO='7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4'
CONTROL_SO='40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453'
CODEC_TEXT='936ce115af95401dda3a37cf310b62baa606003a18910ff46de4707fbaba19cf'
def sha(b):return hashlib.sha256(b).hexdigest()

# Helper source: audit_cjj_vfirst_actual_SO_sass_cpu_20261028.py SHA256 b89b78a8c16ee3f9da99cc7abf5104a7b17499b58ab0ae2edf87aa536151b45e
def llvm_functions(text):
    # ixobjdump wraps long signature lines. Keep the raw display untouched;
    # inspect body SSA without claiming that the display can be recompiled.
    for m in re.finditer(r'^define [^\n]*@([^ (]+)\(',text,re.M):
        begin=m.start();opening=text.index('{\n',m.end());end=text.index('\n}',opening)+2
        yield m[1],text[begin:end],text[:begin].count('\n')+1

def elf_text(data):
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',data);rows=[struct.unpack_from('<IIQQQQIIQQ',data,h[6]+i*h[11]) for i in range(h[12])]
    names=data[rows[h[13]][4]:rows[h[13]][4]+rows[h[13]][5]]
    for row in rows:
        n=names[row[0]:names.index(b'\0',row[0])].decode()
        if n=='.text':return row[3],data[row[4]:row[4]+row[5]]
    raise AssertionError('no .text')

def machine_cfg(rows):
    pcs={x['pc']:i for i,x in enumerate(rows)};start={rows[0]['pc']};targets={}
    for i,row in enumerate(rows):
        op=row['op']
        if op=='sl_jump' or op.startswith('sl_cbr_'):
            displacement=int(row['args'],0);target=(row['pc']+8+displacement)&0xffffffff
            assert target in pcs;targets[row['pc']]=target;start.add(target)
            if i+1<len(rows):start.add(rows[i+1]['pc'])
    blocks={};at={};current=None
    for row in rows:
        if row['pc'] in start:current=row['pc'];blocks[current]=[]
        blocks[current].append(row);at[row['pc']]=current
    edges={}
    for block,body in blocks.items():
        row=body[-1];i=pcs[row['pc']];nextpc=rows[i+1]['pc'] if i+1<len(rows) else None
        if row['op']=='sl_jump':successors={at[targets[row['pc']]]}
        elif row['op'].startswith('sl_cbr_'):successors={at[targets[row['pc']]],at[nextpc]}
        elif row['op'] in ('sl_exit','sl_end','sl_ret'):successors=set()
        else:successors={at[nextpc]} if nextpc is not None else set()
        edges[block]=successors
    entry=rows[0]['pc'];reachable={entry};todo=[entry]
    while todo:
        for x in edges[todo.pop()]:
            if x not in reachable:reachable.add(x);todo.append(x)
    dom={x:({entry} if x==entry else set(reachable)) for x in reachable};changed=True
    while changed:
        changed=False
        for x in reachable-{entry}:
            ps=[p for p in reachable if x in edges[p]];d={x}|set.intersection(*(dom[p] for p in ps))
            if d!=dom[x]:changed=True;dom[x]=d
    return at,dom,targets

# Helper source: review_cjj_native_k8_v_overlap_actual_SO_20261028.py SHA256 dac7555a54197c0fd3ca14c71abce1d32e65e93a136ec33eec7c4146765ab132
def ir_cfg(fn,offset):
    blocks={'entry':[]};at={};current='entry'
    for number,line in enumerate(fn.splitlines(),offset):
        label=re.match(r'^([\w.$-]+):',line)
        if label:current=label[1];blocks[current]=[]
        blocks[current].append((number,line));at[number]=current
    edges={b:{v for _,line in rows if re.match(r'\s*(br|switch)\b',line)
              for v in re.findall(r'label %([\w.$-]+)',line)} for b,rows in blocks.items()}
    reachable={'entry'};todo=['entry']
    while todo:
        for nxt in edges[todo.pop()]:
            if nxt not in reachable:reachable.add(nxt);todo.append(nxt)
    dom={b:({'entry'} if b=='entry' else set(reachable)) for b in reachable};changed=True
    while changed:
        changed=False
        for b in reachable-{'entry'}:
            pred=[p for p in reachable if b in edges[p]]
            value={b}|set.intersection(*(dom[p] for p in pred))
            if value!=dom[b]:dom[b]=value;changed=True
    return lambda a,b:at[a] in dom[at[b]] and (at[a]!=at[b] or a<b)

# Helper source: audit_cjj_native_k8_vfirst_compile_actual_20261028.py SHA256 21305a27d7afaff624dd33eb080d398c38fa41d31f3d1d3c52143f367eab99fe
def llvm_profile(name,fn,offset):
    lines=fn.splitlines();defs=dict(re.findall(r'^\s*(%[\w.]+) = (.*)$',fn,re.M))
    roots={'%0','%1','%2','%3','%4','%5'}
    def origins(value,seen=None):
        seen=set() if seen is None else seen
        if value in seen:return set()
        seen.add(value)
        if value in roots:return {value}
        return set().union(*(origins(v,seen) for v in re.findall(r'%[\w.]+',defs.get(value,''))))
    blocks={'entry':[]};current='entry';at={};edges={}
    for i,line in enumerate(lines,offset):
        label=re.match(r'^([\w.$-]+):',line)
        if label:current=label[1];blocks[current]=[]
        blocks[current].append((i,line));at[i]=current
    for block,body in blocks.items():
        edges[block]=set(v for _,line in body if re.match(r'\s*(br|switch)\b',line) for v in re.findall(r'label %([\w.$-]+)',line))
    reach={'entry'};todo=['entry']
    while todo:
        for target in edges[todo.pop()]:
            assert target in blocks
            if target not in reach:reach.add(target);todo.append(target)
    dom={b:({'entry'} if b=='entry' else set(reach)) for b in reach}
    changed=True
    while changed:
        changed=False
        for b in reach-{'entry'}:
            predecessors=[p for p in reach if b in edges[p]]
            value={b}|set.intersection(*(dom[p] for p in predecessors))
            if value!=dom[b]:changed=True;dom[b]=value
    loads=[];stores=[];sme=[];waits=[];barriers=[]
    for i,line in enumerate(lines,offset):
        m=re.search(r'= load i32, ptr addrspace\(1\) (%[\w.]+)',line)
        if m and '%1' in origins(m[1]):loads.append(i)
        m=re.search(r'^\s*store i32 (%[\w.]+), ptr addrspace\(3\)',line)
        if m and '%1' in origins(m[1]):stores.append(i)
        if '@llvm.bi.sme.load.16x1b64.rowxfb16(' in line:sme.append(i)
        if '@llvm.bi.sl.waitcnt(i64 8)' in line:waits.append(i)
        if '@llvm.bi.sl.barrier(' in line:barriers.append(i)
    assert len(loads)==len(stores)==16 and len(sme)==8 and len(waits)==2
    paths=[]
    for index,kind in enumerate(('full','tail')):
        v=sme[index*4:index*4+4];k=loads[index*8:index*8+8];s=stores[index*8:index*8+8]
        assert len({at[i] for i in v})==1
        block=at[v[0]]
        domination=all(block in dom[at[i]] and (at[i]!=block or max(v)<i) for i in k+s)
        wait=waits[index];before=[i for i in barriers if max(s)<i<wait];after=[i for i in barriers if wait<i]
        paths.append(dict(path=kind,V_SME_lines=v,K_global_I32_load_lines=k,K_shared_I32_store_lines=s,
            V_block=block,V_dominates_all_K_loads_and_stores=domination,
            K_barrier_lines_before_wait=before,V_wait8_line=wait,V_post_wait_barrier_line=min(after)))
    counts=dict(I8_QK=fn.count('@llvm.bi.matrix.mad.i32x4.i8x8('),BF16_PV=fn.count('@llvm.bi.matrix.mad.f32x4.bf16x4('),
        unflagged_F32_divides=len(re.findall(r'= fdiv float ',fn)),Q_integer_RNE=fn.count('@llvm.nvvm.f2i.rn('),
        signed_I32_to_F32=len(re.findall(r'= sitofp i32 ',fn)),BF16_P_casts=len(re.findall(r'fptrunc contract float .* to bfloat',fn)),
        exp2_approx=fn.count('@llvm.nvvm.ex2.approx.f32('),L_FMA=fn.count('@llvm.fma.f32('))
    assert counts==dict(I8_QK=8,BF16_PV=16,unflagged_F32_divides=18,Q_integer_RNE=16,signed_I32_to_F32=4,BF16_P_casts=4,exp2_approx=8,L_FMA=4)
    return dict(symbol=name,body_sha256=sha(fn.encode()),global_lines=[offset,offset+len(lines)-1],paths=paths,
        static_counts_full_plus_tail=counts,standalone_Vfirst=all(x['V_dominates_all_K_loads_and_stores'] for x in paths))

def numeric_lines(fn):
    return [re.sub(r'\s+',' ',x.strip()) for x in fn.splitlines()
            if re.search(r'= (?:fadd|fsub|fmul|fdiv|fptrunc|fpext|sitofp)\b|@llvm\.(?:bi\.matrix|fma|fabs|maxnum|smin|smax|nvvm\.ex2|nvvm\.f2i)',x)]

def alpha(rows):
    names={}
    return [re.sub(r'%\d+',lambda m:names.setdefault(m[0],'%v'+str(len(names))),x) for x in rows]

def normalize_slice(text):
    text=re.sub(r'_ZZ\d+cjj_native_k8_compact(?:_v_overlap)?_producer','_ZZ_KERNEL_STAGE',text)
    text=re.sub(r'\b(nuw|nsw|disjoint|samesign)\b\s*','',text)
    text=re.sub(r'!llvm.loop !\d+','!llvm.loop !LOOP',text)
    rows=[re.sub(r'^(\d+):',r'%\1:',re.sub(r'\s+',' ',x.split(';')[0].strip())) for x in text.splitlines()]
    return '\n'.join(alpha([x for x in rows if x]))

def slices(fn):
    lines=fn.splitlines();wait=[i for i,x in enumerate(lines) if '@llvm.bi.sl.waitcnt(i64 8)' in x]
    pv=[i for i,x in enumerate(lines) if '@llvm.bi.matrix.mad.f32x4.bf16x4(' in x]
    waitline=wait[1]
    join=next(re.match(r'^([\w.$-]+):',lines[i])[1] for i in range(waitline,-1,-1) if re.match(r'^([\w.$-]+):',lines[i]))
    guard=next(i for i in range(waitline+1,len(lines)) if re.search(r'= icmp ult i32 %\d+, 16$',lines[i]))
    begin=next(i for i in range(guard+1,len(lines)) if ' = phi i32 ' in lines[i])
    end=next(i for i in range(begin+1,len(lines)) if '!llvm.loop ' in lines[i])
    # Compare the unchanged clear-loop body. Its outer merge/barrier placement
    # deliberately changes with the six reversible late-wait source edits.
    return dict(tail_clear_guard=lines[guard],tail_clear_loop=re.sub(r'%'+re.escape(join)+r'\b','%TAIL_JOIN','\n'.join(lines[begin:end+1])),
                final_PV_output_store='\n'.join(lines[pv[-1]:]))

def loop_metadata(module,node):
    metadata=dict(re.findall(r'^!(\d+) = (.*)$',module,re.M))
    def visit(current,stack):
        if current in stack:return 'SELF'
        return re.sub(r'!(\d+)',lambda m:visit(m[1],stack|{current}),metadata[current])
    return visit(node,set())

def must_reach(rows,targets,start,stop):
    """All finite CFG paths after start hit stop, with no avoiding cycle/exit."""
    bypc={r['pc']:i for i,r in enumerate(rows)};visited=set();edges={};todo=[rows[start]['pc']];end=rows[stop]['pc']
    while todo:
        pc=todo.pop()
        if pc==end or pc in visited:continue
        visited.add(pc);i=bypc[pc];r=rows[i]
        assert r['op'] not in ('sl_exit','sl_end','sl_ret') and i+1<len(rows)
        nxt=rows[i+1]['pc']
        edges[pc]=({targets[pc]} if r['op']=='sl_jump' else {targets[pc],nxt} if r['op'].startswith('sl_cbr_') else {nxt})
        todo.extend(edges[pc]-visited)
    # Topological elimination also rejects a path that can cycle forever before the boundary.
    finished={end};changed=True
    while changed:
        new={pc for pc,successors in edges.items() if successors<=finished};changed=bool(new-finished);finished|=new
    assert visited<=finished,(rows[start],rows[stop])

def raw_manifest(root):
    manifest=json.loads((root/'download-manifest.json').read_text())
    files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert files=={'download-manifest.json',*(r['relative'] for r in manifest['files'])}
    for r in manifest['files']:
        b=(root/r['relative']).read_bytes();assert (len(b),sha(b))==(r['bytes'],r['sha256'])
    assert len(manifest['files'])==manifest['raw_files']
    assert sum(r['bytes'] for r in manifest['files'])==manifest['raw_bytes']
    return manifest

def actual_exports(root,expected_so,library_name):
    out=root/'out';summary=json.loads((out/'compile-summary.json').read_text())
    inspection=json.loads((out/'actual-so-inspection.json').read_text())
    lib=(out/'build'/library_name).read_bytes()
    assert sha(lib)==summary['library']['sha256']==inspection['library']['sha256']==expected_so
    assert inspection['completed'] and not inspection['errors']
    texts={}
    for mode in ('IR','SASS'):
        filename='actual-so-'+('sass' if mode=='SASS' else mode)+'.stdout.txt'
        row=next(r for r in inspection['artifact_manifest'] if r['relative']==filename);b=(out/filename).read_bytes()
        assert (len(b),sha(b))==(row['bytes'],row['sha256'])
        command=next(c for c in inspection['commands'] if c['argv'][1]==('--sass' if mode=='SASS' else '--IR'))
        assert command['exit_code']==0 and command['argv']==[inspection['objdump'],'--sass' if mode=='SASS' else '--IR',inspection['library']['path']]
        assert inspection['objdump_sha256']=='b229415fa402db66fcd1355cd68f1ed0efbb3259ef3d9fc97f3884189e79d707'
        assert inspection['compile_summary_sha256']==sha((out/'compile-summary.json').read_bytes())
        texts[mode]=b.decode()
    return lib,texts,summary,inspection

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--control-raw',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args()
    m=raw_manifest(a.raw);oldm=raw_manifest(a.control_raw)
    assert (m['raw_files'],m['raw_bytes'])==(33,1408237)
    assert m['original_owner']==dict(pid=60986,start_ticks=355555493,same_start=True,state='Z')
    owner=json.loads((a.raw/'exports/cjj-native-k8-compact-v-overlap-compile-v1-20261029-owner.json').read_text());assert owner['source_commit']==SOURCE
    lib,texts,summary,inspection=actual_exports(a.raw,SO,'cjj_native_k8_compact_v_overlap_probe.so')
    oldlib,oldtexts,_,_=actual_exports(a.control_raw,CONTROL_SO,'cjj_native_k8_compact_probe.so')
    assert summary['completed'] and summary['compile_passed'] and not summary['errors']
    assert summary['torch']['cuda_initialized_before'] is False and summary['torch']['cuda_initialized_after'] is False
    assert all(summary[k]==0 for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    _,factory,contract,_=consumer.modules();binding=contract.compile_binding(a.raw/'out/compile-summary.json')
    metadata=elfutil.embedded(lib);oldmeta=elfutil.embedded(oldlib);objects=[]
    for name in ('qk','codec'):objects+=elfutil.embedded((a.raw/'out/build'/('biv150_native_k8_compact_v_overlap_'+name+'.cuda.o')).read_bytes())
    assert sorted(x['sha256'] for x in metadata)==sorted(x['sha256'] for x in objects)
    codec=next(x for x in metadata if len(x['kernels'])==1)
    assert codec['text_sha256']==next(x for x in oldmeta if len(x['kernels'])==1)['text_sha256']==CODEC_TEXT
    fat=elfutil.elf(lib)['sections']['.nv_fatbin']['bytes'];by_symbol={}
    for i,match in enumerate(re.finditer(b'\x7fELF',fat)):
        b=fat[match.start():];address,blob=elf_text(b)
        for k in metadata[i]['kernels']:by_symbol[k['.name']]=(address,blob,k)
    functions={};current=None
    for number,line in enumerate(texts['SASS'].splitlines(),1):
        mt=re.search(r'Function : (\S+)',line)
        if mt:current=mt[1];functions[current]=[]
        mt=re.match(r'^\s*((?:sl|ml)_\w+)\b(.*?)// ([0-9A-F]+): ([0-9A-F]{8}) ([0-9A-F]{8})',line)
        if mt:
            row=dict(line=number,op=mt[1],args=mt[2].strip(),pc=int(mt[3],16));address,blob,_=by_symbol[current];at=row['pc']-address
            assert blob[at:at+8]==struct.pack('<II',int(mt[4],16),int(mt[5],16));functions[current].append(row)
    assert not re.search(r'^\s*\.long ',texts['SASS'],re.M)
    assert set(functions)==set(by_symbol)
    for rows in functions.values():assert len(rows)*8==rows[-1]['pc']-rows[0]['pc']+8
    resourcekeys=('.vgpr_count','.sgpr_count','.group_segment_fixed_size','.private_segment_fixed_size','.vgpr_spill_count','.sgpr_spill_count','.warp_size')
    machine=[];resources={}
    for name,rows in functions.items():
        if '_producer' not in name:continue
        seg=int(re.search('ILj([24])EE',name)[1]);at,dom,targets=machine_cfg(rows)
        dominates=lambda x,y:at[rows[x]['pc']] in dom[at[rows[y]['pc']]] and (at[rows[x]['pc']]!=at[rows[y]['pc']] or rows[x]['pc']<rows[y]['pc'])
        find=lambda op:[i for i,r in enumerate(rows) if r['op']==op]
        sme=find('sl_sme_load_16x1b64_rowxfb16');wait=[i for i,r in enumerate(rows) if r['op']=='sl_wait' and r['args']=='g2scnt(0)']
        qk=find('ml_matrix_mad_i32_i8_m16n16k32');pv=find('ml_matrix_mad_f32_bf16_m16n16k16')
        assert (len(sme),len(wait),len(qk),len(pv))==(8,2,8,16)
        mult=[r['line'] for r in rows if r['op']=='sl_mul_i64_i32' and r['args'].endswith(', 0x3100')];assert len(mult)==2
        paths=[]
        for n,label in enumerate(('full','tail')):
            vi=sme[4*n:4*n+4];qi=qk[4*n:4*n+4];pi=pv[8*n:8*n+8];wi=wait[n]
            kb=max(i for i in range(vi[0]) if rows[i]['op']=='sl_barrier');vb=next(i for i in range(wi+1,pi[0]) if rows[i]['op']=='sl_barrier')
            kl=[i for i in range(kb) if rows[i]['op']=='ml_lsa_load_dword'][-8:]
            ks=[i for i in range(kb) if rows[i]['op']=='ml_slb_store_b32x1'][-8:]
            assert len(kl)==len(ks)==8
            for i in kl+ks:must_reach(rows,targets,i,kb)
            assert all(dominates(kb,i) for i in vi)
            exp=[i for i in range(qi[-1]+1,len(rows)) if rows[i]['op']=='ml_exp_f32'][:2]
            end=next(i for i in range(qi[-1]+1,len(rows)) if rows[i]['op'].startswith('sl_cbr_')) if label=='full' else wi
            psums=[i for i in range(qi[-1]+1,end) if rows[i]['op']=='ml_add_f32']
            casts=[i for i in range(qi[-1]+1,end) if rows[i]['op']=='ml_cvt_bf16_f32']
            assert (len(exp),len(psums),len(casts))==(2,8,2)
            assert all(dominates(i,wi) for i in vi+qi+exp+psums+casts)
            packing=[i for i in range(casts[-1]+1,len(rows)) if rows[i]['op']=='ml_or_b32'][:2]
            shifts=[i for i in range(casts[-1]+1,packing[-1]) if rows[i]['op']=='ml_shl_b32' and rows[i]['args'].endswith(', 0x10')]
            assert len(packing)==len(shifts)==2
            for i in packing+shifts:must_reach(rows,targets,i,wi)
            assert all(dominates(wi,i) and dominates(vb,i) for i in pi)
            clears=[i for i in range(wi+1,vb) if rows[i]['op']=='ml_slb_store_b32x1']
            assert len(clears)==(0 if label=='full' else 2) and all(dominates(wi,i) for i in clears)
            pvloads=[i for i in range(vb+1,pi[-1]) if rows[i]['op'].startswith('ml_slb_') and 'load' in rows[i]['op']]
            assert len(pvloads)==16 and all(dominates(wi,i) for i in pvloads)
            rho=[i for i in range(max(0,wi-4),vb) if rows[i]['op']=='ml_exp_f32']
            lfma=[i for i in range(wi+1,vb) if rows[i]['op']=='ml_fma_f32']
            assert len(rho)==2 and len(lfma)==2
            paths.append(dict(path=label,K_global_loads=[rows[i]['line'] for i in kl],K_shared_stores=[rows[i]['line'] for i in ks],K_ready_barrier=rows[kb]['line'],
                V_SME=[rows[i]['line'] for i in vi],QK=[rows[i]['line'] for i in qi],P_exp=[rows[i]['line'] for i in exp],P_sums=[rows[i]['line'] for i in psums],
                P_BF16_casts=[rows[i]['line'] for i in casts],P_pack_shifts=[rows[i]['line'] for i in shifts],P_pack_ORs=[rows[i]['line'] for i in packing],
                V_wait=rows[wi]['line'],V_wait_pc=hex(rows[wi]['pc']),tail_clear_stores=[rows[i]['line'] for i in clears],V_ready_barrier=rows[vb]['line'],
                PV_reads=[rows[i]['line'] for i in pvloads],PV_MMA=[rows[i]['line'] for i in pi],rho_exp_before_wait=[rows[i]['line'] for i in rho if i<wi],
                rho_exp_after_wait=[rows[i]['line'] for i in rho if i>wi],running_L_FMA_after_wait=[rows[i]['line'] for i in lfma],
                CFG_dominance_and_conditional_packing_paths_checked=True,physical_wait_before_QK_text=wi<qi[0],all_softmax_before_wait=False))
        k=by_symbol[name][2];oldk=next(k for group in oldmeta for k in group['kernels'] if f'ILj{seg}EE' in k['.name'])
        assert {key:k[key] for key in resourcekeys}=={key:oldk[key] for key in resourcekeys}==dict(zip(resourcekeys,(100,66,8192,0,0,0,64)))
        resources['S'+str(seg)]=dict(vgpr=k['.vgpr_count'],sgpr=k['.sgpr_count'],shared_bytes=k['.group_segment_fixed_size'],private_segment_bytes=k['.private_segment_fixed_size'],vgpr_spills=k['.vgpr_spill_count'],sgpr_spills=k['.sgpr_spill_count'])
        machine.append(dict(symbol=name,segments=seg,instructions=len(rows),branch_targets_checked=len(targets),resources=k,page_12544B_multiply_lines=mult,paths=paths))
    assert len(machine)==2
    oldfunctions=list(llvm_functions(oldtexts['IR']));ir=[]
    for name,fn,offset in llvm_functions(texts['IR']):
        if '_producer' not in name:continue
        seg=int(re.search('ILj([24])EE',name)[1]);oldfn=next(fn for name,fn,_ in oldfunctions if f'ILj{seg}EE' in name)
        math=alpha(numeric_lines(fn));assert len(math)==291 and math==alpha(numeric_lines(oldfn))
        profile=llvm_profile(name,fn,offset);dom=ir_cfg(fn,offset);lines=list(enumerate(fn.splitlines(),offset))
        find=lambda marker:[n for n,line in lines if marker in line]
        sme=find('@llvm.bi.sme.load.16x1b64.rowxfb16(');wait=find('@llvm.bi.sl.waitcnt(i64 8)');qk=find('@llvm.bi.matrix.mad.i32x4.i8x8(');pv=find('@llvm.bi.matrix.mad.f32x4.bf16x4(');exp=find('@llvm.nvvm.ex2.approx.f32(');barrier=find('@llvm.bi.sl.barrier(')
        paths=[]
        for n,label in enumerate(('full','tail')):
            vs=sme[4*n:4*n+4];qs=qk[4*n:4*n+4];ps=pv[8*n:8*n+8];es=exp[4*n:4*n+4];w=wait[n]
            kb=max(b for b in barrier if b<vs[0]);vb=min(b for b in barrier if b>w);kp=profile['paths'][n]
            assert max(kp['K_global_I32_load_lines']+kp['K_shared_I32_store_lines'])<kb
            assert all(dom(x,kb) for x in kp['K_shared_I32_store_lines']) and all(dom(kb,x) for x in vs)
            assert all(dom(x,w) for x in vs+qs+es) and all(dom(w,x) and dom(vb,x) for x in ps)
            clears=[i for i,line in lines if w<i<vb and re.match(r'\s*store i32 ',line)]
            assert len(clears)==(0 if label=='full' else 2) and all(dom(w,x) for x in clears)
            paths.append(dict(path=label,K_dataflow=kp,K_ready_barrier=kb,V_SME=vs,QK=qs,P_and_rho_exp=es,V_wait=w,tail_clear_stores=clears,V_ready_barrier=vb,PV=ps,IR_CFG_checked=True))
        checks={}
        for key,value in slices(fn).items():
            oldvalue=slices(oldfn)[key];normal=normalize_slice(value);assert normal==normalize_slice(oldvalue),(key,seg)
            nodes=re.findall(r'!llvm.loop !(\d+)',value);oldnodes=re.findall(r'!llvm.loop !(\d+)',oldvalue);assert len(nodes)==len(oldnodes)
            for x,y in zip(nodes,oldnodes):assert loop_metadata(texts['IR'].split('Fatbin LLVM IR code:')[1],x)==loop_metadata(oldtexts['IR'].split('Fatbin LLVM IR code:')[1],y)
            checks[key]=dict(alpha_sha256=sha(normal.encode()),lines=len(normal.splitlines()),loop_metadata_preserved=True)
        defs=dict(re.findall(r'^\s*(%\d+) = (.*)$',fn,re.M));muls=[(v,mt[1]) for v,x in defs.items() if (mt:=re.fullmatch(r'mul nsw i64 (%\d+), 12544',x))];assert len(muls)==2
        def one(pattern):
            found=[v for v,x in defs.items() if re.fullmatch(pattern,x)];assert len(found)==1,(pattern,found);return found[0]
        def line_of(v):return offset+next(i for i,x in enumerate(fn.splitlines()) if re.match(r'\s*'+re.escape(v)+r' =',x))
        headbase=one(r'getelementptr i8, ptr addrspace\(1\) %1, i64 %\d+')
        addresses=[]
        for i,(product,page64) in enumerate(muls):
            assert re.fullmatch(r'sext i32 %\d+ to i64',defs[page64])
            kbase=one(r'getelementptr(?: inbounds)? i8, ptr addrspace\(1\) '+re.escape(headbase if i==0 else '%1')+r', i64 '+re.escape(product))
            vbase=one(r'getelementptr inbounds i8, ptr addrspace\(1\) %3, i64 '+re.escape(product));cast=one(r'addrspacecast ptr addrspace\(1\) '+re.escape(vbase)+r' to ptr')
            vhead=one(r'getelementptr inbounds nuw %struct\.__ix_bfloat16, ptr '+re.escape(cast)+r', i64 %\d+');one(r'ptrtoint ptr '+re.escape(vhead)+r' to i64')
            scale=one(r'getelementptr i8, ptr addrspace\(1\) %2, i64 '+re.escape(product));scaleoffset=one(r'getelementptr i8, ptr addrspace\(1\) '+re.escape(scale)+r', i64 4096')
            assert sum('i64 4352' in x for x in defs.values())==0
            addresses.append(dict(path=('full','tail')[i],page_sext=line_of(page64),page_multiply=line_of(product),K_page_GEP=line_of(kbase),V_arg3_page_GEP=line_of(vbase),V_head_GEP=line_of(vhead),scale_arg2_page_GEP=line_of(scale),scale_4096B_once=line_of(scaleoffset),page_stride_bytes=12544,V_stride_BF16=6272,scale_stride_F32=3136))
        descriptors=[n for n,line in lines if 'shufflevector' in line and 'i32 -1, i32 512>' in line];assert len(descriptors)==2
        ir.append(dict(symbol=name,segments=seg,arithmetic_lines=291,arithmetic_alpha_sha256=sha('\n'.join(math).encode()),arithmetic_including_constants_flags_and_operand_sharing_equal_to_actual40f9=True,static_counts=profile['static_counts_full_plus_tail'],paths=paths,tail_and_output_slices=checks,addresses=addresses,SME_512B_descriptor_lines=descriptors))
    assert len(ir)==2
    cpp=(a.raw/'out/biv150_native_k8_compact_v_overlap_qk.cpp').read_text()
    assert cpp==(REPO/'tools/kernels/biv150_native_k8_compact_v_overlap_qk.cpp').read_text()
    for token in ('reinterpret_cast<std::uintptr_t>(v.data_ptr())==kb+4352','reinterpret_cast<std::uintptr_t>(scale_full.data_ptr())==kb','pages*3136','{6272,256,128,1}','{12544,256,128,1}'):assert token in cpp
    assert b'V must already point to +4352B once' in lib
    codecfn=next(fn for name,fn,_ in llvm_functions(texts['IR']) if '_codec_probe' in name)
    assert len(re.findall(r'= fdiv float ',codecfn))==3 and codecfn.count('@llvm.nvvm.f2i.rn(')==2
    assert all('= fdiv float ' in line for line in codecfn.splitlines() if '= fdiv ' in line)
    assert all(token in codecfn for token in ('0x3810000000000000','1.000000e+00','i32 -127','i32 127'))
    report=dict(CPU_only=True,source_commit=SOURCE,SO_sha256=SO,control_SO_sha256=CONTROL_SO,raw_files_rehashed=m['raw_files'],raw_bytes_rehashed=m['raw_bytes'],manifest_sha256=sha((a.raw/'download-manifest.json').read_bytes()),control_manifest_sha256=sha((a.control_raw/'download-manifest.json').read_bytes()),compile_binding=binding,
        all_decoded_instruction_words_bound_to_actual_SO=True,instructions_bound=sum(len(x) for x in functions.values()),actual_embedded_ELFs_equal_to_built_objects=True,machine=machine,IR=ir,
        codec_text_equal_to_actual40f9=True,codec_text_sha256=CODEC_TEXT,codec_actual_IR_unflagged_F32_divides=3,codec_actual_IR_RNE_conversions=2,
        static_resources_equal_to_actual40f9=True,host_compiled_source_and_binary_guard_bound=True,host_machine_disassembled=False,
        conclusion='Actual 7892 S2/S4 full/tail preserve K-before-V issue, QK/P softmax/packing precede V wait, wait dominates V tail clear and PV. Compact 12544B layout and codec preserved. No resource regression relative to actual40f9.',
        limitations=['Full-loop blocks are physically reordered; CFG rather than line ordering establishes execution order.',
            'Machine full-page rho has one exp before wait and one after; tail has both rho exp after wait. Both running-L FMAs remain after wait in each path. Do not assert all softmax precedes wait.',
            'Actual-SO IR is the hash-bound ixobjdump export, not independently decoded LLVM bitcode. Arithmetic/slice/source comparisons and CFG checks are not a whole-program equivalence proof.',
            'Host offset/stride guards are checked in exact compiled C++ plus binary guard text, not x86 disassembly.',
            'Binary metadata does not measure runtime occupancy, hide latency, or establish throughput.',
            'No GPU numerical, Graph, side-stream, fullstep, model, service, capacity, N40/N48, or performance qualification. Root must independently review before a device consumer.'],
        reviewer_GPU_SSH_JIT_compile_or_Torch_executed=False,GPU_numeric_verified=False,Graph_verified=False,side_stream_verified=False,fullstep_verified=False,model_verified=False,service_verified=False,capacity_reclaimed=False,precision_gate=False,performance_verified=False,manual_Root_approval=False,
        script_sha256=sha(Path(__file__).read_bytes()),ELF_parser_sha256=sha(Path(elfutil.__file__).read_bytes()))
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    review=contract.review_template(a.raw/'out/compile-summary.json')
    verified_flags=('actual_machine_code_bound_to_SO','actual_compact_addresses_and_SME_descriptor_verified',
        'actual_wait_tail_QK_PV_math_verified','actual_codec_preserved',
        'actual_K_then_V_issue_order_preserved','actual_V_wait_after_QK_and_P_softmax_packing')
    assert set(contract.REVIEW_FLAGS)==set(verified_flags),'Unknown review field must not be granted automatically'
    for flag in verified_flags:review[flag]=True
    review['static_resources']=resources
    review['notes']='Independent CPU review of actual7892 final SO: all S2/S4 full+tail K/V, QK/P packing, wait/tail/PV CFG and compact layout/codec checks passed. rho exp/running-L FMA may remain after wait; no device numeric or model claim. Root must independently review before GPU.'
    review['independent_audit_report_sha256']=sha(a.report.read_bytes());review['manual_Root_approval']=False
    a.review.write_text(json.dumps(review,sort_keys=True,indent=2)+'\n')
    contract.admission(a.raw/'out/compile-summary.json',a.review,sha(a.review.read_bytes()),a.raw/'out/build/cjj_native_k8_compact_v_overlap_probe.so')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(dict(report=str(a.report),report_sha256=sha(a.report.read_bytes()),review=str(a.review),review_sha256=sha(a.review.read_bytes()),instructions_bound=report['instructions_bound'],manual_Root_approval=False,GPU_numeric_verified=False),sort_keys=True))

if __name__=='__main__':main()
