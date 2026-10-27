#!/usr/bin/env python3
"""CPU review of real ixobjdump SO output; produces recommendations, no license."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0,'/private/tmp/flagos-cjj-native-k8-vfirst-20261028/tools')
import audit_biv150_native_k8_compile_cpu as elfutil
import probe_biv150_native_k8_vfirst as frontend
spec=importlib.util.spec_from_file_location('vfirst_prior_audit','/private/tmp/audit_cjj_native_k8_vfirst_compile_actual_20261028.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

def sha(b):return hashlib.sha256(b).hexdigest()
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

def source_assembly(text):
    result={};current=None
    for line_number,line in enumerate(text.splitlines(),1):
        m=re.match(r'\s*\.iluvatar_kernel (\S+)',line)
        if m:current=m[1];result[current]=dict(ops=[],labels={})
        m=re.match(r'(BB\d+_\d+):',line)
        if m:result[current]['labels'][m[1]]=len(result[current]['ops'])
        m=re.match(r'\s*((?:sl|ml)_\w+)\b(.*)',line)
        if m:result[current]['ops'].append(dict(op=m[1],args=m[2].strip(),line=line_number))
    return result

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

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();a.out.mkdir();files=elfutil.from_raw(a.raw)
    m=json.loads(files['download-manifest.json']);assert m['CPU_only'] and not m['GPU_test_rerun']
    assert set(files)=={'download-manifest.json',*(x['relative'] for x in m['files'])}
    for x in m['files']:assert (len(files[x['relative']]),sha(files[x['relative']]))==(x['bytes'],x['sha256'])
    assert len(m['files'])==9 and sum(x['bytes'] for x in m['files'])==496623
    result=json.loads(files['result.json']);assert result['CPU_only'] and result['GPU_calls'] is False
    assert [x['exit'] for x in result['commands']]==[0,0,255,255]
    compileroot=Path('/private/tmp/cjj-native-k8-vfirst-compile-v1-raw-20261028/out')
    compile_owner=json.loads((compileroot.parent/'exports/cjj-native-k8-vfirst-compile-v1-20261028-owner.json').read_text())
    assert compile_owner['source_commit']=='e46fa381ccc0d10b92251c73ba7cd24abf0b9bfb'
    binding=frontend.compile_record(compileroot/'compile-summary.json')
    so=(compileroot/'build/cjj_native_k8_vfirst_probe.so').read_bytes();assert sha(so)==binding['variant_library_sha256']==result['library_sha256']
    fat=elfutil.elf(so)['sections']['.nv_fatbin']['bytes'];by_symbol={};gpu_meta=elfutil.embedded(so)
    for i,match in enumerate(re.finditer(b'\x7fELF',fat)):
        b=fat[match.start():];b=b[:elfutil.elf(b)['extent']];address,text=elf_text(b)
        for k in gpu_meta[i]['kernels']:by_symbol[k['.name']]=dict(ELF_sha256=sha(b),text_address=address,text=text,metadata=k)
    sass=files['actual-so-sass.stdout.txt'].decode();functions={};current=None
    for line_no,line in enumerate(sass.splitlines(),1):
        m=re.search(r'Function : (\S+)',line)
        if m:current=m[1];functions[current]=[];assert current in by_symbol
        m=re.match(r'^\s*((?:sl|ml)_\w+)\b(.*?)// ([0-9A-F]+): ([0-9A-F]{8}) ([0-9A-F]{8})',line)
        if m:
            row=dict(line=line_no,op=m[1],args=m[2].strip(),pc=int(m[3],16),words=[m[4],m[5]])
            raw=struct.pack('<II',int(m[4],16),int(m[5],16));elf=by_symbol[current];at=row['pc']-elf['text_address']
            assert elf['text'][at:at+8]==raw;functions[current].append(row)
    assert sum(len(x) for x in functions.values())==2768 and not re.search(r'^\s*\.long ',sass,re.M)
    standalone=source_assembly((compileroot/'biv150_native_k8_vfirst_qk.s').read_text());profiles=[]
    for symbol,source in standalone.items():
        rows=functions[symbol];assert len(rows)==len(source['ops'])==1341
        assert [r['op'] for r in rows]==[r['op'] for r in source['ops']]
        assert [x['pc'] for x in rows]==list(range(rows[0]['pc'],rows[-1]['pc']+8,8))
        branches=[];float_match=0;display_differences=[]
        for source_op,actual in zip(source['ops'],rows):
            target=re.fullmatch(r'BB\d+_\d+',source_op['args'])
            if target:
                expected=rows[source['labels'][target[0]]]['pc'];decoded=(actual['pc']+8+int(actual['args'],0))&0xffffffff
                assert expected==decoded;branches.append(dict(line=actual['line'],pc=hex(actual['pc']),target=hex(decoded)))
            else:
                clean=lambda x:re.sub(r'\s+','',x)
                if clean(source_op['args'])!=clean(actual['args']):
                    display_differences.append(dict(line=actual['line'],opcode=actual['op'],assembly=source_op['args'],disassembly=actual['args']))
                if re.search(r'_f32|matrix_mad|cvt_',actual['op']):
                    assert clean(source_op['args'])==clean(actual['args']);float_match+=1
        assert len(branches)==14
        at,dom,targets=machine_cfg(rows)
        sme=[i for i,x in enumerate(rows) if x['op']=='sl_sme_load_16x1b64_rowxfb16'];assert len(sme)==8
        paths=[]
        for index,name in enumerate(('full','tail')):
            quad=[rows[i] for i in sme[index*4:index*4+4]];last=sme[index*4+3]
            wait=next(i for i in range(last+1,len(rows)) if rows[i]['op']=='sl_wait' and 'g2scnt(0)' in rows[i]['args'])
            loads=[x for x in rows[last+1:wait] if x['op']=='ml_lsa_load_dword']
            stores=[x for x in rows[last+1:wait] if x['op']=='ml_slb_store_b32x1'];assert len(loads)==len(stores)==8
            vblock=at[quad[0]['pc']];assert all(at[x['pc']]==vblock for x in quad)
            assert all(vblock in dom[at[x['pc']]] and (at[x['pc']]!=vblock or x['pc']>quad[-1]['pc']) for x in loads+stores)
            barrier=next(i for i in range(wait+1,len(rows)) if rows[i]['op']=='sl_barrier')
            explicit=next(i for i in range(barrier+1,len(rows)) if rows[i]['op']=='sl_wait' and rows[i]['args']=='g2scnt(0)')
            post=next(i for i in range(explicit+1,len(rows)) if rows[i]['op']=='sl_barrier')
            paths.append(dict(path=name,V_SME=[dict(line=x['line'],pc=hex(x['pc'])) for x in quad],
                K_global_load_lines=[x['line'] for x in loads],K_shared_store_lines=[x['line'] for x in stores],
                V_dominates_K_in_actual_machine_CFG=True,K_flush_wait_line=rows[wait]['line'],K_barrier_line=rows[barrier]['line'],
                explicit_original_V_wait_line=rows[explicit]['line'],post_V_barrier_line=rows[post]['line']))
        profiles.append(dict(symbol=symbol,code_body_sha256=sha(b''.join(struct.pack('<II',*(int(x,16) for x in r['words'])) for r in rows)),
            instruction_count=len(rows),all_instructions_byte_bound_to_original_SO=True,all_opcode_order_matches_standalone=True,
            actual_relative_branch_targets_match_all_standalone_labels=True,branches=branches,
            float_MMA_conversion_operands_exactly_match_standalone=float_match,
            non_float_display_differences=display_differences,actual_ELF_metadata=by_symbol[symbol]['metadata'],paths=paths))
    actual_ir=files['actual-so-IR.stdout.txt'].decode();ir_profiles=[]
    for name,fn,line in llvm_functions(actual_ir):
        if 'vfirst_gqa' in name:
            profile=prior.llvm_profile(name,fn,line);profile['actual_SO_IR_Vfirst']=profile.pop('standalone_Vfirst');ir_profiles.append(profile)
    assert len(ir_profiles)==2 and all(x['actual_SO_IR_Vfirst'] for x in ir_profiles)
    # Verify actual embedded IR floating/numerical lines against standalone
    # SSA. Ignore integer optimization annotations only; preserve FP flags.
    standalone_ir_text=(compileroot/'biv150_native_k8_vfirst_qk.ll').read_text()
    standalone_ir=dict(elfutil.llvm_functions(standalone_ir_text))
    qk_actual_module=actual_ir.split('Fatbin LLVM IR code:')[1]
    def canonical_loop(module,node):
        metadata=dict(re.findall(r'^!(\d+) = (.*)$',module,re.M))
        def visit(current,stack):
            if current in stack:return 'SELF'
            return re.sub(r'!(\d+)',lambda m:visit(m[1],stack|{current}),metadata[current])
        return visit(node,set())
    def normalized_integer_annotations(text):
        text=re.sub(r'\b(nuw|nsw|disjoint|samesign)\b\s*','',text)
        text=re.sub(r'!llvm.loop !\d+','!llvm.loop !EQUIVALENT_LOOP_NODE',text)
        return '\n'.join(re.sub(r'\s+',' ',x.split(';')[0].strip()) for x in text.splitlines() if x.split(';')[0].strip())
    ir_math=[]
    for name,fn,line in llvm_functions(actual_ir):
        if name not in standalone_ir:continue
        def math_lines(t):
            return [re.sub(r'\s+',' ',x.strip()) for x in t.splitlines() if re.search(r'= (?:fadd|fsub|fmul|fdiv|fptrunc|fpext|sitofp)\b|@llvm\.(?:bi\.matrix|fma|nvvm\.ex2|nvvm\.f2i)',x)]
        aa=math_lines(fn);bb=math_lines(standalone_ir[name]);assert aa==bb
        actual_slices=prior.mathematical_slices(fn);source_slices=prior.mathematical_slices(standalone_ir[name]);slices={}
        for k in actual_slices:
            for actual_node,source_node in zip(re.findall(r'!llvm.loop !(\d+)',actual_slices[k]),re.findall(r'!llvm.loop !(\d+)',source_slices[k])):
                assert canonical_loop(qk_actual_module,actual_node)==canonical_loop(standalone_ir_text,source_node)
            x=normalized_integer_annotations(actual_slices[k]);y=normalized_integer_annotations(source_slices[k]);assert x==y
            slices[k]=dict(integer_optimization_annotation_and_proven_loop_metadata_normalized_SSA_exact=True,sha256=sha(x.encode()))
        ir_math.append(dict(symbol=name,math_lines=len(aa),exact_F32_BF16_I32_math_SSA_and_FP_flags=True,sha256=sha('\n'.join(aa).encode()),slices=slices))
    old_binary=Path('/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so').read_bytes()
    assert sha(old_binary)==elfutil.LIBRARY_SHA
    old=elfutil.embedded(old_binary)
    codec=next(x for x in gpu_meta if len(x['kernels'])==1);assert codec['text_sha256']==next(x for x in old if len(x['kernels'])==1)['text_sha256']
    recommendation=dict(actual_V_issue_before_K_packing=True,actual_wait_barrier_tail_preserved=True,original_I8_QK_BF16_PV_and_codec_preserved=True)
    failed_root=Path('/private/tmp/cjj-native-k8-vfirst-elf-review-v1-raw-20261028')
    failed_manifest_bytes=(failed_root/'download-manifest.json').read_bytes();failed_manifest=json.loads(failed_manifest_bytes)
    for meta in failed_manifest['files']:
        b=(failed_root/meta['relative']).read_bytes();assert (len(b),sha(b))==(meta['bytes'],meta['sha256'])
    failed_summary=json.loads((failed_root/'evidence-summary.json').read_text())
    assert next(x['exit'] for x in failed_summary['commands'] if x['label']=='exact-ninja-save-temps')==1
    report=dict(CPU_only=True,script_sha256=sha(Path(__file__).read_bytes()),raw_files_verified=9,raw_bytes_verified=496623,
        manifest_sha256=sha(files['download-manifest.json']),source_commit=compile_owner['source_commit'],binding=binding,SO_sha256=sha(so),
        actual_SO_SASS_sha256=sha(files['actual-so-sass.stdout.txt']),actual_SO_IR_sha256=sha(files['actual-so-IR.stdout.txt']),
        decoded_instruction_bytes_independently_bound=2768*8,actual_SO_profiles=profiles,actual_SO_IR_profiles=ir_profiles,
        actual_embedded_IR_math_vs_standalone=ir_math,actual_codec_machine_text_identical_to_old22e=True,
        actual_machine_schedule_supported=True,recommended_review_flags=recommendation,manual_Root_approval=False,
        approved_sidecar_generated=False,numeric_licensed=False,original_failed_supplements_preserved=True,
        prior_failed_supplement_files_rehashed=len(failed_manifest['files']),prior_failed_supplement_manifest_sha256=sha(failed_manifest_bytes),
        IR_display_limit='ixobjdump wraps long signatures, including a cut attribute token; raw display retained, body SSA audited rather than recompiled',
        reviewer_Torch_JIT_GPU_SSH_or_compile_executed=False,model_verified=False,precision_gate=False,service_verified=False,performance_verified=False,capacity_reclaimed=False)
    (a.out/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    template=frontend.review_template(compileroot/'compile-summary.json');template.update(proposed_values_only=recommendation,
        actual_SO_SASS_sha256=report['actual_SO_SASS_sha256'],actual_SO_IR_sha256=report['actual_SO_IR_sha256'],independent_report_sha256=sha((a.out/'review.json').read_bytes()))
    (a.out/'UNAPPROVED-sidecar-template.json').write_text(json.dumps(template,sort_keys=True,indent=2)+'\n')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(dict(report=str(a.out/'review.json'),sha256=sha((a.out/'review.json').read_bytes()),recommended_flags=recommendation,manual_approval=False),sort_keys=True))

if __name__=='__main__':main()
