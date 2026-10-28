#!/usr/bin/env python3
"""N48-only CPU audit of actual2ad versus actual7892; never load either SO."""
import argparse, ast, hashlib, json, re, struct, subprocess, sys
from pathlib import Path

SOURCE='83b6c8c9282445219782ee1377680778a2aa8d3a'
SO='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
OLD_SO='7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4'
HELPER_SHA='02fc4aff949091969a058bd1ffed64465b9a5baa73b1e7d57d6cf9178bcaa271'
ELF_SHA='09ac1c98f2f1a1ff25ad02fcc819b25fc00bdb442a55c2ec1ad69defc61ea6e7'
OLD_REPORT_SHA='f5a40726b7ae1c1db122e710c1921815db887a1f06112901c842f976a1c9d0dc'
RENAMES=(('cjj_native_k8_compact_v_overlap_producer','cjj_native_k8_compact_n48_producer'),
 ('launch_cjj_native_k8_compact_v_overlap_s2','launch_cjj_native_k8_compact_n48_s2_unused'),
 ('launch_cjj_native_k8_compact_v_overlap_s4','launch_cjj_native_k8_compact_n48_s4'),
 ('cjj_launch_native_k8_compact_v_overlap_codec','cjj_launch_native_k8_compact_n48_codec'),
 ('cjj_native_k8_compact_v_overlap_codec_probe','cjj_native_k8_compact_n48_codec_probe'),
 ('cjj_native_k8_compact_v_overlap_probe','cjj_native_k8_compact_n48_s4_probe'))
def sha(b):return hashlib.sha256(b).hexdigest()
def reuse(path,pin,names):
    raw=path.read_bytes();assert sha(raw)==pin
    tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
def rename(text):
    for old,new in RENAMES:
        text=text.replace(str(len(old))+old,str(len(new))+new).replace(old,new)
    return text
def full_metadata(blob):
    # Same note walk as the existing ELF reader, retaining .args for ABI equality.
    note=elf(blob)['sections']['.note']['bytes'];at=0
    while at<len(note):
        ns,ds,kind=struct.unpack_from('<III',note,at);at+=12
        name=note[at:at+ns].rstrip(b'\0');at=(at+ns+3)//4*4
        desc=note[at:at+ds];at=(at+ds+3)//4*4
        if name==b'Iluvatar' and kind==10:return msgpack(desc)
    raise AssertionError('Missing Iluvatar metadata')
def inspect(lib,texts):
    metas=embedded(lib);fat=elf(lib)['sections']['.nv_fatbin']['bytes'];symbols={};modules=[]
    for i,match in enumerate(re.finditer(b'\x7fELF',fat)):
        data=fat[match.start():];data=data[:elf(data)['extent']]
        assert sha(data)==metas[i]['sha256']
        address,blob=elf_text(data);full=full_metadata(data)
        modules.append(full)
        for kernel in full['iluvatar.kernels']:
            assert kernel['.name'] not in symbols
            symbols[kernel['.name']]=(address,blob,kernel)
    functions={};current=None
    for number,line in enumerate(texts['SASS'].splitlines(),1):
        mt=re.search(r'Function : (\S+)',line)
        if mt:current=mt[1];assert current not in functions;functions[current]=[]
        mt=re.match(r'^\s*((?:sl|ml)_\w+)\b(.*?)// ([0-9A-F]+): ([0-9A-F]{8}) ([0-9A-F]{8})',line)
        if mt:
            pc=int(mt[3],16);address,blob,_=symbols[current];offset=pc-address
            word=struct.pack('<II',int(mt[4],16),int(mt[5],16))
            assert blob[offset:offset+8]==word
            functions[current].append(dict(line=number,op=mt[1],args=mt[2].strip(),pc=pc,word=word))
    assert not re.search(r'^\s*\.long ',texts['SASS'],re.M)
    assert set(functions)==set(symbols) and len(functions)==3
    for rows in functions.values():
        assert [r['pc'] for r in rows]==list(range(rows[0]['pc'],rows[-1]['pc']+8,8))
        machine_cfg(rows) # Independently ensure every direct branch target is in this actual function.
    ir={name:(fn,line) for name,fn,line in llvm_functions(texts['IR'])}
    return metas,symbols,functions,ir,modules
def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference-raw',type=Path,required=True)
    p.add_argument('--helper',type=Path,required=True);p.add_argument('--elf-helper',type=Path,required=True)
    p.add_argument('--reference-report',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--review-out',type=Path,required=True)
    a=p.parse_args();assert not a.report.exists() and not a.review_out.exists()
    reuse(a.elf_helper,ELF_SHA,('sha','msgpack','elf','embedded'))
    reuse(a.helper,HELPER_SHA,('sha','elf_text','machine_cfg','llvm_functions','raw_manifest','actual_exports'))
    assert sha(a.reference_report.read_bytes())==OLD_REPORT_SHA
    prior=json.loads(a.reference_report.read_text());assert prior['SO_sha256']==OLD_SO
    m=raw_manifest(a.raw);oldm=raw_manifest(a.reference_raw)
    for manifest in (m,oldm):assert len(manifest['files'])==len({r['relative'] for r in manifest['files']})
    assert (m['raw_files'],m['raw_bytes'])==(33,1412002)
    assert m['original_owner']==dict(pid=61412,start_ticks=355787063,same_start=True,state='Z')
    owner=json.loads((a.raw/'exports/cjj-compact-native-n48-compile-v1-20261029-owner.json').read_text())
    assert owner['source_commit']==SOURCE and owner['pid']==61412 and owner['start_ticks']==355787063
    lib,texts,summary,inspection=actual_exports(a.raw,SO,'cjj_native_k8_compact_n48_s4_probe.so')
    oldlib,oldtexts,oldsummary,_=actual_exports(a.reference_raw,OLD_SO,'cjj_native_k8_compact_v_overlap_probe.so')
    assert summary['completed'] and summary['compile_passed'] and not summary['errors']
    assert all(summary[x]==0 for x in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'))
    assert summary['torch']['cuda_initialized_before'] is False and summary['torch']['cuda_initialized_after'] is False
    assert summary['explicit_cuda_flags']==oldsummary['explicit_cuda_flags'] and summary['compiler']==oldsummary['compiler']
    assert summary['torch']==oldsummary['torch']
    assert len(summary['commands'])==4 and all(c['exit_code']==0 for c in summary['commands'])
    for entry in summary['artifact_manifest']:
        b=(a.raw/'out'/entry['relative']).read_bytes();assert (len(b),sha(b))==(entry['bytes'],entry['sha256'])
    git=lambda *args:subprocess.check_output(['git','-C',str(a.source_root),*args])
    for rel,pin in summary['private_source'].items():
        raw=(a.raw/'out'/Path(rel).name).read_bytes();assert sha(raw)==pin
        assert raw==git('show',SOURCE+':'+rel)
    # Actual new ELF and its separately archived GPU objects must contain identical complete images.
    meta,symbols,functions,ir,modules=inspect(lib,texts)
    ometa,osymbols,ofunctions,oir,omodules=inspect(oldlib,oldtexts)
    objects=[]
    for role in ('qk','codec'):objects+=embedded((a.raw/'out/build'/('biv150_native_k8_compact_n48_'+role+'.cuda.o')).read_bytes())
    assert sorted(x['sha256'] for x in meta)==sorted(x['sha256'] for x in objects)
    assert sorted(x['text_sha256'] for x in meta)==sorted(x['text_sha256'] for x in ometa)
    assert rename(json.dumps(omodules,sort_keys=True))==json.dumps(modules,sort_keys=True)
    # ixobjdump wraps/truncates long printed signatures at a fixed buffer width.
    # Keep that raw display; never repair it or claim reassemblable LLVM. The
    # actual full .args metadata binds the ABI; body, attributes and metadata
    # definitions are independently compared without removing semantic flags.
    for pattern in (r'^attributes #\d+ = .*$',r'^!\d+ = .*$'):
        assert rename('\n'.join(re.findall(pattern,oldtexts['IR'],re.M)))=='\n'.join(re.findall(pattern,texts['IR'],re.M))
    comparisons=[]
    for oldname,oldrows in ofunctions.items():
        name=rename(oldname);rows=functions[name]
        assert b''.join(r['word'] for r in oldrows)==b''.join(r['word'] for r in rows)
        assert [(r['op'],r['args']) for r in oldrows]==[(r['op'],r['args']) for r in rows]
        assert rename(json.dumps(osymbols[oldname][2],sort_keys=True))==json.dumps(symbols[name][2],sort_keys=True)
        oldfn=oir[oldname][0];newfn=ir[name][0]
        oldbody=oldfn.split('{\n',1)[1];newbody=newfn.split('{\n',1)[1]
        assert rename(oldbody)==newbody, 'Actual IR body differs beyond the declared kernel renames'
        comparisons.append(dict(symbol=name,reference_symbol=oldname,instructions=len(rows),
          machine_words_sha256=sha(b''.join(r['word'] for r in rows)),machine_equal_to_actual7892=True,
          IR_body_rename_only_equal_to_actual7892=True,IR_body_sha256=sha(newbody.encode()),IR_function_line=ir[name][1],
          raw_IR_full_function_display_rename_only_equal=rename(oldfn)==newfn,
          SASS_first_line=rows[0]['line'],SASS_last_line=rows[-1]['line'],metadata_including_args_equal=True,
          host_admitted=bool('_producerILj4EE' in name or '_codec_probe' in name)))
    cpp=(a.raw/'out/biv150_native_k8_compact_n48_qk.cpp').read_text()
    for token in ('TORCH_CHECK(segments==4,"Private N48/S4 only");','const std::int64_t rows=48;',
      'reinterpret_cast<std::uintptr_t>(v.data_ptr())==kb+4352','reinterpret_cast<std::uintptr_t>(scale_full.data_ptr())==kb',
      'pages*3136','{6272,256,128,1}','{12544,256,128,1}'):
        assert token in cpp
    for token in (b'Private N48/S4 only',b'V must already point to +4352B once',b'cjj_native_k8_compact_n48_s4_probe'):assert token in lib
    ninja=(a.raw/'out/build/build.ninja').read_text()
    assert 'build cjj_native_k8_compact_n48_s4_probe.so: link biv150_native_k8_compact_n48_qk.o biv150_native_k8_compact_n48_qk.cuda.o biv150_native_k8_compact_n48_codec.cuda.o' in ninja
    assert '-DCOREX_NATIVE_GQA_V_OVERLAP=1' in ninja and '--cuda-gpu-arch=ivcore11' in ninja
    s4=next(name for name in functions if '_producerILj4EE' in name)
    resource=symbols[s4][2]
    expected={'.vgpr_count':100,'.sgpr_count':66,'.group_segment_fixed_size':8192,'.private_segment_fixed_size':0,
      '.vgpr_spill_count':0,'.sgpr_spill_count':0,'.warp_size':64,'.kernarg_segment_size':144,'.kernarg_segment_align':8}
    assert all(resource[k]==v for k,v in expected.items())
    template=json.loads(a.template.read_text())
    flags=('actual_machine_code_bound_to_SO','actual_compact_addresses_and_SME_descriptor_verified',
      'actual_wait_tail_QK_PV_math_verified','actual_codec_preserved','actual_K_then_V_issue_order_preserved',
      'actual_V_wait_after_QK_and_P_softmax_packing')
    assert all(template[k] is False for k in flags)
    assert set(template['static_resources'])=={'S4'}
    assert SO in json.dumps(template) and 'cjj_native_k8_compact_n48_s4_probe' in json.dumps(template)
    assert not {'torch','triton','numpy'}&set(sys.modules)
    report=dict(schema=1,CPU_only=True,source_commit=SOURCE,SO_sha256=SO,reference_SO_sha256=OLD_SO,
      source_namespace='cjj_native_k8_compact_n48_s4_probe',reviewed_admission_shapes=[[48,4,2048],[48,4,2560]],
      source_and_actual_compile_review_passed=True,raw_files_rehashed=m['raw_files'],raw_bytes_rehashed=m['raw_bytes'],
      manifest_sha256=sha((a.raw/'download-manifest.json').read_bytes()),reference_manifest_sha256=sha((a.reference_raw/'download-manifest.json').read_bytes()),
      original_owner=m['original_owner'],source_pins=summary['private_source'],compile_summary_sha256=sha((a.raw/'out/compile-summary.json').read_bytes()),
      inspection_sha256=sha((a.raw/'out/actual-so-inspection.json').read_bytes()),
      compiler=summary['compiler'],CUDA_initialized_before=False,CUDA_initialized_after=False,frontend_launch_and_initialization_counts_zero=True,
      actual_embedded_ELFs_equal_to_built_GPU_objects=True,all_decoded_instruction_words_bound_to_actual_SO=True,
      instructions_bound=sum(len(x) for x in functions.values()),actual_GPU_images=meta,
      actual_full_metadata_including_argument_ABI_rename_only_equal=True,IR_attributes_and_numbered_metadata_definitions_rename_only_equal=True,functions=comparisons,
      actual_S4_resources=expected,resources_scope='Actual ELF metadata; not runtime occupancy or postlaunch observation',
      parent_CFG_reused_for_S4=True,parent_report_sha256=OLD_REPORT_SHA,
      inherited_S4_CFG=next(x for x in prior['machine'] if x['segments']==4),
      inherited_S4_IR_proof=next(x for x in prior['IR'] if x['segments']==4),
      inherited_CFG_line_numbers_refer_to_parent7892_raw=True,
      compact_addresses=dict(K_page_bytes=12544,V_page_BF16=6272,V_external_offset_bytes_once=4352,scale_page_F32=3136,scale_internal_offset_F32_once=1024,SME_row_bytes=512),
      zero_length='Same actual GPU code as7892: early return leaves O/M/L poison; reducer separately owns final zero output',
      host_compiled_source_and_binary_guard_text_bound=True,host_machine_disassembled=False,
      unused_S2_code_compared_but_admission_or_qualification_granted=False,
      proof_limits=['The prior CFG proof is transferred only after complete actual instruction words, per-function operations, full metadata including argument ABI and actual IR bodies/attributes/metadata definitions match modulo names.',
        'Prior proof line numbers remain parent7892 locations; new actual IR/SASS ranges are in functions.',
        'V issue remains after K load/store barrier. QK and P softmax/packing precede V wait; some rho exp/running-L FMA remain after wait. No claim that all softmax precedes wait.',
        'Actual-SO IR is an ixobjdump export with hash-bound provenance, not independent bitcode decoding.',
        'The raw printed producer signatures wrap/truncate differently after symbol renames (including a broken noundef token); these are retained as-is. Full printed functions are not equal and are not claimed reassemblable. Actual argument ABI is checked in complete ELF metadata.',
        'Host guard checking is source/binary-text/compile provenance; no x86 disassembly or actual TORCH_CHECK execution.',
        'Zero frontend kernels and CUDA_initialized=false do not prove absence of every CUDA API call.',
        'No N48 numeric, Graph, side stream, fullstep, writer, capacity, 42-layer model, service, precision or throughput qualification. No S2 admission.'],
      GPU_numeric_verified=False,Graph_verified=False,side_stream_verified=False,fullstep_verified=False,
      model_verified=False,service_verified=False,capacity_reclaimed=False,precision_gate=False,performance_verified=False,
      manual_Root_approval=False,reviewer_GPU_SSH_JIT_compile_or_Torch_executed=False,
      helper_sha256=HELPER_SHA,ELF_helper_sha256=ELF_SHA,script_sha256=sha(Path(__file__).read_bytes()))
    a.report.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    template['static_resources']['S4']=dict(vgpr=100,sgpr=66,shared_bytes=8192,private_segment_bytes=0,vgpr_spills=0,sgpr_spills=0)
    template['independent_audit_report_sha256']=sha(a.report.read_bytes())
    template['manual_Root_approval']=False
    template['notes']='Independent CPU audit of actual2ad N48/S4 matched actual7892 machine words, full argument metadata and actual IR bodies/attributes modulo names. Printed long signatures retain the ixobjdump wrap/truncation limitation. All review flags remain false pending Root independent approval; no device qualification.'
    a.review_out.write_text(json.dumps(template,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(report=str(a.report),report_sha256=sha(a.report.read_bytes()),review=str(a.review_out),review_sha256=sha(a.review_out.read_bytes()),instructions_bound=report['instructions_bound'],manual_Root_approval=False)))
if __name__=='__main__':main()
