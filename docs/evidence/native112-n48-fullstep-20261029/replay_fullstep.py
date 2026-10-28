#!/usr/bin/env python3
"""Plan by default; exact old numeric audit adapted to the N48 compact pair.
Actual summary/manifest hashes must be supplied after Root collection. No GPU.
"""
import argparse,ast,copy,hashlib,json,math,re,struct,subprocess,sys,types
from pathlib import Path
FROZEN='e1c5a8fc3c267f1faf9aa790f3c8f80cc65c12a8'
PREPARED='f5695ab58394b4a46da76beaa539afb625e7b46f'
CONSUMER='493e210f8eeaac6006b25a94ed55e08fbe932337d6481fe3d8ec6e86d0033272'
LIBRARY='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
AUDIT_SHA='1c527358d545110a462866c2cc675334e5ea9baf9ccf4a047bec1bcd1f6c060f'
ELF_SHA='09ac1c98f2f1a1ff25ad02fcc819b25fc00bdb442a55c2ec1ad69defc61ea6e7'
sha=lambda b:hashlib.sha256(b).hexdigest()

def adapt(original):
    edits=[]
    def edit(old,new):edits.append((old,new))
    edit("COMMIT = '3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'",'COMMIT = '+repr(FROZEN))
    edit("CONSUMER = 'c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128'",'CONSUMER = '+repr(CONSUMER))
    edit("need(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==COMMIT, 'source HEAD')","need(CODE_IDENTITIES_VERIFIED, 'f569/e1c5 tools and production byte identity')")
    edit('import biv150_native_k8_fullstep_fixture as f','import biv150_native_k8_compact_n48_fullstep_fixture as f')
    edit("dict(pid=706887,start_ticks=347999537,same_start=True,state='Z')","dict(pid=62378,start_ticks=356248262,same_start=True,state='Z')")
    edit("len(paths)==manifest['raw_files']==192 and total==manifest['raw_bytes']==144847098","len(paths)==manifest['raw_files']==EXPECTED_FILES and total==manifest['raw_bytes']==EXPECTED_BYTES")
    edit("raw/'exports/cjj-native-k8-fullstep-v1-20261028-owner.json'","raw/'exports/cjj-native112-n48-fullstep-v1-20261029-owner.json'")
    edit("owner['pid']==706887 and owner['start_ticks']==347999537","owner['pid']==62378 and owner['start_ticks']==356248262")
    edit('json.loads(json.dumps(f.plan()))','json.loads(json.dumps(N48_PLAN))')
    edit("need(summary['actual_consumer']['actual_consumer_summary_sha256']==CONSUMER and summary['actual_consumer']['all_full_D128_snapshots_replayed'], 'consumer identity')\n    need(sha((source/'docs/evidence/native-k8-qk-feasibility-20261028/consumer-v2/original-summary.json').read_bytes())==CONSUMER, 'frozen actual consumer summary bytes')",
         "need(summary['actual_consumer']==ACTUAL_CONSUMER, 'actual493e/c079/2ad prerequisite files')")
    edit('raw_files_verified=192','raw_files_verified=EXPECTED_FILES')
    edit("K8_SO=summary['K8_library_sha256'],BF16_SO=summary['BF16_library_sha256'],","native_SO=summary['actual_native_library']['sha256'],TL_reader_sha256=N48_PLAN['production_reader']['sha256'],")
    edit("views['k8']['key']['storage_offset']==256 and views['k8']['scale']['storage_offset']==64 and views['k8']['value']['storage_offset']==4224", "all(views[arm]['key']['storage_offset']==256 and views[arm]['scale']['storage_offset']==64 and views[arm]['value']['storage_offset']==2304 for arm in ('bf16','k8'))")
    edit("views['k8']['key']['stride']==[16384,256,128,1] and views['k8']['value']['stride']==[8192,256,128,1]", "all(views[arm]['key']['stride']==[12544,256,128,1] and views[arm]['value']['stride']==[6272,256,128,1] for arm in ('bf16','k8'))")
    edit('dict(initial_and_consumer_V_is_offset_bytes=8192,append83_V_full_base=True,K_and_scale_same_full_base=True)','dict(consumer_V_is_offset_bytes=4352,append83_V_full_base=True,K_and_scale_same_full_base=True)')
    edit("for role in ('k8_append','bf16_append','reducer'):","for role in ('initial_encoder','k8_append','bf16_append','tl_producer','reducer'):")
    edit("('bf16_append','tools/kernels/biv150_pair_append_support.py','append_kv'),", "('bf16_append',str(COMPACT_WRITER),'encode_append'),\n                                   ('tl_producer',str(TL_READER),'_gqa_decode_konly_i8'),")
    edit("('initial_encoder','tools/kernels/biv150_pair_append_support.py','encode_initial'),", "('initial_encoder',str(COMPACT_WRITER),'encode_append'),")
    edit("('k8_append','tools/kernels/biv150_konly_strided_writer.py','encode_append'),", "('k8_append',str(COMPACT_WRITER),'encode_append'),")
    edit("signature={ast.literal_eval(k):v for k,v in rec['actual_compiled']['k8_append']['signature'].items()}", "signature=rec['actual_compiled']['k8_append']['signature']\n        need(signature==rec['actual_compiled']['bf16_append']['signature'],'same typed writer ABI for both arms')")
    edit("check['guards_exact'] and check['all_bytes_compared']==f.PAGES*16384", "check['guards_exact'] and check['all_bytes_compared']==f.PAGES*(12544 if 'arm' in check else 16384)")
    edit("sum(x['bytes'] for x in chunks)==f.PAGES*16384", "sum(x['bytes'] for x in chunks)==f.PAGES*(12544 if 'arm' in check else 16384)")
    edit('full_cache_bytes_per_arm=f.PAGES*16384,','full_cache_bytes_per_arm=f.PAGES*12544,')
    edit("for arm,initial in (('bf16',cold),('k8',packed)):","for arm,initial in (('bf16',packed),('k8',packed)):")
    edit("data=part.view(np.uint8).reshape(count,16384) if phase is None else f.patched_chunk(arm,start,part,controls[name,phase])", "data=COMPACT_REMAP(part if phase is None else f.patched_chunk(arm,start,part,controls[name,phase]))")
    anchor="            for check in rec['initial_source_readonly_checks']:need(check['chunks']==expected['bf16',None],'regenerated source digests')"
    replacement="""            initial_chunks=[]
            for start in range(0,f.PAGES,128):
                count=min(128,f.PAGES-start);data=cold[start:start+count].tobytes()
                initial_chunks.append(dict(start=start,bytes=len(data),sha256=sha(data)))
            for check in rec['initial_source_readonly_checks']:need(check['chunks']==initial_chunks,'regenerated original BF16 source digests')
            for check in rec['compact_initial_source_checks']:
                need(check['stage'] in ('after_initial_pack','after_timing'),'linear source stage')
                need({x['source'] for x in check['sources']}=={'key','value','slots'},'linear source roles')
                for item in check['sources']:
                    role=item['source'];width=128 if role=='slots' else 8192
                    need(item['guards_exact'] and item['all_bytes_compared']==f.PAGES*width,'linear source coverage')
                    chunks=[]
                    for start in range(0,f.PAGES,128):
                        count=min(128,f.PAGES-start)
                        data=(np.arange(start*16,(start+count)*16,dtype='<i8').tobytes() if role=='slots' else cold[start:start+count,0 if role=='key' else 1].tobytes())
                        chunks.append(dict(start=start,bytes=len(data),sha256=sha(data)))
                    need(item['chunks']==chunks,'full separately packed K/V/identity slots')"""
    edit(anchor,replacement)
    # Symbols/labels change only after physical remap edits, preserving cold BF16 provenance.
    text=original
    for old,new in edits:
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
    labels=[("'bf16'","'tl'"),("'k8'","'native'"),('bf16_append','tl_append'),('k8_append','native_append'),('event_ratio_BF16_over_K8','event_ratio_TL_over_Native'),('host_wall_ratio_BF16_over_K8','host_wall_ratio_TL_over_Native'),('event_BF16_over_K8','event_TL_over_Native'),('host_wall_BF16_over_K8','host_wall_TL_over_Native'),('all_four_actual_JIT_function_sources_match_frozen','all_five_actual_JIT_function_sources_match_frozen'),('actual_K8_append_six_pointer_six_I64_ABI','actual_stride2560_append_six_pointer_six_I64_ABI')]
    for old,new in labels:
        assert old in text and new not in text;text=text.replace(old,new)
    restored=text
    for old,new in reversed(labels):restored=restored.replace(new,old)
    for old,new in reversed(edits):
        assert restored.count(new)==1;(restored:=restored.replace(new,old))
    assert restored==original;ast.parse(text)
    before=ast.parse(original);after=ast.parse(text)
    helpers={}
    for name in ('metric_same','check_snapshot','load','array','equal_fields'):
        one=next(n for n in ast.walk(before) if isinstance(n,ast.FunctionDef) and n.name==name)
        two=next(n for n in ast.walk(after) if isinstance(n,ast.FunctionDef) and n.name==name)
        a=ast.get_source_segment(original,one);b=ast.get_source_segment(text,two);assert a==b,name;helpers[name]=sha(a.encode())
    return text,dict(original_auditor_sha256=AUDIT_SHA,adapted_auditor_sha256=sha(text.encode()),inverse_byte_exact=True,edits=[dict(old=x,new=y) for x,y in edits],labels=labels,unchanged_math_and_array_helpers=helpers)


def extra_review(front,source,raw,summary,report):
    """New pair identity/ABI/resource layer; no replacement numeric mathematics."""
    out=raw/'out'
    def load(meta):
        data=(out/meta['path']).read_bytes();assert sha(data)==meta['sha256'] and len(data)==meta['bytes'];return data
    helper=source/'tools/audit_biv150_native_k8_compile_cpu.py';assert sha(helper.read_bytes())==ELF_SHA
    nodes=[n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('elf','msgpack')]
    env=dict(struct=struct);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper),'exec'),env)
    def metadata(blob):
        parsed=env['elf'](blob);assert parsed['machine']==248
        note=parsed['sections']['.note']['bytes'];at=0
        while at<len(note):
            ns,ds,kind=struct.unpack_from('<III',note,at);at+=12;name=note[at:at+ns].rstrip(b'\0');at=(at+ns+3)//4*4;desc=note[at:at+ds];at=(at+ds+3)//4*4
            if name==b'Iluvatar' and kind==10:
                ks=env['msgpack'](desc)['iluvatar.kernels'];assert len(ks)==1;return ks[0]
        raise AssertionError('missing actual cubin metadata')
    loaded=summary['actual_TL_reader_and_quant_helper']
    assert loaded['reader_sha256']==front.source.READER_SHA and loaded['quant_helper_sha256']==front.source.HELPER_SHA and loaded['quantizer_object_identity_verified'] is True
    assert loaded['reader_file']==summary['imports']['production_compact_reader'] and Path(loaded['quant_helper_file']).name==Path(front.source.HELPER).name
    assert summary['actual_native_library']['sha256']==LIBRARY and summary['actual_native_library']['loaded_library_set_confirmed']
    actual_resources=[]
    for rec in summary['records']:
        name=rec['case'];n,base,s,pitch=front.f.CASES[name]
        assert rec['page_budget']==front.f.budget(front.f.LIVE[name],base)
        expected={str(p):{k:v for k,v in front.f.host_plan(name,p).items() if k!='table'} for p in range(3)}
        assert rec['phase_page_proofs']==expected
        for field in ('query','new_k','new_v'):assert rec['source_views'][field]['stride'][0]==2560
        assert rec['source_views']['query']['shape']==[48,16,128]
        for field in ('new_k','new_v'):assert rec['source_views'][field]['shape']==[48,2,128]
        for phase,binding in enumerate(rec['writer_bindings']):
            assert set(binding)=={'tl','native'} and binding['tl']==binding['native']
            assert binding['tl']==dict(cache_pages=44769,format=front.compact.FORMAT,grid=[48,2],key_row_stride=2560,key_rows=48,launch=True,live_gpu_mapping_certified=False,requires_finite_valid_K_external=True,requires_gpu_slots_equal_snapshot=True,slot_count=48,slot_epoch=phase,value_row_stride=2560,value_rows=48)
        assert len(rec['writer_bindings'])==3
        assert set(rec['compact_roles'])=={'tl','native'}
        for arm in ('tl','native'):
            assert rec['compact_roles'][arm]==dict(format=front.compact.FORMAT,page_bytes=12544,pages=44769,V_consumer_offset=4352,writer_three_full_bases=True,frozen_views_sha256=front.compact.FROZEN['views'][1])
            assert rec['warm'][arm]['replays']==128 and rec['warm'][arm]['unroll']==8
        assert rec['stride2560_actual_JIT_and_resources_verified'] is True
        kernels=rec['actual_compiled'];assert set(kernels)=={'initial_encoder','tl_append','native_append','tl_producer','reducer'}
        for role,kernel in kernels.items():
            md=metadata(load(kernel['artifacts']['cubin']));asm=load(kernel['artifacts']['asm']).decode();llir=load(kernel['artifacts']['llir']).decode()
            assert type(kernel['n_regs']) is int and kernel['n_regs']>0 and kernel['n_spills']==0
            assert md['.vgpr_count']==kernel['n_regs'] and md['.vgpr_spill_count']==0
            assert type(md['.sgpr_spill_count']) is int and md['.sgpr_spill_count']>=0
            for field in ('.vgpr_spill_count','.sgpr_spill_count'):
                values=re.findall(r'^\s*'+re.escape(field)+r':\s*(\d+)',asm,re.M);assert len(values)==1 and int(values[0])==md[field]
            assert md['.private_segment_fixed_size']==0 and md['.warp_size']==64
            for key,field in (('.vgpr_count','workitem_vgpr_count'),('.sgpr_count','warp_sgpr_count'),('.group_segment_fixed_size','workgroup_group_segment_byte_size'),('.private_segment_fixed_size','workitem_private_segment_byte_size')):
                values=re.findall(r'^\s*'+re.escape(field)+r' = (\d+)',asm,re.M);assert len(values)==1 and int(values[0])==md[key]
            entries=re.findall(r'^define iluvatar_kernel void @(\w+)\((.*?)\)',llir,re.M)
            assert entries and entries[0][0]==md['.name']
            if role in ('initial_encoder','tl_append','native_append'):
                args=md['.args'];assert len(args)==14 and md['.kernarg_segment_size']==112
                for idx,arg in enumerate(args):assert arg['.offset']==idx*8 and arg['.size']==8 and arg['.value_kind']==('by_value' if 6<=idx<12 else 'global_buffer')
                assert len(re.findall(r'= fdiv float ',llir))>0 and not re.search(r'= fdiv (?:fast|arcp)',llir)
            if role=='tl_producer':
                sig=kernel['signature'];fields=('segm_output_ptr','segm_max_ptr','segm_expsum_ptr','query_ptr','query_code_ptr','query_scale_ptr','key_scale_ptr','key_cache_ptr','value_cache_ptr','seq_lens_ptr','block_table_ptr')
                assert [sig[x] for x in fields]==['*fp32']*3+['*bf16','*i8','*fp32','*fp32','*i8','*bf16','*i32','*i32']
                assert 'matrix.mad.i32' in llir and '.i8' in llir and 'matrix.mad.f32' in llir and 'bf16' in llir
                assert '= fdiv float ' in llir and not re.search(r'= fdiv (?:fast|arcp)',llir)
            actual_resources.append(dict(case=name,role=role,kernel=md['.name'],VGPR=md['.vgpr_count'],SGPR=md['.sgpr_count'],ELF_static_shared_bytes=md['.group_segment_fixed_size'],JIT_dynamic_shared_bytes=ast.literal_eval(kernel['metadata']['shared']),private_bytes=md['.private_segment_fixed_size'],runtime_n_spills=kernel['n_spills'],ELF_VGPR_spills=md['.vgpr_spill_count'],ELF_SGPR_spills=md['.sgpr_spill_count'],cubin_sha256=kernel['artifacts']['cubin']['sha256'],metadata_and_ASM_agree=True,signature=kernel['signature'],constants=kernel['constants'],attrs=kernel['attrs'],metadata=kernel['metadata']))
        assert all(key in rec['initial'] for key in ('compact_initial_key_linear_pack','compact_initial_value_linear_pack','compact_initial_slots','TL_GPU_full_encode','Native_GPU_full_encode','source_GPU_upload','CPU_full_counter_and_oracle_total_seconds'))
    owner_path=raw/'exports/cjj-native112-n48-fullstep-v1-20261029-owner.json';owner=json.loads(owner_path.read_text())
    assert summary['GPU_executed'] is False and '--gpu' in owner['command'] and '--root-owned-exclusive' in owner['command']
    native_static=summary['actual_consumer']['actual_compile_and_SO']['review']['static_resources']
    assert set(native_static)=={'S4'} and native_static['S4']['vgpr_spills']==native_static['S4']['sgpr_spills']==native_static['S4']['private_segment_bytes']==0
    assert all(x['ELF_SGPR_spills']==(26 if x['role']=='tl_producer' else 0) for x in actual_resources)
    report.update(actual_cubin_metadata_ASM_and_typed_ABI=actual_resources,actual_TL_reader_and_quant_helper=loaded,
        actual_native2ad_static_resources_from_fixed_approved_SO=native_static,
        runtime_n_spills_zero_gate_preserved=True,ELF_SGPR_spill_all_zero=False,
        resource_review_erratum=dict(rejected_extra_assumption='The earlier offline assertion that every ELF SGPR spill count must be zero exceeded the frozen frontend gate.',
            original_gate='Every loaded JIT handle has positive n_regs and n_spills==0; no frontend SGPR-zero rule.',
            observed_TL=dict(runtime_n_regs=144,runtime_n_spills=0,ELF_VGPR=144,ELF_SGPR=104,ELF_VGPR_spills=0,ELF_SGPR_spills=26,ELF_private_bytes=0,ELF_static_shared_bytes=0,JIT_dynamic_shared_bytes=6144),
            original_failure_retained=True,numerical_gates_or_tolerances_changed=False,Root_confirmed_scope_correction=True,
            interpretation_limit='Do not equate runtime n_spills with ELF scalar-register spill metadata or infer local-memory traffic/occupancy from one field.'),
        GPU_execution_record_erratum=dict(raw_GPU_executed=False,reason='Frozen plan default was copied into the summary and never updated by run(). Original raw bytes are unchanged.',
            owner_command_contains_gpu=True,owner_source_commit=owner['source_commit'],owner_sha256=sha(owner_path.read_bytes()),
            actual_library_load=summary['actual_native_library'],actual_five_role_JIT_artifacts_verified=True,
            full_saved_output_buffers_verified=36,original_numeric_checks_verified=144,Graph_and_side_and_ABBA_clocks_independently_verified=True),
        scope_limits=['Two same-shape TL48/native2ad compact arenas with44769 logical pages, not BF16 or twoN32 performance denominator.','Stride2560 Q/K/V are separate allocations; shared model QKV path is untested.','Three independently changed prefix snapshots, not a continuous-generation proof.','Complete GPU input/cache bodies are represented by runtime byte comparisons and reconstructible hashes, not a fresh GPU readback.','No42-layer capacity, service, model or stable end-to-end throughput qualification.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--source',type=Path,default=Path('/private/tmp/flagos-cjj-compact-native-n48-fullstep-20261029'))
    p.add_argument('--raw',type=Path);p.add_argument('--summary-sha256');p.add_argument('--manifest-sha256')
    p.add_argument('--resume-original-audit',type=Path,help='Only the pinned db985 already-completed original numerical audit; avoids rerunning its math while finalizing resource observations.')
    p.add_argument('--consumer-raw',type=Path,default=Path('/private/tmp/cjj-native112-n48-consumer-v1-raw-20261029'))
    p.add_argument('--consumer-review',type=Path,default=Path('/private/tmp/cjj-agent-native112-consumer-independent-review-20261029.json'))
    p.add_argument('--compile-raw',type=Path,default=Path('/private/tmp/cjj-native112-n48-compile-v1-raw-20261029'));a=p.parse_args();sys.dont_write_bytecode=True
    source=a.source.resolve();audit=source/'docs/evidence/native-k8-compact-fullstep-v1-20261028/original_fullstep_auditor.py'
    original=audit.read_bytes();assert sha(original)==AUDIT_SHA
    text,proof=adapt(original.decode())
    # Exact executable tree comparison allows docs-only merges after f569.
    git=lambda *args:subprocess.check_output(['git',*args],cwd=source)
    assert not git('diff',PREPARED,FROZEN,'--','tools','vllm_fl','csrc')
    for name in git('ls-tree','-r','--name-only',FROZEN,'tools','vllm_fl','csrc').decode().splitlines():
        if (source/name).is_file():
            data=str((source/name).readlink()).encode() if (source/name).is_symlink() else (source/name).read_bytes()
            assert data==git('show',FROZEN+':'+name),name
    sys.path.insert(0,str(source/'tools'));import probe_biv150_native_k8_compact_n48_fullstep as front
    plan=front.plan();assert plan['actual_source_sha256']=='3678f41e6df0e6e11fa52efa2c4174648928534c30da89806a4f4870b90a1d6f'
    if not a.run:
        print(json.dumps(dict(CPU_plan=True,review_executed=False,qualification_granted=False,pending_actual_summary_and_manifest=True,source_commit=FROZEN,adaptation=proof),indent=2,sort_keys=True));return
    assert a.raw and a.out and not a.out.exists()
    for pin in (a.summary_sha256,a.manifest_sha256):assert re.fullmatch('[0-9a-f]{64}',pin or '')
    raw=a.raw.resolve();assert sha((raw/'out/fullstep-summary.json').read_bytes())==a.summary_sha256 and sha((raw/'download-manifest.json').read_bytes())==a.manifest_sha256
    summary=json.loads((raw/'out/fullstep-summary.json').read_text());manifest=json.loads((raw/'download-manifest.json').read_text())
    assert summary['source_adapter']==json.loads(json.dumps(front.source.adapted_source()[1]))
    for name,pin in summary['frontend_sources'].items():assert sha((source/'tools'/name).read_bytes())==pin==sha((raw/'out/frontends'/name).read_bytes())
    for role,(name,pin) in front.compact.FROZEN.items():assert sha((raw/'out/frozen-sources'/Path(name).name).read_bytes())==pin
    assert (raw/'out/frozen-sources/biv150_konly_compact_decode.py').read_bytes()==front.source.production_bytes()
    cp=a.compile_raw/'out';ca=types.SimpleNamespace(consumer_raw=a.consumer_raw,consumer_archive_sha256='51ba899cc0b0767750b6a7018027448b01a925a26ea7963ba21ddf5e647c250f',consumer_review=a.consumer_review,consumer_review_sha256=front.CONSUMER_REVIEW_SHA,compile_summary=cp/'compile-summary.json',assembly_review=a.consumer_raw/'out/candidate_review-input.json',k8_library=cp/'build/cjj_native_k8_compact_n48_s4_probe.so')
    admission=front.actual_consumer_gate(ca)
    env=dict(__name__='original_fullstep_auditor_N48_adapter',__file__=str(audit),CODE_IDENTITIES_VERIFIED=True,EXPECTED_FILES=manifest['raw_files'],EXPECTED_BYTES=manifest['raw_bytes'],N48_PLAN=plan,ACTUAL_CONSUMER=admission,COMPACT_WRITER=raw/'out/frozen-sources/biv150_konly_compact_writer.py',TL_READER=raw/'out/frozen-sources/biv150_konly_compact_decode.py',COMPACT_REMAP=front.compact.compact_chunk)
    if a.resume_original_audit:
        intermediate=a.resume_original_audit
        assert sha(intermediate.read_bytes())=='db985c037f33727c238ba4c7b885692f6ac548c702083f88f2ec3c941006699b'
        prior=json.loads(intermediate.read_text())
        assert prior['passed'] and prior['cold_counter_and_gold_regenerated'] and prior['source_commit']==FROZEN
        assert prior['summary_sha256']==a.summary_sha256 and prior['manifest_sha256']==a.manifest_sha256
    else:
        intermediate=a.out.with_name(a.out.name+'.original-audit-only.json');assert not intermediate.exists()
        before=sys.argv;sys.argv=[str(audit),'--raw-root',str(raw),'--source-root',str(source),'--out',str(intermediate),'--regenerate']
        try:exec(compile(text,str(audit)+'::<N48 geometry/identity adapter>','exec'),env);env['main']()
        finally:sys.argv=before
    report=json.loads(intermediate.read_text());extra_review(front,source,raw,summary,report)
    report.update(N48_adapter=proof,N48_adapter_script_sha256=sha(Path(__file__).read_bytes()),original_numeric_audit_report_sha256=sha(intermediate.read_bytes()),original_numeric_audit_reused_without_repeating_math=bool(a.resume_original_audit),consumer_array_math_repeated=False,consumer_original_report_sha256=front.CONSUMER_REVIEW_SHA)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(report=str(a.out),sha256=sha(a.out.read_bytes()),passed=report['passed'])))
if __name__=='__main__':main()
