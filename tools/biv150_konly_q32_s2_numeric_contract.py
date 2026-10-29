"""Exact actual N1 S1/S2 cache admission; original e4dd gates remain unchanged."""
import ast
import base64
import json
from pathlib import Path
import sys
import types
import biv150_konly_q32_s2_numeric_source as s

ROLES=('s1','s2')
MODES=('reader','append-reader')
STAGES={'s1':1,'s2':2}
LENGTHS={0:6144,1:6143}
RESOURCES={'s1':dict(n_regs=122,n_spills=0,shared=33792,ELF_sgpr_spills=96),
           's2':dict(n_regs=124,n_spills=0,shared=41216,ELF_sgpr_spills=93)}
HASHES={'s1':'53ef08b799eddb1170100ca6a7fca2171a54443acbf5a5e88549b35ceffd1603',
        's2':'2d8938635879f904f3d844b27cb8919ec3d6d4e71cf9a5b990362059065d90e1'}
CUBINS={'s1':'7eadbdbf854be149083065cb30f0f8c1bdc593aea6d2b9baf454502adb6fbb43',
        's2':'dab726c48cabe17bd795e2b35f094c5fefcb40db0e802f655bff51470b8715ce'}
OVERRIDES={'s1':'3c750774af90b1c94eae06a2b047295318812b4a7ef669f81c45864b88629ffc',
           's2':'9f45fd740d0eca78f00e80f43b890a1a624301f12aee84664c3707a41e9326db'}
ARCHIVES={'s1':('60476bbcbaf13ea4d44fcfc0770d586f2241c8bfdb67cff6dbe79124891589c1',3219,84437304,206472,364351602,s.BASE),
          's2':('a7583d8afb11bf0614adbeaeeb05349716fc3f7793f827b5932da1d5f6c0316c',27,2227237,219924,364718401,s.PARENT)}
BINDING='konly-q32-hoist/206782/6a927892585b09d9dfd2635c7e188ca35920f9f62e2de430954b32ce19cc12a7/binding.json'
BINDING_SHA='65e94b7e66f7e79ccaa3b6f7fb6f57b3c23a498cc3612b1c00da0a1c02c369e3'
SUMMARY_SHA='c35d0001e07b2b0e59879eee790d32e9745aeecff98d5dca0823c45991eeac32'
REVIEW_SHA='80f1b3388822eb50c7db1d953430d726d15b0269dd927a02638c5c2328cc58a0'
FUNCTION_SHA='308a02ba6bef2e50c4c3d439809fd73eb2ca1e9fe87d47d4772f5fcef7ebde23'
FIXTURE='289afc781f17e80465626eef045cd4bfb3db5ed7c516ce662d8e75c06a698132'
WRITER='16afc46585f950f4b9c06c831b0e0dbf5515ee941a476a9ed300c492ebf947f9'
LOADED_COMPILER=Path('/usr/local/lib/python3.12/site-packages/triton/spec/iluvatar/triton/compiler/compiler.py')
need=s.need
sha=s.sha


def plan():
    d=s.dependencies()
    return dict(schema='cjj-q32-S1-S2-preheader-numeric-v1',base=s.BASE,compile_source=s.PARENT,
        immutable_screen=s.TEMPLATE,screen_source_pins=s.PINS,derived_numeric_driver_sha256=sha(s.run_source()),
        roles=list(ROLES),stages=STAGES,hashes=HASHES,cubins=CUBINS,resources=RESOURCES,
        query_rows=2048,query_pitch=2048,pages=773,page_bytes=16384,table_pitch=8192,writer_pitch=2560,
        heads=dict(Q=16,KV=2,D=128),num_seqs=1,grid=[65,2],modes=list(MODES),
        cases=[dict(pattern=k,total_KV=v,prefix=v-2048,valid_tail=v%16,KV_tile_tail=v%64) for k,v in LENGTHS.items()],
        seed=20261025,pattern0_all_rows=True,pattern1_rows=[0,1,15,16,31,32,63,64,1024,2047],
        reference_commit=d.contract.REFERENCE_COMMIT,reference_pins=d.contract.REFERENCES,
        unchanged_gate='original pointwise .005+.005*abs(reference); original BF16 relative RMSE <=.005',
        original_BF16_diagnostic_failure_allowed_only_with_explicit_flag=True,
        reference_is_ideal_FP64_not_pagewise_BF16P=True,Graph_B_before_fresh_eager_B=True,
        side_stream_required=True,reader_cache_miss_compile_prohibited=True,
        captured_steps=4,planned_reader_cache_warmups=2,planned_writer_warmups=1,
        default_enabled=False,GPU_executed=False,numeric_verified=False,Graph_verified=False,
        side_stream_verified=False,performance_measured=False,precision_gate=False,service_verified=False,
        Root_numeric_admission=False,context16k_tested=False,ABBA_pending=True,
        ABBA_future='same complete S1/S2 reader and unchanged-writer+reader; no raw-S2 or N2 control',
        limits=['Isolated original N1 descriptor only; no service/host/paired-prefill/capacity qualification.',
                'Pattern0 is aligned, pattern1 is a valid 6143-token tail with 2048 queries/prefix4095; invalid page lanes are poisoned.',
                'All outputs/cache/inputs saved. Random FP64 references are selected rows; uniform FP64 reference covers all rows.',
                'Reader warmups must hit the reviewed cache; no candidate IR compilation. Writer warmup may compile the unchanged writer.'])


def check_pair(records,metadata):
    need(set(records)==set(metadata)==set(ROLES),'exact two N1 roles')
    for role in ROLES:
        r=records[role];m=metadata[role]
        need(all(type(m.get(k)) is int for k in ('num_warps','num_stages','warp_size','shared','use_sme')) and all(type(v) is int for v in r['launch_geometry']),'known integer options '+role)
        need(r['hash']==m['hash']==HASHES[role] and r['artifacts']['cubin']==CUBINS[role],'actual binary '+role)
        need(r['source_digest']==FUNCTION_SHA and r['function_name']=='kernel_cjj_konly_q32','same N1 function '+role)
        need(r['launch_geometry']==[16,STAGES[role],64] and (m['num_warps'],m['num_stages'],m['warp_size'])==(16,STAGES[role],64),'stage geometry '+role)
        need(m['target']==dict(backend='corex',arch=71,warp_size=64) and m['shared']==RESOURCES[role]['shared'],'actual target/shared '+role)
    need(records['s1']['ABI']==records['s2']['ABI'],'FULL typed ABI/constexpr/attributes differ')
    changed={k for k in metadata['s1'].keys()|metadata['s2'].keys() if metadata['s1'].get(k)!=metadata['s2'].get(k)}
    need(changed=={'hash','num_stages','shared','ir_override'},'other actual options changed')


def verified_raw(raw,role):
    raw=raw.resolve();pin,count,size,pid,start,commit=ARCHIVES[role]
    data=(raw/'download-manifest.json').read_bytes();need(sha(data)==pin,'exact raw manifest '+role)
    m=json.loads(data);need(m['raw_files']==len(m['files'])==count and m['raw_bytes']==size,'raw geometry '+role)
    need(m['original_owner']==dict(pid=pid,start_ticks=start,same_start=True,state='Z'),'terminal owner '+role)
    need(len({r['relative'] for r in m['files']})==count,'duplicate archive member')
    for row in m['files']:
        path=raw/row['relative'];need(path.resolve().is_relative_to(raw),'archive escape')
        body=path.read_bytes();need(len(body)==row['bytes'] and sha(body)==row['sha256'],'raw bytes '+row['relative'])
    owner=json.loads(next((raw/'exports').glob('*owner.json')).read_bytes())
    need((owner['pid'],owner['start_ticks'],owner['source_commit'])==(pid,start,commit),'original owner identity')
    return m,owner


def preflight(s1_raw,s2_raw,review_path):
    need(sha(review_path.read_bytes())==REVIEW_SHA,'independently replayed actual S2 static report')
    review=json.loads(review_path.read_bytes())
    need(review['review_passed'] is True and review['source']==s.PARENT and review['new_compiled']['cubin_sha256']==CUBINS['s2'],'S2 actual report binding')
    need(review['Q_operand_hoist_TTGIR_LLVM_compiler_ASM_static_verified'] is True and review['S2_bank_publish_reuse_acquire_static_verified'] is True,'S2 static evidence')
    d=s.dependencies();records={};metadata={};runtime={};cache={};provenance={}
    for name,digest in ((s.ROOT/'tools/biv150_prefill_probe_fixtures.py',FIXTURE),(s.ROOT/'vllm_fl/patches/biv150_konly_encoded_writer.py',WRITER)):
        need(sha(name.read_bytes())==digest,'unchanged fixture/writer')
    for role,raw in (('s1',s1_raw),('s2',s2_raw)):
        manifest,owner=verified_raw(raw,role);out=raw/'out';remote=Path(owner['out'])
        if role=='s1':
            path=out/BINDING;need(sha(path.read_bytes())==BINDING_SHA,'selected actual production N1 binding')
            binding=json.loads(path.read_bytes());r=binding['hoisted'];folder=path.parent/'hoisted'
            service=json.loads((out/'manifest.json').read_bytes())
            need(service['source_commit']==s.BASE and service['source_hashes']['vllm_fl/patches/biv150_konly_encoded_q32.py']==d.c.PINS['biv150_konly_encoded_q32.py'],'actual main899 production source')
            am=binding['actual_metadata'];need(am['query_rows']==2048 and am['grid']==[65,2] and am['constants']['num_seqs']==1,'actual N1, not N2 substitute')
            need(binding['resources']['hoisted']['known'] is True,'production loaded-resource receipt')
            selected=[v for v in service['actual_runtime_after_throughput']['selected'] if v.get('hash')==HASHES['s1']]
            need(len(selected)==1 and selected[0]['query_rows']==2048 and selected[0]['sequences']==1 and selected[0]['I8_fragment_hoist_verified'] is True,'actual service-selected N1 S1')
            # The immutable proposal has pending flags even after _prepare's
            # separate actual audit. Preserve them; do not turn a static cache
            # selection into inherited numerical/service qualification.
            need(binding['proposal']['runtime_binding_qualified'] is False and binding['numerical_verified'] is False,'original pending labels retained')
            source=Path(service['source'])/'vllm_fl/patches/biv150_konly_encoded_q32.py';module='vllm_fl.patches.biv150_konly_encoded_q32'
            override=Path(binding['override_path']);native=d.c.module_from_source('exact_current_S1_native_audit',d.c.original('biv150_konly_encoded_hoist_native.py'))
            audit=native.audit((path.parent/'override.ttgir').read_text(),(folder/'actual.ttgir').read_text(),(folder/'actual.llir').read_text(),(folder/'actual.asm').read_text())
            need(audit==binding['native'],'replay actual S1 Q/read/sync/math audit')
        else:
            need(sha((out/'summary.json').read_bytes())==SUMMARY_SHA,'actual S2 summary')
            summary=json.loads((out/'summary.json').read_bytes());item=summary['records']['preheader-q32-stage2'];r=item['preheader-stage2']
            need(summary['commit']==s.PARENT and summary['completed'] is summary['compile_passed'] is True and summary['errors']==[],'actual S2 compile completion')
            need(all(summary[k]==0 for k in ('attention_launches','writer_launches','encoder_launches')),'preserve compile-only zero launches')
            folder=out/'preheader-q32-stage2/preheader-stage2';source=remote/'sources/candidate_q32_s2.py';module='cjj_private_q32_s2_scout'
            override=Path(summary['override_path']);need(item['actual_metadata_ir_override']==str(override),'actual S2 override')
        need(json.loads((folder/'record.json').read_bytes())==r,'record mirror '+role)
        for ext,digest in r['artifacts'].items():need(sha((folder/('actual.'+ext)).read_bytes())==digest,'preserved artifact '+role+'/'+ext)
        need(sha((folder/'actual.jit_python_source.py').read_bytes())==FUNCTION_SHA,'same complete kernel Python source')
        directory=base64.b32encode(bytes.fromhex(r['hash'])).decode().rstrip('=');group=out/'triton-cache'/directory
        index=group/'__grp__kernel_cjj_konly_q32.json';children=json.loads(index.read_bytes())['child_paths']
        need(set(children)=={'kernel_cjj_konly_q32.'+e for e in d.cache.EXTENSIONS},'complete actual seven-child cache')
        selected={}
        for path in [index]+[group/n for n in sorted(children)]:
            relative=path.relative_to(out).as_posix();row=next(v for v in manifest['files'] if v['relative']=='out/'+relative)
            selected[relative]=dict(bytes=row['bytes'],sha256=row['sha256'])
            if path.name in children:
                ext=path.suffix[1:]
                if ext in r['artifacts']:need(sha(path.read_bytes())==r['artifacts'][ext],'actual cache/artifact '+role+'/'+ext)
        m=json.loads((group/'kernel_cjj_konly_q32.json').read_bytes());metadata[role]=m;records[role]=r
        need(m['ir_override']==str(override),'original source/debug/override path stays fixed')
        local_override=(out/Path(BINDING).parent/'override.ttgir') if role=='s1' else out/'q-local-preheader.ttgir'
        need(sha(local_override.read_bytes())==OVERRIDES[role],'exact role override '+role)
        runtime[role]=dict(source=str(source),source_sha256=d.c.PINS['biv150_konly_encoded_q32.py'],module_name=module,
            override=str(override),override_sha256=OVERRIDES[role])
        cache[role]=dict(original_out=str(remote),directory=directory,members=selected)
        provenance[role]=dict(manifest_sha256=ARCHIVES[role][0],raw_files=manifest['raw_files'],raw_bytes=manifest['raw_bytes'],owner=manifest['original_owner'])
    check_pair(records,metadata)
    return dict(schema='cjj-N1-S1-S2-prepared-actual-cache-v1',plan=plan(),records=records,metadata=metadata,
        runtime_paths=runtime,cache=cache,provenance=provenance,static_review_sha256=REVIEW_SHA,
        actual_N1_control_from_main899_service=True,complete_typed_ABI_equal=True,Root_numeric_admission=False)


def admission_template(commit,preflight_sha256):
    return dict(schema='cjj-124-N1-S1-S2-numeric-root-admission-v1',Root_numeric_approved=False,
        frontend_commit=commit,preflight_sha256=preflight_sha256,static_review_sha256=REVIEW_SHA,
        hashes=HASHES,cubins=CUBINS,query_rows=2048,num_seqs=1,stages=STAGES,
        timing_allowed=False,service_qualified=False,precision_qualified=False)


def admit(args,pre):
    need(pre['plan']==plan() and pre['static_review_sha256']==REVIEW_SHA,'exact prepared source/proof scope')
    check_pair(pre['records'],pre['metadata'])
    need(sha(args.admission.read_bytes())==args.admission_sha256,'Root admission SHA')
    receipt=json.loads(args.admission.read_bytes())
    need(receipt.get('Root_numeric_approved') is True and all(receipt.get(k) is False for k in ('timing_allowed','service_qualified','precision_qualified')),'strict Root decision booleans')
    expected=dict(admission_template(args.commit,args.preflight_sha256),Root_numeric_approved=True)
    need(receipt==expected,'explicit Root numeric-only decision missing or scope differs')
    for paths in pre['runtime_paths'].values():
        need(sha(Path(paths['source']).read_bytes())==paths['source_sha256'],'original role source path/bytes')
        need(sha(Path(paths['override']).read_bytes())==paths['override_sha256'],'original role override path/bytes')


def prepare_cache(destination,pre):
    """522c's seven-child copy/relocation logic, with two explicit source roots."""
    d=s.dependencies();destination=destination.resolve();destination.mkdir()  # fresh, no overwrite
    result={}
    for role,r in pre['records'].items():
        original=pre['cache'][role];root=Path(original['original_out']);directory=original['directory']
        def read(relative):
            path=root/relative;need(path.resolve().is_relative_to(root.resolve()),'live cache path escape')
            row=original['members'][relative];body=path.read_bytes()
            need(len(body)==row['bytes'] and sha(body)==row['sha256'],'original cache bytes '+role+'/'+relative)
            return body
        name=r['function_name'];index='__grp__'+name+'.json';prefix='triton-cache/'+directory+'/'
        old_index=read(prefix+index);children=json.loads(old_index)['child_paths']
        need(set(children)=={name+'.'+ext for ext in d.cache.EXTENSIONS},'seven actual cache children')
        target=destination/directory;target.mkdir();relocated={};digests={}
        for filename,oldpath in children.items():
            need(Path(oldpath)==root/prefix/filename,'actual original cache group path')
            body=read(prefix+filename);ext=filename.rsplit('.',1)[1]
            if ext in r['artifacts']:need(sha(body)==r['artifacts'][ext],'reviewed record/cache artifact')
            if ext=='json':need(json.loads(body)==pre['metadata'][role],'entire actual metadata/options')
            out=target/filename;out.write_bytes(body);relocated[filename]=str(out);digests[filename]=sha(body)
        (target/index).write_text(json.dumps({'child_paths':relocated},sort_keys=True)+'\n')
        result[role]=dict(hash=r['hash'],name=name,cache_root=str(destination),paths=relocated,sha256=digests,
            original_index_sha256=sha(old_index),relocated_index_sha256=sha((target/index).read_bytes()))
    return result


def export_references(folder):
    d=s.dependencies();folder.mkdir()
    for name,digest in d.contract.REFERENCES.items():
        body=s.git('show',d.contract.REFERENCE_COMMIT+':tools/'+name)
        need(sha(body)==digest,'immutable e4dd reference '+name);(folder/name).write_bytes(body)


def reference_source(text):
    old=s.dependencies().contract
    return s.replace(text,[(old.ROWS_BEFORE,
        'rows=[0,1,15,16,31,32,63,64,1024,2047];length=int(cpu[\'lengths\'][0])\n    if length not in (6143,6144) or cpu[\'cu_q\'].tolist()!=[0,2048]:raise ValueError(\'only Q2048/KV6143 or6144\')'),
        ('mask=torch.arange(length)[None,:]<4096+torch.tensor(rows)[:,None]+1',
         'mask=torch.arange(length)[None,:]<(length-2048)+torch.tensor(rows)[:,None]+1')])


def load_reference(folder):
    d=s.dependencies();sources,_=d.contract.reference_sources(folder);expanded=reference_source(sources['biv150_konly_q32_numeric_contract.py'])
    for name in ('biv150_konly_i8_qk_reference','biv150_konly_decode_quant_reference'):
        need(name not in sys.modules,'unrelated reference already loaded');d.c.load_file(name,folder/(name+'.py'))
    ref=d.c.module_from_source('cjj_original_e4dd_reference_N1_valid_tail',expanded)
    tree=ast.parse(sources['bench_biv150_konly_service_q32_bitwise.py']);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='policy')
    ns={'h':types.SimpleNamespace(need=need)};exec(compile(ast.Module(body=[fn],type_ignores=[]),'unchanged-e4dd-policy','exec'),ns)
    return ref,ns['policy'],expanded


def snapshot_module(role):
    d=s.dependencies()
    if role=='s1':return d.c.load_file('cjj_S1_original_snapshot',s.ROOT/'vllm_fl/patches/biv150_konly_encoded_q32_adapter.py')
    need(role=='s2','unknown role snapshot')
    import biv150_konly_q32_s2_preheader_source as candidate
    return d.c.module_from_source('cjj_S2_private_snapshot',candidate.snapshot_source())


def check_loaded(role,kernel,snapshot,expected):
    d=s.dependencies();source=s.original('biv150_konly_q64_screen_contract.py');tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_loaded')
    before='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
    after=s.replace(before,[("==(16,1,64),'loaded geometry'","==(16,STAGES[role],64),'loaded geometry'")])
    ns=dict(c=d.c,RESOURCES=RESOURCES,STAGES=STAGES,execution_snapshot=d.contract.execution_snapshot)
    exec(compile(after,'<unchanged-522c-loader-plus-per-role-stage>','exec'),ns)
    return ns['check_loaded'](role,kernel,snapshot,expected)


def execution_sources(root):return s.dependencies().contract.execution_sources(root)
