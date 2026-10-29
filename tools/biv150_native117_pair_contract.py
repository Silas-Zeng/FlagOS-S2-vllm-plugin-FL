"""CPU-only fixed 16KiB N64/S2 pair admission and exact frozen-host adapters."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import biv150_native_k8_klookahead_factory as factory
import probe_biv150_native_k8_v_overlap_consumer as old_consumer
import probe_biv150_native_k8_v_overlap_fullstep as old_fullstep

MAIN='a455a3de364051fbea80226870c239de3a48e716'
SOURCE='2c5b558c2238915443eda010da2450360d5161b8'
MAIN_SHA='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
NEW_SHA='5898a00a4a4e3314ee8ed1fc40dffde02a766ca6d0e5c116dbd161dccaa3286b'
COMPILE_SHA='678f26d589f3cfa9148009d56c9df028a6c3fd4fa4489a9816e7a9b1eafbb6d6'
REVIEW_SHA='176feaad8fa861f56d210a94829c05532ca801d2541b4f9efb571f1e4dffeabe'
CODEC_SHA='3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263'
CASES=((64,2,2048),(64,2,2560))
STAGES=('A-before','A-producer','A-after','graphB-before','graphB-after',
        'eagerB-before','eagerB-producer','eagerB-after','sideC-before','sideC-after')
ROLES={'main8b':(MAIN_SHA,'cjj_native_k8_v_overlap_probe'),
       'lookahead':(NEW_SHA,'cjj_native_k8_klookahead_probe')}
FLAGS=('Knext_dependency_store_then_end_barrier_before_next_QK',
       'Knext_global_after_current_V_wait_and_ready_barrier','Knext_global_issued_before_current_PV',
       'Knext_shared_publish_after_current_PV','actual_SO_ELF_bytes_bound',
       'both_S2_S4_full_next_last_and_original_tail_audited',
       'no_new_global_pending_in_current_V_wait8_window','no_speculative_tail_or_segment_read',
       'original_V_SME_wait_tail_QK_softmax_PV_math_preserved','Root_source_and_actual_SASS_review')

def sha(path):return factory.sha(Path(path).read_bytes())

def parent(root):return old_fullstep.parent_files(root)

def function(root,file,name):
    text=(parent(root)/'tools'/(file+'.py')).read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(text,node)

def edit(original,edits):
    changed=original
    for old,new,count in edits:
        assert changed.count(old)==count,(old,changed.count(old),count)
        changed=changed.replace(old,new)
    inverse=changed
    for old,new,count in reversed(edits):
        assert inverse.count(new)==count,(new,inverse.count(new),count)
        inverse=inverse.replace(new,old)
    assert inverse==original
    return changed,dict(parent=old_consumer.PARENT,original_sha256=factory.sha(original.encode()),
        adapted_sha256=factory.sha(changed.encode()),inverse_byte_exact=True,
        edits=[dict(old=o,new=n,count=c) for o,n,c in edits])

def consumer_source(root):
    text=function(root,'probe_biv150_native_k8_consumer_v2','run')
    return edit(text,[
        ('f.sha(args.library.read_bytes())==f.LIBRARY_SHA','f.sha(args.library.read_bytes())==args.role_sha256',2),
        ('torch.ops.cjj_native_k8_qk_probe.produce','getattr(torch.ops,args.namespace).produce',1),
        ('result=dict(f.plan(),codec_precondition=receipt,','result=dict(role_plan(),codec_precondition=receipt,',1),
        ('for n,s,pitch in f.CASES:','for n,s,pitch in PAIR_CASES:',1)])

def receipt_source(root):
    text=function(root,'biv150_native_k8_consumer_receipt','require_consumer')
    return edit(text,[
        ("expected_sha=='c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128' and f.sha(raw)==expected_sha",'f.sha(raw)==expected_sha',1),
        ('f.LIBRARY_SHA','ROLE_SHA',2),('set(f.CASES)','set(PAIR_CASES)',1),
        ("len(j['records'])==4","len(j['records'])==len(PAIR_CASES)",1),('cases=4,','cases=len(PAIR_CASES),',1)])

def load(root,file,name):
    assert name not in sys.modules
    sys.path.insert(0,str(parent(root)/'tools'))
    spec=importlib.util.spec_from_file_location(name,root/'tools'/(file+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);parent(root)
    return m

def plan():
    # Current-main reference is the exact 8b source carried by a455, not the old
    # BF16 aace control and not a service performance claim.
    import biv150_native_k8_v_overlap_factory as base
    for role in base.FILES:
        current=base.path(role).read_bytes()
        assert current==subprocess.check_output(['git','-C',str(factory.ROOT),'show',MAIN+':'+str(base.path(role).relative_to(factory.ROOT))])
    result=dict(main_source=MAIN,kernel_source=SOURCE,parent_fixture=old_consumer.PARENT,
        roles={r:dict(library_sha256=p[0],namespace=p[1]) for r,p in ROLES.items()},
        kernel_source_proof=factory.verify(),main8b_source_proof=base.verify(),
        consumer_cases=CASES,fullstep_case={'4k':(64,4096,2,2048)},page_bytes=16384,
        code_byte_page_stride=16384,scale_FP32_page_stride=4096,V_BF16_page_stride=8192,
        V_byte_offset=8192,codec='unchanged RN32/RNE',private_component=True,
        default_stdlib_only=True,numeric_verified=False,performance_verified=False,
        precision_gate=False,model_verified=False,service_verified=False,capacity_reclaimed=False,
        original_BF16_proxy_gate_not_changed=True,Root_exclusive_scheduling_required=True)
    return json.loads(json.dumps(result))

def admission(a):
    assert a.root_owned_exclusive,'Root must confirm prior owner terminal and exclusive schedule'
    root=parent(a.parent_source)
    for path in (a.compile_summary,a.assembly_review,a.main8b_library,a.library,a.codec_summary):
        assert path.is_absolute() and path.is_file()
    assert a.assembly_review_sha256==REVIEW_SHA and sha(a.assembly_review)==REVIEW_SHA
    assert sha(a.compile_summary)==COMPILE_SHA
    j=json.loads(a.compile_summary.read_text());r=json.loads(a.assembly_review.read_text())
    assert j['completed'] is True and j['compile_passed'] is True and not j['errors']
    assert j['source_commit']==r['source_commit']==SOURCE and j['source_root']==r['source_root']
    assert j['source_transform']==factory.verify() and r['compile_summary_sha256']==COMPILE_SHA
    assert j['library']==r['library'] and j['library']['sha256']==NEW_SHA
    assert r['approved'] is True and all(r[k] is True for k in FLAGS)
    for row in r['resources'].values():
        for field,value in (('.vgpr_count',85),('.sgpr_count',70),('.group_segment_fixed_size',8192),
            ('.vgpr_spill_count',0),('.sgpr_spill_count',0),('.private_segment_fixed_size',0),('.warp_size',64)):
            assert type(row[field]) is int and row[field]==value
    assert all(r[k] is False for k in ('numeric_verified','performance_verified','service_verified'))
    assert j['frontend_explicit_kernel_launches']==j['input_initializations']==0
    assert j['torch']['cuda_initialized_before'] is j['torch']['cuda_initialized_after'] is False
    for role,pin in j['source_transform']['files'].items():
        filename=factory.path(role).name
        assert (a.compile_summary.parent/filename).read_bytes()==factory.path(role).read_bytes()
    for row in j['artifact_manifest']:
        name=Path(row['relative']);assert not name.is_absolute() and '..' not in name.parts
        p=a.compile_summary.parent/name
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    assert sha(a.library)==NEW_SHA and sha(a.main8b_library)==MAIN_SHA
    assert a.codec_proof_sha256==CODEC_SHA and sha(a.codec_summary)==CODEC_SHA
    assert a.out.is_absolute() and not a.out.exists()
    assert not a.out.resolve().is_relative_to(root) and not a.out.resolve().is_relative_to(factory.ROOT)
    return root,dict(compile_summary_sha256=COMPILE_SHA,assembly_review_sha256=REVIEW_SHA,
        library_sha256=NEW_SHA,main8b_library_sha256=MAIN_SHA,
        actual_compile_source_root=j['source_root'],actual_compile_source_commit=SOURCE,
        validator_source_root=str(factory.ROOT),actual_library_path=str(a.library),
        static_resources=r['resources'],
        main8b_library_path=str(a.main8b_library),source_proof=factory.verify(),
        exclusion_authority='Root serial scheduling; no namespace PID or empty-table inference')

def safe_blob(root,meta):
    rel=Path(meta['path']);assert not rel.is_absolute() and '..' not in rel.parts
    path=root/rel;assert path.resolve().is_relative_to(root.resolve())
    raw=path.read_bytes();assert len(raw)==meta['bytes'] and factory.sha(raw)==meta['sha256'];return raw

def compare_pair(summaries):
    left,right=(summaries[r] for r in ROLES)
    lm={tuple(r['case']):r for r in left['records']};rm={tuple(r['case']):r for r in right['records']}
    assert set(lm)==set(rm)==set(CASES) and len(left['records'])==len(right['records'])==len(CASES)
    counts=[]
    for case in CASES:
        a,b=lm[case],rm[case];assert set(a['snapshots'])==set(b['snapshots'])==set(STAGES)
        for stage in a['snapshots']:
            x,y=a['snapshots'][stage],b['snapshots'][stage];assert x.keys()==y.keys()
            assert {'partial','maximum','summation','output'}<=set(x)
            # Content-addressed full arrays include all guards and inactive poison.
            for field in x:assert (x[field]['bytes'],x[field]['sha256'])==(y[field]['bytes'],y[field]['sha256']),(case,stage,field)
        counts.append(dict(case=case,full_snapshot_stages=len(a['snapshots']),all_fields_bitwise=True))
    return counts

def require_pair(path,expected,root,binding):
    assert len(expected)==64 and sha(path)==expected
    pair=json.loads(path.read_text());assert pair['completed'] is True and not pair['errors']
    recorded=pair['binding']
    kernel_binding=lambda d:{k:v for k,v in d.items() if k!='validator_source_root'}
    assert kernel_binding(recorded)==kernel_binding(binding)
    assert pair['plan']==plan() and pair['full_OML_final_bitwise'] is True
    assert set(pair['arms'])==set(ROLES)
    source,adapter=receipt_source(root);module=load(root,'biv150_native_k8_consumer_receipt','cjj117_receipt_'+expected[:12])
    summaries={};receipts={}
    for role,(digest,namespace) in ROLES.items():
        p=path.parent/pair['arms'][role]['summary_relative'];raw=p.read_bytes()
        assert factory.sha(raw)==pair['arms'][role]['summary_sha256']
        j=json.loads(raw);assert j['pair_binding']==recorded and j['pair_role']==role and j['pair_plan']==plan()
        assert j['library_sha256']==digest and j['consumer_adapter']==consumer_source(root)[1]
        ns=dict(vars(module),ROLE_SHA=digest,PAIR_CASES=CASES)
        exec(compile(source,'<117-original-consumer-receipt-identity-scope>','exec'),ns)
        receipts[role]=ns['require_consumer'](p,factory.sha(raw));summaries[role]=j
    comparisons=compare_pair(summaries)
    return dict(summary_sha256=expected,arms=receipts,full_comparisons=comparisons,
        actual_consumer_validator_root=recorded['validator_source_root'],
        current_validator_root=binding['validator_source_root'],
        receipt_adapter=adapter,all_full_arrays_and_FP64_replayed=True,no_performance_or_service_qualification=True)
