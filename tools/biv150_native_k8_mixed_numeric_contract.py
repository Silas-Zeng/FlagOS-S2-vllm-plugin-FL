"""Private PR118 component admission and exact 3eb consumer adaptation; stdlib only."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import types
import biv150_native_k8_mixed_factory as factory
import compile_biv150_native_k8_mixed as build
import probe_biv150_native_k8_v_overlap_consumer as original_consumer

MAIN_SHA='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
MAIN_NAMESPACE='cjj_native_k8_v_overlap_probe'
CODEC_SHA='3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263'
PAIRS=((29,4),(30,4),(31,4),(60,2),(61,2),(63,2))
CASES=tuple((n,s,pitch) for n,s in PAIRS for pitch in (2048,2560))
STAGES=('A-before','A-producer','A-after','graphB-before','graphB-after',
        'eagerB-before','eagerB-producer','eagerB-after','sideC-before','sideC-after')
FLAGS=('actual_SO_ELF_IR_SASS_bound','normalized_GPU_math_matches_accepted8b',
       'new_rows31_63_host_ABI_verified','new_rows29_30_host_ABI_verified',
       'new_rows60_61_host_ABI_verified','full_tail_wait_and_codec_verified')

def sha(p):return hashlib_sha(Path(p).read_bytes())
def hashlib_sha(raw):return factory.sha(raw)
def need(ok,why):
    if not ok:raise ValueError('PR118 component: '+why)
def edit(source,changes):
    result=source
    for a,b,count in changes:
        need(result.count(a)==count,'source edit count '+a[:80]);result=result.replace(a,b)
    inverse=result
    for a,b,count in reversed(changes):
        need(inverse.count(b)==count,'source inverse count '+b[:80]);inverse=inverse.replace(b,a)
    need(inverse==source,'source inverse')
    return result,dict(original_sha256=hashlib_sha(source.encode()),adapted_sha256=hashlib_sha(result.encode()),
        inverse_byte_exact=True,edits=[dict(old=a,new=b,count=n) for a,b,n in changes])

def private_module(module,name,**overrides):
    """Rebind original function globals without mutating the frozen parent module."""
    result=types.ModuleType(name);result.__dict__.update(vars(module));result.__dict__.update(overrides)
    result.__dict__['__name__']=name
    for key,value in vars(module).items():
        if isinstance(value,types.FunctionType) and value.__module__==module.__name__:
            fn=types.FunctionType(value.__code__,result.__dict__,value.__name__,value.__defaults__,value.__closure__)
            fn.__kwdefaults__=value.__kwdefaults__;result.__dict__[key]=fn
    return result

def parent(root):return original_consumer.parent_files(root)
def source_function(root,file,name):
    text=(parent(root)/'tools'/(file+'.py')).read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(text,node)

def consumer_source(root):
    text=source_function(root,'probe_biv150_native_k8_consumer_v2','run')
    changes=[('f.sha(args.library.read_bytes())==f.LIBRARY_SHA','f.sha(args.library.read_bytes())==args.role_sha256',2),
        ('torch.ops.cjj_native_k8_qk_probe.produce','getattr(torch.ops,args.namespace).produce',1),
        ('result=dict(f.plan(),codec_precondition=receipt,','result=dict(role_plan(),codec_precondition=receipt,',1),
        ("shapes={'partial':(n,16,s,128),'maximum':(n,16,s),'summation':(n,16,s),'output':(n,16,128)}",
         "shapes={'partial':(n,16,16,128),'maximum':(n,16,16),'summation':(n,16,16),'output':(n,16,128)}",1),
        ("views[field]=full[field].narrow(0,f.GUARD_ELEMENTS,count).reshape(shape)",
         "wide=full[field].narrow(0,f.GUARD_ELEMENTS,count).reshape(shape)\n                views[field]=wide if field=='output' else wide.narrow(2,0,s)",1),
        ("for field,tensor in views.items():tensor.view(torch.int16 if field=='output' else torch.int32).fill_(f.BF16_POISON if field=='output' else f.F32_POISON)",
         "for field,tensor in full.items():tensor.narrow(0,f.GUARD_ELEMENTS,tensor.numel()-2*f.GUARD_ELEMENTS).view(torch.int16 if field=='output' else torch.int32).fill_(f.BF16_POISON if field=='output' else f.F32_POISON)",1)]
    old="""                        mask=np.broadcast_to(g['active'][:,None,:,None] if field=='partial' else g['active'][:,None,:],shape)
                        checks[field+'_inactive_poison']=bool((bits[~mask]==f.F32_POISON).all())
                        metrics[field]=f.metrics(actual[mask],g[field][mask]);checks[field+'_FP64']=metrics[field]['passed']
                        if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(actual[mask],g['ideal_partial_diagnostic'][mask])"""
    new="""                        mask=np.zeros(shape,dtype=bool)
                        mask[:,:,:s]=np.broadcast_to(g['active'][:,None,:,None] if field=='partial' else g['active'][:,None,:],mask[:,:,:s].shape)
                        wanted=mask[:,:,:s]
                        checks[field+'_inactive_poison']=bool((bits[~mask]==f.F32_POISON).all())
                        metrics[field]=f.metrics(actual[:,:,:s][wanted],g[field][wanted]);checks[field+'_FP64']=metrics[field]['passed']
                        if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(actual[:,:,:s][wanted],g['ideal_partial_diagnostic'][wanted])"""
    changes.append((old,new,1))
    old="""            assert key.data_ptr()==scale.data_ptr()==inputs['cache'].data_ptr()+f.LAYER_OFFSET and value.data_ptr()==key.data_ptr()+8192"""
    new=old+"""
            assert tuple(table.stride())==(8192,1)
            assert tuple(views['partial'].stride())==(32768,2048,128,1)
            assert all(tuple(views[k].stride())==(256,16,1) for k in ('maximum','summation'))
            record['binding']['actual_output_strides']={k:list(v.stride()) for k,v in views.items()}
            record['binding']['physical_rows']=n
            record['binding']['valid_rows']=args.valid_rows"""
    changes.append((old,new,1))
    text,proof=edit(text,changes);compile(text,'<PR118-adapted-original-consumer>','exec')
    proof['parent']=original_consumer.PARENT;return text,proof

def load(root,file,name):
    path=parent(root)/'tools'/(file+'.py');need(name not in sys.modules,'module already loaded '+name)
    sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    parent(root);return module

def fixture(module,physical_rows,valid_rows,segments,pitch):
    need((valid_rows,segments) in PAIRS and physical_rows in (valid_rows,32 if segments==4 else 64),'fixture scope')
    base=module.f.original
    logical=private_module(base,'cjj118_logical_fixture',CASES=((physical_rows,segments,pitch),))
    # Gold stays on the original16-column logical table; only the upload table
    # expands to the real8192 pitch. The entire unused physical table stays -1.
    result=types.SimpleNamespace(**vars(module.f));result.CASES=((physical_rows,segments,pitch),);result.COLS=8192
    def make(n,s,p,phase):
        case=logical.fixture(n,s,p,phase)
        for row in range(valid_rows,n):
            case['lengths'][logical.GUARD_ELEMENTS+row]=0
            a=logical.GUARD_ELEMENTS+row*logical.COLS
            case['table'][a:a+logical.COLS]=[-1]*logical.COLS
        need(all(x==0 for x in case['lengths'][logical.GUARD_ELEMENTS+valid_rows:-logical.GUARD_ELEMENTS]),'reference zero lengths')
        return case
    def inputs(case):
        import struct
        values=logical.input_bytes(case);table=[-1]*(2*logical.GUARD_ELEMENTS+physical_rows*8192)
        for row in range(physical_rows):
            a=logical.GUARD_ELEMENTS+row*logical.COLS;b=logical.GUARD_ELEMENTS+row*8192
            table[b:b+logical.COLS]=case['table'][a:a+logical.COLS]
        values['table']=struct.pack('<'+str(len(table))+'i',*table);return values
    def golden(case):
        # The exact old ideal/online-page algorithms are retained separately.
        from biv150_native_k8_pagewise_reference import from_inputs
        value=logical.golden(case)
        online=from_inputs(logical.input_bytes(case),physical_rows,segments,pitch)
        value['ideal_partial_diagnostic']=value['partial'];value['partial']=online['partial']
        return value
    result.fixture=make;result.input_bytes=inputs;result.golden=golden
    return result

def review_fields(r):
    need(r.get('approved') is True and all(r.get(k) is True for k in FLAGS),'actual static review pending')
    need(r.get('required_new_rows_segments')==[list(x) for x in PAIRS],'review exact six-row scope')
    need(set(r['resources'])=={'S2','S4'},'known S2/S4 resources')
    for x in r['resources'].values():
        for k in ('.vgpr_count','.sgpr_count'):
            need(type(x.get(k)) is int and x[k]>0,'unknown resource '+k)
        for k,v in (('.group_segment_fixed_size',8192),('.vgpr_spill_count',0),('.sgpr_spill_count',0),('.private_segment_fixed_size',0),('.warp_size',64)):
            need(type(x.get(k)) is int and x[k]==v,'resource '+k)

def admission(a):
    need(a.root_owned_exclusive,'Root must confirm original owner terminal and serial exclusion')
    root=parent(a.parent_source)
    for p in (a.compile_summary,a.assembly_review,a.library,a.main8b_library,a.codec_summary):need(p.is_absolute() and p.is_file(),'actual absolute input '+str(p))
    need(re.fullmatch('[0-9a-f]{64}',a.assembly_review_sha256 or '') and sha(a.assembly_review)==a.assembly_review_sha256,'external Root review SHA')
    j=json.loads(a.compile_summary.read_text());r=json.loads(a.assembly_review.read_text());review_fields(r)
    need(j['completed'] is True and j['compile_passed'] is True and not j['errors'],'actual successful compile')
    need(j['source_transform']==factory.verify() and j['explicit_host_flags']==build.HOST_FLAGS and j['explicit_cuda_flags']==build.CUDA_FLAGS,'actual exact source/options')
    need(j['actual_SO_export_completed'] is j['actual_symbol_binding_verified'] is True,'actual final SO exports/symbols')
    need(j['torch']['cuda_initialized_before'] is j['torch']['cuda_initialized_after'] is False,'compile-only CUDA state')
    for k in ('attention_launches','encoder_launches','codec_launches','reducer_launches','input_initializations'):need(j[k]==0,'actual compile counters')
    need(r['compile_summary_sha256']==sha(a.compile_summary) and r['source_commit']==j['source_commit'] and r['library']==j['library'],'review actual object binding')
    files={}
    for row in j['artifact_manifest']:
        rel=Path(row['relative']);need(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in files,'compile artifact path')
        p=a.compile_summary.parent/rel;need(p.resolve().is_relative_to(a.compile_summary.parent.resolve()) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'actual artifact SHA '+str(rel));files[str(rel)]=row
    for role in factory.FILES:
        p=factory.path(role);need((a.compile_summary.parent/p.name).read_bytes()==p.read_bytes(),'actual compile input '+role)
    lib=Path(j['library']['path']);need(lib.is_absolute() and lib.resolve().is_relative_to(a.compile_summary.parent.resolve()),'actual compiled library path')
    need(lib.name==factory.NAMESPACE+'.so' and sha(a.library)==j['library']['sha256']==sha(lib),'actual new SO bytes')
    rel=str(lib.resolve().relative_to(a.compile_summary.parent.resolve()))
    need(rel in files and files[rel]['sha256']==j['library']['sha256'],'actual SO in artifact manifest')
    for name in ('actual-SO-IR.stdout.txt','actual-SO-sass.stdout.txt'):
        need(name in files and files[name]['bytes']>0,'actual final SO export '+name)
    need(j['symbol_renames']==r['symbol_renames']==build.SYMBOL_RENAMES,'actual isolated symbol map')
    need(sha(a.main8b_library)==MAIN_SHA and sha(a.library)!=MAIN_SHA,'distinct original main8b and new mixed SO')
    need(sha(a.codec_summary)==a.codec_proof_sha256==CODEC_SHA,'original actual codec receipt')
    need(a.out.is_absolute() and not a.out.exists() and not a.out.resolve().is_relative_to(root) and not a.out.resolve().is_relative_to(factory.ROOT),'fresh independent out')
    return root,dict(compile_summary_sha256=sha(a.compile_summary),assembly_review_sha256=sha(a.assembly_review),
        actual_compile_source_root=j['source_root'],actual_compile_source_commit=j['source_commit'],validator_source_root=str(factory.ROOT),
        library=j['library'],main8b_library_sha256=MAIN_SHA,main8b_library_path=str(a.main8b_library),actual_static_resources=r['resources'],source_proof=factory.verify(),
        exclusion_authority='Root explicit serial coordination; no namespace-idle claim')

def safe_blob(root,m):
    p=Path(m['path']);need(not p.is_absolute() and '..' not in p.parts,'blob path');p=root/p
    need(p.resolve().is_relative_to(root.resolve()),'blob escape');raw=p.read_bytes()
    need(len(raw)==m['bytes'] and hashlib_sha(raw)==m['sha256'],'full blob SHA');return raw

def compare_records(candidate_root,candidate,reference_root,reference,valid_rows,pitch):
    need(len(candidate['records'])==len(reference['records'])==1,'one exact scoped case per arm')
    left,right=candidate['records'][0],reference['records'][0]
    need(left['case'][0]==valid_rows and right['case'][0] in (32,64) and left['case'][1:]==right['case'][1:],'candidate/reference scope')
    need(set(left['snapshots'])==set(right['snapshots'])==set(STAGES),'original full stage set')
    rowsize=dict(partial=16*16*128*4,maximum=16*16*4,summation=16*16*4,output=16*128*2)
    for stage in STAGES:
        a,b=left['snapshots'][stage],right['snapshots'][stage]
        for field,size in rowsize.items():
            ar=safe_blob(candidate_root,a[field]);br=safe_blob(reference_root,b[field]);guard=128 if field=='output' else 256
            need(ar[guard:guard+valid_rows*size]==br[guard:guard+valid_rows*size],f'all real-row OML/final bits {stage}/{field}')
        for field,rowbytes,guard in [('query',pitch*2,128),('lengths',4,256),('table',8192*4,256)]:
            ar=safe_blob(candidate_root,a[field]);br=safe_blob(reference_root,b[field])
            need(ar[guard:guard+valid_rows*rowbytes]==br[guard:guard+valid_rows*rowbytes],'same actual real-row input '+field)
        need(safe_blob(candidate_root,a['cache'])==safe_blob(reference_root,b['cache']),'same encoded cache bytes')
    return dict(valid_rows=valid_rows,reference_rows=right['case'][0],reference_zero_length_rows=right['case'][0]-valid_rows,
        stages=len(STAGES),all_real_OML_final_bits_equal=True,only_valid_rows_compared=True,reference_dummy_legality_checked_by_original_zero_and_poison_gates=True)

def plan():
    return dict(source=factory.verify(),parent_fixture=original_consumer.PARENT,accepted_main8b_sha256=MAIN_SHA,
        cases=CASES,table_pitch=8192,scratch_width=16,query_pitches=[2048,2560],candidate_padding=False,
        reference_only_padding=True,reference_physical_rows={'S2':64,'S4':32},codec_receipt_sha256=CODEC_SHA,
        actual_SO_static_review_required=FLAGS,producer_math_source='unchanged main8b; no117 lookahead',
        default_CPU=True,numeric_verified=False,precision_gate=False,performance_verified=False,service_verified=False)


def receipt_source(root):
    import probe_biv150_native_k8_v_overlap_fullstep as old
    root=old.parent_files(root);text=source_function(root,'biv150_native_k8_consumer_receipt','require_consumer')
    changes=[("expected_sha=='c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128' and f.sha(raw)==expected_sha","f.sha(raw)==expected_sha",1),
        ("len(j['records'])==4","len(j['records'])==len(f.CASES)",1),('cases=4,','cases=len(f.CASES),',1),
        ("('partial',(n,16,s,128)),('maximum',(n,16,s)),('summation',(n,16,s))",
         "('partial',(n,16,16,128)),('maximum',(n,16,16)),('summation',(n,16,16))",1)]
    a="""                    bits=np.frombuffer(middle,dtype='<u4').reshape(shape);active=np.broadcast_to(refs[p]['active'][:,None,:,None] if field=='partial' else refs[p]['active'][:,None,:],shape)
                    assert bool((bits[~active]==f.F32_POISON).all());assert f.metrics(bits.view('<f4')[active],refs[p][field][active])['passed']"""
    b="""                    bits=np.frombuffer(middle,dtype='<u4').reshape(shape);active=np.zeros(shape,dtype=bool)
                    active[:,:,:s]=np.broadcast_to(refs[p]['active'][:,None,:,None] if field=='partial' else refs[p]['active'][:,None,:],active[:,:,:s].shape)
                    wanted=active[:,:,:s]
                    assert bool((bits[~active]==f.F32_POISON).all());assert f.metrics(bits.view('<f4')[:,:,:s][wanted],refs[p][field][wanted])['passed']"""
    changes.append((a,b,1));return edit(text,changes)


def consumer_plan(actual):
    # JSON wire arrays are lists; the in-memory case declaration uses tuples.
    need(actual==json.loads(json.dumps(plan())), 'consumer exact serialized plan')


def require_actual_consumer(a,root,binding):
    need(a.consumer_summary.is_absolute() and re.fullmatch('[0-9a-f]{64}',a.consumer_sha256 or '') and sha(a.consumer_summary)==a.consumer_sha256,'actual complete consumer summary SHA')
    pair=json.loads(a.consumer_summary.read_text())
    need(pair['completed'] and pair['errors']==[] and pair['implementation_hard_passed'] and pair['all_real_OML_final_bitwise'],'actual consumer completion')
    same=lambda d:{k:v for k,v in d.items() if k!='validator_source_root'}
    need(same(pair['binding'])==same(binding),'consumer dual-root/source binding');consumer_plan(pair['plan'])
    need({tuple(x['case']) for x in pair['cases']}==set(CASES) and len(pair['cases'])==len(CASES),'complete twelve-case consumer scope')
    text,proof=receipt_source(root);replayed=[]
    for rec in pair['cases']:
        n,s,pitch=rec['case'];summaries={};folders={}
        for role,physical,digest in [('reference',32 if s==4 else 64,MAIN_SHA),('candidate',n,binding['library']['sha256'])]:
            info=rec['arms'][role];rel=Path(info['summary']);need(not rel.is_absolute() and '..' not in rel.parts,'consumer sub-summary path')
            path=a.consumer_summary.parent/rel;need(sha(path)==info['sha256'],'actual sub-summary SHA')
            j=json.loads(path.read_text());need(j['mixed_consumer_binding']==pair['binding'] and j['role']==role and j['mixed_case']==[n,s,pitch],'actual role/case provenance')
            namespace=MAIN_NAMESPACE if role=='reference' else factory.NAMESPACE
            need(j['schema'].startswith(namespace+'::produce(') and j['table_columns']==8192 and j['scratch_physical_width']==16,'actual role schema/layout')
            need(j['physical_rows']==physical and j['valid_rows']==n and j['reference_zero_length_rows']==physical-n,'actual physical/real row counts')
            for label,wanted,nsname in [('reference',MAIN_SHA,MAIN_NAMESPACE),('candidate',binding['library']['sha256'],factory.NAMESPACE)]:
                loaded=j['actual_dual_library_load'][label]
                need(loaded['sha256']==wanted and loaded['namespace']==nsname and loaded['loaded_library_set_verified'] is True,'actual dual library receipt '+label)
            module=load(root,'biv150_native_k8_consumer_receipt',f'cjj118_receipt_{n}_{pitch}_{role}')
            proxy=fixture(module,physical,n,s,pitch);proxy.LIBRARY_SHA=digest
            ns=dict(vars(module),f=proxy);exec(compile(text,'<118-exact-original-full-consumer-receipt>','exec'),ns)
            replayed.append(ns['require_consumer'](path,info['sha256']))
            summaries[role]=j;folders[role]=path.parent
        compare_records(folders['candidate'],summaries['candidate'],folders['reference'],summaries['reference'],n,pitch)
    need(not {'torch','triton'}.intersection(sys.modules),'receipt before Torch')
    return dict(summary_sha256=a.consumer_sha256,original_receipt_inverse=proof,full_cases_replayed=len(CASES),
        all_full_arrays_and_original_FP64_gates_replayed=True,old_validator_root=pair['binding']['validator_source_root'],
        current_validator_root=binding['validator_source_root'],no_performance_or_service_qualification=True)
