"""Finite source/counter/RPC payload CPU fixtures; never CUDA or service qualification."""
import argparse,ast,copy,hashlib,importlib.util,json,os,re,struct,sys,types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import biv150_hidden_gather_service_bridge as b
import biv150_hidden_gather_snapshot as s
import biv150_hidden_gather_service_observer as observer
import biv150_hidden_gather_path_identity as identity
ROOT=b.ROOT

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def relocate_artifacts(original,root):
    """Explicit synthetic CPU fixture, independently rewrite original ELF headers.

    This is NOT a compiled artifact or runtime admission path; no file is modified.
    """
    old=identity.OLD_ROOT.encode();new=root.encode();raw=original['cubin']
    h=list(struct.unpack_from('<16sHHIQQQIHHHHHH',raw));rows=[list(struct.unpack_from('<IIQQQQIIQQ',raw,h[6]+i*h[11])) for i in range(h[12])]
    data=[raw[r[4]:r[4]+r[5]] for r in rows];info=data[5].replace(old,new);data[5]=struct.pack('<I',len(info)-4)+info[4:]
    pos=h[5]+h[9]*h[10]
    for i,row in enumerate(rows[1:],1):
        pos+=(-pos)%row[8];row[4]=pos;row[5]=len(data[i]);pos+=row[5]
    h[6]=pos+(-pos)%8;out=bytearray(h[6]+h[11]*h[12]);out[:64]=struct.pack('<16sHHIQQQIHHHHHH',*h)
    out[h[5]:h[5]+h[9]*h[10]]=raw[h[5]:h[5]+h[9]*h[10]]
    for i,(row,d) in enumerate(zip(rows,data)):
        out[row[4]:row[4]+row[5]]=d;struct.pack_into('<IIQQQQIIQQ',out,h[6]+i*h[11],*row)
    result={k:v.replace(old,new) for k,v in original.items() if k!='cubin'};result['cubin']=bytes(out)
    result['asm']=re.sub(rb'(?m)^(\s*\.long\s+)171([ \t]+; Length of Unit)$',lambda m:m[1]+str(len(data[5])-4).encode()+m[2],result['asm'])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--component-out',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();checks=[]
    def ck(name,value):assert value,name;checks.append(name)
    def refuse(name,fn):
        try:fn()
        except (ValueError,RuntimeError,AssertionError):checks.append(name)
        else:raise AssertionError('accepted '+name)
    helper=load(ROOT/s.HELPER,'cpu122_helper');raw=(ROOT/s.HELPER).read_bytes()
    parent_raw=b.inverse_default_on(s.HELPER,raw)
    ck('only_two_helper_defaults_from_tested5b',parent_raw==b.git(ROOT,'show',b.DEFAULT_PARENT+':'+s.HELPER))
    restored=b.inverse(parent_raw,json.loads((ROOT/'tools/biv150_hidden_gather_diagnostic_edits.json').read_text()))
    ck('helper_exact_inverse_a101',restored==b.git(ROOT,'show',b.COMPONENT+':'+s.HELPER))
    ck('GPU_kernel_bytes_unchanged',b.sha(ROOT/s.KERNEL)==b.PINS['kernel_source_sha256'])
    entry=(ROOT/b.CONTROLLER).read_bytes();edits=json.loads((ROOT/'tools/biv150_hidden_gather_controller_edits.json').read_text())
    ck('entire_controller_inverse637d',b.inverse(entry,edits)==(ROOT/b.PARENT_CONTROLLER).read_bytes())
    for name in ('tools/biv150_runtime_stats_snapshot.py','tools/biv150_runtime_stats_controller.py','tools/runtime_stats_bootstrap/sitecustomize.py'):
        ck('119_unchanged_'+name,(ROOT/name).read_bytes()==b.git(ROOT,'show',b.MAIN+':'+name))
    ck('old_SME5_pins_preserved',(ROOT/'tools/biv150_hidden_gather_actual_pins.json').read_bytes()==b.git(ROOT,'show',b.NUMERIC_COMPONENT+':tools/biv150_hidden_gather_actual_pins.json'))
    for name in (s.HELPER,s.KERNEL,s.RUNNER,'vllm_fl/patches/biv150_compute_order.py',b.CONTROLLER):
        ck('exact_inverse7347_'+name,b.inverse_default_on(name,(ROOT/name).read_bytes())==b.git(ROOT,'show',b.NUMERIC_COMPONENT+':'+name))
    for name in b.PRODUCTION:
        ck('exact_production_inverse5b_'+name,b.inverse_default_on(name,(ROOT/name).read_bytes())==b.git(ROOT,'show',b.DEFAULT_PARENT+':'+name))
    for name,edits in b.DEFAULT_EDITS.items():
        for before,after in edits:
            reverted=(ROOT/name).read_bytes().replace(after.encode(),before.encode(),1)
            refuse('missing_default_or_OFF_edit_refused_'+after,lambda name=name,reverted=reverted:b.inverse_default_on(name,reverted))
    # Execute the real prepare path up to the existing pre-cold boundary.
    # These are explicitly fake CPU metadata, never SDK/cold/device qualification.
    parent=types.ModuleType('cpu122_tested5b_helper');parent.__file__=str(ROOT/s.HELPER);sys.modules[parent.__name__]=parent
    exec(compile(parent_raw,parent.__file__,'exec'),parent.__dict__)
    int32=object()
    class Array:
        dtype=int32;shape=(2,);flags=NS(c_contiguous=True)
        def __iter__(self):return iter((2,3))
    np=NS(ndarray=Array,int32=int32);torch=NS(cuda=NS(is_current_stream_capturing=lambda:False))
    branches=[]
    for module,flag,expected in ((parent,'1',1),(helper,None,1),(helper,'1',1),(helper,'0',0),(helper,'',0),(helper,'true',0)):
        calls=[]
        def cold(*args):calls.append('cold attempted');raise ValueError('CPU sentinel; not real cold')
        runner=NS(speculative_config=None,input_batch=NS(num_reqs=2))
        environment={} if flag is None else {helper.ENV:flag}
        with patch.dict(os.environ,environment,clear=True),patch.dict(sys.modules,torch=torch,numpy=np),patch.object(module,'_cold',cold):
            ck('prepare_branch_'+module.__name__+'_'+str(flag),module.prepare(runner,None,Array(),5,None,None) is None and len(calls)==expected)
        branches.append(dict(module=module.__name__,flag=flag,cold_attempts=len(calls)))
    with patch.dict(os.environ,{helper.ENV:'0'},clear=True):
        ck('explicit_OFF_does_not_touch_prepare_metadata',helper.prepare(object(),object(),object(),8,None,None) is None)
        receipt=helper._Receipt(object(),object(),(),8,(),'',{},0,helper._SEAL)
        ck('explicit_OFF_before_consumer_tensor_checks_or_launch',helper.try_gather(object(),object(),object(),receipt,True,None) is None and receipt.consumed)
    semantics=load(ROOT/'vllm_fl/compile_semantics.py','cpu122_default_semantics')
    config=NS(model_config=NS(hf_config=NS(**semantics._GEOMETRY)))
    env={'VLLM_FL_BIV150_NATIVE_W8A8':'0','VLLM_FL_BIV150_NATIVE_GQA':'0'}
    fingerprints=[semantics.build_fingerprint(config,'iluvatar',environ={**env,**extra},abi={'CPU':True}) for extra in ({},{helper.ENV:'1'},{helper.ENV:'0'})]
    ck('actual_default_and_explicit_ON_C3_equal_OFF_distinct',fingerprints[0]==fingerprints[1]!=fingerprints[2])
    ck('new_bootstrap_default_OFF',s.install({})=={'enabled':False})
    refuse('bad_token_refused',lambda:s.configuration({s.ENABLE:'1',s.TOKEN:'short',s.RUN_ID:'cpu'}))
    dirty={b.ENV:'1',b.DIAGNOSTICS:'1',s.ENABLE:'1',s.TOKEN:'never-export',s.RUN_ID:'cpu','CJJ_RUNTIME_STATS_SNAPSHOT':'1',
           'PYTHONPATH':os.pathsep.join((str(ROOT/'tools/hidden_gather_bootstrap'),str(ROOT/'tools/runtime_stats_bootstrap'),str(ROOT))),
           'VLLM_FL_BIV150_KONLY_ENCODED_SERVICE':'1','VLLM_FL_BIV150_KONLY_NATIVE_DECODE':'1'}
    clean=b.clear_environment(dirty);ck('OFF_client_explicit_zero_and_clear_private_keys',clean[b.ENV]=='0' and not any(k in clean for k in (*observer.PRIVATE_KEYS,'CJJ_RUNTIME_STATS_SNAPSHOT')) and clean['PYTHONPATH']==str(ROOT))
    ck('empty_probe_environment_also_explicit_OFF',b.clear_environment({})[b.ENV]=='0')
    enabled=b.service_environment(dirty,Path('/private/tmp/exact-reviewed-ebbc-placeholder.so'))
    ck('service_enables_gather_and_preserves_native',enabled[b.ENV]=='1' and enabled[b.old.ENV]=='1' and enabled['VLLM_FL_BIV150_KONLY_NATIVE_DECODE']=='1' and not any(k in enabled for k in observer.PRIVATE_KEYS))
    args=NS(baseline=False,q32_hoist=True,native_gqa_library=Path('/CPU-fixture/0c66.so'),native_gqa_sha256='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8',native_k8_library=Path('/CPU-fixture/5898.so'),native_k8_sha256=b.PARENT,
        mixed_native_library=Path('/CPU-fixture/ebbc.so'),mixed_native_sha256=b.MIXED,graph_coverage_bootstrap_dir=None,
        mixed_source_bridge=Path('/CPU-fixture/bridge.json'),mixed_source_bridge_sha256='0'*64,
        math105=False,supplemental_native_math105_64=False,throughput=False,gather_diagnostics=True)
    b.options(args);ck('plain_smoke_requires_and_accepts_diagnostics',True)
    args.gather_diagnostics=False;refuse('plain_without_diagnostics_refused',lambda:b.options(args))
    args.throughput=True;b.options(args);ck('performance_diagnostics_OFF',True)
    args.gather_diagnostics=True;refuse('performance_diagnostics_ON_refused',lambda:b.options(args))
    args.throughput=False;args.math105=True;refuse('Math_diagnostics_ON_refused',lambda:b.options(args))
    args.gather_diagnostics=False;b.options(args);ck('Math_diagnostics_OFF',True)
    transport_old=ast.parse((ROOT/'tools/biv150_runtime_stats_snapshot.py').read_bytes())
    transport_new=ast.parse((ROOT/'tools/biv150_hidden_gather_snapshot.py').read_bytes())
    for name in ('configuration','sdk_check','process_identity','install_worker','install_api','install'):
        old=next(x for x in transport_old.body if isinstance(x,ast.FunctionDef) and x.name==name)
        new=next(x for x in transport_new.body if isinstance(x,ast.FunctionDef) and x.name==name)
        ck('same_fixed119_transport_'+name,ast.dump(new).replace('_cjj_hidden_gather_wrapped','_cjj_runtime_stats_wrapped')==ast.dump(old))
    # Execute the actual five post-validation statements, not a rewritten counter model.
    fn=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef) and x.name=='try_gather')
    shell=ast.parse('def tail(torch,n,hidden,indices,state,receipt,_launch):\n pass').body[0]
    shell.body=copy.deepcopy(fn.body[-5:]);module=ast.fix_missing_locations(ast.Module(body=[shell],type_ignores=[]));ns={};exec(compile(module,'actual-helper-tail-CPU','exec'),ns)
    ck('tail_starts_actual_allocation_then_launch',isinstance(shell.body[0],ast.Assign) and 'torch.empty' in ast.unparse(shell.body[0]) and '_launch(' in ast.unparse(shell.body[1]))
    output=object();compiled=object();torch=NS(empty=lambda *x,**k:output);hidden=NS(dtype='BF16',device='CPU-fixture',shape=(8,2048))
    def call(state,source,launch=lambda *x:compiled):return ns['tail'](torch,3,hidden,object(),state,NS(source=source),launch)
    observed={k:0 for k in s.COUNTERS};observed.update(compiled={},overflow=0);state={'diagnostics':observed}
    for source in s.COUNTERS:ck('actual_success_counter_'+source,call(state,source) is output and observed[source]==1)
    ck('one_compiled_reference_retained',observed['compiled']=={id(compiled):compiled})
    before=dict(observed)
    def bad(*x):raise RuntimeError('CPU launch-failure fixture')
    refuse('launch_failure_propagates',lambda:call(state,s.COUNTERS[0],bad));ck('failed_launch_not_counted',observed==before)
    ck('diagnostics_OFF_preserves_return',call({'diagnostics':None},s.COUNTERS[0]) is output)
    for _ in range(6):call(state,s.COUNTERS[0],lambda *x:object())
    ck('compiled_reference_bound_and_explicit_overflow',len(observed['compiled'])==4 and observed['overflow']==3)
    # Reuse actual saved artifact bytes in an explicitly fake compiled-object CPU fixture.
    summary=json.loads((a.component_out/'helper-summary.json').read_bytes());rec=summary['records'][0]
    ck('actual_component_summary_sha',b.sha(a.component_out/'helper-summary.json')==b.PINS['summary_sha256'])
    def untype(v):
        t=v['type']
        if t in ('tuple','list'):return tuple(untype(x) for x in v['items']) if t=='tuple' else [untype(x) for x in v['items']]
        if t=='dict':return {untype(k):untype(x) for k,x in v['items']}
        return v['value']
    function=NS(fn=NS(__code__=NS(co_filename=str(ROOT/s.KERNEL))));src=NS(fn=function,**{k:untype(v) for k,v in b.PINS['ABI']['typed'].items()})
    folder=a.component_out/'actual-8-ordinary'
    if not folder.is_dir():folder=next(p.parent for p in a.component_out.glob('actual-*-ordinary/actual.cubin'))
    original={ext:(folder/('actual.'+ext)).read_bytes() for ext in identity.EXTS}
    ck('original_reference_self_identity',identity.describe(original,identity.OLD_ROOT)['verified'])
    for name,root in [('same_length',identity.OLD_ROOT[:-1]+'x'),('longer','/CPU-fixture/'+'x'*100),('actual_source_directory',str(ROOT))]:
        proof=identity.describe(relocate_artifacts(original,root),root)
        ck('synthetic_path_only_'+name,proof['verified'] and proof['canonical_files']==b.PINS['files'])
    synthetic=relocate_artifacts(original,str(ROOT));proof=identity.describe(synthetic,str(ROOT));ck('raw_SHA_not_overwritten',proof['raw_files']!=proof['canonical_files'])
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',synthetic['cubin']);rows=[struct.unpack_from('<IIQQQQIIQQ',synthetic['cubin'],h[6]+i*64) for i in range(12)]
    for name,offset in [('text',rows[3][4]),('descriptor',rows[2][4]),('note',rows[1][4]),('symbols',rows[9][4]),('debug_frame',rows[6][4]),('debug_line',rows[7][4]),('debug_abbrev',rows[4][4]),('padding',1224),('debug_ALLOC',h[6]+5*64+8),('PH',64+56+4),('section_offset',h[6]+8*64+24)]:
        changed=dict(synthetic);data=bytearray(changed['cubin']);data[offset]^=1;changed['cubin']=bytes(data)
        ck('reject_nonpath_ELF_'+name,not identity.describe(changed,str(ROOT))['verified'])
    changed=dict(synthetic);changed['cubin']+=b'\0';ck('reject_ELF_trailing_byte',not identity.describe(changed,str(ROOT))['verified'])
    changed=dict(synthetic);changed['asm']=changed['asm'].replace(b'; Length of Unit',b'; wrong length marker');ck('reject_ASM_length_disagreement',not identity.describe(changed,str(ROOT))['verified'])
    changed=dict(synthetic);changed['ttir']+=b'\n';ck('reject_other_IR_difference',not identity.describe(changed,str(ROOT))['verified'])
    ck('reject_wrong_source_root',not identity.describe(synthetic,str(ROOT)+'-wrong')['verified'])
    owner=NS(__file__=str(folder/'actual-launcher.so'))
    k=NS(src=src,hash=b.PINS['hash'],metadata=b.PINS['options'],n_regs=6,n_spills=0,
        asm=synthetic,
        _run=NS(launch=types.MethodType(lambda self:None,owner)))
    binding=rec['actual_handler']['C_function_binding']
    refuse('Python_function_is_not_C_pointer',lambda:s.loaded_launcher_binding(lambda:None))
    # Actual Linux binding saved on device, injected ONLY at the resolver boundary.
    # This CPU fixture does not load that ELF or claim a live device C pointer.
    resolver=patch.object(s,'loaded_launcher_binding',lambda launch:copy.deepcopy(binding))
    with resolver:row=s.compiled_record(k,function)
    function.fn.__code__.co_filename='/CPU-fixture/foreign_kernel.py'
    with patch.object(s,'loaded_launcher_binding',lambda launch:copy.deepcopy(binding)):rejected=s.compiled_record(k,function)
    ck('actual_source_path_reject_keeps_raw_fields',not rejected['source_path_identity']['verified'] and rejected['files']==row['files'])
    function.fn.__code__.co_filename=str(ROOT/s.KERNEL)
    ck('actual_artifact_schema_matches_review_with_explicit_CPU_address_fixture',observer.compiled_verdict(row))
    changed=copy.deepcopy(row);changed['launcher_binding']['legacy_module_path']='/CPU-fixture/foreign-module-label';changed['launcher_binding']['legacy_module_sha256']='0'*64
    ck('legacy_module_label_does_not_define_C_ownership',observer.compiled_verdict(changed))
    for field,value in [('DSO_sha256','0'*64),('DSO_offset',5217),('method_flags',True),('method_name','other'),('DSO_path','relative.so')]:
        changed=copy.deepcopy(row);changed['launcher_binding'][field]=value;ck('C_binding_refuses_'+field,not observer.compiled_verdict(changed))
    changed=copy.deepcopy(row);changed.pop('launcher_binding');ck('old_module_only_row_refused',not observer.compiled_verdict(changed))
    changed=copy.deepcopy(row);changed['files']['asm']['sha256']='0'*64;ck('cold_ASM_byte_change_refused',not observer.compiled_verdict(changed))
    changed=copy.deepcopy(row);changed['files']['cubin']['sha256']='0'*64;ck('changed_cubin_refused',not observer.compiled_verdict(changed))
    changed=copy.deepcopy(row);changed['ABI']['constants']['5']=16;ck('changed_ABI_refused',not observer.compiled_verdict(changed))
    refuse('foreign_JIT_object_refused',lambda:s.compiled_record(k,object()))
    # Real snapshot function with labelled CPU worker metadata fixtures, no CUDA import/call.
    Runner=type('ModelRunnerFL',(),{'__module__':'vllm_fl.worker.model_runner',
       'execute_model':lambda self:None,'_prepare_inputs':lambda self:None,'_get_cumsum_and_arange':lambda self:None})
    Worker=type('WorkerFL',(),{'__module__':'vllm_fl.worker.worker'});runner=Runner();worker=Worker();worker.model_runner=runner
    runner.prepare_inputs_event=object();wrapper=NS(worker=worker,rpc_rank=0)
    native={'enabled':True,'library':{'path':'CPU-fixture-5898','sha256':b.PARENT}}
    enc=NS(registry=NS(armed=True,entries={1:True}),actual_target=['corex',71,64],attribute_bindings=[(Runner,'initialize_kv_cache',object())],final_export=None,
           stats=lambda:{'native_decode':native,'q32_hoist':{'enabled':True}},mixed_native=NS(library={'path':'CPU-fixture-ebbc','sha256':b.MIXED}),installed_digest='CPU-fixture-encoded')
    modules={'vllm_fl.patches.biv150_dynamic_hidden_gather':helper,'vllm_fl.worker.model_runner':NS(__file__=str(ROOT/s.RUNNER)),
        'vllm_fl.patches.biv150_compute_order':NS(EXPECTED_RUNNER=b.sha(ROOT/s.RUNNER)),
        s.SERVICE:NS(__file__=str(ROOT/'vllm_fl/patches/biv150_konly_encoded_service.py'),_STATE=enc),
        'vllm_fl.patches.biv150_dynamic_hidden_gather_kernel':NS(__file__=str(ROOT/s.KERNEL),gather_hidden_bits=function)}
    args=dict(expected_run='cpu',source_sha256=b.sha(s.__file__),sdk_pins=s.SDK_PINS)
    with patch.dict(sys.modules,modules),patch.dict(os.environ,{b.ENV:'1',b.DIAGNOSTICS:'1'}),patch.object(s,'process_identity',lambda:(101,201)),patch.object(s,'loaded_launcher_binding',lambda launch:copy.deepcopy(binding)):
        refuse('worker_missing_refused',lambda:s.snapshot(NS(),'smoke-before','cpu',**args))
        before=json.loads(s.snapshot(wrapper,'smoke-before','cpu',**args));ck('before_real_cold_has_zero_not_fabricated_success',before['counts']==dict.fromkeys(s.COUNTERS,0) and not before['cold_initialized'])
        diagnostics={s.COUNTERS[0]:2,s.COUNTERS[1]:3,'compiled':{id(k):k},'overflow':0}
        setattr(runner,helper._STATE,dict(codes=helper._codes(runner),event=runner.prepare_inputs_event,diagnostics=diagnostics))
        after=json.loads(s.snapshot(wrapper,'smoke-after','cpu',**args));d=observer.delta(before,after)
        ck('same_worker_both_sources_delta',d['both_sources_submitted'] and d['counts']==dict(zip(s.COUNTERS,(2,3))))
        ck('snapshot_reads_only_supplied_compiled_object',after['compiled']==[row])
        wrong=copy.deepcopy(after);wrong['start_ticks']+=1;refuse('worker_epoch_change_refused',lambda:observer.delta(before,wrong))
        wrong=copy.deepcopy(after);wrong['counts'][s.COUNTERS[0]]=-1;refuse('counter_regression_refused',lambda:observer.delta(before,wrong))
        diagnostics['overflow']=1;refuse('compiled_overflow_refused',lambda:s.snapshot(wrapper,'smoke-after','cpu',**args));diagnostics['overflow']=0
    ck('no_GPU_framework_import',not {'torch','triton','vllm'}.intersection(sys.modules))
    report=dict(passed=True,CPU_only=True,checks=checks,count=len(checks),source_commit=b.git(ROOT,'rev-parse','HEAD').decode().strip(),
        files={p:b.sha(ROOT/p) for p in (*b.PRODUCTION,*b.TOOLS)},script_sha256=b.sha(__file__),
        CPU_fake_worker_and_compiled_objects=True,CPU_C_address_resolver_stub=True,actual_saved_artifact_bytes_used=True,synthetic_debug_path_fixtures_not_device_compile=True,service_started=False,GPU_executed=False,service_verified=False,
        default_parent=b.DEFAULT_PARENT,pre_cold_branches=branches,default_equals_explicit_ON_fingerprint=True,real_unset_worker_verified=False,
        scope='Actual post-launch counter fragment and snapshot function; fake worker/Tensor metadata are not natural service or CUDA execution.')
    assert a.out.is_absolute() and not a.out.exists();a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(b.sha(a.out))
if __name__=='__main__':main()
