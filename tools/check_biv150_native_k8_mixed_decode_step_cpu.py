#!/usr/bin/env python3
"""CPU source/protocol checks; optional tiny NumPy oracle-layout consistency, never GPU."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import probe_biv150_native_k8_mixed_decode_step as b

def fn(text,name):
    tree=ast.parse(text);node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name)
    return ast.get_source_segment(text,node)

def run(root,arrays=False):
    source,proof=b.adapted_source(root);receipt,receipt_proof=b.c.receipt_source(root)
    original=b.c.source_function(root,'probe_biv150_native_k8_fullstep','run')
    assert fn(source,'timed')==fn(original,'timed')
    assert "for arm in f.ORDER:" in source and "f.REPLAYS" in source
    assert 'graphB' not in source or 'changed_graph_before_eager' in source
    roles=('tl_append','mixed_append','tl_producer','tl_reducer','mixed_reducer')
    assert 'for role in '+repr(roles)+':' in source.replace("','", "', '")
    assert "s=SPLITS[arm][case]" in fn(source,'capture') and "s=SPLITS[arm][case]" in fn(source,'reducer')
    assert 'SPLITS[arm][case]' in source and "current_s=SPLITS[arm][case]" in fn(source,'producer')
    assert 'original_BF16_proxy' in fn(source,'capture')
    # Only the per-arm S selection and handle bookkeeping differ from3eb.
    assert fn(source,'reducer').replace('s=SPLITS[arm][case];','').replace("handles[arm+'_reducer']","handles['reducer']")==fn(original,'reducer')
    assert "handles['reducer']" not in source
    # Exercise the unchanged compiled exporter with two distinct fake JITs.
    # These are CPU evidence of bookkeeping, never device artifact receipts.
    def blob(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def handle(name):
        return SimpleNamespace(hash=name,n_regs=1,n_spills=0,metadata={'name':name},
            src=SimpleNamespace(fn=SimpleNamespace(src='same reducer source'),signature={0:'*fp32'}),
            asm={'cubin':name.encode(),'llir':('IR '+name).encode()})
    namespace=dict(f=SimpleNamespace(sha=lambda raw:hashlib.sha256(raw).hexdigest()),blob=blob,actual_metadata=b.actual_metadata)
    exec(fn(source,'compiled'),namespace)
    handles={role:handle(role) for role in roles};exports={role:namespace['compiled'](value) for role,value in handles.items()}
    assert exports['tl_reducer']['artifacts']['cubin']!=exports['mixed_reducer']['artifacts']['cubin']
    for role in ('tl_reducer','mixed_reducer'):
        assert exports[role]['actual_metadata']['name']==role
        assert exports[role]['actual_generated_source']==blob(b'same reducer source')
    gate=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
        and n.target.id=='role' and isinstance(n.iter,ast.Tuple) and ast.literal_eval(n.iter)==roles)
    gate_code=compile(ast.fix_missing_locations(ast.Module(body=[gate],type_ignores=[])),'<original-resource-gate>','exec')
    exec(gate_code,{'handles':handles})
    rejections=[]
    for role in ('tl_reducer','mixed_reducer'):
        for field,value in [('n_regs',None),('n_regs',0),('n_spills',1),('n_spills',None),('n_spills',False)]:
            old=getattr(handles[role],field);setattr(handles[role],field,value)
            try:exec(gate_code,{'handles':handles})
            except AssertionError:rejections.append([role,field,value])
            else:raise AssertionError('unknown/nonzero reducer resource accepted')
            finally:setattr(handles[role],field,old)
    del handles['tl_reducer']
    try:exec(gate_code,{'handles':handles})
    except KeyError:pass
    else:raise AssertionError('missing TL reducer accepted')
    assert 'all2048_token_writer_included=False' in Path(b.__file__).read_text()
    b.c.consumer_plan(json.loads(json.dumps(b.c.plan())))
    mutant=json.loads(json.dumps(b.c.plan()));mutant['cases'][0][0]=59
    try:b.c.consumer_plan(mutant)
    except ValueError:pass
    else:raise AssertionError('wrong consumer case accepted')
    assert b.actual_metadata(type('Loaded',(),{'metadata':{'num_warps':1,'shared':8192}})())=={'num_warps':1,'shared':8192}
    try:b.actual_metadata(type('Unknown',(),{'metadata':None})())
    except ValueError:pass
    else:raise AssertionError('unknown actual metadata accepted')
    source_pins=b.main_sources();assert b.SPLITS=={'tl':{'4k':4,'16k':8},'mixed':{'4k':2,'16k':4}}
    record=dict(passed=True,CPU_only=True,source_adapter=proof,receipt_adapter=receipt_proof,
        original_timed_function_byte_exact=True,consumer_plan_JSON_roundtrip_and_wrong_case_refusal=True,
        loaded_metadata_export_and_unknown_refusal=True,actual_main_TL_files=source_pins,major_cases=b.CASES,per_arm_splits=b.SPLITS,
        reducer_call_inverse_byte_exact=True,independent_reducer_exports_CPU_only={k:exports[k] for k in ('tl_reducer','mixed_reducer')},
        independent_reducer_resource_refusals=rejections,missing_TL_reducer_refused=True,
        actual_new_SO_or_GPU_verified=False,numeric_verified=False,performance_verified=False,service_verified=False,
        paired_prefill_included=False,host_ticket_included=False,full_mixed_step_timing=None)
    if arrays:
        import numpy as np
        module=b.c.load(root,'probe_biv150_native_k8_fullstep','cpu118_original_step_helpers')
        base=module.f;proxy=b.fixture(base)
        original_control=base.control(3,33,2048,0);changed=proxy.control(3,33,2048,0)
        assert np.array_equal(changed['new_v'][:,:2],original_control['new_v'])
        assert changed['new_k'].strides==(512,256,2) and changed['new_v'].strides==(5120,256,2)
        assert bool((changed['new_v'][:,2:]==0x4255).all())
        cold,*aux=base.chunk_sources(0,7);packed=base.encode_chunk(0,cold,*aux,base.codebook())
        changed['lengths']=np.array([0,17,33],dtype='<i4')
        changed['table'].fill(-1)
        for row,L in enumerate(changed['lengths']):changed['table'][row,:(int(L)+15)//16]=np.arange((int(L)+15)//16)%7
        changed['slots']=np.array([-1 if not L else int(changed['table'][row,(int(L)-1)//16])*16+(int(L)-1)%16 for row,L in enumerate(changed['lengths'])],dtype='<i8')
        narrow=dict(changed,new_v=changed['new_v'][:,:2].copy())
        for arm in ('tl','mixed'):
            for s in (2,4,8):
                got=proxy.golden(cold,packed,changed,s,arm);wanted=base.golden(cold,packed,narrow,s,'k8')
                for k in wanted:assert np.array_equal(got[k],wanted[k],equal_nan=True)
                assert np.array_equal(got['original_BF16_output'],base.golden(cold,packed,narrow,s,'bf16')['output'])
            assert np.array_equal(proxy.patched_chunk(arm,0,packed,changed),base.patched_chunk('k8',0,packed,narrow))
        record['optional_NumPy_helper_consistency']=dict(passed=True,tiny_pages=7,rows=3,not_device_shape_qualification=True,
            exact_original_K8_and_BF16_oracles_preserved=True,K256_V2560_input_layout=True,
            complete_original_patched_chunk_equal=True)
    else:assert 'numpy' not in sys.modules
    assert not {'torch','triton','vllm_fl'}.intersection(sys.modules)
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent-source',type=Path,required=True);p.add_argument('--arrays',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.parent_source,a.arrays),sort_keys=True,indent=2))
