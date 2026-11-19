#!/usr/bin/env python3
"""Decode-step diagnostic only: shipping TL S4/S8 vs mixed Native S2/S4, not full mixed."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import biv150_native_k8_mixed_numeric_contract as c
from probe_biv150_native_k8_mixed_consumer import arguments,dispatch_binding
import probe_biv150_native_k8_v_overlap_fullstep as old

CASES={'4k':(61,4096,2,2048),'16k':(30,16384,4,2048)}
SPLITS={'tl':{'4k':4,'16k':8},'mixed':{'4k':2,'16k':4}}

def main_sources():
    files={}
    for role,name in [('tl','biv150_konly_encoded_decode.py'),('codec','biv150_konly_encoded_writer.py')]:
        p=c.factory.ROOT/'vllm_fl/patches'/name
        original=subprocess.check_output(['git','-C',str(c.factory.ROOT),'show',c.factory.BASE+':'+str(p.relative_to(c.factory.ROOT))])
        c.need(p.read_bytes()==original,'current accepted main TL source '+name);files[role]=dict(path=str(p),sha256=c.sha(p))
    return files

def load_tl(out):
    proof=main_sources();out=Path(out)
    import importlib.util
    name='cjj118_frozen_main_writer'
    path=Path(proof['codec']['path']);spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    source=Path(proof['tl']['path']).read_text();a='from vllm_fl.patches.biv150_konly_encoded_writer import quantize_bf16_rne';b='from cjj118_frozen_main_writer import quantize_bf16_rne'
    adapted,inverse=c.edit(source,[(a,b,1)])
    path=out/'frozen-main-tl.py';path.write_text(adapted)
    name='cjj118_frozen_main_tl';spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module._gqa_decode_konly_i8,dict(files=proof,import_only_inverse=inverse,service_installed=False)

def tl_produce(fn,z,q,lens,table,s):
    p,m,l=z['partial'],z['maximum'],z['summation']
    return fn[(q.shape[0],2,s)](segm_output_ptr=p,segm_max_ptr=m,segm_expsum_ptr=l,query_ptr=q,
        query_code_ptr=z['key'],query_scale_ptr=z['scale'],key_scale_ptr=z['scale'],key_cache_ptr=z['key'],value_cache_ptr=z['value'],
        seq_lens_ptr=lens,block_table_ptr=table,softmax_scale_log2e=128**-.5*1.4426950408889634,queries_per_kv=8,
        block_table_stride=table.stride(0),query_stride_token=q.stride(0),query_stride_head=128,
        k_stride_block=16384,k_stride_slot=256,k_stride_head=128,v_stride_block=8192,v_stride_slot=256,v_stride_head=128,
        segment_stride_token=p.stride(0),segment_stride_head=p.stride(1),segment_stride_segment=128,
        max_stride_token=m.stride(0),max_stride_head=m.stride(1),HEAD_SIZE=128,PAGE_SIZE=16,ROWS=8,SEGMENTS=s,num_warps=1,num_stages=1)

def actual_metadata(k):
    value=getattr(k,'metadata',None);c.need(value is not None,'actual loaded TL metadata missing')
    fields=value._asdict() if hasattr(value,'_asdict') else (value if isinstance(value,dict) else vars(value))
    c.need(isinstance(fields,dict) and bool(fields),'actual loaded metadata fields unknown')
    return json.loads(json.dumps(fields,default=repr))

def adapted_source(root):
    root=old.parent_files(root);original=c.source_function(root,'probe_biv150_native_k8_fullstep','run');edits=[]
    def change(a,b,count=1):edits.append((a,b,count))
    change('pre=require_consumer(a.consumer_summary,a.consumer_proof_sha256)','pre=ACTUAL_CONSUMER')
    change('f.sha(a.k8_library.read_bytes())==f.LIBRARY_SHA and f.sha(a.bf16_library.read_bytes())==f.BF16_SHA',
           'f.sha(a.k8_library.read_bytes())==NEW_SHA and f.sha(a.bf16_library.read_bytes())==MAIN_SHA')
    change('result=dict(f.plan(),actual_consumer=pre,','result=dict(step_plan(),actual_consumer=pre,')
    change("'fullstep-summary.json'","'decode-step-summary.json'",2)
    change("for path in (a.bf16_library,a.k8_library):torch.ops.load_library(str(path))", "_,result['actual_dual_library_load']=dispatch_binding(torch,a)\n        tl_kernel,result['actual_TL_source']=load_tl(out)")
    change("q=qb.as_strided((n,16,128),(pitch,128,1));nk=inputviews['new_k'];nv=inputviews['new_v'];", "q=qb.as_strided((n,16,128),(pitch,128,1));nk=inputviews['new_k'].as_strided((n,2,128),(256,128,1));nv=inputviews['new_v'].as_strided((n,2,128),(2560,128,1));")
    a="""                if arm=='bf16':
                    viewed=resident.view(torch.bfloat16).reshape(f.PAGES,2,16,2,128);key,value=viewed.unbind(1);scale=dummy
                else:
                    key=resident.view(torch.int8).as_strided((f.PAGES,16,2,128),(16384,256,128,1));scale=resident.view(torch.float32)
                    value=full.view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(8192,256,128,1),storage_offset=(f.GUARD+8192)//2)"""
    change(a,'\n'.join(line[4:] for line in a.splitlines()[3:]))
    change("if arm=='k8':assert key.data_ptr()","assert key.data_ptr()")
    change("wide if field=='output' else wide.narrow(2,0,s)","wide if field=='output' else wide.narrow(2,0,SPLITS[arm][case])")
    a="""                if arm=='bf16':arenas[arm]['raw'].copy_(gpu_source.view(torch.uint8).reshape(-1))
                else:
                    arenas[arm]['raw'].fill_(165)
                    handles['initial_encoder']=support.encode_initial[(f.PAGES*32,)](gpu_source[:,0],gpu_source[:,1],cases[arm]['key'],cases[arm]['scale'],cases[arm]['value'],f.PAGES,num_warps=1,num_stages=1)"""
    change(a,'\n'.join(line[4:] for line in a.splitlines()[2:]))
    a="""                if arm=='bf16':handles['bf16_append']=support.append_kv[(n,2)](nk,nv,slots,z['key'],dummy,z['value'],f.PAGES*16,False,num_warps=1,num_stages=1)
                else:handles['k8_append']=writer.encode_append[(n,2)](nk,nv,slots,z['key'],z['scale'],arenas[arm]['raw'].view(torch.bfloat16),nk.stride(0),nv.stride(0),nk.shape[0],nv.shape[0],slots.shape[0],f.PAGES,num_warps=1,num_stages=1)"""
    change(a,"                handles[arm+'_append']="+a.split("else:handles['k8_append']=")[1])
    a="""                z=cases[arm];args=(q,z['key'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634)
                if arm=='bf16':getattr(torch.ops,'corex_native_gqa_s'+str(s)+'_probe').produce(*args)
                else:torch.ops.cjj_native_k8_qk_probe.produce(q,z['key'],z['scale'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634,s)"""
    b="""                z=cases[arm];current_s=SPLITS[arm][case]
                if arm=='bf16':handles['tl_producer']=tl_produce(tl_kernel,z,q,lens,table,current_s)
                else:getattr(torch.ops,NEW_NAMESPACE).produce(q,z['key'],z['scale'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634,current_s)"""
    change(a,b)
    change("z=cases[arm];o,m,l=z['partial'],z['maximum'],z['summation'];handles['reducer']", "s=SPLITS[arm][case];z=cases[arm];o,m,l=z['partial'],z['maximum'],z['summation'];handles[arm+'_reducer']")
    change("initial=cold[start:start+count] if arm=='bf16' else packed[start:start+count]","initial=packed[start:start+count]")
    a="z=controls[p];return contract.qualify(descriptor(nk),descriptor(nv),descriptor(slots),descriptor(arenas['k8']['raw']),f.PAGES,contract.SlotSnapshot(tuple(int(x) for x in z['slots']),'component-owned-cpu-preupload',p))"
    change(a,"z=controls[p];return {arm:contract.qualify(descriptor(nk),descriptor(nv),descriptor(slots),descriptor(arenas[arm]['raw']),f.PAGES,contract.SlotSnapshot(tuple(int(x) for x in z['slots']),'component-owned-cpu-preupload',p)) for arm in ('bf16','k8')}")
    change("payload={};metrics={}\n                arrays=", "s=SPLITS[arm][case];payload={};metrics={}\n                arrays=")
    change("metrics[field]=f.metrics(values,gold[field]);assert", "metrics[field]=f.metrics(values,gold[field]);metrics['original_BF16_proxy']=f.metrics(values,gold['original_BF16_output']);assert")
    change("gold[arm,p]=f.golden(cold,packed,controls[p],s,arm)","gold[arm,p]=f.golden(cold,packed,controls[p],SPLITS[arm][case],arm)")
    start=original.index('            signature={};names=');end=original.index('            for role in (',start);block=original[start:end]
    wrapped=block.replace("handles['k8_append']","handles[writer_arm+'_append']")
    change(block,"            for writer_arm in ('bf16','k8'):\n"+''.join('    '+line+'\n' for line in wrapped.splitlines()))
    change("for role in ('k8_append','bf16_append','reducer'):","for role in ('tl_append','mixed_append','tl_producer','tl_reducer','mixed_reducer'):")
    change("type(handle.n_regs) is int and handle.n_regs>0 and handle.n_spills==0", "type(handle.n_regs) is int and handle.n_regs>0 and type(handle.n_spills) is int and handle.n_spills==0")
    change("record['artifacts']={ext:blob(","record['actual_metadata']=actual_metadata(k);record['actual_generated_source']=blob(k.src.fn.src.encode())\n            record['artifacts']={ext:blob(")
    change('event_ratio_BF16_over_K8','event_ratio_TL_over_mixed');change('host_wall_ratio_BF16_over_K8','host_wall_ratio_TL_over_mixed')
    change('BF16_GPU_pack','TL_GPU_full_encode');change('K8_GPU_full_encode','mixed_GPU_full_encode')
    text,proof=c.edit(original,edits)
    text,rename=c.edit(text,[("'bf16'","'tl'",text.count("'bf16'")),("'k8'","'mixed'",text.count("'k8'"))])
    proof['arm_literal_rename']=rename;proof['final_adapted_sha256']=c.hashlib_sha(text.encode());compile(text,'<118-original-decode-step>','exec');return text,proof

def fixture(base):
    f=SimpleNamespace(**vars(base));f.CASES=CASES;f.ORDER=('tl','mixed','mixed','tl')
    def control(n,length,pitch,phase):
        import numpy as np
        x=base.control(n,length,pitch,phase);wide=np.full((n,20,128),0x4255,dtype='<u2');wide[:,:2]=x['new_v'];x['new_v']=wide;return x
    def narrow(control):return dict(control,new_v=control['new_v'][:,:2].copy())
    def golden(cold,packed,control,s,arm):
        c.need(arm in ('tl','mixed'),'step arm');control=narrow(control)
        result=base.golden(cold,packed,control,s,'k8');result['original_BF16_output']=base.golden(cold,packed,control,s,'bf16')['output'];return result
    def patched(arm,start,initial,control):return base.patched_chunk('k8',start,initial,narrow(control))
    f.control=control;f.golden=golden;f.patched_chunk=patched;return f

def execute(a):
    root,binding=c.admission(a);root=old.parent_files(root)
    actual=c.require_actual_consumer(a,root,binding)
    source,proof=adapted_source(root);mod=c.load(root,'probe_biv150_native_k8_fullstep','cjj118_original_decode_step')
    f=fixture(mod.f);a.k8_library=a.library;a.bf16_library=a.main8b_library
    a.main_library=a.main8b_library;a.mixed_library=a.library;a.mixed_sha256=binding['library']['sha256'];a.namespace=c.factory.NAMESPACE
    def step_plan():
        return dict(c.plan(),component='decode-step only: K8 append+TL/native producer+reducer',
            decode_step_cases=CASES,per_arm_segments=SPLITS,binding=binding,source_adapter=proof,current_main_TL_sources=main_sources(),
            original_protocol=mod.f.plan(),clock_protocol=dict(order=f.ORDER,unroll=f.UNROLL,warm=f.WARM,replays=f.REPLAYS),
            full_mixed_step_timing=None,paired_prefill_included=False,host_metadata_ticket_included=False,
            all2048_token_writer_included=False,append_source_rows='decode N only',new_K_source_pitch=256,new_V_source_pitch=2560,
            Graph_numeric_is_isolated_component_not_natural_mixed_service=True,service_verified=False,precision_gate=False)
    ns=dict(vars(mod),f=f,ACTUAL_CONSUMER=actual,step_plan=step_plan,
        NEW_SHA=binding['library']['sha256'],MAIN_SHA=c.MAIN_SHA,NEW_NAMESPACE=c.factory.NAMESPACE,
        SPLITS=SPLITS,dispatch_binding=dispatch_binding,load_tl=load_tl,tl_produce=tl_produce,actual_metadata=actual_metadata)
    exec(compile(source,'<118-original-fullstep-functions-decode-only>','exec'),ns);return ns['run'](a)

def main():
    p=arguments();p.description=__doc__;p.add_argument('--consumer-summary',type=Path);p.add_argument('--consumer-sha256');p.add_argument('--case',choices=CASES,action='append');a=p.parse_args()
    if a.gpu:
        if not all(v for k,v in vars(a).items() if k not in ('gpu','case')):p.error('Root actual compile/static review plus full six-N consumer arrays required')
        if a.case and len(set(a.case))!=len(a.case):p.error('duplicate case')
        return execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('device arguments require --gpu')
    c.need(not {'torch','triton','numpy'}.intersection(sys.modules),'stdlib CPU default')
    print(json.dumps(dict(c.plan(),decode_step_cases=CASES,per_arm_segments=SPLITS,current_main_TL_sources=main_sources(),
        full_mixed_step_timing=None,paired_prefill_included=False,host_metadata_ticket_included=False,all2048_token_writer_included=False,
        original_3eb_numeric_and_ABBA_reused=True),sort_keys=True,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
