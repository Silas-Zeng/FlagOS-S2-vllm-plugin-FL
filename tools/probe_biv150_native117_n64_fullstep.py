#!/usr/bin/env python3
"""Default CPU; original 3eb width16 full-step ABBA, two 16KiB encoded readers."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import biv150_native117_pair_contract as c
from probe_biv150_native117_n64_consumer import arguments

def adapted_source(root):
    original=c.function(root,'probe_biv150_native_k8_fullstep','run')
    edits=[]
    def change(old,new,count=1):edits.append((old,new,count))
    change('pre=require_consumer(a.consumer_summary,a.consumer_proof_sha256)','pre=ACTUAL_PAIR')
    change('f.sha(a.k8_library.read_bytes())==f.LIBRARY_SHA and f.sha(a.bf16_library.read_bytes())==f.BF16_SHA',
           'f.sha(a.k8_library.read_bytes())==NEW_SHA and f.sha(a.bf16_library.read_bytes())==MAIN_SHA')
    change('result=dict(f.plan(),actual_consumer=pre,','result=dict(fullstep_plan(),actual_consumer=pre,')
    old="""                if arm=='bf16':
                    viewed=resident.view(torch.bfloat16).reshape(f.PAGES,2,16,2,128);key,value=viewed.unbind(1);scale=dummy
                else:
                    key=resident.view(torch.int8).as_strided((f.PAGES,16,2,128),(16384,256,128,1));scale=resident.view(torch.float32)
                    value=full.view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(8192,256,128,1),storage_offset=(f.GUARD+8192)//2)"""
    new='\n'.join(x[4:] for x in old.splitlines()[3:])
    change(old,new)
    change("if arm=='k8':assert key.data_ptr()","assert key.data_ptr()")
    old="""                if arm=='bf16':arenas[arm]['raw'].copy_(gpu_source.view(torch.uint8).reshape(-1))
                else:
                    arenas[arm]['raw'].fill_(165)
                    handles['initial_encoder']=support.encode_initial[(f.PAGES*32,)](gpu_source[:,0],gpu_source[:,1],cases[arm]['key'],cases[arm]['scale'],cases[arm]['value'],f.PAGES,num_warps=1,num_stages=1)"""
    change(old,'\n'.join(x[4:] for x in old.splitlines()[2:]))
    old="""                if arm=='bf16':handles['bf16_append']=support.append_kv[(n,2)](nk,nv,slots,z['key'],dummy,z['value'],f.PAGES*16,False,num_warps=1,num_stages=1)
                else:handles['k8_append']=writer.encode_append[(n,2)](nk,nv,slots,z['key'],z['scale'],arenas[arm]['raw'].view(torch.bfloat16),nk.stride(0),nv.stride(0),nk.shape[0],nv.shape[0],slots.shape[0],f.PAGES,num_warps=1,num_stages=1)"""
    new="                handles[arm+'_append']="+old.split('else:handles[\'k8_append\']=')[1]
    change(old,new)
    old="""                z=cases[arm];args=(q,z['key'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634)
                if arm=='bf16':getattr(torch.ops,'corex_native_gqa_s'+str(s)+'_probe').produce(*args)
                else:torch.ops.cjj_native_k8_qk_probe.produce(q,z['key'],z['scale'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634,s)"""
    change(old,"                z=cases[arm];getattr(torch.ops,PAIR_NAMESPACES[arm]).produce(q,z['key'],z['scale'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634,s)")
    change("initial=cold[start:start+count] if arm=='bf16' else packed[start:start+count]","initial=packed[start:start+count]")
    old="z=controls[p];return contract.qualify(descriptor(nk),descriptor(nv),descriptor(slots),descriptor(arenas['k8']['raw']),f.PAGES,contract.SlotSnapshot(tuple(int(x) for x in z['slots']),'component-owned-cpu-preupload',p))"
    change(old,"z=controls[p];return {arm:contract.qualify(descriptor(nk),descriptor(nv),descriptor(slots),descriptor(arenas[arm]['raw']),f.PAGES,contract.SlotSnapshot(tuple(int(x) for x in z['slots']),'component-owned-cpu-preupload',p)) for arm in ('bf16','k8')}")
    # Replay the exact original writer ABI validator separately for both actual
    # append handles; only the handle role lookup is changed.
    begin=original.index('            signature={};names=');end=original.index('            for role in (',begin)
    block=original[begin:end]
    changed=block.replace("handles['k8_append']","handles[writer_arm+'_append']")
    change(block,"            for writer_arm in ('bf16','k8'):\n"+''.join('    '+x+'\n' for x in changed.splitlines()))
    change("for role in ('k8_append','bf16_append','reducer'):","for role in ('main8b_append','lookahead_append','reducer'):")
    change('            # Actual append/reducer identity',
           "            rec['cross_arm_full_OML_final_bitwise']=cross_arm(rec)\n            # Actual append/reducer identity")
    change("            rec['warm']={}",
           "            assert rec['timing_bitwise_reference']['bf16']==rec['timing_bitwise_reference']['k8'],'Cross-arm full timing arrays'\n            rec['warm']={}")
    change('event_ratio_BF16_over_K8','event_ratio_main8b_over_lookahead')
    change('host_wall_ratio_BF16_over_K8','host_wall_ratio_main8b_over_lookahead')
    change('BF16_GPU_pack','main8b_GPU_full_encode')
    change('K8_GPU_full_encode','lookahead_GPU_full_encode')
    change('result.update(completed=True,numerical_verified=True,',
           'result.update(completed=True,numerical_verified=True,numeric_verified=True,')
    text,proof=c.edit(original,edits)
    # Rename only exact arm string literals. Dtype strings such as '*bf16',
    # reference source bytes, library constants and all timing math are retained.
    renames=[("'bf16'","'main8b'",text.count("'bf16'")),("'k8'","'lookahead'",text.count("'k8'"))]
    text,rename_proof=c.edit(text,renames)
    inverse=text
    for old,new,count in reversed(renames):inverse=inverse.replace(new,old)
    for old,new,count in reversed(edits):
        assert inverse.count(new)==count;inverse=inverse.replace(new,old)
    assert inverse==original
    proof['arm_rename']=rename_proof;proof['final_adapted_sha256']=c.factory.sha(text.encode())
    return text,proof

def cross_arm(record):
    values={role:{(x['phase'],x['stage']):x['full_arrays'] for x in record['numeric'] if x['arm']==role} for role in c.ROLES}
    assert set(values['main8b'])==set(values['lookahead'])=={
        (0,'initial_eager'),(0,'new_eager_or_repeat'),
        (1,'changed_graph_before_eager'),(1,'new_eager_or_repeat'),
        (2,'changed_side_before_eager'),(2,'new_eager_or_repeat')}
    for key in values['main8b']:
        assert set(values['main8b'][key])=={'partial','maximum','summation','output'}
        assert values['main8b'][key]==values['lookahead'][key],('Cross-arm full OML/final bits',key)
    return dict(phases=3,stages_per_arm=6,all_full_arrays_bitwise=True)

def fixture(original):
    proxy=SimpleNamespace(**vars(original))
    proxy.CASES={'4k':original.CASES['4k']}
    proxy.ORDER=('main8b','lookahead','lookahead','main8b')
    def patched(arm,*args):
        assert arm in c.ROLES;return original.patched_chunk('k8',*args)
    def golden(*args):
        assert args[-1] in c.ROLES;return original.golden(*args[:-1],'k8')
    proxy.patched_chunk=patched;proxy.golden=golden
    return proxy

def execute(a):
    root,binding=c.admission(a)
    assert a.case in (None,'4k')
    assert a.consumer_pair_summary.is_absolute() and len(a.consumer_pair_sha256 or '')==64
    actual=c.require_pair(a.consumer_pair_summary,a.consumer_pair_sha256,root,binding)
    assert not {'torch','triton'}.intersection(sys.modules)
    source,adapter=adapted_source(root)
    module=c.load(root,'probe_biv150_native_k8_fullstep','cjj117_fullstep_parent')
    f=fixture(module.f);assert f.CASES['4k']==(64,4096,2,2048)
    a.k8_library=a.library;a.bf16_library=a.main8b_library;a.case=['4k']
    def fullstep_plan():
        return dict(c.plan(),component='current main8b vs117 complete same-format component',
            fullstep_adapter=adapter,fullstep_binding=binding,fullstep_protocol=dict(
                order=f.ORDER,unroll=f.UNROLL,warm=f.WARM,replays=f.REPLAYS,scratch_builder_width=16,
                arena_pages=f.PAGES,arena_bytes=f.PAGES*16384,writer='same exact PR83 both arms',
                reducer='same exact original b3 stride-aware reducer both arms',
                partial_reference='unchanged FP32 score/rho, page16 RN32/BF16 P implementation oracle',
                final_reference='unchanged ideal quantized-score FP64 .005 both arms'),
            original_protocol=f.plan(),numeric_verified=False,timing_executed=False)
    ns=dict(vars(module),f=f,fullstep_plan=fullstep_plan,ACTUAL_PAIR=actual,
        NEW_SHA=c.NEW_SHA,MAIN_SHA=c.MAIN_SHA,cross_arm=cross_arm,
        PAIR_NAMESPACES={role:value[1] for role,value in c.ROLES.items()})
    exec(compile(source,'<117-original-width16-complete-step-pair>','exec'),ns)
    return ns['run'](a)

def main():
    p=arguments();p.description=__doc__
    p.add_argument('--consumer-pair-summary',type=Path);p.add_argument('--consumer-pair-sha256')
    p.add_argument('--case',choices=('4k',));a=p.parse_args()
    if a.gpu:
        if not all(v for k,v in vars(a).items() if k not in ('gpu','case')):p.error('All actual receipts, both libraries, replayable consumer pair and exclusive Root schedule required')
        return execute(a)
    if any(v for k,v in vars(a).items() if k!='gpu'):p.error('Device arguments require --gpu')
    assert not {'torch','triton','numpy'}.intersection(sys.modules)
    print(json.dumps(dict(c.plan(),fullstep_entry=True,new_reference_framework=False,
        existing_3eb_fullstep_body_reused=True,actual_pair_replay_required=True,
        existing_ABBA_unroll8_warm128_replays64=True),indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
