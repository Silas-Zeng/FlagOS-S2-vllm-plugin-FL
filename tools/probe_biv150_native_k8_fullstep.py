#!/usr/bin/env python3
"""CPU default; explicit full-step native pair after actual consumer qualification."""
import argparse
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import biv150_native_k8_fullstep_fixture as f
from biv150_native_k8_consumer_receipt import require_consumer


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result


def run(a):
    pre=require_consumer(a.consumer_summary,a.consumer_proof_sha256)
    assert f.sha(a.k8_library.read_bytes())==f.LIBRARY_SHA and f.sha(a.bf16_library.read_bytes())==f.BF16_SHA
    f.pins();assert not os.environ.get('TRITON_KERNEL_OVERRIDE') and not os.environ.get('TRITON_OVERRIDE_DIR'),'external IR override not allowed'
    out=a.out.resolve();assert not out.exists();out.mkdir(parents=True);(out/'blobs').mkdir()
    result=dict(f.plan(),actual_consumer=pre,completed=False,errors=[],records=[],performance_acceptance=False)
    def save(path,j):path.write_text(json.dumps(j,sort_keys=True,indent=2)+'\n')
    checkpoint=lambda:save(out/'fullstep-summary.json',result)
    def blob(raw):
        digest=f.sha(raw);path=out/'blobs'/(digest+'.bin')
        if not path.exists():path.write_bytes(raw)
        return dict(path=str(path.relative_to(out)),sha256=digest,bytes=len(raw))
    def npblob(v):
        b=io.BytesIO();np.save(b,v,allow_pickle=False);return blob(b.getvalue())
    checkpoint()
    try:
        import numpy as np
        import torch
        os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache');torch.set_num_threads(4)
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target();assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        props=torch.cuda.get_device_properties(0);assert 'BI-V150' in props.name and props.multi_processor_count==16
        assert torch.__version__=='2.10.0'
        for path in (a.bf16_library,a.k8_library):torch.ops.load_library(str(path))
        base=module(f.ROOT/'vllm_fl/patches/biv150_gqa_decode.py','cjj_fullstep_b3_reducer')
        support=module(f.ROOT/'tools/kernels/biv150_pair_append_support.py','cjj_fullstep_94_support')
        writer=module(f.ROOT/'tools/kernels/biv150_konly_strided_writer.py','cjj_fullstep_exact83_writer')
        contract=module(f.ROOT/'tools/biv150_konly_strided_writer_contract.py','cjj_fullstep_83_contract')
        result['imports']=dict(torch=torch.__file__,numpy=np.__file__,triton=triton.__file__,reducer=base.__file__,writer83=writer.__file__,support94=support.__file__)
        def raw(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        def tensor(bits):return torch.from_numpy(np.array(bits,copy=True).view(np.int16)).view(torch.bfloat16).to('cuda')
        def descriptor(t):
            storage=t.untyped_storage();return contract.View(str(t.dtype).removeprefix('torch.'),tuple(t.shape),tuple(t.stride()),storage.data_ptr(),storage.nbytes(),t.storage_offset()*t.element_size(),str(t.device))
        def timed(fn,repeats):
            wall=time.monotonic();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
            for _ in range(repeats):fn()
            end.record();end.synchronize();return dict(gpu_ms=start.elapsed_time(end)/repeats,host_wall_ms=(time.monotonic()-wall)*1000/repeats)
        def compiled(k):
            record=dict(hash=str(k.hash),n_regs=getattr(k,'n_regs',None),n_spills=getattr(k,'n_spills',None),source_sha256=f.sha(k.src.fn.src.encode()))
            record['artifacts']={ext:blob(data.encode() if isinstance(data,str) else data) for ext,data in k.asm.items() if isinstance(data,(str,bytes))}
            record['signature']={repr(key):str(value) for key,value in k.src.signature.items()};return record
        for case in a.case or f.CASES:
            n,length,s,pitch=f.CASES[case];rec=dict(case=case,geometry=[n,length,s,pitch],numeric=[],cache_checks=[],timing=None);result['records'].append(rec);checkpoint()
            begin=time.monotonic();cold,packed,cost=f.build_initial();rec['initial']=cost
            rec['initial']['CPU_full_counter_and_oracle_total_seconds']=time.monotonic()-begin
            cold_sha=f.sha(cold.tobytes());packed_sha=f.sha(packed.tobytes());rec['initial']['CPU_BF16_sha256']=cold_sha;rec['initial']['CPU_K8_oracle_sha256']=packed_sha
            host_source=torch.from_numpy(cold.view(np.int16)).view(torch.bfloat16)
            alloc=time.monotonic();gpu_source_back=torch.full((cold.nbytes+2*f.GUARD,),109,dtype=torch.uint8,device='cuda');gpu_source=gpu_source_back.narrow(0,f.GUARD,cold.nbytes).view(torch.bfloat16).reshape(cold.shape);rec['initial']['GPU_source_allocation_wall_seconds']=time.monotonic()-alloc
            rec['initial']['source_GPU_upload']=timed(lambda:gpu_source.copy_(host_source),1)
            rec['initial']['upload_included_end_to_end_wall_seconds']=time.monotonic()-begin
            rec['initial']['source_upload_in_same_dtype_bits']=True
            arenas={};cases={};back={};handles={};graphs={};controls={p:f.control(n,length,pitch,p) for p in (0,1,2)}
            c=controls[0];inputbacks={};inputviews={}
            for field,bits in c.items():
                dtype=torch.bfloat16 if field in ('query','new_k','new_v') else (torch.int64 if field=='slots' else torch.int32)
                fullin=torch.full((bits.nbytes+2*f.GUARD,),109,dtype=torch.uint8,device='cuda')
                payload=torch.from_numpy(bits.view(np.uint8).reshape(-1).copy());fullin.narrow(0,f.GUARD,bits.nbytes).copy_(payload)
                inputbacks[field]=fullin;inputviews[field]=fullin.narrow(0,f.GUARD,bits.nbytes).view(dtype).reshape(bits.shape)
            qb=inputviews['query'];q=qb.as_strided((n,16,128),(pitch,128,1));nk=inputviews['new_k'];nv=inputviews['new_v'];table=inputviews['table'];lens=inputviews['lengths'];slots=inputviews['slots'];dummy=torch.empty(1,dtype=torch.float32,device='cuda')
            for arm in ('bf16','k8'):
                full=torch.full((f.PAGES*16384+2*f.GUARD,),90,dtype=torch.uint8,device='cuda');resident=full.narrow(0,f.GUARD,f.PAGES*16384)
                arenas[arm]=dict(full=full,raw=resident)
                assert max(full.data_ptr(),gpu_source_back.data_ptr())>=min(full.data_ptr()+full.numel(),gpu_source_back.data_ptr()+gpu_source_back.numel()),'initial source/destination overlap'
                if arm=='bf16':
                    viewed=resident.view(torch.bfloat16).reshape(f.PAGES,2,16,2,128);key,value=viewed.unbind(1);scale=dummy
                else:
                    key=resident.view(torch.int8).as_strided((f.PAGES,16,2,128),(16384,256,128,1));scale=resident.view(torch.float32)
                    value=full.view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(8192,256,128,1),storage_offset=(f.GUARD+8192)//2)
                cases[arm]=dict(key=key,value=value,scale=scale);back[arm]={}
                if arm=='k8':assert key.data_ptr()==scale.data_ptr()==resident.data_ptr() and value.data_ptr()==resident.data_ptr()+8192
                for field,shape,dtype in [('partial',(n,16,16,128),torch.float32),('maximum',(n,16,16),torch.float32),('summation',(n,16,16),torch.float32),('output',(n,16,128),torch.bfloat16)]:
                    fullout=torch.empty(math.prod(shape)+128,dtype=dtype,device='cuda');fullout.view(torch.int16 if field=='output' else torch.int32).fill_(0x4255 if field=='output' else 0x428e0000)
                    wide=fullout.narrow(0,64,math.prod(shape)).reshape(shape);cases[arm][field]=wide if field=='output' else wide.narrow(2,0,s);back[arm][field]=(fullout,wide)
            def geometry(t):return dict(dtype=str(t.dtype),shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),pointer_alignment64=t.data_ptr()%64,device=str(t.device))
            rec['actual_views']={arm:{name:geometry(t) for name,t in values.items()} for arm,values in cases.items()}
            rec['source_views']={name:geometry(t) for name,t in dict(query=q,new_k=nk,new_v=nv,table=table,lengths=lens,slots=slots).items()}
            rec['K8_pointer_roles']=dict(initial_and_consumer_V_is_offset_bytes=cases['k8']['value'].data_ptr()-arenas['k8']['raw'].data_ptr(),append83_V_full_base=arenas['k8']['raw'].view(torch.bfloat16).data_ptr()==arenas['k8']['raw'].data_ptr(),K_and_scale_same_full_base=cases['k8']['key'].data_ptr()==cases['k8']['scale'].data_ptr()==arenas['k8']['raw'].data_ptr())
            def reset(arm):
                if arm=='bf16':arenas[arm]['raw'].copy_(gpu_source.view(torch.uint8).reshape(-1))
                else:
                    arenas[arm]['raw'].fill_(165)
                    handles['initial_encoder']=support.encode_initial[(f.PAGES*32,)](gpu_source[:,0],gpu_source[:,1],cases[arm]['key'],cases[arm]['scale'],cases[arm]['value'],f.PAGES,num_warps=1,num_stages=1)
            jit_begin=time.monotonic()
            handles['initial_encoder']=support.encode_initial.warmup(gpu_source[:,0],gpu_source[:,1],cases['k8']['key'],cases['k8']['scale'],cases['k8']['value'],f.PAGES,grid=(f.PAGES*32,),num_warps=1,num_stages=1)
            handles['initial_encoder']._init_handles();torch.cuda.synchronize()
            rec['initial']['existing_initial_encoder_JIT_load_wall_seconds']=time.monotonic()-jit_begin
            rec['initial']['BF16_GPU_pack']=timed(lambda:reset('bf16'),1);rec['initial']['K8_GPU_full_encode']=timed(lambda:reset('k8'),1)
            rec['initial']['total_through_resident_GPU_pack_wall_seconds']=time.monotonic()-begin
            def load(p):
                z=controls[p]
                for field,bits in z.items():inputbacks[field].narrow(0,f.GUARD,bits.nbytes).copy_(torch.from_numpy(bits.view(np.uint8).reshape(-1).copy()))
            def qualify(p):
                z=controls[p];return contract.qualify(descriptor(nk),descriptor(nv),descriptor(slots),descriptor(arenas['k8']['raw']),f.PAGES,contract.SlotSnapshot(tuple(int(x) for x in z['slots']),'component-owned-cpu-preupload',p))
            def append(arm):
                z=cases[arm]
                if arm=='bf16':handles['bf16_append']=support.append_kv[(n,2)](nk,nv,slots,z['key'],dummy,z['value'],f.PAGES*16,False,num_warps=1,num_stages=1)
                else:handles['k8_append']=writer.encode_append[(n,2)](nk,nv,slots,z['key'],z['scale'],arenas[arm]['raw'].view(torch.bfloat16),nk.stride(0),nv.stride(0),nk.shape[0],nv.shape[0],slots.shape[0],f.PAGES,num_warps=1,num_stages=1)
            def producer(arm):
                z=cases[arm];args=(q,z['key'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634)
                if arm=='bf16':getattr(torch.ops,'corex_native_gqa_s'+str(s)+'_probe').produce(*args)
                else:torch.ops.cjj_native_k8_qk_probe.produce(q,z['key'],z['scale'],z['value'],lens,table,z['partial'],z['maximum'],z['summation'],128**-.5*1.4426950408889634,s)
            def reducer(arm):
                z=cases[arm];o,m,l=z['partial'],z['maximum'],z['summation'];handles['reducer']=base._reduce_partials[(n,16)](output_ptr=z['output'],segm_output_ptr=o,segm_max_ptr=m,segm_expsum_ptr=l,seq_lens_ptr=lens,segment_stride_token=o.stride(0),segment_stride_head=o.stride(1),segment_stride_segment=o.stride(2),segment_stride_dim=1,max_stride_token=m.stride(0),max_stride_head=m.stride(1),max_stride_segment=1,output_stride_token=z['output'].stride(0),output_stride_head=128,page_size=16,num_segments=s,HEAD_SIZE=128,num_warps=4)
            def step(arm):append(arm);producer(arm);reducer(arm)
            def poison(arm):
                for field,(_,wide) in back[arm].items():wide.view(torch.int16 if field=='output' else torch.int32).fill_(0x7fc1 if field=='output' else 0x7fc12345)
            def fingerprints():return {name:f.sha(raw(t)) for name,t in inputbacks.items()}
            def verify_inputs(p):
                for field,bits in controls[p].items():assert raw(inputbacks[field])==bytes([109])*f.GUARD+bits.tobytes()+bytes([109])*f.GUARD,'source/guard/padding input mismatch'
            def check_cache(arm,p,stage):
                z=arenas[arm];guard=raw(z['full'].narrow(0,0,f.GUARD))+raw(z['full'].narrow(0,f.GUARD+f.PAGES*16384,f.GUARD));assert guard==bytes([90])*(2*f.GUARD)
                chunks=[]
                for start in range(0,f.PAGES,128):
                    count=min(128,f.PAGES-start);initial=cold[start:start+count] if arm=='bf16' else packed[start:start+count]
                    expected=(initial.view(np.uint8).reshape(count,16384).copy() if p is None else f.patched_chunk(arm,start,initial,controls[p]));actual=raw(z['raw'].narrow(0,start*16384,count*16384))
                    if actual!=expected.tobytes():
                        rec.setdefault('failed_cache',[]).append(dict(arm=arm,phase=p,start_page=start,actual=blob(actual),expected=blob(expected.tobytes())));raise AssertionError('Complete cache byte mismatch')
                    chunks.append(dict(start=start,bytes=len(actual),sha256=f.sha(actual)))
                rec['cache_checks'].append(dict(arm=arm,phase=p,stage=stage,all_bytes_compared=f.PAGES*16384,guards_exact=True,chunks=chunks))
            def check_source(stage):
                guard=raw(gpu_source_back.narrow(0,0,f.GUARD))+raw(gpu_source_back.narrow(0,f.GUARD+cold.nbytes,f.GUARD));assert guard==bytes([109])*(2*f.GUARD)
                chunks=[];flat=gpu_source.view(torch.uint8).reshape(-1)
                for start in range(0,f.PAGES,128):
                    count=min(128,f.PAGES-start);actual=raw(flat.narrow(0,start*16384,count*16384));expected=cold[start:start+count].tobytes()
                    if actual!=expected:
                        rec['failed_initial_source']=dict(stage=stage,start=start,actual=blob(actual),expected=blob(expected));raise AssertionError('Full initial source readonly mismatch')
                    chunks.append(dict(start=start,bytes=len(actual),sha256=f.sha(actual)))
                rec.setdefault('initial_source_readonly_checks',[]).append(dict(stage=stage,all_bytes_compared=cold.nbytes,guards_exact=True,chunks=chunks))
            def capture(arm,p,stage,gold,readonly):
                payload={};metrics={}
                arrays={field:raw(values[0]) for field,values in back[arm].items()}
                payload={field:blob(data) for field,data in arrays.items()}
                rec.setdefault('snapshots',[]).append(dict(stage=stage,arm=arm,phase=p,full_arrays=payload))
                assert fingerprints()==readonly,'readonly source changed'
                verify_inputs(p)
                for field,(fullout,wide) in back[arm].items():
                    b=arrays[field];width=2 if field=='output' else 4;marker=0x4255 if width==2 else 0x428e0000;guard=struct.pack('<H' if width==2 else '<I',marker)*64
                    assert b[:len(guard)]==b[-len(guard):]==guard,'full output canary'
                    middle=b[len(guard):-len(guard)]
                    if field=='output':
                        bits=np.frombuffer(middle,dtype='<u2').reshape(n,16,128);values=(bits.astype(np.uint32)<<16).view('<f4').astype(np.float64);metrics[field]=f.metrics(values,gold[field]);assert bool((bits[controls[p]['lengths']==0]==0).all())
                    else:
                        shape=(n,16,16,128) if field=='partial' else (n,16,16);bits=np.frombuffer(middle,dtype='<u4').reshape(shape)
                        mask=np.zeros(shape,dtype=bool);mask[:,:,:s]=np.broadcast_to(gold['active'][:,None,:,None] if field=='partial' else gold['active'][:,None,:],mask[:,:,:s].shape)
                        assert bool((bits[~mask]==0x7fc12345).all()),'inactive poison changed'
                        wanted=np.broadcast_to(gold['active'][:,None,:,None] if field=='partial' else gold['active'][:,None,:],gold[field].shape)
                        metrics[field]=f.metrics(bits.view('<f4')[:,:,:s][wanted],gold[field][wanted])
                        if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(bits.view('<f4')[:,:,:s][wanted],gold['ideal_partial_diagnostic'][wanted])
                    if not metrics[field]['passed']:rec.setdefault('failed_numeric',[]).append(dict(arm=arm,phase=p,stage=stage,field=field,metrics=metrics[field],full_actual=payload[field]));raise AssertionError('Full FP64 .005 '+arm+' '+field)
                return dict(stage=stage,arm=arm,phase=p,metrics=metrics,full_arrays=payload)
            check_cache('bf16',None,'initial_pack');check_cache('k8',None,'initial_pack');check_source('after_initial_pack')
            gold={}
            for p in (0,1,2):
                load(p);binding=qualify(p);verify_inputs(p);readonly=fingerprints()
                rec.setdefault('writer_bindings',[]).append(binding)
                for arm in ('bf16','k8'):
                    gold[arm,p]=f.golden(cold,packed,controls[p],s,arm)
                    rec.setdefault('golden',{})[arm+str(p)]={k:npblob(v) for k,v in gold[arm,p].items()}
                    reset(arm);poison(arm);torch.cuda.synchronize()
                    if p==1:
                        # Both append sources and metadata changed after capture.
                        # The first B work is graph replay; no eager B prewrite.
                        graphs[arm].replay();torch.cuda.synchronize()
                        first=capture(arm,p,'changed_graph_before_eager',gold[arm,p],readonly)
                        rec['numeric'].append(first);check_cache(arm,p,'changed_graph')
                    elif p==2:
                        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):step(arm)
                        side.synchronize();first=capture(arm,p,'changed_side_before_eager',gold[arm,p],readonly)
                        rec['numeric'].append(first);check_cache(arm,p,'side')
                    else:
                        step(arm);torch.cuda.synchronize();first=capture(arm,p,'initial_eager',gold[arm,p],readonly)
                        rec['numeric'].append(first);check_cache(arm,p,'eager')
                    poison(arm);step(arm);torch.cuda.synchronize()
                    second=capture(arm,p,'new_eager_or_repeat',gold[arm,p],readonly)
                    assert first['full_arrays']==second['full_arrays'];rec['numeric'].append(second)
                    check_cache(arm,p,'fresh_eager')
                    if p==0:
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):step(arm)
                        graphs[arm]=graph
            # Actual append/reducer identity and postlaunch resource gates happen
            # before any warmed measurements, not after a performance result.
            rec['actual_compiled']={role:compiled(handle) for role,handle in handles.items()}
            signature={};names=['key_source','value_source','slots','code_full','scale_full','value_full','key_row_stride','value_row_stride','key_rows','value_rows','slot_count','cache_pages']
            for key,value in handles['k8_append'].src.signature.items():
                if isinstance(key,tuple):assert len(key)==1;key=key[0]
                if isinstance(key,str) and key in names:key=names.index(key)
                assert type(key) is int and 0<=key<12 and key not in signature
                signature[key]=str(value)
            assert [signature[i] for i in range(12)]==['*bf16','*bf16','*i64','*i8','*fp32','*bf16']+['i64']*6,'actual83 typed ABI'
            for role in ('k8_append','bf16_append','reducer'):
                handle=handles[role];assert type(handle.n_regs) is int and handle.n_regs>0 and handle.n_spills==0,'unknown/failed postlaunch resources'
            # Dense state restored; all compulsory phases have passed before timing.
            load(0);qualify(0);readonly=fingerprints();timed_graph={}
            for arm in ('bf16','k8'):
                reset(arm)
                for _ in range(8):step(arm)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(f.UNROLL):step(arm)
                timed_graph[arm]=graph
                poison(arm);graph.replay();torch.cuda.synchronize();tg=capture(arm,0,'timing_graph_guard',gold[arm,0],readonly)
                poison(arm);step(arm);torch.cuda.synchronize();te=capture(arm,0,'timing_fresh_eager',gold[arm,0],readonly);assert tg['full_arrays']==te['full_arrays']
                rec.setdefault('timing_bitwise_reference',{})[arm]=te['full_arrays']
                check_cache(arm,0,'pre_timing')
            rec['warm']={}
            for arm in timed_graph:
                warm=timed(timed_graph[arm].replay,f.WARM);rec['warm'][arm]=dict(gpu_ms_per_step=warm['gpu_ms']/f.UNROLL,graph_launch_sync_wall_ms_per_step=warm['host_wall_ms']/f.UNROLL,replays=f.WARM,unroll=f.UNROLL)
            rec['timing']=[]
            for arm in f.ORDER:
                clocks=timed(timed_graph[arm].replay,f.REPLAYS);rec['timing'].append(dict(arm=arm,gpu_ms_per_step=clocks['gpu_ms']/f.UNROLL,graph_launch_sync_wall_ms_per_step=clocks['host_wall_ms']/f.UNROLL))
            for arm in ('bf16','k8'):
                post=capture(arm,0,'post_timing',gold[arm,0],readonly);assert post['full_arrays']==rec['timing_bitwise_reference'][arm];check_cache(arm,0,'post_timing')
            check_source('after_timing')
            assert f.sha(cold.tobytes())==cold_sha and f.sha(packed.tobytes())==packed_sha,'CPU initial sources changed'
            rec['event_ratio_BF16_over_K8']=sum(x['gpu_ms_per_step'] for x in rec['timing'] if x['arm']=='bf16')/sum(x['gpu_ms_per_step'] for x in rec['timing'] if x['arm']=='k8')
            rec['host_wall_ratio_BF16_over_K8']=sum(x['graph_launch_sync_wall_ms_per_step'] for x in rec['timing'] if x['arm']=='bf16')/sum(x['graph_launch_sync_wall_ms_per_step'] for x in rec['timing'] if x['arm']=='k8')
            rec['passed']=True;checkpoint();del cases,back,arenas,gpu_source,gpu_source_back,host_source,cold,packed,graphs,timed_graph,inputbacks,inputviews,qb,q,nk,nv,table,lens,slots,dummy,graph;torch.cuda.empty_cache()
        result.update(completed=True,numerical_verified=True,timing_executed=True,requested_component_gates_passed=True,Graph_verified=True,side_stream_verified=True,loaded_helper_resources_verified=True,large_context_all_OML_and_ideal_final_FP64_gates_verified=True)
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifacts']=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=f.sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='fullstep-summary.json'];checkpoint();print(json.dumps(result,sort_keys=True,indent=2));return 0 if result['completed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--consumer-summary',type=Path);p.add_argument('--consumer-proof-sha256');p.add_argument('--k8-library',type=Path);p.add_argument('--bf16-library',type=Path);p.add_argument('--out',type=Path);p.add_argument('--case',choices=f.CASES,action='append');a=p.parse_args()
    if not a.gpu:
        assert not any((a.consumer_summary,a.consumer_proof_sha256,a.k8_library,a.bf16_library,a.out,a.case));assert not {'torch','triton','numpy'}.intersection(sys.modules);print(json.dumps(f.plan(),sort_keys=True,indent=2));return 0
    if not all((a.consumer_summary,a.consumer_proof_sha256,a.k8_library,a.bf16_library,a.out)):raise ValueError('Actual consumer pass and both exact existing libraries + new out required')
    if a.case and len(set(a.case))!=len(a.case):raise ValueError('Duplicate case')
    return run(a)

if __name__=='__main__':raise SystemExit(main())
