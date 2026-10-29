#!/usr/bin/env python3
"""One isolated Q2048 screen; default CPU plan, explicit Root-only GPU execution."""
import argparse
import errno
import gzip
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import biv150_konly_q64_compile_source as c
import biv150_konly_q64_screen_contract as f
import probe_biv150_konly_q64_compile as compiler
from biv150_konly_q64_compile_arguments import kwargs


def run(a,pre):
    c.need(a.root_owned_exclusive and a.commit and a.out and a.out.is_absolute() and not a.out.exists(),'Root ownership/exact commit/new absolute output')
    c.need(not a.out.resolve().is_relative_to(c.ROOT),'output outside checkout')
    c.need(compiler.git('rev-parse','HEAD')==a.commit and not compiler.git('status','--porcelain'),'exact clean frontend')
    c.need(not compiler.git('diff',c.BASE,'--','vllm_fl','csrc'),'production/kernel changes prohibited')
    c.need(not {'torch','triton','numpy','vllm','vllm_fl'}.intersection(sys.modules),'cold admission before framework imports')
    a.out.mkdir(parents=True)
    result=dict(f.plan(),commit=a.commit,completed=False,errors=[],records={},stages=[],timings={},
        actual_preflight=pre,reader_launch_attempts=0,writer_launch_attempts=0,Graph_replays=0,
        counter_meaning='Python launches include capture; graph replays counted separately, not measured kernel profiling')
    def save():compiler.save(a.out/'summary.json',result)
    def persist(name,value):
        path=a.out/(name+'.pt.gz')
        with gzip.open(path,'wb',compresslevel=3) as stream:
            torch.save(value,stream)
        return dict(path=path.name,bytes=path.stat().st_size,sha256=c.sha(path.read_bytes()))
    save()
    try:
        h=c.proposal(32);base=Path('/usr/local/lib/python3.12/site-packages')
        result['execution_path_source_proof']=f.execution_sources(base/'triton');save()
        installed=dict(sdk=base/'vllm/v1/attention/ops/triton_unified_attention.py',helpers=base/'vllm/v1/attention/ops/triton_attention_helpers.py',jit=base/'triton/spec/iluvatar/triton/runtime/jit.py')
        for name,path in installed.items():c.need(c.sha(path.read_bytes())==h.PINS[name],'installed pin '+name)
        smi=subprocess.run(['/usr/local/corex/bin/ixsmi','-i','0'],capture_output=True,text=True,timeout=15)
        pids=subprocess.run(['/usr/local/corex/bin/ixsmi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        compiler.save(a.out/'prelaunch-observation.json',dict(smi=vars(smi),pids=vars(pids)))
        c.need(smi.returncode==pids.returncode==0 and 'No running processes' in smi.stdout and pids.stdout.strip() in ('','No running processes','No running processes found','No running processes found.'),'GPU not confirmed idle')
        with socket.socket() as sock:c.need(sock.connect_ex(('127.0.0.1',9031))==errno.ECONNREFUSED,'service9031 not closed')
        for name in ('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE','VLLM_FL_BIV150_KONLY_NATIVE_DECODE','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'):
            c.need(os.environ.get(name,'0')!='1','service activation prohibited '+name)
        os.environ['TRITON_CACHE_DIR']=str(a.out/'triton-cache')
        import numpy as np
        import torch
        from vllm.triton_utils import triton
        import biv150_prefill_probe_fixtures as fixture
        fixture.torch=torch;torch.set_num_threads(4);torch.set_flush_denormal(False)
        reference,policy,expanded=f.load_reference(a.references)
        (a.out/'reference-with-63-64.py').write_text(expanded)
        c.need(sys.byteorder=='little','little-endian oracle')
        target=triton.runtime.driver.active.get_current_target()
        c.need((target.backend,target.arch,target.warp_size)==('corex',71,64),'target')
        result.update(GPU_executed=True,CUDA_API_entered=True);save()
        tensors,owner=compiler.allocate(torch)
        guarded={r:torch.empty((2050,16,128),device='cuda',dtype=torch.bfloat16) for r in f.ROLES}
        backing={name:torch.empty((2048,2560),device='cuda',dtype=torch.bfloat16) for name in ('key','value')}
        writer_views={name:value.as_strided((2048,2,128),(2560,128,1)) for name,value in backing.items()}
        slots=torch.empty((2048,),device='cuda',dtype=torch.int64)
        def interval(t):
            storage=t.untyped_storage();return storage.data_ptr(),storage.data_ptr()+storage.nbytes()
        readonly_tensors={k:v for k,v in dict(tensors,cache=owner,writer_key=backing['key'],writer_value=backing['value'],slots=slots).items() if k!='out'}
        for role,out in guarded.items():
            lo,hi=interval(out)
            c.need(all(hi<=x or y<=lo for x,y in map(interval,readonly_tensors.values())),'output/source storage alias')
        c.need(interval(guarded['q32'])!=interval(guarded['q64']),'role output storage alias')
        result['tensor_descriptors']={k:dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),storage_offset=v.storage_offset(),storage_interval=list(interval(v))) for k,v in dict(readonly_tensors,output_q32=guarded['q32'],output_q64=guarded['q64']).items()}
        stream=torch.cuda.Stream();graphs={};calls={};functions={};handles={};grids={};launch_options={}
        c.need(stream.cuda_stream!=torch.cuda.default_stream().cuda_stream,'explicit nondefault side stream')
        result['streams']=dict(actual=stream.cuda_stream,default=torch.cuda.default_stream().cuda_stream)
        patch=c.ROOT/'vllm_fl/patches'
        adapter=c.load_file('cjj_q64_screen_snapshot',patch/'biv150_konly_encoded_q32_adapter.py')
        for role in f.ROLES:
            bq=int(role[1:]);paths=pre['runtime_paths'][role];source=Path(paths['source']);override=Path(paths['override'])
            c.need(source.read_text()==(c.original('biv150_konly_encoded_q32.py') if bq==32 else c.candidate_source()),'original compile source '+role)
            c.need(override.read_bytes()==(a.raw/'out'/role/'override.ttgir').read_bytes(),'original override path/content '+role)
            module=c.load_file('cjj_q64_screen_control' if bq==32 else 'cjj_q64_screen_candidate',source)
            fn=getattr(module,'kernel_cjj_konly_q32' if bq==32 else c.NAME);functions[role]=fn
            pins=adapter.Adapter.installed_pins(fn,source,{'vllm.v1.attention.ops.triton_unified_attention':h.PINS['sdk'],'vllm.v1.attention.ops.triton_attention_helpers':h.PINS['helpers']})
            c.need(pins==h.PINS and Path(inspect.getfile(type(fn))).resolve()==installed['jit'].resolve(),'live JIT/codec binding')
            for key,name in (('sdk','vllm.v1.attention.ops.triton_unified_attention'),('helpers','vllm.v1.attention.ops.triton_attention_helpers')):
                c.need(Path(sys.modules[name].__file__).resolve()==installed[key].resolve(),'live import '+name)
            config,grid=c.geometry(2048,bq);grids[role]=grid
            calls[role]=kwargs(dict(profile=role,sequences=1,segments=1,config=config),dict(tensors,out=guarded[role][1:2049]),True)
            launch_options[role]=dict(num_warps=16,num_stages=1,ir_override=str(override))
            kernel=fn.warmup(grid=grid,**calls[role],**launch_options[role]);handles[role]=kernel
            snapshot=adapter.kernel_snapshot(kernel,fn);adapter.Adapter._preserve(a.out/role,kernel,snapshot)
            result.setdefault('actual_loader_paths',{})[role]=str(Path(inspect.getfile(type(kernel))).resolve());save()
            c.need(Path(result['actual_loader_paths'][role])==f.LOADED_COMPILER.resolve(),'actual CompiledKernel loader implementation')
            resources=f.check_loaded(role,kernel,snapshot,pre['records'][role])
            result['records'][role]=dict(snapshot=snapshot,loaded=resources,actual_source_path=str(source),actual_source_sha256=c.sha(source.read_bytes()))
            save()
        writer_module=c.load_file('cjj_q64_same_accepted_writer',patch/'biv150_konly_encoded_writer.py')
        writer=writer_module.encode_append
        c.need(writer.fn.__globals__['quantize_bf16_rne'] is writer_module.quantize_bf16_rne,'actual same-module writer codec')
        c.need(Path(inspect.getfile(type(writer))).resolve()==installed['jit'].resolve(),'actual writer JIT')
        writer_args=(writer_views['key'],writer_views['value'],slots,owner.view(torch.int8),owner.view(torch.float32),owner.view(torch.bfloat16),2560,2560,2048,2048,2048,773)
        wk=writer.warmup(*writer_args,grid=(2048,2),num_warps=1,num_stages=1);wk._init_handles()
        c.need(type(wk.n_regs) is int and wk.n_regs>0 and type(wk.n_spills) is int and wk.n_spills==0,'same writer actual loaded resources')
        c.need((wk.metadata.num_warps,wk.metadata.num_stages,wk.metadata.warp_size)==(1,1,64),'same writer launch geometry')
        wd=a.out/'writer';wd.mkdir();writer_artifacts={}
        for ext,value in wk.asm.items():
            if isinstance(value,(str,bytes)):
                raw=value.encode() if isinstance(value,str) else value;(wd/('actual.'+ext)).write_bytes(raw);writer_artifacts[ext]=c.sha(raw)
        result['writer']=dict(source_sha256=f.WRITER,source_path=str(patch/'biv150_konly_encoded_writer.py'),hash=wk.hash,source_digest=c.sha(writer.src),n_regs=wk.n_regs,n_spills=wk.n_spills,shared=wk.metadata.shared,artifacts=writer_artifacts,
            qualification='unchanged writer; this invocation still requires complete CPU-codec cache equality before timing')
        save()
        def byte_cpu(t):return t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        def snapshot(t):return t.detach().cpu().contiguous().clone()
        def fingerprints():
            return {k:c.sha(byte_cpu(v)) for k,v in dict(q=tensors['q'],table=tensors['table'],lengths=tensors['lengths'],cu_q=tensors['cu_q'],k_descale=tensors['k_descale'],v_descale=tensors['v_descale'],writer_key=backing['key'],writer_value=backing['value'],slots=slots).items()}
        def reader(role):
            result['reader_launch_attempts']+=1
            got=functions[role][grids[role]](**calls[role],**launch_options[role])
            c.need(got is handles[role],'actual reader handle changed')
        def step(role,mode):
            if mode=='append-reader':
                result['writer_launch_attempts']+=1
                c.need(writer[(2048,2)](*writer_args,num_warps=1,num_stages=1) is wk,'writer handle changed')
            reader(role)
        def replay(role,mode):graphs[role,mode].replay();result['Graph_replays']+=1
        inputs={};references={};gold_outputs={};uploaded={};append_poison={}
        for pattern in (0,1):
            cpu=fixture.cpu_inputs([2048],[6144],20261025,pattern)
            encoded,codes,scales,codec=reference.encoded_bytes(cpu['packed'][:,0],cpu['packed'][:,1],torch,np)
            refs=reference.references(cpu,codes,scales,torch,np)
            if pattern==0:
                values=cpu['packed'][cpu['table'][0,:384].long(),1].reshape(-1,2,128).double()
                visible=4096+torch.arange(2048)+1
                full=(values.cumsum(0)[visible-1]/visible[:,None,None]).repeat_interleave(8,dim=1)
                refs=dict(refs,rows=list(range(2048)),original=full,quantized=full)
            positions=torch.arange(4096,6144,dtype=torch.int64)
            physical=cpu['table'][0,positions//16].long();cols=positions%16;slot_cpu=physical*16+cols
            source_cpu={name:torch.full((2048,2560),-61,dtype=torch.bfloat16) for name in backing}
            for name,index in (('key',0),('value',1)):source_cpu[name][:,:256].copy_(cpu['packed'][physical,index,cols].reshape(2048,256))
            poisoned=bytearray(encoded)
            for slot in slot_cpu.tolist():
                page,col=divmod(slot,16)
                for start,count in ((page*16384+col*256,256),(page*16384+4096+col*8,8),(page*16384+8192+col*512,512)):
                    poisoned[start:start+count]=bytes([215])*count
            uploaded[pattern]=torch.from_numpy(np.frombuffer(encoded,dtype=np.uint8).copy())
            append_poison[pattern]=torch.from_numpy(np.frombuffer(poisoned,dtype=np.uint8).copy())
            owner.copy_(uploaded[pattern])
            for dest,source in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]:tensors[dest].copy_(cpu[source])
            for name in backing:backing[name].copy_(source_cpu[name])
            slots.copy_(slot_cpu);torch.cuda.synchronize()
            c.need(byte_cpu(owner)==encoded,'uploaded full cache differs from CPU codec')
            immutable=fingerprints();references[pattern]=refs
            expected_readonly=dict(q=cpu['q'],table=cpu['table'],lengths=cpu['lengths'],cu_q=cpu['cu_q'],k_descale=cpu['k_scale'],v_descale=cpu['v_scale'],writer_key=source_cpu['key'],writer_value=source_cpu['value'],slots=slot_cpu)
            c.need(immutable=={k:c.sha(byte_cpu(v)) for k,v in expected_readonly.items()},'complete uploaded inputs differ from CPU originals')
            inputs[pattern]=persist('pattern-'+str(pattern)+'-inputs',dict(original_CPU=cpu,expected_encoded_bytes=uploaded[pattern],append_poison_bytes=append_poison[pattern],expected_codes=codes,expected_scales=scales,reference=refs,writer_sources=source_cpu,writer_slots=slot_cpu,codec=codec,readonly=immutable))
            def stage(label,mode,use_graph=False):
                outputs={};record=dict(label=label,pattern=pattern,mode=mode,actual_roles='hoisted Q32 and hoisted Q64',metrics={},artifacts={},prelaunch_artifacts={},readonly={},cache_after={})
                result['stages'].append(record);save()
                for role in f.ROLES:
                    owner.copy_(append_poison[pattern] if mode=='append-reader' else uploaded[pattern])
                    guarded[role].fill_(71);calls[role]['output_ptr'].fill_(float('nan'));torch.cuda.synchronize()
                    before=snapshot(guarded[role]);cache_before_sha=c.sha(byte_cpu(owner))
                    record['prelaunch_artifacts'][role]=persist(str(pattern)+'-'+label+'-'+mode+'-'+role+'-prelaunch',dict(before=before,cache_before_sha256=cache_before_sha,input_archive=inputs[pattern]))
                    save()
                    c.need(cache_before_sha==c.sha(byte_cpu(append_poison[pattern] if mode=='append-reader' else uploaded[pattern])),'complete prelaunch cache differs')
                    c.need(bool((before[0]==71).all() and (before[2049]==71).all() and torch.isnan(before[1:2049]).all()),'poison/canary setup')
                    with torch.cuda.stream(stream):
                        if use_graph:replay(role,mode)
                        else:step(role,mode)
                    stream.synchronize();after=snapshot(guarded[role]);output=snapshot(calls[role]['output_ptr']);cache_after=snapshot(owner);readonly=fingerprints()
                    record['artifacts'][role]=persist(str(pattern)+'-'+label+'-'+mode+'-'+role,dict(before=before,after=after,output=output,cache_after=cache_after,readonly_before=immutable,readonly_after=readonly,input_archive=inputs[pattern]))
                    record['readonly'][role]=readonly;record['cache_after'][role]=c.sha(byte_cpu(cache_after));save()
                    c.need(bool((after[0]==71).all() and (after[2049]==71).all() and torch.isfinite(after[1:2049]).all()),'full output guard/finite')
                    c.need(byte_cpu(output)==byte_cpu(after[1:2049]),'output view/full-buffer mismatch')
                    c.need(readonly==immutable and byte_cpu(cache_after)==encoded,'readonly inputs or full cache differs')
                    sample=output[refs['rows']].double().reshape(-1).tolist()
                    quant=reference.metrics(sample,refs['quantized'].reshape(-1).tolist());original=reference.metrics(sample,refs['original'].reshape(-1).tolist())
                    record['metrics'][role]=dict(quantized_operand=quant,original_BF16=original);outputs[role]=output
                hard=dict(finite=True,codec=True,output_canaries=True,read_only_inputs=True,actual_IR=True,
                    hoisted_unhoisted_bitwise=byte_cpu(outputs['q32'])==byte_cpu(outputs['q64']),
                    quantized_reference=all(m['quantized_operand']['finite'] and m['quantized_operand']['pointwise_passed'] for m in record['metrics'].values()))
                quality=all(m['original_BF16']['finite'] and m['original_BF16']['relative_rmse']<=.005 for m in record['metrics'].values())
                record.update(hard_gates=hard,legacy_policy_bitwise_key_means='actual hoisted Q32 versus actual hoisted Q64',**policy(hard,quality,a.numeric_diagnostic));save()
                return outputs
            if pattern==0:
                for mode in f.MODES:gold_outputs[0,mode]=stage('eager-A',mode)
                for mode in f.MODES:
                    for role in f.ROLES:
                        with torch.cuda.stream(stream):
                            for _ in range(3):step(role,mode)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,stream=stream):
                                for _ in range(4):step(role,mode)
                        stream.synchronize();graphs[role,mode]=graph
                    actual=stage('Graph-A',mode,True)
                    c.need(all(byte_cpu(actual[r])==byte_cpu(gold_outputs[0,mode][r]) for r in f.ROLES),'GraphA differs from eagerA')
            else:
                # Both actual-handle graphs consume B before any fresh eager B.
                graph_b={mode:stage('Graph-B-first',mode,True) for mode in f.MODES}
                for mode in f.MODES:
                    c.need(all(byte_cpu(graph_b[mode][r])!=byte_cpu(gold_outputs[0,mode][r]) for r in f.ROLES),'changed-input Graph counterexample absent')
                for mode in f.MODES:
                    gold_outputs[1,mode]=stage('eager-B-after-Graph',mode)
                    c.need(all(byte_cpu(graph_b[mode][r])==byte_cpu(gold_outputs[1,mode][r]) for r in f.ROLES),'GraphB differs from fresh eagerB')
                    # All eager stage launches use the explicit nondefault stream.
                result.update(numeric_verified=True,Graph_verified=True,side_stream_verified=True)
                save()
            c.need(all(byte_cpu(gold_outputs[pattern,'reader'][r])==byte_cpu(gold_outputs[pattern,'append-reader'][r]) for r in f.ROLES),'append plus reader differs from same-input reader')
        for mode in f.MODES:
            with torch.cuda.stream(stream):
                for role in f.ROLES:
                    for _ in range(8):replay(role,mode)
            stream.synchronize();rounds=[];result['timings'][mode]=rounds;save()
            for index,role in enumerate(('q32','q64','q64','q32')):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start=time.perf_counter()
                with torch.cuda.stream(stream):
                    begin.record()
                    for _ in range(16):replay(role,mode)
                    end.record()
                end.synchronize();wall=time.perf_counter()-start
                post=snapshot(guarded[role]);cache_after=byte_cpu(owner)
                row=dict(index=index,role=role,event_ms_per_step=begin.elapsed_time(end)/64,wall_ms_per_step=wall*1000/64,
                    output_sha256=c.sha(byte_cpu(post[1:2049])),full_guarded_sha256=c.sha(byte_cpu(post)),cache_sha256=c.sha(cache_after),readonly=fingerprints())
                rounds.append(row);save()
                c.need(bool((post[0]==71).all() and (post[2049]==71).all()),'timed guard changed')
                c.need(byte_cpu(post[1:2049])==byte_cpu(gold_outputs[1,mode][role]) and cache_after==byte_cpu(uploaded[1]) and row['readonly']==immutable,'timed complete result/input/cache differs')
            average={r:sum(z['event_ms_per_step'] for z in rounds if z['role']==r)/2 for r in f.ROLES}
            result.setdefault('observed_event_means',{})[mode]=dict(average,Q32_over_Q64=average['q32']/average['q64'])
        result.update(completed=True,performance_measured=True,precision_gate=False,service_verified=False,
            original_BF16_quality_passed=all(r['original_BF16_quality_passed'] for r in result['stages']),input_archives=inputs,
            timing_scope='reader: complete Q quantization/staging/QK/softmax/PV/epilogue; append-reader: unchanged stride2560 K/V writer plus same complete reader; no service claim')
        save();return result
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,message=str(error)));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--cpu-preflight',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--numeric-diagnostic',action='store_true')
    p.add_argument('--commit');p.add_argument('--out',type=Path);p.add_argument('--raw',type=Path);p.add_argument('--review',type=Path);p.add_argument('--references',type=Path);a=p.parse_args()
    c.need(not (a.gpu and a.cpu_preflight),'choose CPU preflight or explicit GPU')
    if a.gpu or a.cpu_preflight:
        c.need(all((a.raw,a.review,a.references)),'actual raw/review/original reference paths required')
        pre=f.preflight(a.raw,a.review,a.references)
        if a.gpu:value=run(a,pre)
        else:
            c.need(not any((a.root_owned_exclusive,a.numeric_diagnostic,a.commit,a.out)),'CPU preflight rejects device options');value=pre
    else:
        c.need(not any((a.root_owned_exclusive,a.numeric_diagnostic,a.commit,a.out,a.raw,a.review,a.references)),'paths/options need explicit mode');value=f.plan()
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
if __name__=='__main__':main()
