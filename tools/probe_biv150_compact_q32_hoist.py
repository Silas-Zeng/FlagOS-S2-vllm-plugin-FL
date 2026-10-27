#!/usr/bin/env python3
"""Default stdlib plan; explicit Root-only compact Q32 compile/numeric falsifier."""
import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
sys.dont_write_bytecode=True
import biv150_compact_q32_probe_contract as c
from biv150_compact_q32_probe_arguments import kwargs


def save(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def source_identity():
    p=c.ROOT/'vllm_fl/patches';m=json.loads((p/'biv150_konly_compact_sources.json').read_text())
    hashes={}
    for row in m['generated'].values():
        path=c.ROOT/row['path'];c.need(c.sha(path.read_bytes())==row['sha256'],'private source pin '+str(path))
        hashes[row['path']]=row['sha256']
    for name,pin in m['hoist_dependencies'].items():
        path=p/name;c.need(c.sha(path.read_bytes())==pin,'hoist helper pin '+name);hashes[str(path.relative_to(c.ROOT))]=pin
    for name,pin in m['binding_files'].items():
        path=c.ROOT/'vllm_fl'/name;c.need(c.sha(path.read_bytes())==pin,'binding source '+name);hashes['vllm_fl/'+name]=pin
    for name,pin in json.loads((c.ROOT/'tools/compact_reference/source-pins.json').read_text())['paths'].items():
        c.need(c.sha((c.ROOT/'tools/compact_reference'/name).read_bytes())==pin,'CPU reference pin '+name)
    return dict(base_source_commit=c.BASE,format=c.FORMAT,source_hashes=hashes,
                manifest_sha256=c.sha((p/'biv150_konly_compact_sources.json').read_bytes()))


def admit(args):
    c.need(args.commit and args.out and args.out.is_absolute() and not args.out.exists(),'exact commit/new absolute output required')
    c.need(not args.out.resolve().is_relative_to(c.ROOT),'output must be outside checkout')
    def git(*a):return subprocess.check_output(['git','-C',str(c.ROOT),*a],text=True).strip()
    c.need(git('rev-parse','HEAD')==args.commit and not git('status','--porcelain'),'exact clean frontend freeze')
    c.need(not git('diff',c.BASE,'--','vllm_fl','csrc'),'103 frozen source/GPU math changed')
    c.need(not {'torch','triton','numpy','vllm_fl'}.intersection(sys.modules),'admission must precede frameworks')
    return source_identity()


def compile_receipt(folder,identity,commit):
    folder=folder.resolve();r=json.loads((folder/'summary.json').read_text())
    c.need(r['completed'] is True and r['compile_passed'] is True and r['errors']==[],'compile prerequisite failed')
    c.need(r['mode']=='compile-only' and r['commit']==commit and r['source_identity']==identity,'compile receipt scope/source mismatch')
    c.need(r['attention_launches']==r['writer_launches']==r['encoder_launches']==r['initialized_tensor_elements']==0,'compile-only zero-launch/init protocol')
    b=r['binding'];c.need(b['native']['actual_I8_operand_hoist_verified'] is True,'native QCode move prerequisite')
    root=Path(b['artifact_directory']).resolve();c.need(root.is_relative_to(folder),'compile artifact path outside receipt')
    for role,key in [('unhoisted','unhoisted'),('hoisted','hoisted')]:
        c.need({'ttir','ttgir','llir','asm','cubin'}<=set(b[key]['artifacts']),'complete original compiler artifacts')
        for ext,pin in b[key]['artifacts'].items():
            path=root/role/('actual.'+ext);c.need(path.is_file() and c.sha(path.read_bytes())==pin,'original compiler artifact '+role+'/'+ext)
    c.need(c.sha(Path(b['override_path']).read_bytes())==b['override_sha256'],'original override changed')
    return dict(summary_sha256=c.sha((folder/'summary.json').read_bytes()),path=str(folder),
                fresh_numeric_reaudit_required=True,old_hash_not_an_admission_license=True)



def preserve_compiled(kernel, folder):
    folder.mkdir()
    files={}
    for ext,value in getattr(kernel,'asm',{}).items():
        if isinstance(value,(str,bytes)):
            raw=value.encode() if isinstance(value,str) else value
            (folder/('actual.'+ext)).write_bytes(raw);files[ext]=dict(bytes=len(raw),sha256=c.sha(raw))
    src=getattr(kernel,'src',None)
    record=dict(hash=str(getattr(kernel,'hash',None)),files=files,prelaunch_only=True,
                loaded_resources_or_occupancy_claimed=False)
    for key in ('signature','constants','attrs'):
        # JSON-compatible typed ABI is also saved by the actual adapter. Raw
        # artifacts above survive even an unknown metadata representation.
        try:
            from vllm_fl.patches.biv150_konly_encoded_q32_adapter import typed
            record[key]=typed(getattr(src,key))
        except Exception as error:record[key]=dict(unrepresented=type(error).__name__)
    save(folder/'record.json',record)
    return record

def run(args):
    identity=admit(args)
    prior=compile_receipt(args.compile_receipt,identity,args.commit) if args.gpu else None
    args.out.mkdir();result=dict(c.plan(),mode='gpu-numeric' if args.gpu else 'compile-only',
        commit=args.commit,source_identity=identity,compile_prerequisite=prior,completed=False,
        compile_passed=False,errors=[],attention_launches=0,Graph_replays=0,writer_launches=0,encoder_launches=0,
        initialized_tensor_elements=0,CUDA_API_entered=False,context_or_allocations_attempted=False,numerical_verified=False,
        stages=[],Graph_hoisted_qualification=False,
        attention_count_meaning='Python attention launch invocations incl capture; Graph replays counted separately, not execution profiling',
        initialized_count_meaning='explicit frontend input/output fill/copy elements; allocator internals not profiled')
    def checkpoint():save(args.out/'summary.json',result)
    checkpoint()
    try:
        base=Path('/usr/local/lib/python3.12/site-packages')
        installed=dict(sdk=base/'vllm/v1/attention/ops/triton_unified_attention.py',
            helpers=base/'vllm/v1/attention/ops/triton_attention_helpers.py',jit=base/'triton/spec/iluvatar/triton/runtime/jit.py')
        for name,path in installed.items():c.need(c.sha(path.read_bytes())==c.PINS[name],'installed source '+name)
        smi=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        (args.out/'prelaunch-ixsmi.txt').write_text(smi);c.need('No running processes found' in smi,'GPU must be idle')
        with socket.socket() as sock:c.need(sock.connect_ex(('127.0.0.1',9031))!=0,'service9031 must be closed')
        # No service activation, dependency installation or unrelated native SO.
        for key in ('VLLM_FL_BIV150_KONLY_COMPACT_SERVICE','VLLM_FL_BIV150_KONLY_ENCODED_SERVICE',
                    'VLLM_FL_BIV150_KONLY_NATIVE_DECODE','VLLM_FL_BIV150_KONLY_SERVICE_Q32_HOIST'):
            c.need(os.environ.get(key,'0')!='1','component rejects service activation '+key)
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import torch
        from vllm.triton_utils import triton
        c.need(sys.byteorder=='little','byte oracle requires little endian')
        result['CUDA_initialized_after_torch_import']=bool(torch.cuda.is_initialized())
        result.update(CUDA_API_entered=True,GPU_executed=True);checkpoint()
        target=triton.runtime.driver.active.get_current_target()
        c.need((target.backend,target.arch,target.warp_size)==('corex',71,64),'actual CoreX target')
        prefix='vllm_fl.patches.biv150_konly_'
        module=importlib.import_module(prefix+'compact_q32')
        runtime_module=importlib.import_module(prefix+'compact_runtime');broker=importlib.import_module(prefix+'compact_views')
        adapter_module=importlib.import_module(prefix+'compact_q32_adapter')
        original_adapter=importlib.import_module(prefix+'encoded_q32_adapter')
        proposal=importlib.import_module(prefix+'compact_hoist_proposal');native=importlib.import_module(prefix+'encoded_hoist_native')
        for value in (module,runtime_module,broker,adapter_module,original_adapter,proposal,native):
            expected=c.ROOT/(value.__name__.replace('.','/')+'.py')
            c.need(Path(value.__file__).resolve()==expected.resolve(),'actual private import path '+value.__name__)
        for name,path in installed.items():
            value=sys.modules['vllm.v1.attention.ops.triton_unified_attention' if name=='sdk' else
                'vllm.v1.attention.ops.triton_attention_helpers' if name=='helpers' else type(module.kernel_cjj_konly_q32).__module__]
            c.need(Path(value.__file__).resolve()==path.resolve() and c.sha(path.read_bytes())==c.PINS[name],'actual loaded '+name)
        result['actual_imports']={v.__name__:dict(path=str(Path(v.__file__).resolve()),sha256=c.sha(Path(v.__file__).read_bytes()))
            for v in (torch,module,runtime_module,broker,adapter_module,original_adapter,proposal,native)}
        fn=module.kernel_cjj_konly_q32
        runtime=runtime_module.Runtime(torch,broker,{'q32':fn},None);runtime.registry.armed=True
        result.update(CUDA_API_entered=True,context_or_allocations_attempted=True);checkpoint()
        raw=torch.empty(c.PAGES*12544+2*c.GUARD,dtype=torch.uint8,device='cuda')
        layer=torch.as_strided(raw.view(torch.bfloat16),(c.PAGES,2,16,2,128),(6272,2176,256,128,1),c.GUARD//2)
        source_k=torch.empty((1,2,128),dtype=torch.bfloat16,device='cuda');source_v=torch.empty_like(source_k)
        entry=runtime.registry.claim(object(),layer,source_k,source_v);roles=entry.roles['prefill80']
        carrier_k=torch.as_strided(raw.view(torch.bfloat16),(c.PAGES,16,2,128),(6272,256,128,1),c.GUARD//2)
        carrier_v=torch.as_strided(raw.view(torch.bfloat16),(c.PAGES,16,2,128),(6272,256,128,1),c.GUARD//2+2176)
        def output():return torch.empty((c.Q+2*c.OUT_GUARD_ROWS,16,128),device='cuda',dtype=torch.bfloat16)
        guarded={r:output() for r in c.ROLES}
        def active(value):return value[c.OUT_GUARD_ROWS:c.OUT_GUARD_ROWS+c.Q]
        def empty(shape,dtype):return torch.empty(shape,dtype=dtype,device='cuda')
        t=dict(q=empty((c.Q,16,128),torch.bfloat16),out=active(guarded['hoisted']),k=roles['key_cache_ptr'],
            v=roles['value_cache_ptr'],kscale=roles['key_code_scale_ptr'],table=empty((1,8192),torch.int32),
            lengths=empty((1,),torch.int32),cu_q=empty((2,),torch.int32),k_descale=empty((),torch.float32),v_descale=empty((),torch.float32))
        p=dict(profile='q32',sequences=1,segments=1,config=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,IS_3D=False))
        launch=kwargs(p,t,True);launch.update(roles,num_warps=16,num_stages=1);grid=(c.Q//32+1,2)
        pins=adapter_module.Adapter.installed_pins(fn,module.__file__,{
            'vllm.v1.attention.ops.triton_unified_attention':c.PINS['sdk'],
            'vllm.v1.attention.ops.triton_attention_helpers':c.PINS['helpers']})
        adapter=adapter_module.Adapter(proposal,native,directory=args.out/'adapter',
            target=dict(backend=target.backend,arch=target.arch,warp_size=target.warp_size),pins=pins,
            implementation_digest=c.sha(json.dumps(identity,sort_keys=True)),jit=importlib.import_module(type(fn).__module__),
            device=triton.runtime.driver.active.get_current_device(),fn=fn,
            source_module=module,source_sha256=identity['source_hashes']['vllm_fl/patches/biv150_konly_compact_q32.py'])
        adapter.validate_call(runtime,entry,launch)
        result['frontend_unhoisted_warmups_attempted']=1;checkpoint()
        first=fn.warmup(grid=grid,**launch)
        result['first_unhoisted']=preserve_compiled(first,args.out/'first-unhoisted-before-admission')
        checkpoint()
        override,binding=adapter.prepare(fn,launch,grid)
        result['adapter_preparation']=adapter.stats();checkpoint()
        c.need(binding is not None,'fresh actual compact hoist/native/resources unavailable; no reader permitted')
        result['binding']={k:v for k,v in binding.items() if not k.startswith('_')}
        result['owner_epoch']=dict(generation=entry.generation,shape=list(entry.observation.shape),storage_offset=entry.observation.storage_offset,registry_component_only=True)
        inside=native.llvm_observation(binding['_unhoisted_kernel'].asm['llir'])['Q_shared_reads_inside_KV']
        c.need(len(inside)==16,'unhoisted baseline QCode no longer inside KV')
        result.update(compile_passed=True,fresh_actual_compact_IR_native_resources_verified=True,
            loaded_resources_are_not_occupancy=True,actual_scope='one seq/Q1024/nonzero offset; no service/capacity proof')
        checkpoint()
        if not args.gpu:
            result.update(completed=True,GPU_executed=True,numerical_verified=False,
                initialized_tensor_elements=0,attention_launches=0);checkpoint();return result
        # All subsequent array/device work is explicit --gpu and only after the
        # separate compile receipt plus this process's independent fresh audit.
        import numpy as np
        normal=importlib.import_module(prefix+'compact_normalized2d128').kernel_cjj_konly_normalized2d128
        runtime.kernels['normalized2d128']=normal
        normal_launch=dict(launch,BLOCK_M=128,BLOCK_Q=16,num_warps=8)
        normal_grid=(c.Q//16+1,2)
        normal_kernel=normal.warmup(grid=normal_grid,**normal_launch)
        c.need(normal_kernel.src.fn is normal,'actual normalized source object')
        c.need({'ttir','ttgir','llir','asm','cubin'}<=set(normal_kernel.asm),'normalized actual artifacts incomplete')
        resources=original_adapter.loaded(normal_kernel)
        c.need((resources['num_warps'],resources['num_stages'],resources['warp_size'])==(8,1,64),'actual capture-bypass geometry')
        folder=args.out/'normalized2d128';folder.mkdir()
        for ext,value in normal_kernel.asm.items():
            if isinstance(value,(str,bytes)):(folder/('actual.'+ext)).write_bytes(value.encode() if isinstance(value,str) else value)
        save(folder/'actual-metadata.json',dict(hash=str(normal_kernel.hash),signature=original_adapter.typed(normal_kernel.src.signature),
            constants=original_adapter.typed(normal_kernel.src.constants),attrs=original_adapter.typed(normal_kernel.src.attrs),
            resources=resources,hoist_verified=False))
        class Parent:
            def __getitem__(self,grid):
                def reject(**kw):raise ValueError('owned compact reached stock fallback')
                return reject
        facade=runtime.facade(Parent(),'sdk');runtime.q32_hoist=adapter
        fresh={r:output() for r in c.ROLES};graphs={}
        def interval(v):
            storage=v.untyped_storage();return storage.data_ptr(),storage.data_ptr()+storage.nbytes()
        readonly_intervals=[interval(v) for v in (raw,t['q'],t['table'],t['lengths'],t['cu_q'],t['k_descale'],t['v_descale'],source_k,source_v)]
        outputs=[*guarded.values(),*fresh.values()]
        for i,v in enumerate(outputs):
            lo,hi=interval(v)
            c.need(all(hi<=a or b<=lo for a,b in readonly_intervals+[interval(x) for x in outputs[:i]]),'output/source/cache storage alias')
        def cpu_bits(value):return value.detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16).copy()
        def cpu_bytes(value):return value.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().copy()
        def readonly():return {k:cpu_bytes(v) for k,v in dict(q=t['q'],table=t['table'],lengths=t['lengths'],
                cu_q=t['cu_q'],k_descale=t['k_descale'],v_descale=t['v_descale'],writer_source_k=source_k,writer_source_v=source_v).items()}
        def poison(buf):
            buf.fill_(71);active(buf).fill_(float('nan'))
            result['initialized_tensor_elements']+=int(buf.numel()+active(buf).numel())
        def buffers_ok(buf):
            v=cpu_bits(buf);g=c.OUT_GUARD_ROWS
            return v,bool((v[:g]==0x428e).all() and (v[-g:]==0x428e).all() and np.isfinite(c.decode(v[g:g+c.Q],np)).all())
        expected_raw=None;before_readonly=None;current=None;refs=None;refproof=None
        def initialize(phase):
            nonlocal expected_raw,before_readonly,current,refs,refproof
            current=c.canonical(phase,np);encoded,kc,ks,repairs=c.packed(current,np)
            refs,refproof=c.references(current,kc,ks,np)
            expected_raw=np.full(raw.numel(),0xa5,dtype=np.uint8)
            expected_raw[c.GUARD:-c.GUARD]=encoded.reshape(-1)
            raw.copy_(torch.from_numpy(expected_raw))
            t['q'].copy_(torch.from_numpy(current['q_bits'].view(np.int16)).view(torch.bfloat16))
            for name in ('table','lengths','cu_q'):t[name].copy_(torch.from_numpy(current[name]))
            t['k_descale'].fill_(1);t['v_descale'].fill_(1)
            source_k.fill_(3);source_v.fill_(5)
            torch.cuda.synchronize();result['initialized_tensor_elements']+=int(raw.numel()+sum(v.numel() for k,v in t.items() if k in ('q','table','lengths','cu_q','k_descale','v_descale'))+source_k.numel()+source_v.numel())
            c.need(np.array_equal(cpu_bytes(raw),expected_raw),'CPU encoded full byte upload mismatch')
            before_readonly=readonly();return dict(phase=phase,codec_boundary_repairs=repairs,reference=refproof)
        def stage(name,path_kind,buffers,action,phase_meta,compare=None):
            artifact={**current,**refs,'cache_before_u8':cpu_bytes(raw),'expected_cache_u8':expected_raw,
                **{'readonly_before_'+k:v for k,v in before_readonly.items()}}
            for r in c.ROLES:poison(buffers[r]);artifact['guarded_before_'+r]=cpu_bits(buffers[r])
            location=args.out/(name+'.npz')
            np.savez_compressed(location,**artifact)
            g=c.OUT_GUARD_ROWS
            c.need(all((artifact['guarded_before_'+r][:g]==0x428e).all() and (artifact['guarded_before_'+r][-g:]==0x428e).all() and np.isnan(c.decode(artifact['guarded_before_'+r][g:g+c.Q],np)).all() for r in c.ROLES),'output setup poison/guard failed')
            actual=[]
            pending=dict(name=name,artifact=location.name,observed=actual,completed=False)
            result.setdefault('incomplete_stage_records',[]).append(pending);checkpoint()
            for r in c.ROLES:
                handle=action(r)
                torch.cuda.synchronize()
                whole,ok=buffers_ok(buffers[r]);after=readonly();cache_after=cpu_bytes(raw)
                artifact['guarded_after_'+r]=whole;artifact['output_'+r]=whole[c.OUT_GUARD_ROWS:c.OUT_GUARD_ROWS+c.Q]
                artifact['cache_after_'+r]=cache_after
                artifact.update({'readonly_after_'+r+'_'+k:v for k,v in after.items()})
                np.savez_compressed(location,**artifact)
                actual.append(dict(role=r,hash=str(handle.hash) if handle is not None else str(normal_kernel.hash),
                    actual_path=path_kind,hoist_verified=path_kind=='q32' and r=='hoisted',
                    full_guard_ok=ok,whole_cache_unchanged=bool(np.array_equal(cache_after,expected_raw)),
                    readonly_unchanged=all(np.array_equal(before_readonly[k],v) for k,v in after.items())))
                checkpoint()
                c.need(ok and actual[-1]['whole_cache_unchanged'] and actual[-1]['readonly_unchanged'],name+'/'+r+': full buffer/cache/readonly hard gate')
            metric={};quant=True;quality=True
            for r in c.ROLES:
                output=c.decode(artifact['output_'+r],np)
                qm=c.metric(output,refs['quantized_FP64'],np);om=c.metric(output,refs['original_FP64'],np)
                quant=quant and qm['finite'] and qm['pointwise_passed']
                quality=quality and om['finite'] and om['relative_rmse'] is not None and om['relative_rmse']<=.005
                metric[r]=dict(quantized_FP64=qm,original_BF16=om)
            equal=bool(np.array_equal(artifact['output_unhoisted'],artifact['output_hoisted']))
            if compare is not None:
                c.need(all(np.array_equal(artifact['output_'+r],compare['output_'+r]) for r in c.ROLES),'GraphB versus same-path fresh eagerB bitwise')
            hard=dict(finite=True,quantized_FP64_pointwise=bool(quant),pair_bitwise=equal,
                full_output_guards=True,whole_cache=True,readonly=True,fresh_actual_IR_native_resources=True,actual_path_bound=True)
            row=dict(name=name,phase=phase_meta,records=actual,hard_gates=hard,metrics=metric,artifact=location.name,
                Graph_hoisted_qualification=False,original_BF16_proxy_passed=bool(quality))
            result['stages'].append(row);checkpoint()
            row.update(c.policy(hard,bool(quality),args.numeric_diagnostic));pending['completed']=True;checkpoint();return artifact
        def q32(r):
            selected=dict(launch,output_ptr=active(guarded[r]));adapter.validate_call(runtime,entry,selected)
            result['attention_launches']+=1;checkpoint()
            handle=fn[grid](**selected,**(override if r=='hoisted' else {}))
            c.need(handle is binding['_kernel' if r=='hoisted' else '_unhoisted_kernel'],'different actual Q32 compiler object')
            if r=='hoisted':adapter.after_launch(handle,fn,binding)
            return handle
        a=initialize(0);stage('eager-A','q32',guarded,q32,a)
        # Warmup/compile occurred above. Captured calls use the actual facade,
        # whose capture policy chooses normalized2D128, never forced Q32 hoist.
        stream=torch.cuda.Stream()
        def capture_action(r):
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            selected=dict(launch,output_ptr=active(guarded[r]),key_cache_ptr=carrier_k,value_cache_ptr=carrier_v)
            with torch.cuda.graph(graph,stream=stream):handle=facade[grid](**selected)
            result['attention_launches']+=1
            c.need(handle is normal_kernel,'capture must use exact unhoisted normalized2D128 object')
            graphs[r]=graph
            graph.replay();result['Graph_replays']+=1
            return handle
        graphA=stage('capture-normalized-A-and-replay','normalized2d128',guarded,capture_action,a)
        result['capture_records']=[dict(role=r,actual_path='normalized2d128',hash=str(normal_kernel.hash),hoist_verified=False,service_hoisted_capture_supported=False) for r in c.ROLES];checkpoint()
        b=initialize(1)
        def replay(r):graphs[r].replay();result['Graph_replays']+=1;return normal_kernel
        graphB=stage('Graph-B-first','normalized2d128',guarded,replay,b)
        c.need(all(not np.array_equal(graphA['output_'+r],graphB['output_'+r]) for r in c.ROLES),'changed-input Graph counterexample missing')
        result['Graph_changed_output_counterexample']=True;checkpoint()
        def normal_eager(r):
            result['attention_launches']+=1
            handle=normal[normal_grid](**dict(normal_launch,output_ptr=active(fresh[r])))
            c.need(handle is normal_kernel,'same-path normalized eager changed compiler object');return handle
        stage('fresh-normalized-eager-B','normalized2d128',fresh,normal_eager,b,compare=graphB)
        q32B=stage('fresh-Q32-eager-B','q32',guarded,q32,b)
        d=initialize(2);side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        c.need(side.cuda_stream!=torch.cuda.current_stream().cuda_stream,'side stream must be distinct')
        result['side_stream']=dict(stream=int(side.cuda_stream),default=int(torch.cuda.current_stream().cuda_stream),waited_for_input_stream=True);checkpoint()
        def side_call(r):
            with torch.cuda.stream(side):return q32(r)
        q32C=stage('side-Q32-C','q32',guarded,side_call,d)
        c.need(all(not np.array_equal(q32B['output_'+r],q32C['output_'+r]) for r in c.ROLES),'changed side-input counterexample missing')
        result.update(completed=True,requested_component_gates_passed=True,
            full_1024_final_output_verified=True,Graph_bypass_verified=True,Graph_verified=True,
            Graph_hoisted_qualification=False,side_stream_Q32_verified=True,numerical_verified=True,
            original_BF16_proxy_passed=all(x['original_BF16_proxy_passed'] for x in result['stages']),
            full_validation_matrix_verified=False,GPU_executed=True)
        checkpoint();return result
    except Exception as error:
        # Bounded failure salvage, no compile/launch/init_handles/retry.
        if 'fn' in locals() and 'triton' in locals():
            try:
                cache=fn.device_caches[triton.runtime.driver.active.get_current_device()][0]
                result['failure_cached_compiled_objects']=[]
                for index,kernel in enumerate(list(cache.values())[:8]):
                    result['failure_cached_compiled_objects'].append(preserve_compiled(kernel,args.out/('failure-compiler-'+str(index))))
                result['failure_cache_salvage_truncated']=len(cache)>8
            except Exception as salvage:result['failure_cache_salvage_error']=type(salvage).__name__+': '+str(salvage)
        result['errors'].append(dict(type=type(error).__name__,message=str(error)));checkpoint();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--compile-only',action='store_true');mode.add_argument('--gpu',action='store_true')
    p.add_argument('--commit');p.add_argument('--out',type=Path);p.add_argument('--compile-receipt',type=Path)
    p.add_argument('--numeric-diagnostic',action='store_true');a=p.parse_args()
    if not(a.gpu or a.compile_only):
        c.need(not any((a.commit,a.out,a.compile_receipt,a.numeric_diagnostic)),'device options require explicit mode')
        value=c.plan()
    else:
        c.need(not a.compile_only or not(a.compile_receipt or a.numeric_diagnostic),'compile mode cannot use numeric options')
        c.need(not a.gpu or a.compile_receipt is not None,'numeric mode requires completed fresh compile receipt')
        value=run(a)
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
