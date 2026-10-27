#!/usr/bin/env python3
"""Default CPU plan; explicit bounded fresh-service Q32 bitwise/numerical falsifier."""
import argparse,gzip,importlib.util,json,os,socket,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
import bench_biv150_konly_service_q32_fresh_compile as parent
import biv150_konly_service_q32_cache as cache
import biv150_konly_service_q32_fresh_binding as binding
import biv150_konly_service_q32_native as native
import biv150_konly_q32_numeric_contract as reference
h=cache.h


def plan():
    return dict(CPU_only=True,profile='q32',query_rows=2048,kv_rows=6144,num_seqs=1,pages=773,
        geometry=dict(BLOCK_M=256,BLOCK_Q=32,TILE_SIZE=64,num_warps=16,num_stages=1),grid=[65,2],patterns=[0,1],
        roles=list(binding.ROLES),planned_reader_launches=4,encoder_launches=0,timing_attempted=False,
        original_BF16_quality_limit=.005,precision_gate=False,Graph_verified=False,side_stream_verified=False,
        actual_I8_hoist_verified=False,service_verified=False,GPU_executed=False)


def policy(hard,quality,diagnostic):
    required={'finite','codec','output_canaries','read_only_inputs','actual_IR','hoisted_unhoisted_bitwise','quantized_reference'}
    h.need(set(hard)==required and all(v is True for v in hard.values()),'bitwise/numerical hard gate failed')
    h.need(type(quality) is bool and type(diagnostic) is bool,'quality flags must be bool')
    h.need(quality or diagnostic,'original BF16 quality failed; explicit diagnostic required')
    return dict(requested_component_gates_passed=True,original_BF16_quality_passed=quality,numeric_diagnostic=diagnostic,
        precision_gate=False,model_verified=False,service_verified=False,performance_measured=False)


def run(args):
    h.need(all((args.commit,args.out,args.service_source,args.compile_out)) and not args.out.exists(),'exact commit/source/compile record/fresh output required')
    h.need(not args.out.resolve().is_relative_to(parent.ROOT),'output must be outside checkout')
    h.need(parent.git('rev-parse','HEAD')==args.commit and not parent.git('status','--porcelain'),'exact clean numeric frontend')
    h.need(not parent.git('diff',parent.MAIN,'--','vllm_fl','csrc'),'production differs from DAC')
    reviewed=binding.load_compile(args.compile_out);meta=reviewed['metadata']
    h.need(args.service_source.resolve()==Path(reviewed['proposal']['provenance']['actual_source_path']) and args.service_source.read_text()==reviewed['source'],'original actual359 source required')
    h.need(not {'torch','triton','numpy'}.intersection(sys.modules),'admission before framework import')
    args.out.mkdir(parents=True);result=dict(plan(),CPU_only=False,GPU_executed=False,commit=args.commit,completed=False,errors=[],reader_launches=0,reader_launches_attempted=0,CUDA_API_entered=False,
        compile_record=str(args.compile_out.resolve()),native_audit=reviewed['audit'],records={},patterns=[])
    def save():parent.save(args.out/'bitwise-summary.json',result)
    save()
    try:
        base=Path('/usr/local/lib/python3.12/site-packages')
        for name,path in [('sdk',base/'vllm/v1/attention/ops/triton_unified_attention.py'),('helpers',base/'vllm/v1/attention/ops/triton_attention_helpers.py'),('jit',base/'triton/spec/iluvatar/triton/runtime/jit.py')]:
            h.need(h.sha(path.read_bytes())==h.PINS[name],'installed source '+name)
        smi=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True);(args.out/'prelaunch-ixsmi.txt').write_text(smi)
        h.need('No running processes found' in smi,'GPU must be idle')
        with socket.socket() as sock:h.need(sock.connect_ex(('127.0.0.1',9031))!=0,'service9031 must be closed')
        os.environ['TRITON_CACHE_DIR']=str(args.out/'triton-cache')
        import numpy as np
        import torch
        from vllm.triton_utils import triton
        import biv150_prefill_probe_fixtures as fixture
        fixture.torch=torch;torch.set_num_threads(4);torch.set_flush_denormal(False)
        h.need(sys.byteorder=='little','little-endian byte oracle required')
        result.update(CUDA_API_entered=True,GPU_executed=True);save()
        target=triton.runtime.driver.active.get_current_target();h.need((target.backend,target.arch,target.warp_size)==('corex',71,64),'actual target')
        spec=importlib.util.spec_from_file_location('cjj_actual359_private_q32',args.service_source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        fn=getattr(module,meta['function_name'])
        imports={}
        for label,module_name,path in [('sdk','vllm.v1.attention.ops.triton_unified_attention',base/'vllm/v1/attention/ops/triton_unified_attention.py'),
                ('helpers','vllm.v1.attention.ops.triton_attention_helpers',base/'vllm/v1/attention/ops/triton_attention_helpers.py'),
                ('jit',type(fn).__module__,base/'triton/spec/iluvatar/triton/runtime/jit.py')]:
            loaded_path=Path(sys.modules[module_name].__file__).resolve()
            h.need(loaded_path==path.resolve() and h.sha(loaded_path.read_bytes())==h.PINS[label],'actual loaded module '+label)
            imports[label]=dict(path=str(loaded_path),sha256=h.sha(loaded_path.read_bytes()))
        imports['torch']=dict(path=str(Path(torch.__file__).resolve()));imports['private359']=dict(path=str(args.service_source.resolve()),sha256=h.sha(args.service_source.read_bytes()))
        result['actual_imports']=imports
        p,t,owner=parent.allocate(torch)
        guarded={r:torch.empty((2050,16,128),device='cuda',dtype=torch.bfloat16) for r in binding.ROLES}
        options={r:dict(meta['launch_options'],**({'ir_override':reviewed['override_path']} if r=='proposed-hoist' else {})) for r in binding.ROLES}
        def storage_interval(tensor):
            storage=tensor.untyped_storage();begin=storage.data_ptr();return begin,begin+storage.nbytes()
        source_intervals=[storage_interval(v) for k,v in t.items() if k!='out']
        for role,output in guarded.items():
            lo,hi=storage_interval(output)
            h.need(all(hi<=a or b<=lo for a,b in source_intervals),'output aliases read-only source/cache')
        h.need(storage_interval(guarded[binding.ROLES[0]])!=storage_interval(guarded[binding.ROLES[1]]),'role outputs alias')
        result['tensor_descriptors']={k:dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),device=str(v.device),
            storage_offset=v.storage_offset(),storage_interval=list(storage_interval(v))) for k,v in t.items()}
        calls={};kernels={}
        for role in binding.ROLES:
            current=dict(t,out=guarded[role][1:2049]);kw=parent.kwargs(p,current,True);calls[role]=kw
            kernel=fn.warmup(grid=tuple(meta['grid']),**kw,**options[role]);kernels[role]=kernel
            record=parent.preserve(kernel,args.out/role)
            observed={e:(args.out/role/('actual.'+e)).read_bytes() for e in record['files']}
            proof=binding.verify_recompile(record,reviewed['records'][role],observed,reviewed['actual'][role])
            if role=='proposed-hoist':proof['native']=native.audit(reviewed['override'],*(observed[e].decode() for e in ('ttgir','llir','asm')))
            else:h.need(len(native.llvm_observation(observed['llir'].decode())['Q_shared_reads_inside_KV'])==16,'unhoisted placement changed')
            record['binding']=proof;record['loaded_resources']=binding.loaded(kernel);result['records'][role]=record
            parent.save(args.out/role/'record-loaded.json',record);save()
        result.update(actual_binding_verified=True,actual_I8_hoist_verified=True,loaded_resources_verified=True,GPU_executed=True)
        def bytes_cpu(tensor):return tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
        def fingerprints():return {k:h.sha(bytes_cpu(v)) for k,v in dict(q=t['q'],encoded=owner,table=t['table'],lengths=t['lengths'],cu_q=t['cu_q'],k_descale=t['k_descale'],v_descale=t['v_descale']).items()}
        def snapshot(value):return value.detach().cpu().contiguous().clone()
        def persist(path,value):
            with gzip.open(path,'wb',compresslevel=3) as stream:torch.save(value,stream)
        for pattern in (0,1):
            cpu=fixture.cpu_inputs([2048],[6144],20261025,pattern)
            encoded,codes,scales,codec=reference.encoded_bytes(cpu['packed'][:,0],cpu['packed'][:,1],torch,np)
            refs=reference.references(cpu,codes,scales,torch,np)
            if pattern==0:
                values=cpu['packed'][cpu['table'][0,:384].long(),1].reshape(-1,2,128).double()
                visible=4096+torch.arange(2048)+1
                full=(values.cumsum(0)[visible-1]/visible[:,None,None]).repeat_interleave(8,dim=1)
                refs=dict(refs,rows=list(range(2048)),original=full,quantized=full)
            owner.copy_(torch.from_numpy(np.frombuffer(encoded,dtype=np.uint8).copy()))
            for name,source in [('q','q'),('table','table'),('lengths','lengths'),('cu_q','cu_q'),('k_descale','k_scale'),('v_descale','v_scale')]:t[name].copy_(cpu[source])
            torch.cuda.synchronize();codec_passed=bytes_cpu(owner)==encoded
            before=fingerprints();outputs={};artifact=dict(original_CPU=cpu,expected_encoded_bytes=torch.from_numpy(np.frombuffer(encoded,dtype=np.uint8).copy()),
                actual_encoded_bytes=snapshot(owner),expected_codes=codes,expected_scales=scales,reference=refs,
                outputs={},guarded_before={},guarded_after={},readonly_before=before,readonly_after={})
            path=args.out/('pattern-'+str(pattern)+'.pt.gz');persist(path,artifact)
            h.need(codec_passed,'uploaded encoded cache differs from CPU codec')
            for role in binding.ROLES:
                guarded[role].fill_(71);calls[role]['output_ptr'].fill_(float('nan'));pre=snapshot(guarded[role])
                artifact['guarded_before'][role]=pre;persist(path,artifact)
                h.need(bool((pre[0]==71).all() and (pre[2049]==71).all() and torch.isnan(pre[1:2049]).all()),'output setup canary/NaN failed')
                result['reader_launches_attempted']+=1;save()
                launched=fn[tuple(meta['grid'])](**calls[role],**options[role]);result['reader_launches']+=1;torch.cuda.synchronize()
                h.need(launched is kernels[role],'launched different compiled object')
                post=snapshot(guarded[role]);output=snapshot(calls[role]['output_ptr']);after=fingerprints()
                artifact['guarded_after'][role]=post;artifact['outputs'][role]=output;artifact['readonly_after'][role]=after;outputs[role]=output
                persist(path,artifact)
                h.need(bool((post[0]==71).all() and (post[2049]==71).all() and torch.isfinite(post[1:2049]).all()),'full CPU output canary/finite failed')
                h.need(bytes_cpu(output)==bytes_cpu(post[1:2049]),'output view/full storage mismatch')
                h.need(after==before,'read-only input modified')
                save()
            equal=bytes_cpu(outputs['unhoisted'])==bytes_cpu(outputs['proposed-hoist'])
            hard=dict(finite=True,codec=codec_passed,output_canaries=True,read_only_inputs=True,actual_IR=True,hoisted_unhoisted_bitwise=equal,quantized_reference=True)
            metrics={}
            for role,output in outputs.items():
                sample=output[refs['rows']].double().reshape(-1).tolist()
                quant=reference.metrics(sample,refs['quantized'].reshape(-1).tolist());original=reference.metrics(sample,refs['original'].reshape(-1).tolist())
                hard['quantized_reference']=hard['quantized_reference'] and quant['finite'] and quant['pointwise_passed']
                metrics[role]=dict(quantized_operand=quant,original_BF16=original)
            quality=all(m['original_BF16']['finite'] and m['original_BF16']['relative_rmse']<=.005 for m in metrics.values())
            row=dict(pattern=pattern,hard_gates=hard,metrics=metrics,codec=codec,artifact=path.name,**policy(hard,quality,args.numeric_diagnostic))
            result['patterns'].append(row);save()
        h.need(result['reader_launches']==4,'bounded reader launch count changed')
        result.update(completed=True,requested_component_gates_passed=True,hoisted_unhoisted_full_output_bitwise=True,
            original_BF16_quality_passed=all(r['original_BF16_quality_passed'] for r in result['patterns']),full_numerical_matrix_verified=False,requested_component_numerical_verified=True,model_verified=False)
        save();return result
    except Exception as exc:
        result['errors'].append(dict(type=type(exc).__name__,message=str(exc)));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--commit');p.add_argument('--out',type=Path)
    p.add_argument('--service-source',type=Path);p.add_argument('--compile-out',type=Path);p.add_argument('--numeric-diagnostic',action='store_true');a=p.parse_args()
    if a.gpu:value=run(a)
    else:
        h.need(not any((a.commit,a.out,a.service_source,a.compile_out,a.numeric_diagnostic)),'device options require explicit --gpu')
        value=plan()
    print(json.dumps(value,sort_keys=True,indent=2))
if __name__=='__main__':main()
