#!/usr/bin/env python3
"""Default stdlib plan. Explicit Root GPU needs an independently replayed codec pass."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import traceback
import biv150_native_k8_consumer_v2_fixture as f
from biv150_native_k8_codec_receipt import require_codec


def run(args):
    # This is deliberately before Torch import, input construction or CUDA use.
    receipt=require_codec(args.codec_summary,args.codec_proof_sha256)
    assert f.sha(args.library.read_bytes())==f.LIBRARY_SHA
    out=args.out.resolve();assert not out.exists(),'new output directory required';out.mkdir(parents=True)
    (out/'blobs').mkdir()
    result=dict(f.plan(),codec_precondition=receipt,completed=False,errors=[],records=[],producer_calls=0,reducer_calls=0,capture_calls=0,graph_replays=0)
    def write_json(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    def blob(raw):
        digest=f.sha(raw);path=out/'blobs'/(digest+'.bin')
        if path.exists():assert path.read_bytes()==raw
        else:path.write_bytes(raw)
        return dict(path=str(path.relative_to(out)),bytes=len(raw),sha256=digest)
    def npblob(array):
        stream=io.BytesIO();np.save(stream,array,allow_pickle=False);return blob(stream.getvalue())
    try:
        import numpy as np
        import torch
        os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache')
        assert torch.__version__=='2.10.0'
        torch.ops.load_library(str(args.library));op=torch.ops.cjj_native_k8_qk_probe.produce
        assert f.sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
        spec=importlib.util.spec_from_file_location('cjj_native_k8_original_b3_reducer',f.REDUCER_FILE)
        base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target()
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        props=torch.cuda.get_device_properties(0);assert 'BI-V150' in props.name and props.multi_processor_count==16
        result['imports']=dict(torch=torch.__file__,numpy=np.__file__,reducer=base.__file__,triton=triton.__file__)
        result['target']=[target.backend,target.arch,target.warp_size];result['schema']=str(op.default._schema)
        def host(raw,dtype):return torch.frombuffer(bytearray(raw),dtype=dtype).clone()
        def bytes_gpu(tensor):return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
        def bf16_float(raw):return (np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view('<f4').astype(np.float64)
        for n,s,pitch in f.CASES:
            name='n'+str(n)+'s'+str(s)+'q'+str(pitch);folder=out/name;folder.mkdir()
            cases={p:f.fixture(n,s,pitch,p) for p in (0,1,2)};gold={p:f.golden(cases[p]) for p in cases}
            assert all(f.input_bytes(cases[0])[k]!=f.input_bytes(cases[1])[k] for k in ('query','cache','lengths','table'))
            assert cases[0]['qscales']!=cases[1]['qscales']
            record=dict(case=[n,s,pitch],snapshots={},checks=[],golden={});result['records'].append(record)
            for p,g in gold.items():
                record['golden'][str(p)]={k:npblob(v) if isinstance(v,np.ndarray) else v for k,v in g.items()}
            inputs={k:host(raw,torch.uint8).to('cuda') for k,raw in f.input_bytes(cases[0]).items()}
            q=inputs['query'].view(torch.bfloat16).as_strided((n,16,128),(pitch,128,1),storage_offset=f.GUARD_ELEMENTS)
            key=inputs['cache'].view(torch.int8).as_strided((f.PAGES,16,2,128),(16384,256,128,1),storage_offset=f.LAYER_OFFSET)
            scale=inputs['cache'].view(torch.float32).narrow(0,f.LAYER_OFFSET//4,f.PAGES*4096)
            value=inputs['cache'].view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(8192,256,128,1),storage_offset=(f.LAYER_OFFSET+8192)//2)
            lengths=inputs['lengths'].view(torch.int32).narrow(0,f.GUARD_ELEMENTS,n)
            table=inputs['table'].view(torch.int32).as_strided((n,f.COLS),(f.COLS,1),storage_offset=f.GUARD_ELEMENTS)
            shapes={'partial':(n,16,s,128),'maximum':(n,16,s),'summation':(n,16,s),'output':(n,16,128)}
            full={};views={}
            for field,shape in shapes.items():
                width=2 if field=='output' else 4;dtype=torch.bfloat16 if width==2 else torch.float32
                count=1
                for size in shape:count*=size
                full[field]=torch.empty(count+2*f.GUARD_ELEMENTS,dtype=dtype,device='cuda')
                full[field].view(torch.int16 if width==2 else torch.int32).fill_(f.BF16_GUARD if width==2 else f.F32_GUARD)
                views[field]=full[field].narrow(0,f.GUARD_ELEMENTS,count).reshape(shape)
            record['binding']=dict(query_shape=list(q.shape),query_stride=list(q.stride()),query_offset=q.storage_offset(),
              key_stride=list(key.stride()),key_pointer=key.data_ptr(),scale_pointer=scale.data_ptr(),V_pointer=value.data_ptr(),
              cache_base_pointer=inputs['cache'].data_ptr(),layer_offset=f.LAYER_OFFSET)
            assert key.data_ptr()==scale.data_ptr()==inputs['cache'].data_ptr()+f.LAYER_OFFSET and value.data_ptr()==key.data_ptr()+8192
            def poison():
                for field,tensor in views.items():tensor.view(torch.int16 if field=='output' else torch.int32).fill_(f.BF16_POISON if field=='output' else f.F32_POISON)
                torch.cuda.synchronize()
            def change(p):
                for field,raw in f.input_bytes(cases[p]).items():inputs[field].copy_(host(raw,torch.uint8))
                poison()
            def snapshot(stage):
                arrays={**{k:bytes_gpu(t) for k,t in inputs.items()},**{k:bytes_gpu(t) for k,t in full.items()}}
                record['snapshots'][stage]={k:blob(raw) for k,raw in arrays.items()};return arrays
            def check(stage,p,producer_only=False):
                arrays=snapshot(stage);checks={};metrics={};g=gold[p]
                for field,expected in f.input_bytes(cases[p]).items():checks[field+'_readonly']=arrays[field]==expected
                for field,shape in shapes.items():
                    width=2 if field=='output' else 4;fmt='<H' if width==2 else '<I';marker=f.BF16_GUARD if width==2 else f.F32_GUARD
                    checks[field+'_guards']=arrays[field][:f.GUARD_ELEMENTS*width]==struct.pack(fmt,marker)*f.GUARD_ELEMENTS and arrays[field][-f.GUARD_ELEMENTS*width:]==struct.pack(fmt,marker)*f.GUARD_ELEMENTS
                    middle=arrays[field][f.GUARD_ELEMENTS*width:-f.GUARD_ELEMENTS*width]
                    if field=='output':
                        if producer_only:checks['output_still_poison']=middle==struct.pack('<H',f.BF16_POISON)*(n*16*128)
                        else:
                            actual=bf16_float(middle).reshape(shape);metrics['output']=f.metrics(actual,g['output']);checks['output_FP64']=metrics['output']['passed']
                            metrics['original_BF16_proxy']=f.metrics(actual,g['original_BF16_output'])
                            zero=np.array(cases[p]['lengths'][f.GUARD_ELEMENTS:-f.GUARD_ELEMENTS])==0
                            checks['empty_exact_zero']=bool((np.frombuffer(middle,dtype='<u2').reshape(shape)[zero]==0).all())
                    else:
                        bits=np.frombuffer(middle,dtype='<u4').reshape(shape);actual=bits.view('<f4').astype(np.float64)
                        mask=np.broadcast_to(g['active'][:,None,:,None] if field=='partial' else g['active'][:,None,:],shape)
                        checks[field+'_inactive_poison']=bool((bits[~mask]==f.F32_POISON).all())
                        metrics[field]=f.metrics(actual[mask],g[field][mask]);checks[field+'_FP64']=metrics[field]['passed']
                        if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(actual[mask],g['ideal_partial_diagnostic'][mask])
                item=dict(stage=stage,pattern=p,checks=checks,metrics=metrics);record['checks'].append(item);write_json(folder/(stage+'-checks.json'),item)
                assert all(checks.values()),'Full numerical/guard/readonly gate: '+json.dumps(item,sort_keys=True)
                return arrays
            def producer():
                op(q,key,scale,value,lengths,table,views['partial'],views['maximum'],views['summation'],128**-.5*1.4426950408889634,s)
            reducer_handle=None
            def reducer():
                nonlocal reducer_handle
                reducer_handle=base._reduce_partials[(n,16)](output_ptr=views['output'],segm_output_ptr=views['partial'],
                  segm_max_ptr=views['maximum'],segm_expsum_ptr=views['summation'],seq_lens_ptr=lengths,
                  segment_stride_token=views['partial'].stride(0),segment_stride_head=views['partial'].stride(1),segment_stride_segment=views['partial'].stride(2),segment_stride_dim=1,
                  max_stride_token=views['maximum'].stride(0),max_stride_head=views['maximum'].stride(1),max_stride_segment=1,
                  output_stride_token=views['output'].stride(0),output_stride_head=128,page_size=16,num_segments=s,HEAD_SIZE=128,num_warps=4)
            def eager(stage,p):
                snapshot(stage+'-before');producer();result['producer_calls']+=1;torch.cuda.synchronize();mid=check(stage+'-producer',p,True)
                reducer();result['reducer_calls']+=1;torch.cuda.synchronize();after=check(stage+'-after',p)
                assert all(mid[k]==after[k] for k in ('partial','maximum','summation')),'Reducer mutated OML'
                return after
            poison();eager('A',0)
            graph=torch.cuda.CUDAGraph();capture=torch.cuda.Stream();capture.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph,stream=capture):producer();reducer()
            result['capture_calls']+=1;capture.synchronize()
            change(1);pre=snapshot('graphB-before')
            for field,shape in shapes.items():
                width=2 if field=='output' else 4;count=1
                for x in shape:count*=x
                expected=struct.pack('<H' if width==2 else '<I',f.BF16_POISON if width==2 else f.F32_POISON)*count
                assert pre[field][f.GUARD_ELEMENTS*width:-f.GUARD_ELEMENTS*width]==expected
            graph.replay();result['graph_replays']+=1;torch.cuda.synchronize();gb=check('graphB-after',1)
            poison();eb=eager('eagerB',1);assert all(gb[k]==eb[k] for k in gb),'Graph changed-input result differs from new eager'
            change(2);snapshot('sideC-before');side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):producer();reducer()
            result['producer_calls']+=1;result['reducer_calls']+=1;side.synchronize();check('sideC-after',2)
            assert reducer_handle.src.fn is base._reduce_partials
            record['reducer']=dict(hash=str(reducer_handle.hash),n_regs=getattr(reducer_handle,'n_regs',None),n_spills=getattr(reducer_handle,'n_spills',None),source_sha256=f.sha(reducer_handle.src.fn.src.encode()))
            for extension,data in reducer_handle.asm.items():
                if isinstance(data,(str,bytes)):record['reducer'].setdefault('artifacts',{})[extension]=blob(data.encode() if isinstance(data,str) else data)
            record['completed']=True;write_json(folder/'record.json',record);write_json(out/'consumer-summary.json',result)
        assert f.sha(args.library.read_bytes())==f.LIBRARY_SHA and f.sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
        result.update(completed=True,consumer_numeric_verified=True,Graph_verified=True,side_stream_verified=True,requested_consumer_gates_passed=True)
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifact_manifest']=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=f.sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='consumer-summary.json']
    write_json(out/'consumer-summary.json',result);print(json.dumps(result,sort_keys=True,indent=2));return 0 if result['completed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true');parser.add_argument('--library',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--codec-summary',type=Path);parser.add_argument('--codec-proof-sha256');args=parser.parse_args()
    if not args.gpu:
        if any((args.library,args.out,args.codec_summary,args.codec_proof_sha256)):raise ValueError('Device arguments require explicit --gpu')
        assert not {'torch','triton','numpy'}.intersection(sys.modules);print(json.dumps(f.plan(),sort_keys=True,indent=2));return 0
    if not all((args.library,args.out,args.codec_summary,args.codec_proof_sha256)):raise ValueError('Actual codec pass receipt+SHA, existing hard-SHA library, and new out are mandatory')
    return run(args)

if __name__=='__main__':raise SystemExit(main())
