#!/usr/bin/env python3
"""Stdlib CPU default; explicit reviewed Root consumer, no timing/service entry."""
import argparse
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import traceback

import biv150_native_k8_compact_factory as factory
import biv150_native_k8_compact_consumer_contract as contract
import biv150_native_k8_consumer_v2_fixture as f
from biv150_native_k8_codec_receipt import require_codec


def compact_bytes(canonical):
    assert type(canonical) is bytes and len(canonical)==f.LAYER_OFFSET+f.PAGES*16384+f.CACHE_GUARD
    result=bytearray(canonical[:f.LAYER_OFFSET]+b'\xa5'*(f.PAGES*12544)+canonical[-f.CACHE_GUARD:])
    for page in range(f.PAGES):
        old=f.LAYER_OFFSET+page*16384;new=f.LAYER_OFFSET+page*12544
        result[new:new+4224]=canonical[old:old+4224]
        result[new+4352:new+12544]=canonical[old+8192:old+16384]
    return bytes(result)


def plan():
    pins={}
    for name in ('biv150_native_k8_consumer_fixture.py','biv150_native_k8_consumer_v2_fixture.py',
                 'biv150_native_k8_pagewise_reference.py','biv150_native_k8_codec_fixture.py'):
        raw=(factory.ROOT/'tools'/name).read_bytes()
        assert raw==factory.blob(factory.PARENT,'tools/'+name)
        pins[name]=factory.sha(raw)
    assert factory.sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
    return dict(component='compact12544 vs original22e full O/M/L/final consumer equivalence',
        source_proof=factory.verify(),reference=f.plan(),reference_source_pins=pins,
        cases=[list(c) for c in f.CASES],maximum_context=193,cache_pages=f.PAGES,
        scratch_backing_batch=64,scratch_backing_segments=16,layer_offset_bytes=f.LAYER_OFFSET,
        full_field_bitwise_required=True,reference_version='pagewise_BF16_P_v2',thresholds_changed=False,
        original_BF16_proxy_separate=True,original_library_sha256=f.LIBRARY_SHA,
        compact_compile_and_actual_SO_review_required=True,live_process_epoch_required=True,
        old_library_qualification_is_not_compact_qualification=True,
        attention_launches=0,encoder_launches=0,reducer_launches=0,timing=None,
        implementation_verified=False,Graph_verified=False,side_stream_verified=False,
        full_field_bitwise_verified=False,eager_numeric_passed=False,original_BF16_proxy_passed=None,
        precision_gate=False,model_verified=False,service_installed=False,capacity_reclaimed=False,
        default_imports_framework=False,N40_supported=False)


def run(a):
    assert a.root_owned_exclusive,'Root must coordinate previous job terminal state and serialize device ownership'
    out=a.out.resolve();assert not out.exists(),'new output directory required';out.mkdir(parents=True)
    (out/'blobs').mkdir()
    result=dict(plan(),completed=False,errors=[],records=[],eager_producer_calls=0,eager_reducer_calls=0,
                capture_pairs=0,graph_replay_pairs=0,exclusion_authority='Root external coordination, no namespace/PID inference')
    def write(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    def checkpoint():
        result['attention_launches']=result['eager_producer_calls']+result['graph_replay_pairs']
        result['reducer_launches']=result['eager_reducer_calls']+result['graph_replay_pairs']
        result['attention_launch_counter_semantics']='eager dispatches plus replay nodes; capture node registrations separate'
        write(out/'consumer-summary.json',result)
    def blob(raw):
        digest=factory.sha(raw);path=out/'blobs'/(digest+'.bin.gz');packed=gzip.compress(raw,compresslevel=1,mtime=0)
        if path.exists():assert gzip.decompress(path.read_bytes())==raw
        else:path.write_bytes(packed)
        return dict(path=str(path.relative_to(out)),bytes=len(raw),sha256=digest,encoding='gzip',
                    stored_bytes=len(packed),stored_sha256=factory.sha(packed))
    def load(meta):
        packed=(out/meta['path']).read_bytes();assert factory.sha(packed)==meta['stored_sha256']
        raw=gzip.decompress(packed);assert len(raw)==meta['bytes'] and factory.sha(raw)==meta['sha256'];return raw
    try:
        # All binary/review/codec admission occurs before any device imports.
        result['requested_admission_paths']={key:str(value) for key,value in vars(a).items() if isinstance(value,Path)}
        for label,path in [('compile-summary',a.compile_summary),('actual-so-review',a.assembly_review),('original-codec-summary',a.codec_summary)]:
            (out/(label+'-input.json')).write_bytes(path.read_bytes())
        result['compact_admission']=contract.admission(a.compile_summary,a.assembly_review,a.assembly_review_sha256,a.compact_library)
        result['old_codec_admission']=require_codec(a.codec_summary,a.codec_proof_sha256)
        assert factory.sha(a.original_library.read_bytes())==f.LIBRARY_SHA
        result['source_root']=str(factory.ROOT)
        result['frontend_sources']={name:factory.sha((factory.ROOT/'tools'/name).read_bytes()) for name in
            ('probe_biv150_native_k8_compact_consumer.py','biv150_native_k8_compact_consumer_contract.py')}
        import numpy as np
        import torch
        os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache')
        assert torch.__version__=='2.10.0'
        torch.ops.load_library(str(a.original_library));torch.ops.load_library(str(a.compact_library))
        ops=dict(original=torch.ops.cjj_native_k8_qk_probe.produce,compact=torch.ops.cjj_native_k8_compact_probe.produce)
        schemas={name:str(op.default._schema) for name,op in ops.items()}
        assert schemas['compact'].replace(factory.NAMESPACE,'cjj_native_k8_qk_probe')==schemas['original']
        result['actual_schemas']=schemas
        spec=importlib.util.spec_from_file_location('cjj_compact_original_b3_reducer',f.REDUCER_FILE)
        base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target();props=torch.cuda.get_device_properties(0)
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64)
        assert 'BI-V150' in props.name and props.multi_processor_count==16
        result['target']=[target.backend,target.arch,target.warp_size]
        def host(raw,dtype):return torch.frombuffer(bytearray(raw),dtype=dtype).clone()
        def raw(t):return bytes(t.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
        def observe(t):
            s=t.untyped_storage()
            return dict(dtype=str(t.dtype).removeprefix('torch.'),shape=tuple(t.shape),strides=tuple(t.stride()),
                pointer=t.data_ptr(),storage_id=int(s._cdata),storage_base=s.data_ptr(),storage_bytes=s.nbytes(),
                storage_offset=t.storage_offset(),device=str(t.device))
        def npblob(x):
            stream=io.BytesIO();np.save(stream,x,allow_pickle=False);return blob(stream.getvalue())
        fields=('partial','maximum','summation','output')
        all_proxy=[]
        for n,s,pitch in f.CASES:
            cases={p:f.fixture(n,s,pitch,p) for p in (0,1,2)};gold={p:f.golden(cases[p]) for p in cases}
            expected={p:f.input_bytes(cases[p]) for p in cases}
            assert all(expected[0][key]!=expected[1][key] for key in ('query','cache','lengths','table'))
            for p,case in cases.items():
                for row,length in enumerate(case['lengths'][f.GUARD_ELEMENTS:-f.GUARD_ELEMENTS]):
                    assert type(length) is int and 0<=length<=193
                    ids=case['table'][f.GUARD_ELEMENTS+row*f.COLS:f.GUARD_ELEMENTS+(row+1)*f.COLS]
                    used=(length+15)//16
                    assert all(type(x) is int and 0<=x<f.PAGES for x in ids[:used])
                    assert ids[used:]==[-1]*(f.COLS-used)
            record=dict(case=[n,s,pitch],arms={},golden={});result['records'].append(record)
            for p,g in gold.items():record['golden'][str(p)]={k:npblob(v) if isinstance(v,np.ndarray) else v for k,v in g.items()}
            for arm in ('original','compact'):
                e={p:dict(expected[p]) for p in expected}
                if arm=='compact':
                    for p in e:e[p]['cache']=compact_bytes(e[p]['cache'])
                ar=dict(snapshots={},checks=[],epoch_records=[],completed=False);record['arms'][arm]=ar
                inputs={k:host(value,torch.uint8).to('cuda') for k,value in e[0].items()}
                q=inputs['query'].view(torch.bfloat16).as_strided((n,16,128),(pitch,128,1),storage_offset=f.GUARD_ELEMENTS)
                kstep,stotal,vstep,voffset=(12544,3136,6272,4352) if arm=='compact' else (16384,4096,8192,8192)
                key=inputs['cache'].view(torch.int8).as_strided((f.PAGES,16,2,128),(kstep,256,128,1),storage_offset=f.LAYER_OFFSET)
                scale=inputs['cache'].view(torch.float32).narrow(0,f.LAYER_OFFSET//4,f.PAGES*stotal)
                value=inputs['cache'].view(torch.bfloat16).as_strided((f.PAGES,16,2,128),(vstep,256,128,1),storage_offset=(f.LAYER_OFFSET+voffset)//2)
                lengths=inputs['lengths'].view(torch.int32).narrow(0,f.GUARD_ELEMENTS,n)
                table=inputs['table'].view(torch.int32).as_strided((n,f.COLS),(f.COLS,1),storage_offset=f.GUARD_ELEMENTS)
                shapes=dict(partial=(64,16,16,128),maximum=(64,16,16),summation=(64,16,16),output=(64,16,128))
                full={};views={}
                for field,shape in shapes.items():
                    count=int(np.prod(shape));dtype=torch.bfloat16 if field=='output' else torch.float32
                    t=torch.empty(count+2*f.GUARD_ELEMENTS,dtype=dtype,device='cuda')
                    t.view(torch.int16 if field=='output' else torch.int32).fill_(f.BF16_GUARD if field=='output' else f.F32_GUARD)
                    full[field]=t;core=t.narrow(0,f.GUARD_ELEMENTS,count).reshape(shape)
                    views[field]=core[:n] if field=='output' else core[:n,:,:s]
                assert views['partial'].stride()==(32768,2048,128,1)
                assert views['maximum'].stride()==views['summation'].stride()==(256,16,1)
                gate=None;lease=None
                if arm=='compact':
                    gate=contract.LiveEpoch(inputs['cache'],dict(key=key,scale=scale,value=value),observe,raw,
                        pages=f.PAGES,layer_offset=f.LAYER_OFFSET,format_id=factory.FORMAT)
                    lease=gate.activate(e[0]['cache'])
                ar['binding']=dict(inputs={k:observe(t) for k,t in inputs.items()},query=observe(q),key=observe(key),
                    scale=observe(scale),value=observe(value),scratch={k:observe(t) for k,t in views.items()})
                def admitted():
                    if gate is not None:
                        return gate.admit(lease,readonly=(inputs['query'],inputs['lengths'],inputs['table']),outputs=tuple(full.values()))
                    return dict(original22e_typed_host=True,compact_qualification_inherited=False)
                def poison():
                    for field,t in views.items():t.view(torch.int16 if field=='output' else torch.int32).fill_(f.BF16_POISON if field=='output' else f.F32_POISON)
                    torch.cuda.synchronize()
                def snapshot(stage):
                    arrays={**{k:raw(t) for k,t in inputs.items()},**{k:raw(t) for k,t in full.items()}}
                    ar['snapshots'][stage]={k:blob(value) for k,value in arrays.items()};return arrays
                def check(stage,p,producer_only=False):
                    arrays=snapshot(stage);checks={k+'_readonly':arrays[k]==value for k,value in e[p].items()};metrics={}
                    for field,shape in shapes.items():
                        output=field=='output';dtype='<u2' if output else '<u4';marker=f.BF16_GUARD if output else f.F32_GUARD
                        bits=np.frombuffer(arrays[field],dtype=dtype);checks[field+'_outside_guards']=bool((bits[:f.GUARD_ELEMENTS]==marker).all() and (bits[-f.GUARD_ELEMENTS:]==marker).all())
                        core=bits[f.GUARD_ELEMENTS:-f.GUARD_ELEMENTS].reshape(shape)
                        layout=np.zeros(shape,dtype=bool)
                        if output:layout[:n]=True;active=core[:n]
                        else:layout[:n,:,:s]=True;active=core[:n,:,:s]
                        checks[field+'_physical_padding_guard']=bool((core[~layout]==marker).all())
                        if output:
                            if producer_only:checks['output_still_poison']=bool((active==f.BF16_POISON).all());continue
                            actual=(active.astype(np.uint32)<<16).view('<f4').astype(np.float64)
                            metrics[field]=f.metrics(actual,gold[p][field]);checks[field+'_FP64']=metrics[field]['passed']
                            metrics['original_BF16_proxy']=f.metrics(actual,gold[p]['original_BF16_output'])
                            all_proxy.append(metrics['original_BF16_proxy']['passed'])
                            zero=np.array(cases[p]['lengths'][f.GUARD_ELEMENTS:-f.GUARD_ELEMENTS])==0
                            checks['zero_length_output_positive_zero']=bool((active[zero]==0).all())
                        else:
                            mask=np.broadcast_to(gold[p]['active'][:,None,:,None] if field=='partial' else gold[p]['active'][:,None,:],active.shape)
                            checks[field+'_inactive_poison']=bool((active[~mask]==f.F32_POISON).all())
                            actual=active.view('<f4').astype(np.float64)
                            metrics[field]=f.metrics(actual[mask],gold[p][field][mask]);checks[field+'_FP64']=metrics[field]['passed']
                            if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(actual[mask],gold[p]['ideal_partial_diagnostic'][mask])
                    ar['checks'].append(dict(stage=stage,pattern=p,checks=checks,metrics=metrics));checkpoint()
                    assert all(checks.values()),'saved full-array hard gate failed: '+arm+'/'+stage
                    return arrays
                handle=None
                def producer():
                    admitted()
                    ops[arm](q,key,scale,value,lengths,table,views['partial'],views['maximum'],views['summation'],128**-.5*1.4426950408889634,s)
                def reducer():
                    nonlocal handle
                    handle=base._reduce_partials[(n,16)](output_ptr=views['output'],segm_output_ptr=views['partial'],segm_max_ptr=views['maximum'],segm_expsum_ptr=views['summation'],seq_lens_ptr=lengths,
                        segment_stride_token=32768,segment_stride_head=2048,segment_stride_segment=128,segment_stride_dim=1,
                        max_stride_token=256,max_stride_head=16,max_stride_segment=1,output_stride_token=2048,output_stride_head=128,
                        page_size=16,num_segments=s,HEAD_SIZE=128,num_warps=4)
                def eager(stage,p):
                    snapshot(stage+'-before');producer();result['eager_producer_calls']+=1;torch.cuda.synchronize();mid=check(stage+'-producer',p,True)
                    reducer();result['eager_reducer_calls']+=1;torch.cuda.synchronize();after=check(stage+'-after',p)
                    assert all(mid[k]==after[k] for k in fields[:3]),'Reducer mutated O/M/L'
                    return after
                def change(p):
                    nonlocal lease
                    for field,value_bytes in e[p].items():inputs[field].copy_(host(value_bytes,torch.uint8))
                    poison()
                    upload_check(p)
                    if gate is not None:lease=gate.activate(e[p]['cache'])
                    ar['epoch_records'].append(dict(pattern=p,admission=admitted()))
                def upload_check(p):
                    got={k:raw(t) for k,t in inputs.items()}
                    ar.setdefault('upload_readbacks',{})[str(p)]={k:blob(b) for k,b in got.items()}
                    assert all(got[k]==b for k,b in e[p].items()),'Actual full input upload mismatch before reader'
                poison();upload_check(0);ar['epoch_records'].append(dict(pattern=0,admission=admitted()));eager('A',0)
                graph_tensors=dict(query=q,key=key,scale=scale,value=value,lengths=lengths,table=table,**views)
                graph_binding={k:observe(t) for k,t in graph_tensors.items()}
                ar['graph_capture_binding']=graph_binding
                graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(graph,stream=stream):producer();reducer()
                result['capture_pairs']+=1;stream.synchronize()
                change(1);before=snapshot('graphB-before')
                for field in fields:
                    active=views[field].detach().cpu().contiguous().view(torch.int16 if field=='output' else torch.int32)
                    assert bool((active==(f.BF16_POISON if field=='output' else f.F32_POISON)).all())
                assert all(before[k]==v for k,v in e[1].items())
                assert {k:observe(t) for k,t in graph_tensors.items()}==graph_binding,'Graph pointer/layout changed'
                admitted();graph.replay();result['graph_replay_pairs']+=1;torch.cuda.synchronize();gb=check('graphB-after',1)
                poison();eb=eager('eagerB',1);assert all(gb[k]==eb[k] for k in gb),'Changed Graph B differs from fresh eager B'
                change(2);snapshot('sideC-before');side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
                admitted()
                with torch.cuda.stream(side):producer();reducer()
                result['eager_producer_calls']+=1;result['eager_reducer_calls']+=1;side.synchronize();check('sideC-after',2)
                ar['reducer']=dict(hash=str(handle.hash),n_regs=getattr(handle,'n_regs',None),n_spills=getattr(handle,'n_spills',None),
                    source_sha256=factory.sha(handle.src.fn.src.encode()),artifacts={})
                assert handle.src.fn is base._reduce_partials
                for extension,data in handle.asm.items():
                    if isinstance(data,(str,bytes)):ar['reducer']['artifacts'][extension]=blob(data.encode() if isinstance(data,str) else data)
                if gate is not None:
                    old_lease=lease;gate.close()
                    try:gate.admit(old_lease)
                    except AssertionError:ar['closed_epoch_rejected']=True
                    else:raise AssertionError('closed epoch accepted')
                ar['completed']=True;checkpoint()
            record['cross_arm_full_bitwise']={}
            for stage in ('A-producer','A-after','graphB-after','eagerB-producer','eagerB-after','sideC-after'):
                rows={field:load(record['arms']['original']['snapshots'][stage][field])==load(record['arms']['compact']['snapshots'][stage][field]) for field in fields}
                record['cross_arm_full_bitwise'][stage]=rows;checkpoint()
                assert all(rows.values()),'compact vs original22e full O/M/L/final bit mismatch: '+stage
            record['completed']=True
        assert factory.sha(a.original_library.read_bytes())==f.LIBRARY_SHA
        assert factory.sha(a.compact_library.read_bytes())==result['compact_admission']['library_sha256']
        assert factory.sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
        result.update(completed=True,implementation_verified=True,full_field_bitwise_verified=True,Graph_verified=True,side_stream_verified=True,
                      original_BF16_proxy_passed=all(all_proxy),eager_numeric_passed=all(all_proxy),precision_gate=False)
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifact_manifest']=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=factory.sha(p.read_bytes()))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='consumer-summary.json']
    checkpoint();print(json.dumps(result,sort_keys=True,indent=2));return 0 if result['completed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--gpu',action='store_true');mode.add_argument('--review-template',action='store_true')
    p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--compile-summary',type=Path);p.add_argument('--assembly-review',type=Path);p.add_argument('--assembly-review-sha256')
    p.add_argument('--compact-library',type=Path);p.add_argument('--original-library',type=Path)
    p.add_argument('--codec-summary',type=Path);p.add_argument('--codec-proof-sha256')
    a=p.parse_args()
    if a.review_template:
        assert a.compile_summary and not any((a.root_owned_exclusive,a.out,a.assembly_review,a.assembly_review_sha256,a.compact_library,a.original_library,a.codec_summary,a.codec_proof_sha256))
        print(json.dumps(contract.review_template(a.compile_summary),sort_keys=True,indent=2));return 0
    if not a.gpu:
        assert not any((a.root_owned_exclusive,a.out,a.compile_summary,a.assembly_review,a.assembly_review_sha256,a.compact_library,a.original_library,a.codec_summary,a.codec_proof_sha256))
        assert not {'torch','triton','numpy'}.intersection(sys.modules)
        print(json.dumps(plan(),sort_keys=True,indent=2));return 0
    assert a.out is not None and a.out.is_absolute()
    assert all((a.root_owned_exclusive,a.compile_summary,a.assembly_review,a.assembly_review_sha256,a.compact_library,a.original_library,a.codec_summary,a.codec_proof_sha256))
    return run(a)


if __name__=='__main__':raise SystemExit(main())
