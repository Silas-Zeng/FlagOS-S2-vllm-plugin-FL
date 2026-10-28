#!/usr/bin/env python3
"""Default stdlib plan; explicit N48/S4 four-case numeric/Graph/side component."""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import traceback

import biv150_native_k8_compact_n48_factory as source
import biv150_native_k8_compact_n48_contract as shape
import biv150_native_k8_compact_n48_consumer_contract as admission
import biv150_native_k8_consumer_v2_fixture as f
from biv150_native_k8_codec_receipt import require_codec

CASES=((41,2048),(41,2560),(42,2048),(42,2560))
FIELDS=('partial','maximum','summation','output')
SHAPES=dict(partial=(64,16,16,128),maximum=(64,16,16),summation=(64,16,16),output=(64,16,128))
PHASES=('A-producer','A-after','graphB-after','eagerB-producer','eagerB-after','sideC-after')


def fixtures(live,pitch):
    assert (live,pitch) in CASES
    result={}
    nonzero=[n for n in f.BOUNDARIES if n]
    for phase in (0,1,2):
        item=shape.small_fixture(pitch,phase);case=copy.deepcopy(item['canonical_case']);g=f.GUARD_ELEMENTS
        active={(row+phase*3)%48 for row in range(live)}
        lengths=[nonzero[(row+phase*5)%len(nonzero)] if row in active else 0 for row in range(48)]+[0]*16
        table=[-1]*(64*f.COLS)
        for row,length in enumerate(lengths):
            for col in range((length+15)//16):table[row*f.COLS+col]=(row*3+col*7+phase*5)%f.PAGES
        assert sum(n>0 for n in lengths)==live
        case.update(query_bits=item['query_backing64_bits'],rows=64,
          lengths=[-1234567]*g+lengths+[-1234567]*g,
          table=[-1]*g+table+[-1]*g,
          qcodes=case['qcodes']+[[[0]*128 for _ in range(16)] for _ in range(16)],
          qscales=case['qscales']+[[1.]*16 for _ in range(16)])
        # Both arms read identical complete input buffers. Only compact encoded
        # bytes are uploaded; there is no writer/encoder or implicit16KiB view.
        expected=f.input_bytes(case);expected['cache']=item['compact_cache']
        case48=copy.deepcopy(case)
        case48.update(rows=48,query_bits=case['query_bits'][:g+48*pitch]+case['query_bits'][-g:],
          lengths=[-1234567]*g+lengths[:48]+[-1234567]*g,
          table=[-1]*g+table[:48*f.COLS]+[-1]*g,qcodes=case['qcodes'][:48],qscales=case['qscales'][:48])
        result[phase]=dict(case64=case,case48=case48,expected=expected,live_rows=live)
    assert all(result[0]['expected'][k]!=result[1]['expected'][k] for k in ('query','cache','lengths','table'))
    return result


def plan():
    _,_,bindings=admission.contracts()
    return dict(component='private compact N48/S4 consumer+reducer, twoN32 numeric control',
      source=source.verify(),candidate_library_sha256=None,reference_library_sha256=admission.REFERENCE_LIBRARY_SHA,
      reference_consumer_summary_sha256=admission.REFERENCE_CONSUMER_SHA,
      reference_required='actual PR1107892 full library/compile/SO review and complete passed four-case consumer with Graph/side and all artifacts',
      candidate_required='fresh N48 compile+actual-SO inspection and CJJ review; no inherited library qualification',
      compile_binding_adapters=bindings,cases=[list(c) for c in CASES],rows=48,segments=4,
      pages=29,maximum_context=193,Q_pitches=[2048,2560],layer_offset_bytes=128,
      scratch_backing_rows=64,scratch_backing_segments=16,phases=list(PHASES),
      independent_reference='complete128D I64 dot / FP64; pagewise BF16-P partial; ideal quantized final',
      pointwise_atol=.005,pointwise_rtol=.005,RRMSE_limit=.005,ideal_partial_and_original_BF16_proxy_separate=True,
      reference_control_calls_per_producer=2,reference_control_calls_per_reducer=2,reference_per_call_rows=32,reference_padding_rows=16,
      actual_host_refusal_cases=['N48/S2','N32/S4'],
      bitwise_comparison='first48 rows all O/M/L/final; each arm complete own buffers/guards',
      Graph_changed_B_before_fresh_eager_B=True,side_C=True,writer_launches=0,encoder_launches=0,timing=None,
      thresholds_changed=False,
      framework_imported=False,numeric_verified=False,Graph_verified=False,side_stream_verified=False,
      performance_verified=False,service_verified=False,precision_gate=False)


def checked_admissions(a):
    """The exact device prerequisites, also available without framework imports."""
    candidate_contract,reference_contract,_=admission.contracts()
    c=candidate_contract.admission(a.candidate_compile_summary,a.candidate_review,a.candidate_review_sha256,a.candidate_library)
    r=reference_contract.admission(a.reference_compile_summary,a.reference_review,a.reference_review_sha256,a.reference_library)
    proof=admission.reference_consumer(a.reference_consumer_summary,a.reference_consumer_summary_sha256,r,a.reference_library_sha256)
    assert source.sha(a.reference_library.read_bytes())==a.reference_library_sha256
    codec=require_codec(a.codec_summary,a.codec_proof_sha256)
    return dict(candidate_admission=c,reference_admission=r,reference_consumer_admission=proof,codec_admission=codec)


def run(a):
    assert a.root_owned_exclusive
    out=a.out.resolve();assert not out.exists();out.mkdir(parents=True);(out/'blobs').mkdir()
    result=dict(plan(),completed=False,requested_consumer_gates_passed=False,errors=[],records=[],dispatches={'candidate':0,'reference':0},
      capture_dispatches={'candidate':0,'reference':0},graph_replays={'candidate':0,'reference':0},
      reducer_dispatches={'candidate':0,'reference':0},capture_reducer_dispatches={'candidate':0,'reference':0},
      root_owned_exclusive_acknowledged=True,exclusion_authority='Root external ended-owner and queue checks',
      stage='pre_device_admission',
      launch_count_meaning='producer dispatcher calls; capture registrations and graph replay pairs separate; twoN32 control has2 producers per step')
    def checkpoint():(out/'consumer-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    def blob(raw):
        digest=source.sha(raw);p=out/'blobs'/(digest+'.bin.gz');packed=gzip.compress(raw,compresslevel=1,mtime=0)
        if p.exists():assert gzip.decompress(p.read_bytes())==raw
        else:p.write_bytes(packed)
        return dict(path=str(p.relative_to(out)),bytes=len(raw),sha256=digest,encoding='gzip',stored_bytes=len(packed),stored_sha256=source.sha(packed))
    def load(meta):
        packed=(out/meta['path']).read_bytes();assert source.sha(packed)==meta['stored_sha256']
        raw=gzip.decompress(packed);assert len(raw)==meta['bytes'] and source.sha(raw)==meta['sha256'];return raw
    try:
        # Every prerequisite is evaluated before NumPy/Torch or CUDA imports.
        candidate_contract,reference_contract,_=admission.contracts()
        result.update(checked_admissions(a))
        result['requested_admission_paths']={k:str(v) for k,v in vars(a).items() if isinstance(v,Path)}
        result['source_root']=str(source.ROOT);result['frontend_sources']={}
        (out/'frontends').mkdir()
        for name in ('probe_biv150_native_k8_compact_n48_consumer.py','biv150_native_k8_compact_n48_consumer_contract.py',
          'biv150_native_k8_compact_n48_factory.py','biv150_native_k8_compact_n48_contract.py',
          'biv150_native_k8_consumer_fixture.py','biv150_native_k8_consumer_v2_fixture.py','biv150_native_k8_pagewise_reference.py'):
            raw_source=(source.ROOT/'tools'/name).read_bytes();result['frontend_sources'][name]=source.sha(raw_source)
            (out/'frontends'/name).write_bytes(raw_source)
        for k in ('candidate_compile_summary','candidate_review','reference_compile_summary','reference_review','reference_consumer_summary','codec_summary'):
            (out/(k+'-input.json')).write_bytes(getattr(a,k).read_bytes())
        assert source.sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
        import numpy as np
        import torch
        result['framework_imported']=True;result['stage']='device_setup'
        os.environ['TRITON_CACHE_DIR']=str(out/'triton-cache')
        assert str(torch.__version__)=='2.10.0' and torch._C._GLIBCXX_USE_CXX11_ABI is False
        torch.ops.load_library(str(a.candidate_library));torch.ops.load_library(str(a.reference_library))
        loaded={str(Path(x).resolve()) for x in torch.ops.loaded_libraries}
        assert all(str(path.resolve()) in loaded for path in (a.candidate_library,a.reference_library))
        result['actual_imports']=dict(torch_file=str(torch.__file__),torch_version=str(torch.__version__),cxx11_abi=False,
          candidate_library=dict(path=str(a.candidate_library),sha256=source.sha(a.candidate_library.read_bytes())),
          reference_library=dict(path=str(a.reference_library),sha256=source.sha(a.reference_library.read_bytes())),
          loaded_library_set_confirmed=True)
        ops=dict(candidate=torch.ops.cjj_native_k8_compact_n48_s4_probe.produce,
                 reference=torch.ops.cjj_native_k8_compact_v_overlap_probe.produce)
        schemas={k:str(op.default._schema) for k,op in ops.items()}
        assert schemas['candidate'].replace(source.NAMESPACE,'cjj_native_k8_compact_v_overlap_probe')==schemas['reference']
        result['actual_schemas']=schemas
        spec=importlib.util.spec_from_file_location('cjj_n48_original_reducer',f.REDUCER_FILE)
        reducer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reducer;spec.loader.exec_module(reducer)
        from vllm.triton_utils import triton
        target=triton.runtime.driver.active.get_current_target();props=torch.cuda.get_device_properties(0)
        assert (target.backend,target.arch,target.warp_size)==('corex',71,64) and 'BI-V150' in props.name and props.multi_processor_count==16
        result['actual_target']=[target.backend,target.arch,target.warp_size]
        def host(raw,dtype):return torch.frombuffer(bytearray(raw),dtype=dtype).clone()
        def raw(t):return bytes(t.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
        def observe(t):
            s=t.untyped_storage();return dict(dtype=str(t.dtype).removeprefix('torch.'),shape=tuple(t.shape),strides=tuple(t.stride()),pointer=t.data_ptr(),storage_id=int(s._cdata),storage_base=s.data_ptr(),storage_bytes=s.nbytes(),storage_offset=t.storage_offset(),device=str(t.device))
        def npblob(x):
            stream=io.BytesIO();np.save(stream,x,allow_pickle=False);return blob(stream.getvalue())
        def core_bytes(arrays,field):
            dtype='<u2' if field=='output' else '<u4'
            core=np.frombuffer(arrays[field],dtype=dtype)[64:-64].reshape(SHAPES[field])
            return (core[:48] if field=='output' else core[:48,:,:4]).copy().tobytes()
        proxies=[]
        for live,pitch in CASES:
            cases=fixtures(live,pitch);gold={arm:{p:f.golden(c['case48' if arm=='candidate' else 'case64']) for p,c in cases.items()} for arm in ops}
            record=dict(case=[48,4,pitch,live],arms={},golden={});result['records'].append(record)
            for arm in ops:record['golden'][arm]={str(p):{k:npblob(v) if isinstance(v,np.ndarray) else v for k,v in g.items()} for p,g in gold[arm].items()}
            for arm in ('reference','candidate'):
                n=64 if arm=='reference' else 48
                e={p:c['expected'] for p,c in cases.items()}
                ar=dict(completed=False,snapshots={},checks=[],epoch_records=[]);record['arms'][arm]=ar
                inputs={k:host(v,torch.uint8).to('cuda') for k,v in e[0].items()}
                q64=inputs['query'].view(torch.bfloat16).as_strided((64,16,128),(pitch,128,1),storage_offset=64)
                q=q64[:n]
                key=inputs['cache'].view(torch.int8).as_strided((29,16,2,128),(12544,256,128,1),storage_offset=128)
                scale=inputs['cache'].view(torch.float32).narrow(0,32,29*3136)
                value=inputs['cache'].view(torch.bfloat16).as_strided((29,16,2,128),(6272,256,128,1),storage_offset=(128+4352)//2)
                lengths64=inputs['lengths'].view(torch.int32).narrow(0,64,64);lengths=lengths64[:n]
                table64=inputs['table'].view(torch.int32).as_strided((64,16),(16,1),storage_offset=64);table=table64[:n]
                full={};views={}
                for field,shape_ in SHAPES.items():
                    count=int(np.prod(shape_));output=field=='output';t=torch.empty(count+128,dtype=torch.bfloat16 if output else torch.float32,device='cuda')
                    t.view(torch.int16 if output else torch.int32).fill_(f.BF16_GUARD if output else f.F32_GUARD)
                    full[field]=t;core=t.narrow(0,64,count).reshape(shape_)
                    views[field]=core[:n] if output else core[:n,:,:4]
                epoch=candidate_contract.LiveEpoch(inputs['cache'],dict(key=key,scale=scale,value=value),observe,raw,pages=29,layer_offset=128,format_id='cjj-konly-compact12544-rn32-rne-v1')
                lease=epoch.activate(e[0]['cache'])
                def admitted():return epoch.admit(lease,readonly=(inputs['query'],inputs['lengths'],inputs['table']),outputs=tuple(full.values()))
                def poison():
                    # All64 backing rows in the firstS4 region are poison. The
                    # candidate launches only48; extra16 must remain untouched.
                    # Segments4..15 retain the distinct physical guard marker.
                    for field,t in full.items():
                        core=t.narrow(0,64,int(np.prod(SHAPES[field]))).reshape(SHAPES[field])
                        area=core if field=='output' else core[:,:,:4]
                        area.view(torch.int16 if field=='output' else torch.int32).fill_(f.BF16_POISON if field=='output' else f.F32_POISON)
                    torch.cuda.synchronize()
                def snapshot(stage):
                    arrays={**{k:raw(t) for k,t in inputs.items()},**{k:raw(t) for k,t in full.items()}}
                    ar['snapshots'][stage]={k:blob(v) for k,v in arrays.items()};return arrays
                def check(stage,p,producer_only=False):
                    arrays=snapshot(stage);checks={k+'_readonly':arrays[k]==v for k,v in e[p].items()};metrics={}
                    for field,shape_ in SHAPES.items():
                        output=field=='output';marker=f.BF16_GUARD if output else f.F32_GUARD
                        bits=np.frombuffer(arrays[field],dtype='<u2' if output else '<u4')
                        checks[field+'_outside_guards']=bool((bits[:64]==marker).all() and (bits[-64:]==marker).all())
                        core=bits[64:-64].reshape(shape_);layout=np.zeros(shape_,dtype=bool)
                        if output:layout[:]=True;active=core[:n];unused=core[n:]
                        else:layout[:,:,:4]=True;active=core[:n,:,:4];unused=core[n:,:,:4]
                        checks[field+'_physical_padding_guard']=bool((core[~layout]==marker).all())
                        checks[field+'_unlaunched_rows_poison']=bool((unused==(f.BF16_POISON if output else f.F32_POISON)).all())
                        if output:
                            if producer_only:checks['output_still_poison']=bool((active==f.BF16_POISON).all());continue
                            actual=(active.astype(np.uint32)<<16).view('<f4').astype(np.float64)
                            metrics[field]=f.metrics(actual,gold[arm][p][field]);checks[field+'_FP64']=metrics[field]['passed']
                            metrics['original_BF16_proxy']=f.metrics(actual,gold[arm][p]['original_BF16_output']);proxies.append(metrics['original_BF16_proxy']['passed'])
                            checks['zero_length_output_positive_zero']=bool((active[np.array(cases[p]['case64']['lengths'][64:-64][:n])==0]==0).all())
                        else:
                            mask=np.broadcast_to(gold[arm][p]['active'][:,None,:,None] if field=='partial' else gold[arm][p]['active'][:,None,:],active.shape)
                            checks[field+'_inactive_poison']=bool((active[~mask]==f.F32_POISON).all())
                            actual=active.view('<f4').astype(np.float64);metrics[field]=f.metrics(actual[mask],gold[arm][p][field][mask]);checks[field+'_FP64']=metrics[field]['passed']
                            if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(actual[mask],gold[arm][p]['ideal_partial_diagnostic'][mask])
                    ar['checks'].append(dict(stage=stage,phase=p,checks=checks,metrics=metrics));checkpoint()
                    assert all(checks.values()),'full-array hard gate: '+arm+'/'+stage
                    return arrays
                ar['binding']=dict(inputs={k:observe(v) for k,v in inputs.items()},query=observe(q),key=observe(key),scale=observe(scale),value=observe(value),scratch={k:observe(v) for k,v in views.items()})
                handle=None;capturing=False
                def producer():
                    admitted()
                    for start in ((0,32) if arm=='reference' else (0,)):
                        count=32 if arm=='reference' else 48
                        args_=(q[start:start+count],key,scale,value,lengths[start:start+count],table[start:start+count],views['partial'][start:start+count],views['maximum'][start:start+count],views['summation'][start:start+count])
                        ar.setdefault('producer_call_bindings',{}).setdefault(str(start),dict(row_start=start,rows=count,segments=4,
                          tensors={k:observe(t) for k,t in zip(('query','key','scale','value','lengths','table','partial','maximum','summation'),args_,strict=True)}))
                        ops[arm](*args_,128**-.5*1.4426950408889634,4)
                        result['capture_dispatches' if capturing else 'dispatches'][arm]+=1
                def reduce():
                    nonlocal handle
                    for start in ((0,32) if arm=='reference' else (0,)):
                        count=32 if arm=='reference' else 48
                        handle=reducer._reduce_partials[(count,16)](output_ptr=views['output'][start:start+count],segm_output_ptr=views['partial'][start:start+count],segm_max_ptr=views['maximum'][start:start+count],segm_expsum_ptr=views['summation'][start:start+count],seq_lens_ptr=lengths[start:start+count],
                          segment_stride_token=32768,segment_stride_head=2048,segment_stride_segment=128,segment_stride_dim=1,max_stride_token=256,max_stride_head=16,max_stride_segment=1,
                          output_stride_token=2048,output_stride_head=128,page_size=16,num_segments=4,HEAD_SIZE=128,num_warps=4)
                        result['capture_reducer_dispatches' if capturing else 'reducer_dispatches'][arm]+=1
                def eager(stage,p):
                    snapshot(stage+'-before');producer();torch.cuda.synchronize();middle=check(stage+'-producer',p,True)
                    reduce();torch.cuda.synchronize();after=check(stage+'-after',p)
                    assert all(middle[k]==after[k] for k in FIELDS[:3]),'reducer mutated O/M/L'
                    return after
                def upload_check(p):
                    got={k:raw(t) for k,t in inputs.items()};ar.setdefault('upload_readbacks',{})[str(p)]={k:blob(v) for k,v in got.items()}
                    assert all(got[k]==v for k,v in e[p].items()),'complete input upload mismatch'
                def change(p):
                    nonlocal lease
                    for k,v in e[p].items():inputs[k].copy_(host(v,torch.uint8))
                    poison();upload_check(p);lease=epoch.activate(e[p]['cache']);ar['epoch_records'].append(dict(phase=p,receipt=admitted()))
                poison();upload_check(0);ar['epoch_records'].append(dict(phase=0,receipt=admitted()))
                if arm=='candidate' and (live,pitch)==CASES[0]:
                    # These two new C++ scope guards precede CUDAGuard and the
                    # native launch in the pinned source. Keep full before/after
                    # buffers, and require the precise host rejection reason.
                    for rejected_rows,rejected_segments,message in ((48,2,'Private N48/S4 only'),(32,4,'Q BF16 shape')):
                        label='host-reject-'+str(rejected_rows)+'-S'+str(rejected_segments)
                        before=snapshot(label+'-before')
                        args_=(q[:rejected_rows],key,scale,value,lengths[:rejected_rows],table[:rejected_rows],views['partial'][:rejected_rows],views['maximum'][:rejected_rows],views['summation'][:rejected_rows])
                        try:ops[arm](*args_,128**-.5*1.4426950408889634,rejected_segments)
                        except RuntimeError as exc:
                            assert message in str(exc),'unexpected host rejection'
                            rejected=dict(rows=rejected_rows,segments=rejected_segments,error=str(exc),source_guard_precedes_native_launch=True)
                        else:raise AssertionError('invalid candidate scope dispatched')
                        torch.cuda.synchronize();after=snapshot(label+'-after')
                        assert before==after,'host rejection changed a complete input/output backing'
                        rejected['complete_input_output_bytes_unchanged']=True
                        ar.setdefault('host_rejections',[]).append(rejected);checkpoint()
                eager('A',0)
                graph_tensors=dict(query=q,key=key,scale=scale,value=value,lengths=lengths,table=table,**views)
                binding={k:observe(t) for k,t in graph_tensors.items()};ar['graph_capture_binding']=binding
                graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());capturing=True
                with torch.cuda.graph(graph,stream=stream):producer();reduce()
                capturing=False;stream.synchronize()
                change(1);before=snapshot('graphB-before')
                for field,t in views.items():assert bool((t.detach().cpu().view(torch.int16 if field=='output' else torch.int32)==(f.BF16_POISON if field=='output' else f.F32_POISON)).all())
                assert all(before[k]==v for k,v in e[1].items()) and {k:observe(t) for k,t in graph_tensors.items()}==binding
                admitted();graph.replay();result['graph_replays'][arm]+=1;torch.cuda.synchronize();gb=check('graphB-after',1)
                poison();eb=eager('eagerB',1);assert all(gb[k]==eb[k] for k in gb),'Graph B differs from fresh eager B whole buffers'
                change(2);snapshot('sideC-before');side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());admitted()
                with torch.cuda.stream(side):producer();reduce()
                side.synchronize();check('sideC-after',2)
                ar['reducer']=dict(hash=str(handle.hash),n_regs=getattr(handle,'n_regs',None),n_spills=getattr(handle,'n_spills',None),artifacts={})
                assert handle.src.fn is reducer._reduce_partials
                for ext,data in handle.asm.items():
                    if isinstance(data,(str,bytes)):ar['reducer']['artifacts'][ext]=blob(data.encode() if isinstance(data,str) else data)
                old_lease=lease;epoch.close()
                try:epoch.admit(old_lease)
                except AssertionError:ar['closed_epoch_rejected']=True
                else:raise AssertionError('closed live epoch accepted')
                ar['completed']=True;checkpoint()
            record['first48_complete_field_bitwise']={}
            for phase in PHASES:
                a0={k:load(v) for k,v in record['arms']['reference']['snapshots'][phase].items()};a1={k:load(v) for k,v in record['arms']['candidate']['snapshots'][phase].items()}
                checks={field:core_bytes(a0,field)==core_bytes(a1,field) for field in FIELDS}
                record['first48_complete_field_bitwise'][phase]=checks;checkpoint();assert all(checks.values()),'twoN32 vs N48 full first48 mismatch: '+phase
            record['completed']=True
        assert result['dispatches']==result['reducer_dispatches']=={'candidate':12,'reference':24}
        assert result['capture_dispatches']==result['capture_reducer_dispatches']=={'candidate':4,'reference':8}
        assert result['graph_replays']=={'candidate':4,'reference':4}
        assert sum(len(r['arms']['candidate'].get('host_rejections',[])) for r in result['records'])==2
        assert source.sha(a.candidate_library.read_bytes())==result['candidate_admission']['library_sha256'] and source.sha(a.reference_library.read_bytes())==a.reference_library_sha256
        assert source.sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
        result.update(stage='completed',completed=True,requested_consumer_gates_passed=True,numeric_verified=True,full_first48_bitwise_verified=True,Graph_verified=True,side_stream_verified=True,original_BF16_proxy_passed=all(proxies),precision_gate=False)
    except Exception as exc:
        result['errors'].append(repr(exc));(out/'failure.txt').write_text(traceback.format_exc())
    result['artifact_manifest']=[dict(relative=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=source.sha(p.read_bytes())) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='consumer-summary.json' and '__pycache__' not in p.parts]
    checkpoint();print(json.dumps(result,indent=2,sort_keys=True));return 0 if result['completed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--root-owned-exclusive',action='store_true');p.add_argument('--review-template',action='store_true');p.add_argument('--preflight-cpu',action='store_true')
    for name in ('out','candidate-compile-summary','candidate-review','candidate-library','reference-compile-summary','reference-review','reference-library','reference-consumer-summary','codec-summary'):p.add_argument('--'+name,type=Path)
    for name in ('candidate-review-sha256','reference-review-sha256','reference-library-sha256','reference-consumer-summary-sha256','codec-proof-sha256'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.review_template:
        assert not a.gpu and a.candidate_compile_summary and not any(v for k,v in vars(a).items() if k not in ('review_template','candidate_compile_summary'))
        print(json.dumps(admission.contracts()[0].review_template(a.candidate_compile_summary),indent=2,sort_keys=True));return 0
    if a.preflight_cpu:
        assert not any((a.gpu,a.review_template,a.root_owned_exclusive,a.out))
        assert all(v for k,v in vars(a).items() if k not in ('review_template','gpu','preflight_cpu','root_owned_exclusive','out'))
        result=dict(passed=True,framework_imported=False,device_executed=False,precision_gate=False,**checked_admissions(a))
        assert not {'torch','triton','numpy'}&set(sys.modules)
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    if not a.gpu:
        assert not any(v for k,v in vars(a).items() if k!='gpu')
        result=plan();assert not {'torch','triton','numpy'}&set(sys.modules)
        print(json.dumps(result,indent=2,sort_keys=True));return 0
    assert all(v for k,v in vars(a).items() if k not in ('review_template','gpu','preflight_cpu')) and all(v.is_absolute() for v in vars(a).values() if isinstance(v,Path))
    assert not a.out.exists() and not a.out.resolve().is_relative_to(source.ROOT)
    return run(a)


if __name__=='__main__':raise SystemExit(main())
