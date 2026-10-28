#!/usr/bin/env python3
"""CPU-only complete saved-array replay of frozen PR112 consumer.
Reuses frozen fixture, FP64 oracle, metrics, and exact AST check/core_bytes.
No device import, new numeric reference, threshold change, or fresh GPU readback.
"""
import argparse, ast, copy, gzip, hashlib, io, json, math, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

FROZEN='09e502134fcf9fa7168c9f342c2a3d33754cee1c'
SUMMARY='493e210f8eeaac6006b25a94ed55e08fbe932337d6481fe3d8ec6e86d0033272'
CANDIDATE='2ad51a9812502e7dd8d81b99debd2160086a52510a93063f7ce4b0fbd8c837b1'
REFERENCE='7892a6aad09d247a322a74906312f250a9d253160202fee23259adac7b7260d4'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canon(x):return json.loads(json.dumps(x))
def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b,strict=True):compare(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=4e-14),(a,b)
    else:assert a==b,(a,b)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--candidate-compile',type=Path,required=True);p.add_argument('--reference-compile',type=Path,required=True)
    p.add_argument('--reference-consumer',type=Path,required=True);p.add_argument('--codec',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();sys.dont_write_bytecode=True
    root=a.source.resolve();raw=a.raw.resolve();out=raw/'out';s=read(out/'consumer-summary.json');manifest=read(raw/'download-manifest.json')
    assert sha((out/'consumer-summary.json').read_bytes())==SUMMARY
    assert manifest['raw_files']==170 and manifest['raw_bytes']==59569068
    assert manifest['original_owner']==dict(pid=62050,start_ticks=355994499,same_start=True,state='Z')
    seen=set()
    for row in manifest['files']:
        name=Path(row['relative']);assert not name.is_absolute() and '..' not in name.parts and str(name) not in seen;seen.add(str(name))
        b=(raw/name).read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256']
    assert len(seen)==170 and sum(x['bytes'] for x in manifest['files'])==59569068
    assert seen=={str(x.relative_to(raw)) for x in raw.rglob('*') if x.is_file() and x.name!='download-manifest.json'}
    # All executable frozen tools and reducer match the trusted commit, not just snapshots.
    tree=subprocess.check_output(['git','ls-tree','-r','--name-only',FROZEN,'tools','vllm_fl/patches/biv150_gqa_decode.py'],cwd=root,text=True).splitlines()
    for name in tree:
        if name.endswith(('.py','.cu','.cpp','.cuh')):
            assert (root/name).read_bytes()==subprocess.check_output(['git','show',FROZEN+':'+name],cwd=root)
    for name,digest in s['frontend_sources'].items():assert sha((root/'tools'/name).read_bytes())==digest==sha((out/'frontends'/name).read_bytes())
    sys.path.insert(0,str(root/'tools'))
    import probe_biv150_native_k8_compact_n48_consumer as front
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    def library(d):
        summary=d/'out/compile-summary.json';j=read(summary)
        return summary,d/'out/build'/Path(j['library']['path']).name
    cs,cl=library(a.candidate_compile);rs,rl=library(a.reference_compile)
    assert sha(cl.read_bytes())==CANDIDATE and sha(rl.read_bytes())==REFERENCE
    for input_name,actual in [('candidate_compile_summary',cs),('reference_compile_summary',rs),('reference_consumer_summary',a.reference_consumer/'out/consumer-summary.json'),('codec_summary',a.codec/'out/codec-summary.json')]:
        assert (out/(input_name+'-input.json')).read_bytes()==actual.read_bytes()
    ns=SimpleNamespace(candidate_compile_summary=cs,candidate_library=cl,candidate_review=out/'candidate_review-input.json',candidate_review_sha256=s['candidate_admission']['review_sha256'],
       reference_compile_summary=rs,reference_library=rl,reference_review=out/'reference_review-input.json',reference_review_sha256=s['reference_admission']['review_sha256'],
       reference_consumer_summary=a.reference_consumer/'out/consumer-summary.json',reference_consumer_summary_sha256=front.admission.REFERENCE_CONSUMER_SHA,reference_library_sha256=REFERENCE,
       codec_summary=a.codec/'out/codec-summary.json',codec_proof_sha256=s['codec_admission']['actual_codec_receipt_sha256'])
    admissions=front.checked_admissions(ns)
    for k,v in admissions.items():assert v==s[k],k
    assert s['source']==front.source.verify() and s['actual_imports']['candidate_library']['sha256']==CANDIDATE and s['actual_imports']['reference_library']['sha256']==REFERENCE
    assert s['actual_imports']['loaded_library_set_confirmed'] is True
    assert s['actual_schemas']['candidate'].replace(front.source.NAMESPACE,'cjj_native_k8_compact_v_overlap_probe')==s['actual_schemas']['reference']
    assert s['errors']==[] and s['completed'] and s['stage']=='completed'
    for k in ('requested_consumer_gates_passed','numeric_verified','full_first48_bitwise_verified','Graph_verified','side_stream_verified'):assert s[k] is True
    for k in ('performance_verified','service_verified','precision_gate','thresholds_changed'):assert s[k] is False
    assert s['writer_launches']==s['encoder_launches']==0 and s['timing'] is None
    for row in s['artifact_manifest']:front.admission.original.artifact(out,row)
    import numpy as np
    f=front.f;assert sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
    code=(root/'tools/probe_biv150_native_k8_compact_n48_consumer.py').read_text();tree=ast.parse(code)
    nodes={name:[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name] for name in ('check','core_bytes')}
    assert all(len(v)==1 for v in nodes.values())
    helper_sha={k:sha(ast.get_source_segment(code,v[0]).encode()) for k,v in nodes.items()}
    # Exact original bodies; only GPU snapshot callback is replaced by complete saved bytes.
    helpers={k:compile(ast.fix_missing_locations(ast.Module(body=v,type_ignores=[])),'<frozen112-'+k+'>','exec') for k,v in nodes.items()}
    env=dict(np=np,f=f,SHAPES=front.SHAPES);exec(helpers['core_bytes'],env);core_bytes=env['core_bytes']
    blobs={}
    def load(meta):
        name=meta['path'];assert Path(name).parts[0]=='blobs' and '..' not in Path(name).parts
        if name not in blobs:
            packed=(out/name).read_bytes();assert len(packed)==meta['stored_bytes'] and sha(packed)==meta['stored_sha256']
            blobs[name]=gzip.decompress(packed)
        b=blobs[name];assert len(b)==meta['bytes'] and sha(b)==meta['sha256'];return b
    counts=dict(FP64_hard=0,BF16_proxy=0,ideal_diagnostic=0,ideal_diagnostic_failed=0,full_snapshots=0,readonly_whole_buffers=0,first48_field_pairs=0,golden_arrays=0,epoch_receipts=0,host_refusals=0)
    phases={'A-producer':0,'A-after':0,'graphB-after':1,'eagerB-producer':1,'eagerB-after':1,'sideC-after':2}
    report_records=[];max_drift=0.;all_proxies=[]
    assert [tuple(r['case']) for r in s['records']]==[(48,4,p,l) for l,p in front.CASES]
    for rec in s['records']:
        _,_,pitch,live=rec['case'];cases=front.fixtures(live,pitch);gold={};record_report=dict(case=rec['case'],arms={})
        for arm in ('reference','candidate'):
            gold[arm]={}
            for phase in range(3):
                regenerated=f.golden(cases[phase]['case48' if arm=='candidate' else 'case64'])
                saved=rec['golden'][arm][str(phase)]
                assert set(saved)==set(regenerated)
                for field,value in regenerated.items():
                    if isinstance(value,np.ndarray):
                        saved_array=np.load(io.BytesIO(load(saved[field])),allow_pickle=False)
                        assert saved_array.shape==value.shape and saved_array.dtype==value.dtype
                        if value.dtype.kind in 'f':
                            assert np.array_equal(np.isnan(saved_array),np.isnan(value)) and np.array_equal(np.isinf(saved_array),np.isinf(value))
                            assert np.allclose(saved_array,value,rtol=1e-12,atol=1e-11,equal_nan=True),(arm,phase,field)
                            finite=np.isfinite(value)
                            max_drift=max(max_drift,float(np.max(np.abs(saved_array[finite]-value[finite]))))
                        else:assert np.array_equal(saved_array,value)
                        counts['golden_arrays']+=1
                    else:assert saved[field]==value
                gold[arm][phase]=regenerated
        for arm in ('reference','candidate'):
            ar=rec['arms'][arm];n=48 if arm=='candidate' else 64;e={phase:c['expected'] for phase,c in cases.items()}
            snapshots={stage:{k:load(v) for k,v in vals.items()} for stage,vals in ar['snapshots'].items()}
            assert ar['completed'] and ar['closed_epoch_rejected']
            for stage,arrays in snapshots.items():
                assert set(arrays)==set(e[0])|set(front.FIELDS)
                phase=2 if stage.startswith('sideC') else 1 if stage.startswith(('graphB','eagerB')) else 0
                for k,v in e[phase].items():assert arrays[k]==v,(stage,k);counts['readonly_whole_buffers']+=1
                for field,shape in front.SHAPES.items():assert len(arrays[field])==(int(np.prod(shape))+128)*(2 if field=='output' else 4)
                if stage.endswith('-before') or stage.startswith('host-reject'):
                    for field,shape in front.SHAPES.items():
                        isout=field=='output';bits=np.full(int(np.prod(shape))+128,f.BF16_GUARD if isout else f.F32_GUARD,dtype='<u2' if isout else '<u4');core=bits[64:-64].reshape(shape)
                        if isout:core[:]=f.BF16_POISON
                        else:core[:,:,:4]=f.F32_POISON
                        assert bits.tobytes()==arrays[field],(stage,field,'complete prepoison')
            for phase in range(3):assert {k:load(v) for k,v in ar['upload_readbacks'][str(phase)].items()}==e[phase]
            checked=[];env=dict(np=np,f=f,SHAPES=front.SHAPES,e=e,arm=arm,n=n,gold=gold,cases=cases,proxies=all_proxies,ar={'checks':checked},checkpoint=lambda:None,snapshot=lambda stage:snapshots[stage])
            exec(helpers['check'],env)
            assert [x['stage'] for x in ar['checks']]==list(front.PHASES)
            for saved in ar['checks']:
                env['check'](saved['stage'],saved['phase'],saved['stage'].endswith('-producer'))
                compare(checked[-1],saved)
                counts['full_snapshots']+=1
                for field,metric in checked[-1]['metrics'].items():
                    if field in front.FIELDS:assert metric['passed'];counts['FP64_hard']+=1
                    elif field=='original_BF16_proxy':counts['BF16_proxy']+=1
                    else:counts['ideal_diagnostic']+=1;counts['ideal_diagnostic_failed']+=not metric['passed']
            assert snapshots['graphB-after']==snapshots['eagerB-after']
            for phase in ('A','eagerB'):
                assert all(snapshots[phase+'-producer'][k]==snapshots[phase+'-after'][k] for k in front.FIELDS[:3])
            # Observed full backing layout, offsets and Graph capture inputs.
            b=ar['binding'];capture=ar['graph_capture_binding'];assert capture['query']==b['query']
            for k in ('key','scale','value'):assert capture[k]==b[k]
            for k in front.FIELDS:assert capture[k]==b['scratch'][k]
            specs=dict(query=('bfloat16',[n,16,128],[pitch,128,1],64,'query'),key=('int8',[29,16,2,128],[12544,256,128,1],128,'cache'),scale=('float32',[29*3136],[1],32,'cache'),value=('bfloat16',[29,16,2,128],[6272,256,128,1],2240,'cache'),lengths=('int32',[n],[1],64,'lengths'),table=('int32',[n,16],[16,1],64,'table'))
            for k,(dtype,shape,strides,offset,backing) in specs.items():
                d=capture[k];base=b['inputs'][backing];width={'int8':1,'float32':4,'bfloat16':2,'int32':4}[dtype]
                assert (d['dtype'],d['shape'],d['strides'],d['storage_offset'])==(dtype,shape,strides,offset)
                assert d['storage_base']==base['pointer'] and d['storage_id']==base['storage_id'] and d['storage_bytes']==len(e[0][backing])
                assert d['pointer']==d['storage_base']+offset*width
            starts=(0,32) if arm=='reference' else (0,)
            assert set(ar['producer_call_bindings'])=={str(x) for x in starts}
            for start in starts:
                call=ar['producer_call_bindings'][str(start)];count=32 if arm=='reference' else 48
                assert (call['row_start'],call['rows'],call['segments'])==(start,count,4)
                for k,d in call['tensors'].items():
                    expected=copy.deepcopy(capture[k])
                    if k not in ('key','scale','value'):
                        width={'float32':4,'bfloat16':2,'int32':4}[d['dtype']]
                        expected['shape'][0]=count;expected['storage_offset']+=start*expected['strides'][0];expected['pointer']+=start*expected['strides'][0]*width
                    assert d==expected,(arm,start,k)
            def observe(d):
                return dict(d,shape=tuple(d['shape']),strides=tuple(d['strides']))
            phase_box=[0];contract=front.admission.contracts()[0]
            epoch=contract.LiveEpoch(b['inputs']['cache'],{k:b[k] for k in ('key','scale','value')},observe,lambda _:e[phase_box[0]]['cache'],pages=29,layer_offset=128,format_id='cjj-konly-compact12544-rn32-rne-v1')
            for row in ar['epoch_records']:
                phase_box[0]=row['phase'];token=epoch.activate(e[row['phase']]['cache'])
                receipt=epoch.admit(token,readonly=tuple(b['inputs'][k] for k in ('query','lengths','table')),outputs=tuple(b['scratch'].values()))
                assert canon(receipt)==row['receipt'];counts['epoch_receipts']+=1
            assert [r['phase'] for r in ar['epoch_records']]==[0,1,2];epoch.close()
            try:epoch.admit(token)
            except AssertionError:pass
            else:raise AssertionError('closed epoch accepted in replay')
            for rejection in ar.get('host_rejections',[]):
                rows,segments=rejection['rows'],rejection['segments'];label=f'host-reject-{rows}-S{segments}'
                assert (rows,segments) in ((48,2),(32,4))
                assert ('Private N48/S4 only' if segments==2 else 'Q BF16 shape') in rejection['error']
                assert snapshots[label+'-before']==snapshots[label+'-after'] and rejection['complete_input_output_bytes_unchanged'];counts['host_refusals']+=1
            reducer=ar['reducer'];assert reducer['n_regs']==11 and reducer['n_spills']==0
            for ext,meta in reducer['artifacts'].items():
                data=load(meta);paths=list((out/'triton-cache').rglob('_reduce_partials.'+ext));assert len(paths)==1 and paths[0].read_bytes()==data
            md=read(next((out/'triton-cache').rglob('_reduce_partials.json')))
            assert md['hash']==reducer['hash'] and md['target']==dict(backend='corex',arch=71,warp_size=64) and md['num_warps']==4 and md['shared']==1024
            source_bytes=load(reducer['artifacts']['source'])
            assert b'vllm_fl/patches/biv150_gqa_decode.py\":235:0)' in source_bytes and b'tt.func public @_reduce_partials' in source_bytes
            # .source is Triton IR with debug locations, not Python source bytes.
            # The frozen runtime asserts the full Python module hash before/after,
            # and records handle.src.fn identity before saving these exact artifacts.
            record_report['arms'][arm]=dict(hard_gates_passed=True,complete_backing_padding_and_zero_semantics_verified=True,whole_input_cache_readonly=True,Graph_B_before_eager_B=True,side_C_verified=True,epoch_receipts_replayed=3,reducer_hash=reducer['hash'],checks=checked)
        for stage in front.PHASES:
            arms=[{k:load(v) for k,v in rec['arms'][arm]['snapshots'][stage].items()} for arm in ('reference','candidate')]
            for field in front.FIELDS:
                assert core_bytes(arms[0],field)==core_bytes(arms[1],field)
                assert rec['first48_complete_field_bitwise'][stage][field] is True;counts['first48_field_pairs']+=1
        report_records.append(record_report);print('verified',rec['case'],flush=True)
    assert counts['FP64_hard']==176 and counts['BF16_proxy']==32 and counts['first48_field_pairs']==96 and counts['golden_arrays']==168 and counts['epoch_receipts']==24 and counts['host_refusals']==2
    assert s['original_BF16_proxy_passed']==all(all_proxies)
    assert s['dispatches']==s['reducer_dispatches']==dict(candidate=12,reference=24)
    assert s['capture_dispatches']==s['capture_reducer_dispatches']==dict(candidate=4,reference=8)
    assert s['graph_replays']==dict(candidate=4,reference=4)
    assert 'torch' not in sys.modules and 'triton' not in sys.modules
    report=dict(completed=True,errors=[],review='independent complete saved-array CPU replay',script_sha256=sha(Path(__file__).read_bytes()),frozen_source=FROZEN,summary_sha256=SUMMARY,download_manifest_sha256=sha((raw/'download-manifest.json').read_bytes()),candidate_library_sha256=CANDIDATE,reference_library_sha256=REFERENCE,
      raw_files_rehashed=170,raw_bytes_rehashed=59569068,helper_original_body_sha256=helper_sha,admissions_replayed=True,counts=counts,original_BF16_proxy_passed=all(all_proxies),maximum_host_FP64_golden_drift=max_drift,host_golden_replay_tolerance=dict(rtol=1e-12,atol=1e-11),metric_replay_tolerance=dict(rtol=1e-12,atol=4e-14),device_thresholds_unchanged=dict(pointwise_atol=.005,pointwise_rtol=.005,RRMSE=.005),records=report_records,
      scope=dict(rows=48,segments=4,live_rows=[41,42],Q_pitches=[2048,2560],pages=29,max_context=193,physical_page_bytes=12544,layer_offset_bytes=128,writer=False,encoder=False,fullstep=False,capacity42layers=False,model=False,performance=False),
      limits=['Saved runtime readbacks and exact frozen control flow; no new GPU readback or independent stream profiler.','Epoch CPU replay validates observed layout/value receipt; it does not recreate the remote process-local lease.','Two N32 calls provide numeric control only, never a performance denominator.','Ideal partial diagnostic failures retained separately from original pagewise BF16-P hard reference.'],framework_imported=False)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(report_sha256=sha(a.out.read_bytes()),counts=counts,original_BF16_proxy_passed=all(all_proxies),maximum_host_FP64_golden_drift=max_drift)))
if __name__=='__main__':main()
