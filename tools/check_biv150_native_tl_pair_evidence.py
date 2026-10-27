#!/usr/bin/env python3
"""Read-only SHA/ABI/NPZ/FP64-metric replay of Root's existing paired probe."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from biv150_native_tl_pair_contract import ROOT,MAIN,LIBRARY_SHA,ARENA_PAGES,CASES,ORDER,table,append_slots

SOURCE='36c5b0b04efc1db994ab233365ebdddf35cbda1c'


def sha(b):return hashlib.sha256(b).hexdigest()
def blob(path):return subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+path])
def thaw(d):return {ast.literal_eval(k.split(':',1)[1]):v for k,v in d.items()}
def fn_source(path,name):
    text=blob(path).decode();n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ''.join(text.splitlines(True)[n.lineno-1:n.end_lineno])


def audit(root,library,arrays):
    manifest=json.loads((root/'download-manifest.json').read_text())
    assert manifest['read_only'] is True and manifest['model_rerun'] is False
    assert manifest['original_owner']==dict(pid=688719,start_ticks=347244779,same_start=True,state='Z')
    assert manifest['raw_files']==len(manifest['files'])==94 and manifest['raw_bytes']==15217427
    for item in manifest['files']:
        data=(root/item['relative']).read_bytes()
        assert len(data)==item['bytes'] and sha(data)==item['sha256'],item['relative']
    result=json.loads((root/'out/result.json').read_text())
    assert result['commit']==SOURCE and result['main']==MAIN
    assert result['completed'] is True and result['passed'] is True and result['errors']==[]
    assert result['component_only'] and not result['K8_in_pair'] and not result['model_extrapolation']
    assert result['library']['sha256']==LIBRARY_SHA and sha(library.read_bytes())==LIBRARY_SHA
    binary=library.read_bytes();assert binary[:6]==b'\x7fELF\x02\x01'
    assert all(name in binary for name in (b'corex_native_gqa_s2_probe',b'corex_native_gqa_s4_probe'))
    log=(root/'exports/cjj-b3-native-tl-pair-v2-20261028.log').read_text()
    events={tag:[json.loads(m[1]) for line in log.splitlines()
        if (m:=re.search(tag+r' (\{.*\})$',line))] for tag in ('BIV150_NATIVE_GQA_SELECTED','BIV150_NATIVE_GQA_CAPTURED')}
    for tag,records in events.items():
        assert {(r['rows'],r['segments']) for r in records}=={(64,2),(32,4)}
        assert all(r['library_sha256']==LIBRARY_SHA and r['q_stride']==[2048,128,1]
            and r['k_stride']==r['v_stride']==[8192,256,128,1] for r in records)
    source_pins=json.loads(blob('docs/evidence/native-tl-gqa-pair-20261028/source-pins.json'))
    for name,expected in source_pins.items():assert sha(blob(name))==expected
    rows=[]
    if arrays:import numpy as np
    for record in result['records']:
        name=record['case'];batch,length,segments=CASES[name]
        assert record['batch']==batch and record['length']==length and record['segments']==segments
        assert record['arena_pages']==ARENA_PAGES and record['active_pages']==batch*length//16
        assert record['input_seed']==20261023+batch+length and record['torch_version']=='2.10.0'
        assert record['native_operator']=='corex_native_gqa_s'+str(segments)+'_probe::produce'
        assert record['grid']==[batch,2,segments] and record['passed']
        assert record['Graph_eager_full_bitwise'] and record['timing_source_readonly']
        abi=record['tensor_descriptors']
        for arm in ('native','triton'):
            c=abi[arm]
            assert c['q']['shape']==[batch,16,128] and c['q']['stride']==[2048,128,1]
            for key in ('k','v'):
                assert c[key]['shape']==[ARENA_PAGES,16,2,128] and c[key]['stride']==[8192,256,128,1]
                assert c[key]['dtype']=='torch.bfloat16' and c[key]['pointer_alignment64']==0
            assert c['v']['storage_offset']-c['k']['storage_offset']==4096
            assert c['table']['shape']==[batch,8192] and c['table']['stride']==[8192,1]
            assert c['scratch']['shape']==[batch,16,segments,128] and c['scratch']['stride']==[32768,2048,128,1]
            assert c['maxima']['stride']==c['sums']['stride']==[256,16,1]
        compiled=[]
        for role,path,fn in [('producer','vllm_fl/patches/biv150_gqa_decode.py','_gqa_decode_attention'),
                             ('reducer','vllm_fl/patches/biv150_gqa_decode.py','_reduce_partials'),
                             ('append','tools/kernels/biv150_pair_append_support.py','append_kv')]:
            directory=root/'out'/(name+'-'+role);r=json.loads((directory/'record.json').read_text())
            assert r==record['actual_compiled'][role]
            assert r['source_digest']==sha(fn_source(path,fn).encode())
            assert type(r['n_regs']) is int and r['n_regs']>0 and r['n_spills']==0
            for ext,d in r['files'].items():
                data=(directory/('actual.'+ext)).read_bytes();assert len(data)==d['bytes'] and sha(data)==d['sha256']
            signature=thaw(r['typed_ABI']['signature']);const=thaw(r['typed_ABI']['constants'])
            if role=='producer':
                assert all(signature[k]=='*bf16' for k in ('query_ptr','key_cache_ptr','value_cache_ptr'))
                assert all(signature[k]=='*fp32' for k in ('segm_output_ptr','segm_max_ptr','segm_expsum_ptr'))
                assert const[(24,)]==128 and const[(25,)]==16 and const[(26,)]==8 and const[(27,)]==segments
            elif role=='append':
                assert const=={(6,):548416,(7,):False} and type(const[(7,)]) is bool
                assert signature['slots']=='*i64' and signature['key']==signature['value']=='*bf16'
            else:assert const[(15,)]==segments and const[(16,)]==128 and signature['output_ptr']=='*bf16'
            compiled.append(dict(role=role,hash=r['hash'],n_regs=r['n_regs'],n_spills=r['n_spills'],source_digest=r['source_digest']))
        timing=record['timing'];assert timing['order']==list(ORDER) and timing['warm_replays']==128
        assert [v['arm'] for v in timing['ABBA']]==list(ORDER)
        assert all(v['replays']==64 and v['unroll']==8 for v in timing['ABBA'])
        means={arm:math.fsum(v['complete_append_producer_reducer_ms'] for v in timing['ABBA'] if v['arm']==arm)/2 for arm in ('native','triton')}
        wall={arm:math.fsum(v['Graph_host_launch_and_sync_wall_ms_per_step'] for v in timing['ABBA'] if v['arm']==arm)/2 for arm in means}
        assert means==timing['means'] and means['triton']/means['native']==timing['native_over_triton']
        spread={arm:100*(max(v['complete_append_producer_reducer_ms'] for v in timing['ABBA'] if v['arm']==arm)-
                          min(v['complete_append_producer_reducer_ms'] for v in timing['ABBA'] if v['arm']==arm))/means[arm] for arm in means}
        numeric=[]
        if arrays:
            def bf16(bits):return (bits.view(np.uint16).astype(np.uint32)<<16).view(np.float32).astype(np.float64)
            def metric(actual,ref):
                assert actual.shape==ref.shape and np.isfinite(actual).all() and np.isfinite(ref).all()
                err=actual-ref;maximum=float(np.max(np.abs(err)));rms=float(np.sqrt(np.mean(err*err)))
                denom=float(np.sqrt(np.mean(ref*ref)));relative=rms/denom if denom else (0. if maximum==0 else math.inf)
                violations=int(np.count_nonzero(np.abs(err)>.005+.005*np.abs(ref)))
                assert rms!=0 or maximum==0
                return dict(relative_rmse=relative,max_abs=maximum,rms=rms,pointwise_violations=violations)
            for control in (0,1):
                with np.load(root/'out'/(name+'-control'+str(control)+'.npz'),allow_pickle=False) as z:
                    mapping=table(batch,length)
                    expected=np.full((batch,8192),-1,dtype=np.int32);expected[:,:length//16]=mapping
                    assert np.array_equal(z['table'],expected) and sha(z['table'].tobytes())==record['table_sha256']
                    assert np.array_equal(z['slots'],np.array(append_slots(mapping,z['lengths'].tolist())))
                    assert z['reference'].shape==(batch,16,128) and z['reference'].dtype==np.float64
                    for field in ('scratch','maxima','sums','out'):
                        assert np.array_equal(z['native_'+field],z['triton_'+field])
                        assert record['cross_arm_bitwise'][control]['fields'][field] is True
                    out=bf16(z['native_out']);partial=z['native_scratch'].view(np.float32).astype(np.float64)
                    maxima=z['native_maxima'].view(np.float32).astype(np.float64);sums=z['native_sums'].view(np.float32).astype(np.float64)
                    reduced=np.zeros_like(out)
                    for row,n in enumerate(z['lengths']):
                        n=int(n);width=(n+segments*16-1)//(segments*16)
                        active=np.arange(segments)*width*16<n
                        assert np.isfinite(partial[row][:,active,:]).all()
                        assert np.isnan(partial[row][:,~active,:]).all()
                        assert np.isfinite(maxima[row][:,active]).all() and np.isnan(maxima[row][:,~active]).all()
                        assert (sums[row][:,active]>0).all() and np.isnan(sums[row][:,~active]).all()
                        if n:
                            mm=maxima[row][:,active];weights=np.exp(mm-np.max(mm,axis=1)[:,None])
                            reduced[row]=np.sum(partial[row][:,active,:]*weights[:,:,None],axis=1)/np.sum(sums[row][:,active]*weights,axis=1)[:,None]
                        else:assert np.count_nonzero(out[row])==0 and np.count_nonzero(z['reference'][row])==0
                        if n==1:
                            expected_one=np.repeat(bf16(z['new_v_bits'])[row],8,axis=0)
                            assert np.array_equal(out[row],expected_one) and np.array_equal(z['reference'][row],expected_one)
                    reduction=metric(out,reduced);assert reduction['pointwise_violations']==0 and reduction['relative_rmse']<.005
                    for arm in ('native','triton'):
                        got=metric(bf16(z[arm+'_out']),z['reference'])
                        original=next(v for v in record['numeric'] if v['control']==control and v['arm']==arm)
                        assert original['source_readonly'] and original['repeat_full_output_partial_bitwise']
                        for key,value in got.items():assert math.isclose(value,original['FP64'][key],rel_tol=1e-11,abs_tol=1e-14)
                        assert got['pointwise_violations']==0 and got['relative_rmse']<=.005
                        numeric.append(dict(control=control,arm=arm,full_output_elements=out.size,metrics_recomputed_against_saved_FP64=got))
                    numeric.append(dict(control=control,full_partial_cross_bits=True,independent_FP64_reduction_from_saved_partials=reduction))
        rows.append(dict(case=name,complete_step_ms=means,native_over_triton=means['triton']/means['native'],
            Graph_launch_sync_wall_ms=wall,wall_native_over_triton=wall['triton']/wall['native'],
            within_arm_ABBA_range_percent=spread,initial_pack_copy_ms=record['initial_pack_copy_ms'],compiled=compiled,numeric=numeric))
    return dict(passed=True,CPU_only=True,actual_device_run_repeated=False,raw_files=94,raw_bytes=15217427,
        owner=manifest['original_owner'],manifest_sha256=sha((root/'download-manifest.json').read_bytes()),
        source_commit=SOURCE,main=MAIN,all_SHA_verified=True,local_SO_payload_SHA_verified=LIBRARY_SHA,
        source_and_typed_ABI_verified=True,arrays_replayed=arrays,records=rows,
        actual_native_selected_and_capture_events_verified=True,
        cold_KV_seed_reference_regenerated=False,
        limitation='No local Torch2.10: full cold arena not regenerated; original closed attention FP64 array retained, full metrics independently recomputed. FP64 segment reduction and length0/1 attention regenerated from saved arrays.',
        original_runtime_only_checks=['outer guards','same-arm second-run arrays','Graph-vs-eager separate arrays','full-cache append/readonly hashes; large source/cache payload absent'],
        first_v1_failure_preserved=True,component_only=True,whole_model_speed_or_14percent_cause_claimed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=Path('/private/tmp/cjj-b3-native-tl-pair-v2-raw-20261028'))
    p.add_argument('--library',type=Path,default=Path('/private/tmp/cjj-main-b3-native-gqa-build-raw-v1-20261028/out/corex_native_gqa_s2_probe.so'))
    p.add_argument('--arrays',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.raw_root,a.library,a.arrays),indent=2))
