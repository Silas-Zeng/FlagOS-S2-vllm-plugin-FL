#!/usr/bin/env python3
"""CPU/NumPy replay of one immutable native-K8 fullstep download, never Torch."""
import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

COMMIT = '3ebdbebf61e0e90735cd8ad83eb1605627b6ef46'
CONSUMER = 'c2e13e3766172e9b74342bbcf48b1b4729d1b9604bc249b6974740f7f90cc128'


def sha(b): return hashlib.sha256(b).hexdigest()
def need(ok, message):
    if not ok: raise AssertionError(message)
def safe(root, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts, 'unsafe artifact path')
    result = root / p
    need(result.resolve().is_relative_to(root.resolve()), 'artifact escape')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--regenerate', action='store_true', help='Regenerate full cold counter/cache plus phase0 gold for both lengths and arms')
    a = p.parse_args(); begin = time.monotonic()
    need(not a.out.exists(), 'new report path required')
    source = a.source_root.resolve(); raw = a.raw_root.resolve(); out = raw/'out'
    need(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==COMMIT, 'source HEAD')
    need(not subprocess.check_output(['git','-C',str(source),'status','--porcelain']), 'source tree dirty')
    sys.path.insert(0, str(source/'tools'))
    import numpy as np
    import biv150_native_k8_fullstep_fixture as f
    source_pins = f.pins()
    manifest_bytes = (raw/'download-manifest.json').read_bytes(); manifest=json.loads(manifest_bytes)
    need(manifest['read_only'] and manifest['model_rerun'] is False, 'transport provenance')
    need(manifest['original_owner']==dict(pid=706887,start_ticks=347999537,same_start=True,state='Z'), 'owner terminal')
    total=0; paths=set()
    for meta in manifest['files']:
        file=safe(raw,meta['relative']); b=file.read_bytes(); total+=len(b)
        need(len(b)==meta['bytes'] and sha(b)==meta['sha256'], 'raw hash '+meta['relative']);paths.add(meta['relative'])
    need(len(paths)==manifest['raw_files']==192 and total==manifest['raw_bytes']==144847098, 'raw inventory')
    actual_paths={str(x.relative_to(raw)) for x in raw.rglob('*') if x.is_file() and x.name!='download-manifest.json'}
    need(actual_paths==paths, 'raw file set')
    owner=json.loads((raw/'exports/cjj-native-k8-fullstep-v1-20261028-owner.json').read_text())
    need(owner['source_commit']==COMMIT and owner['pid']==706887 and owner['start_ticks']==347999537 and owner['official_score'] is False, 'owner/source')
    summary_bytes=(out/'fullstep-summary.json').read_bytes(); summary=json.loads(summary_bytes)
    need(summary['completed'] and not summary['errors'], 'run not completed')
    need(summary['source_pins']==source_pins, 'source pins differ')
    for key,value in json.loads(json.dumps(f.plan())).items():
        if key in ('numerical_verified','timing_executed'):need(summary[key] is True,'actual runtime field '+key)
        else:need(summary[key]==value, 'plan field '+key)
    need(summary['actual_consumer']['actual_consumer_summary_sha256']==CONSUMER and summary['actual_consumer']['all_full_D128_snapshots_replayed'], 'consumer identity')
    need(sha((source/'docs/evidence/native-k8-qk-feasibility-20261028/consumer-v2/original-summary.json').read_bytes())==CONSUMER, 'frozen actual consumer summary bytes')
    for meta in summary['artifacts']:
        b=safe(out,meta['path']).read_bytes();need((len(b),sha(b))==(meta['bytes'],meta['sha256']), 'summary artifact')
    def load(meta):
        b=safe(out,meta['path']).read_bytes();need((len(b),sha(b))==(meta['bytes'],meta['sha256']), 'blob');return b
    def array(meta): return np.load(io.BytesIO(load(meta)),allow_pickle=False)
    def equal_fields(first,second):
        need(set(first)==set(second)=={'partial','maximum','summation','output'}, 'four fields')
        for k in first: need(load(first[k])==load(second[k]), 'samearm full bits '+k)
    report=dict(CPU_only=True,source_commit=COMMIT,source_root=str(source),raw_root=str(raw),
        script_sha256=sha(Path(__file__).read_bytes()),manifest_sha256=sha(manifest_bytes),summary_sha256=sha(summary_bytes),
        raw_files_verified=192,raw_bytes_verified=total,source_pins_verified=len(source_pins),consumer_sha256=CONSUMER,
        K8_SO=summary['K8_library_sha256'],BF16_SO=summary['BF16_library_sha256'],
        saved_reference_arithmetic_replay=True,cold_counter_and_gold_regenerated=False,records=[],
        evidence_limits=dict(SO_bytes='Pinned identities and the frozen driver pre-Torch content-hash assertions; SO binaries are not part of this download',
            GPU_input_guards_and_readonly='Frozen runtime verify_inputs/fingerprints assertions; full actual input payloads are not among saved snapshots',
            GPU_cache='Runtime complete-byte comparisons plus saved chunk digests; GPU cache payloads were not saved',
            output_guards_and_inactive_scratch='Independently replayed from every saved full output buffer'),
        qualification=dict(model=False,precision=False,capacity=False,service=False,stable_performance=False))
    metric_float_delta=0.
    def metric_same(actual,saved):
        nonlocal metric_float_delta
        need(set(actual)==set(saved),'metric keys')
        for key,value in actual.items():
            other=saved[key]
            if isinstance(value,dict):metric_same(value,other)
            elif type(value) is float:
                metric_float_delta=max(metric_float_delta,abs(value-other))
                need(math.isclose(value,other,rel_tol=4e-14,abs_tol=1e-30),'saved FP64 metric arithmetic')
            else:need(type(value) is type(other) and value==other,'saved metric state')
    records={r['case']:r for r in summary['records']};need(set(records)==set(f.CASES), 'case set')
    golds={}; controls={}
    def check_snapshot(rec,snap,gold):
        n,L,s,pitch=f.CASES[rec['case']];phase=snap['phase'];arm=snap['arm'];z=controls[rec['case'],phase]
        metrics={}
        for field,meta in snap['full_arrays'].items():
            b=load(meta); width=2 if field=='output' else 4
            marker=0x4255 if width==2 else 0x428e0000
            guard=struct.pack('<H' if width==2 else '<I',marker)*64
            need(b[:64*width]==b[-64*width:]==guard, 'full guard '+field)
            shape=(n,16,128) if field=='output' else ((n,16,16,128) if field=='partial' else (n,16,16))
            need(len(b)==(math.prod(shape)+128)*width, 'full width16 bytes')
            bits=np.frombuffer(b[64*width:-64*width],dtype='<u2' if width==2 else '<u4').reshape(shape)
            if field=='output':
                values=(bits.astype(np.uint32)<<16).view('<f4').astype(np.float64)
                need(bool((bits[z['lengths']==0]==0).all()), 'empty output positive-zero')
                metrics[field]=f.metrics(values,gold[field])
            else:
                mask=np.zeros(shape,dtype=bool)
                wanted=np.broadcast_to(gold['active'][:,None,:,None] if field=='partial' else gold['active'][:,None,:],gold[field].shape)
                mask[:,:,:s]=wanted
                need(bool((bits[~mask]==0x7fc12345).all()), 'inactive/padded poison '+field)
                values=bits.view('<f4')[:,:,:s][wanted]
                metrics[field]=f.metrics(values,gold[field][wanted])
                if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(values,gold['ideal_partial_diagnostic'][wanted])
            need(metrics[field]['passed'], 'original numeric gate '+rec['case']+' '+arm+' '+field)
        return metrics
    for name,rec in records.items():
        n,L,s,pitch=f.CASES[name];need(rec['geometry']==[n,L,s,pitch] and rec['passed'], 'actual geometry')
        for phase in (0,1,2):
            controls[name,phase]=f.control(n,L,pitch,phase)
            for arm in ('bf16','k8'):golds[name,arm,phase]={k:array(v) for k,v in rec['golden'][arm+str(phase)].items()}
        views=rec['actual_views']
        for arm in ('bf16','k8'):
            need(views[arm]['partial']['shape']==[n,16,s,128] and views[arm]['partial']['stride']==[32768,2048,128,1], 'actual width16 O stride')
            for field in ('maximum','summation'):
                need(views[arm][field]['shape']==[n,16,s] and views[arm][field]['stride']==[256,16,1], 'actual width16 ML stride')
        need(views['k8']['key']['storage_offset']==256 and views['k8']['scale']['storage_offset']==64 and views['k8']['value']['storage_offset']==4224, 'typed offset roles')
        need(views['k8']['key']['stride']==[16384,256,128,1] and views['k8']['value']['stride']==[8192,256,128,1], 'cache strides')
        need(rec['K8_pointer_roles']==dict(initial_and_consumer_V_is_offset_bytes=8192,append83_V_full_base=True,K_and_scale_same_full_base=True), 'pointer roles')
        numeric={(x['arm'],x['phase'],x['stage']):x for x in rec['numeric']}
        need(len(rec['snapshots'])==18 and len(numeric)==12, 'snapshot stage count')
        snaps={};metric_count=0;max_rr=0.;max_abs=0.;by_field={k:dict(max_relative_RMSE=0.,max_absolute_error=0.) for k in ('partial','maximum','summation','output')}
        for snap in rec['snapshots']:
            key=(snap['arm'],snap['phase'],snap['stage']);need(key not in snaps,'duplicate stage');snaps[key]=snap
            m=check_snapshot(rec,snap,golds[name,snap['arm'],snap['phase']]);metric_count+=4
            if key in numeric:metric_same(m,numeric[key]['metrics'])
            for k in ('partial','maximum','summation','output'):
                max_rr=max(max_rr,m[k]['relative_RMSE']);max_abs=max(max_abs,m[k]['maximum_absolute_error'])
                by_field[k]['max_relative_RMSE']=max(by_field[k]['max_relative_RMSE'],m[k]['relative_RMSE'])
                by_field[k]['max_absolute_error']=max(by_field[k]['max_absolute_error'],m[k]['maximum_absolute_error'])
        order=[(x['arm'],x['phase'],x['stage']) for x in rec['snapshots']]
        for arm in ('bf16','k8'):
            for phase,first in ((0,'initial_eager'),(1,'changed_graph_before_eager'),(2,'changed_side_before_eager')):
                key=(arm,phase,first);other=(arm,phase,'new_eager_or_repeat')
                need(order.index(key)<order.index(other),'graph/side before fresh eager ledger')
                equal_fields(snaps[key]['full_arrays'],snaps[other]['full_arrays'])
            timing=rec['timing_bitwise_reference'][arm]
            for stage in ('timing_graph_guard','timing_fresh_eager','post_timing'):equal_fields(snaps[arm,0,stage]['full_arrays'],timing)
            for role in ('partial','maximum','summation','output'):
                need(load(snaps[arm,0,'initial_eager']['full_arrays'][role])!=load(snaps[arm,1,'changed_graph_before_eager']['full_arrays'][role]), 'Graph B differs from A '+role)
        for role in ('k8_append','bf16_append','reducer'):
            k=rec['actual_compiled'][role];need(type(k['n_regs']) is int and k['n_regs']>0 and type(k['n_spills']) is int and k['n_spills']==0,'actual helper resources')
        for role,path,function in [('bf16_append','tools/kernels/biv150_pair_append_support.py','append_kv'),
                                   ('initial_encoder','tools/kernels/biv150_pair_append_support.py','encode_initial'),
                                   ('k8_append','tools/kernels/biv150_konly_strided_writer.py','encode_append'),
                                   ('reducer','vllm_fl/patches/biv150_gqa_decode.py','_reduce_partials')]:
            text=(source/path).read_text();node=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name==function)
            expected_source=ast.get_source_segment(text,node)+'\n';actual=rec['actual_compiled'][role]
            need(sha(expected_source.encode())==actual['source_sha256'],'actual JIT function source '+role)
            need(set(actual['artifacts'])=={'source','ttir','ttgir','llir','asm','cubin'},'six actual compiler artifacts')
            need(all(len(load(meta))>0 for meta in actual['artifacts'].values()),'nonempty compiler artifacts')
        signature={ast.literal_eval(k):v for k,v in rec['actual_compiled']['k8_append']['signature'].items()}
        need(signature==dict(key_source='*bf16',value_source='*bf16',slots='*i64',code_full='*i8',scale_full='*fp32',value_full='*bf16',
                             key_row_stride='i64',value_row_stride='i64',key_rows='i64',value_rows='i64',slot_count='i64',cache_pages='i64'),'actual K8 append ABI')
        need(len(rec['cache_checks'])==18 and len(rec['initial_source_readonly_checks'])==2,'complete cache/source ledger')
        for check in rec['cache_checks']+rec['initial_source_readonly_checks']:
            need(check['guards_exact'] and check['all_bytes_compared']==f.PAGES*16384,'runtime full-cache byte check')
            chunks=check['chunks'];need([x['start'] for x in chunks]==list(range(0,f.PAGES,128)), 'chunk coverage')
            need(sum(x['bytes'] for x in chunks)==f.PAGES*16384,'chunk byte coverage')
        timing=rec['timing'];need([x['arm'] for x in timing]==list(f.ORDER),'ABBA order')
        averages={}
        for arm in ('bf16','k8'):
            rows=[x for x in timing if x['arm']==arm];averages[arm]={k:sum(x[k] for x in rows)/2 for k in ('gpu_ms_per_step','graph_launch_sync_wall_ms_per_step')}
            need(all(math.isfinite(v) and v>0 for x in rows for k,v in x.items() if k!='arm'),'finite positive clocks')
        event=averages['bf16']['gpu_ms_per_step']/averages['k8']['gpu_ms_per_step'];wall=averages['bf16']['graph_launch_sync_wall_ms_per_step']/averages['k8']['graph_launch_sync_wall_ms_per_step']
        need(event==rec['event_ratio_BF16_over_K8'] and wall==rec['host_wall_ratio_BF16_over_K8'],'ABBA arithmetic')
        report['records'].append(dict(case=name,snapshot_count=18,full_array_numeric_checks=metric_count,max_relative_RMSE=max_rr,max_absolute_error=max_abs,
            metrics_by_field=by_field,all_four_actual_JIT_function_sources_match_frozen=True,actual_K8_append_six_pointer_six_I64_ABI=True,
            all_four_field_graph_side_and_post_timing_bits=True,actual_width16=True,cache_runtime_checks=18,full_cache_bytes_per_arm=f.PAGES*16384,
            initial_costs_separate=rec['initial'],ABBA=timing,averages=averages,event_BF16_over_K8=event,host_wall_BF16_over_K8=wall,
            helper_resources={k:dict(n_regs=v['n_regs'],n_spills=v['n_spills'],hash=v['hash']) for k,v in rec['actual_compiled'].items()}))
    if a.regenerate:
        print('Regenerating full cold counter and CPU packed oracle',flush=True)
        t=time.monotonic();cold,packed,cost=f.build_initial();counter_time=time.monotonic()-t
        cold_sha=sha(memoryview(cold).cast('B'));packed_sha=sha(memoryview(packed).cast('B'))
        for rec in records.values():
            need(cold_sha==rec['initial']['CPU_BF16_sha256'] and packed_sha==rec['initial']['CPU_K8_oracle_sha256'],'full regenerated cold counter bytes')
        regen=[]
        for name,rec in records.items():
            print('Regenerating '+name+' phase0 gold, both arms',flush=True)
            n,L,s,pitch=f.CASES[name]
            for arm in ('bf16','k8'):
                t=time.monotonic();g=f.golden(cold,packed,controls[name,0],s,arm);saved=golds[name,arm,0];deltas={}
                for field in g:
                    if field=='active':need(np.array_equal(g[field],saved[field]),'regenerated active mask');continue
                    finite=np.isfinite(g[field]);need(np.array_equal(finite,np.isfinite(saved[field])),'gold finite mask')
                    difference=float(np.max(np.abs(g[field][finite]-saved[field][finite]))) if finite.any() else 0.
                    # CPU-library FP64 replay allowance only; all GPU gates below stay .005.
                    need(np.allclose(g[field],saved[field],rtol=2e-12,atol=2e-12,equal_nan=True),'regenerated gold differs '+field)
                    deltas[field]=dict(max_abs=difference,bitwise_equal=bool(np.array_equal(g[field],saved[field],equal_nan=True)))
                for snap in rec['snapshots']:
                    if snap['arm']==arm and snap['phase']==0:check_snapshot(rec,snap,g)
                regen.append(dict(case=name,arm=arm,phase=0,all_rows=n,fields=deltas,actual_against_regenerated_original_005_gates=True,seconds=time.monotonic()-t))
            print('Regenerating '+name+' cache chunk digests for all phases',flush=True)
            expected={}
            for arm,initial in (('bf16',cold),('k8',packed)):
                for phase in (None,0,1,2):
                    chunks=[]
                    for start in range(0,f.PAGES,128):
                        count=min(128,f.PAGES-start);part=initial[start:start+count]
                        data=part.view(np.uint8).reshape(count,16384) if phase is None else f.patched_chunk(arm,start,part,controls[name,phase])
                        chunks.append(dict(start=start,bytes=data.nbytes,sha256=sha(memoryview(data).cast('B'))))
                    expected[arm,phase]=chunks
            for check in rec['cache_checks']:need(check['chunks']==expected[check['arm'],check['phase']],'regenerated expected cache digests')
            for check in rec['initial_source_readonly_checks']:need(check['chunks']==expected['bf16',None],'regenerated source digests')
        report.update(cold_counter_and_gold_regenerated=True,cold_counter_seconds=counter_time,cold_BF16_sha256=cold_sha,cold_K8_sha256=packed_sha,
            regenerated_phase0=regen,all_phases_cache_expected_chunk_digests_regenerated=True,
            cache_evidence_limit='GPU cache bodies were discarded; independent CPU expected chunk digests match the runtime complete-byte-comparison records, not a fresh read of GPU memory',
            FP64_saved_vs_regenerated_comparison_tolerance=dict(rtol=2e-12,atol=2e-12,applies_only_to_CPU_reference_replay=True,GPU_numeric_gates_changed=False))
    need(not {'torch','triton'}.intersection(sys.modules),'framework import forbidden')
    report.update(passed=True,frameworks_loaded=[],seconds=time.monotonic()-begin,
                  saved_metric_max_float_replay_delta=metric_float_delta,
                  saved_metric_FP64_replay_rtol=4e-14,GPU_numeric_thresholds_changed=False)
    a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(passed=True,report=str(a.out),sha256=sha(a.out.read_bytes()),seconds=report['seconds']),sort_keys=True))


if __name__=='__main__': main()
