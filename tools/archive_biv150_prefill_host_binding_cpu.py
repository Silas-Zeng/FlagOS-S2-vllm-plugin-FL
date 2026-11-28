#!/usr/bin/env python3
"""CPU-only preservation/recalculation of the completed e684 two-case screen."""
import argparse
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile

from biv150_resource_records import compare_resource_records

SOURCE='e684ff8e190822771080d5205c262064d58bf900'
MAIN='56f1838192abc940b3fb9accef490022ac125000'
MANIFEST='96fdfc661dd71802c56baff09a33b3f925affdaa2f36458b4b69f5121212f9f4'
RESULT='64bb5bde325aacd32853818e5a1a178990bca4becdf354337bab277ffc6a185b'
RUNS={
    'e684':dict(source=SOURCE,manifest=MANIFEST,result=RESULT,raw_bytes=5902115,
        owner=dict(pid=624624,start_ticks=342942814,same_start=True,state='Z'),archive='v2-raw.tar.gz'),
    '5967':dict(source='5967d49c7e4ef7b19ddf5b3344fb7ff1db9bf30d',
        manifest='34a60e221851ad96372f43bc568476a067e49e5d3d397e70983c37c03efcdb42',
        result='99f18a8abd926680221a8f32013f3ba65795e56f8843dca875e738d07740e417',raw_bytes=5912292,
        owner=dict(pid=626303,start_ticks=343560965,same_start=True,state='Z'),archive='v4-raw.tar.gz')}


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def exact(record):
    need(record['passed']is True and record['finite']is True and record['numeric_equal']is True,'Exact output guard failed')
    need(record['mismatched_elements']==record['outside_tolerance']==0 and record['max_abs_diff']==0,'Nonzero exact error')


def reference(record):
    need(record['passed']is True and record['finite']is True,'Reference guard failed')
    need(record['new_or_worsened_violations']==0,'New oracle violation')


def audit(r,files,source=SOURCE):
    need(r['source_commit']==source and r['production_parent']==MAIN,'Wrong source')
    need(r['completed']is True and r['requested_component_gates_passed']is True and r['errors']==[],'Component incomplete')
    need(r['full_numerical_matrix_verified']is False,'Two requested cases are not full matrix')
    need(r['component_only']is True and all(r[k]is False for k in('official_score','model_precision_gate','service_installed','timing_is_service_throughput')),'Wrong formal scope')
    need(r['parent_restored']is True and r['omitted_boundary_records']==0 and r['parent_M2_retained']is True,'Scope/cleanup failure')
    need(r['timing_includes_component_host_overhead']is True and r['baseline_arm_has_no_new_tap']is True,'Timing/control changed')
    need([(x['name'],x['actual_kv'],x['cpu_upper'],x['expected_route'])for x in r['cases']]==
        [('chunk4k',4096,4096,'reverse'),('chunk16k',16384,16384,'parent')],'Requested matrix changed')
    need(r['source']['reversed_source_sha256']=='38198704c37479e4b5c26778d7dd1da2a894409d1c81ff2626e1d988facca104'and r['source']['route_mode']=='always','Not old fca source')
    rows=[]
    for c in r['cases']:
        need(c['completed']is True and c['query']==2048,'Case incomplete')
        need([x['pattern']for x in c['checks']]==[0,1],'Changed-data coverage missing')
        for check in c['checks']:
            exact(check['direct']);reference(check['reference_guard'])
            need(all(check[k]is True for k in('typed_abi_equal','compiler_options_target_equal','actual_selected_Q_hoist_verified','output_canaries_unchanged','inputs_unchanged')),'ABI/Qhoist/side guard failed')
            need(check['actual_route']==c['expected_route']and check['grid']==[65,2],'Selected scope/grid differs')
            exact(check['stale_contract']);exact(check['side_stream'])
            need(len(check['graphs'])==3,'Graph coverage missing')
            for graph in check['graphs']:exact(graph)
        change=c['changed_metadata'];exact(change['direct']);reference(change['reference_guard'])
        exact(change['stale_contract_after_metadata_change'])
        need(change['actual_kv']==c['actual_kv']-1 and change['metadata_restored_before_timing']is True and change['graph_host_decision_recomputed']is False,'Changed metadata proof differs')
        need(len(change['graphs'])==3,'Changed-metadata graph coverage missing')
        for graph in change['graphs']:exact(graph)
        for label in('baseline_resources','selected_candidate_resources','generic_capture_parent_resources'):
            resource=c[label];need(set(resource['files'])=={'source','ttir','ttgir','llir','asm','cubin'},'Actual artifacts missing')
            for item in resource['files'].values():
                raw=files['out/'+item['path']];need(len(raw)==item['bytes']and sha(raw)==item['sha256'],'Resource artifact differs')
        resources=compare_resource_records(c['baseline_resources'],c['selected_candidate_resources'])
        need(resources==c['resource_comparison']and resources['strict_allowed'],'Actual resource guard differs')
        need(c['baseline_resources']['n_regs']==c['selected_candidate_resources']['n_regs']==113 and
             c['baseline_resources']['n_spills']==c['selected_candidate_resources']['n_spills']==0,'Actual resource counts changed')
        if c['expected_route']=='parent':
            need(c['baseline_resources']['hash']==c['selected_candidate_resources']['hash'],'Long parent binary differs')
            for field in('selected_calls','selected_compiles','fallback_calls','fallback_compiles'):
                need(c['reverse_before'][field]==c['reverse_after'][field],'Long context touched reverse')
        need(len(c['rounds'])==3,'Timing rounds missing')
        for i,timing in enumerate(c['rounds']):
            need(timing['iteration']==i and timing['order']==(['baseline','candidate']if i%2==0 else['candidate','baseline']),'AB/BA/AB order differs')
            for unit in('wall','event'):
                before,after=timing['baseline_'+unit+'_ms'],timing['candidate_'+unit+'_ms']
                need(math.isfinite(before)and math.isfinite(after)and before>0 and after>0,'Invalid timing')
                need(timing[unit+'_speedup']==before/after,'Raw ratio differs')
        for unit in('wall','event'):
            need(c['median_'+unit+'_speedup']==statistics.median(x[unit+'_speedup']for x in c['rounds']),'Median differs')
        rows.append(dict(name=c['name'],query=c['query'],actual_kv=c['actual_kv'],cpu_upper=c['cpu_upper'],route=c['expected_route'],
            rounds=c['rounds'],median_wall_speedup=c['median_wall_speedup'],median_event_speedup=c['median_event_speedup'],
            baseline_hash=c['baseline_resources']['hash'],selected_hash=c['selected_candidate_resources']['hash'],
            n_regs=113,n_spills=0,compiler_shared=65536,resource_guard=resources,
            recorded_numeric_Graph_side_stale_typed_ABI_Qhoist_gates_passed=True))
    for item in r['saved_override_ir']:
        raw=files['out/'+item['path']];need(sha(raw)==item['sha256']and len(raw)==item['bytes'],'Saved actual override changed')
    need(len(r['saved_override_ir'])==2,'Missing actual compiler override')
    for manager in r['manager_proofs'].values():
        need(manager['record_errors']==[]and manager['omitted_records']==0,'Manager proof incomplete')
        for record in manager['compile_records']:
            if record.get('mode')!='selected':continue
            proof=record['mechanism'];need(proof['mechanism_pass']and proof['llvm_q_memory_reads_in_loop']==0 and proof['llvm_q_memory_reads_total']==32,'Manager Qhoist proof failed')
    summary=dict(cpu_only=True,new_GPU_execution=False,source_commit=source,production_parent=MAIN,cases=rows,
        requested_component_gates_passed=True,full_numerical_matrix_verified=False,
        numeric_device_comparisons_reexecuted=False,saved_actual_files_SHA_verified=True,
        new_independent_IR_transform_validation_claimed=False,model105_evaluated=False,service_evaluated=False,
        official_score=False,model_precision_gate=False,
        conclusion='Two requested component cases passed. 4K event ratio is positive but wall median is below1; no host/service gain established. 16K uses the original parent binary. No claim of a full numerical matrix or stable effect.',
        late_round_host_attribution='Raw rounds do not identify CPU phases, GC or scheduler latency; structural extra Python work is a candidate cause, not a measured causal attribution.')
    if source==RUNS['5967']['source']:
        adapter=r['host_adapter'];certificate=adapter['parent_certificate']
        need(all(certificate[k]is True for k in('loaded_policy_and_predicate_code_equal',
            'whole_module_compile_context_preserved','module_not_executed_for_reference',
            'loaded_function_and_globals_paths_equal_pinned_root','all_CodeType_fields_strictly_compared')),'Actual certificate proof failed')
        need(adapter['private_qualification_cache_entries']==1 and adapter['private_qualification_cache_misses']==1
             and adapter['private_qualification_cache_hits']==61,'Actual qualification lifecycle differs')
        summary['actual_host_adapter']=adapter
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run',choices=RUNS,default='e684');a=p.parse_args();run=RUNS[a.run]
    files={x.relative_to(a.snapshot).as_posix():x.read_bytes()for x in a.snapshot.rglob('*')if x.is_file()}
    need(sha(files['download-manifest.json'])==run['manifest']and sha(files['out/result.json'])==run['result'],'Wrong frozen snapshot')
    manifest=json.loads(files['download-manifest.json']);names={'download-manifest.json'}
    for row in manifest['files']:
        name=row['relative'];path=PurePosixPath(name);need(not path.is_absolute()and'..'not in path.parts and name not in names,'Unsafe/duplicate member')
        raw=files[name];need(len(raw)==row['bytes']and sha(raw)==row['sha256'],'Raw mismatch: '+name);names.add(name)
    need(set(files)==names and len(names)==76 and manifest['raw_files']==75 and manifest['raw_bytes']==run['raw_bytes'],'Wrong raw inventory')
    need(sum(len(files[n])for n in names if n!='download-manifest.json')==run['raw_bytes'],'Wrong raw byte count')
    need(manifest['original_owner']==run['owner'],'Wrong owner')
    result=json.loads(files['out/result.json']);summary=audit(result,files,run['source'])
    runtime_manifest=json.loads(files['out/manifest.json'])
    need(runtime_manifest['result_sha256']==run['result']and runtime_manifest['source_commit']==run['source']and runtime_manifest['actual_GPU_result_pending']is False,'Run manifest differs')
    faults=[]
    for name,mutate in [('matrix',lambda r:r.update(full_numerical_matrix_verified=True)),
        ('bitwise',lambda r:r['cases'][0]['checks'][0]['direct'].update(numeric_equal=False)),
        ('reference',lambda r:r['cases'][0]['checks'][0]['reference_guard'].update(new_or_worsened_violations=1)),
        ('graph',lambda r:r['cases'][0]['checks'][0]['graphs'][0].update(passed=False)),
        ('ratio',lambda r:r['cases'][0]['rounds'][0].update(wall_speedup=1.)),
        ('cleanup',lambda r:r.update(parent_restored=False))]:
        bad=copy.deepcopy(result);mutate(bad)
        try:audit(bad,files,run['source'])
        except(ValueError,AssertionError):faults.append(name)
        else:raise ValueError('Fault accepted: '+name)
    summary.update(original_manifest_sha256=run['manifest'],original_result_sha256=run['result'],raw_files=75,raw_bytes=run['raw_bytes'],
        original_owner=manifest['original_owner'],rejected_faults=faults,
        opaque_private_cache_preserved=True,private_cache_or_binary_loaded=False,
        note='Run manifest lists private cache as excluded, but this downloaded snapshot includes it. All75 downloaded originals are preserved, including opaque launch SO/cubin/cache files.')
    buffer=io.BytesIO()
    with gzip.GzipFile(filename='',fileobj=buffer,mode='wb',mtime=0,compresslevel=9)as zipped:
        with tarfile.open(fileobj=zipped,mode='w',format=tarfile.PAX_FORMAT)as archive:
            for name,raw in sorted(files.items()):
                info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.uid=info.gid=info.mtime=0;info.uname=info.gname='';archive.addfile(info,io.BytesIO(raw))
    raw=buffer.getvalue();need(len(raw)<=2*1024*1024,'Archive exceeds publisher bound')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/run['archive']).write_bytes(raw)
    (a.out/'download-manifest.json').write_bytes(files['download-manifest.json']);(a.out/'summary.json').write_bytes(encode(summary))
    (a.out/'archive-manifest.json').write_bytes(encode(dict(archive=run['archive'],bytes=len(raw),sha256=sha(raw),
        archive_members=76,original_manifest_sha256=run['manifest'],original_result_sha256=run['result'],summary_sha256=sha(encode(summary)),files=manifest['files'])))
    need('torch'not in sys.modules and'triton'not in sys.modules,'GPU import occurred')
    print(json.dumps(dict(passed=True,cpu_only=True,archive_sha256=sha(raw),archive_bytes=len(raw),summary_sha256=sha(encode(summary)))))


if __name__=='__main__':main()
