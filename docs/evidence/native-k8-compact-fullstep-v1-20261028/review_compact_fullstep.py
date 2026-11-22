"""CPU compact12544 identity/address adapter around the original fullstep audit."""
import argparse,ast,hashlib,json,math,subprocess,sys,types
from pathlib import Path
SOURCE=Path('/private/tmp/flagos-cjj-compact-fullstep-frozen-fa9-20261028')
PARENT=Path('/private/tmp/flagos-cjj-native-k8-fullstep-frozen-3eb-20261028')
RAW=Path('/private/tmp/cjj-native-k8-compact-fullstep-v1-raw-20261028')
AUDIT=Path('/private/tmp/check_cjj_native_k8_fullstep_actual_20261028.py')
COMMIT='fa92060af9399e60938ddee6dd7e0a603610cd8c'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def change(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--regenerate',action='store_true');a=p.parse_args();assert not a.out.exists()
 assert sha(AUDIT)=='1c527358d545110a462866c2cc675334e5ea9baf9ccf4a047bec1bcd1f6c060f'
 assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
 assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'])
 sys.path.insert(0,str(SOURCE/'tools'));import probe_biv150_native_k8_compact_fullstep as c
 old=c.prepare(PARENT);summary=json.loads((RAW/'out/fullstep-summary.json').read_text())
 assert summary['K8_library_sha256']==c.LIBRARY and summary['BF16_library_sha256']==old.f.BF16_SHA
 assert summary['compact_fullstep_adapter']==c.adapted_source()[1]
 assert summary['compact_writer_sources']==json.loads(json.dumps(c.FROZEN)) and summary['compact_writer_source_commit']==c.WRITER_COMMIT
 assert summary['format']==c.FORMAT and summary['namespace']=='cjj_native_k8_compact_probe'
 for role,(name,pin) in c.FROZEN.items():
  actual=RAW/'out/frozen-sources'/Path(name).name
  assert sha(actual)==pin and actual.read_bytes()==c.blob(c.WRITER_COMMIT,name)
 compiled=Path('/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out')
 args=types.SimpleNamespace(compact_consumer_raw=Path('/private/tmp/cjj-native-k8-compact-consumer-v1-raw-20261028'),compile_summary=compiled/'compile-summary.json',codec_summary=Path('/private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json'),original_library=Path('/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so'),k8_library=compiled/'build/cjj_native_k8_compact_probe.so')
 actual=c.replay_consumer(args)
 # The receipt is independently replayed, not trusted as a boolean. Host-time
 # and host last-bit reference diagnostics are observations, not identity pins.
 for key in ('summary_sha256','compact_library_sha256','original_library_sha256','raw_files','raw_bytes','owner','cases','full_array_numeric_hard_checks','full_field_cross_arm_byte_comparisons','readonly_snapshot_checks','guard_and_physical_padding_checks','original_BF16_final_proxy_checks','source_commit','script_sha256','scope','encoder_launches','writer_verified','model_verified','precision_gate','capacity_reclaimed','service_verified','performance_verified'):
  assert actual[key]==summary['actual_compact_consumer'][key],key
 assert actual['summary_sha256']==c.CONSUMER and actual['compact_library_sha256']==c.LIBRARY
 # Execute the exact nested compact_plan source with the same captured globals.
 execute_node=next(n for n in ast.parse((SOURCE/'tools/probe_biv150_native_k8_compact_fullstep.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='execute')
 node=next(n for n in execute_node.body if isinstance(n,ast.FunctionDef) and n.name=='compact_plan')
 plan_ns=dict(vars(c),old=old,actual=summary['actual_compact_consumer'],transformation=c.adapted_source()[1])
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact compact_plan source>','exec'),plan_ns)
 expected_plan=plan_ns['compact_plan']()
 text=AUDIT.read_text()
 text=change(text,"dict(pid=706887,start_ticks=347999537,same_start=True,state='Z')","dict(pid=38381,start_ticks=354308715,same_start=True,state='Z')")
 text=change(text,"len(paths)==manifest['raw_files']==192 and total==manifest['raw_bytes']==144847098","len(paths)==manifest['raw_files']==181 and total==manifest['raw_bytes']==145937040")
 text=change(text,"raw/'exports/cjj-native-k8-fullstep-v1-20261028-owner.json'","raw/'exports/cjj-native-k8-compact-fullstep-v1-20261028-owner.json'")
 text=change(text,"owner['source_commit']==COMMIT and owner['pid']==706887 and owner['start_ticks']==347999537","owner['source_commit']=="+repr(COMMIT)+" and owner['pid']==38381 and owner['start_ticks']==354308715")
 text=change(text,'json.loads(json.dumps(f.plan()))','json.loads(json.dumps(COMPACT_PLAN))')
 text=change(text,'source_commit=COMMIT,source_root=str(source),raw_root=str(raw),','source_commit='+repr(COMMIT)+',parent_fixture_commit=COMMIT,source_root=str(source),raw_root=str(raw),')
 text=change(text,'raw_files_verified=192','raw_files_verified=181')
 text=change(text,"views['k8']['value']['storage_offset']==4224","views['k8']['value']['storage_offset']==2304")
 text=change(text,"views['k8']['key']['stride']==[16384,256,128,1] and views['k8']['value']['stride']==[8192,256,128,1]","views['k8']['key']['stride']==[12544,256,128,1] and views['k8']['value']['stride']==[6272,256,128,1]")
 text=change(text,"dict(initial_and_consumer_V_is_offset_bytes=8192,append83_V_full_base=True,K_and_scale_same_full_base=True)","dict(consumer_V_is_offset_bytes=4352,append83_V_full_base=True,K_and_scale_same_full_base=True)")
 text=change(text,"for role in ('k8_append','bf16_append','reducer'):","for role in ('initial_encoder','k8_append','bf16_append','reducer'):")
 # The compact initial encoder is the SAME dc5 encode_append function, rather
 # than the original strided-cache encode_initial helper.
 text=change(text,"('initial_encoder','tools/kernels/biv150_pair_append_support.py','encode_initial'),","('initial_encoder',str(COMPACT_WRITER),'encode_append'),")
 text=change(text,"('k8_append','tools/kernels/biv150_konly_strided_writer.py','encode_append'),","('k8_append',str(COMPACT_WRITER),'encode_append'),")
 text=change(text,"check['guards_exact'] and check['all_bytes_compared']==f.PAGES*16384","check['guards_exact'] and check['all_bytes_compared']==f.PAGES*(12544 if check.get('arm')=='k8' else 16384)")
 text=change(text,"sum(x['bytes'] for x in chunks)==f.PAGES*16384","sum(x['bytes'] for x in chunks)==f.PAGES*(12544 if check.get('arm')=='k8' else 16384)")
 text=change(text,"full_cache_bytes_per_arm=f.PAGES*16384,","full_cache_bytes_per_arm={'bf16':f.PAGES*16384,'k8':f.PAGES*12544},")
 text=change(text,"                        chunks.append(dict(start=start,bytes=data.nbytes,sha256=sha(memoryview(data).cast('B'))))","                        if arm=='k8':data=COMPACT_REMAP(data)\n                        chunks.append(dict(start=start,bytes=data.nbytes,sha256=sha(memoryview(data).cast('B'))))")
 # Recheck the two separately packed full linear inputs and identity slots.
 # These costs are outside timing and may not silently disappear from the ledger.
 extra='''
            for check in rec['compact_initial_source_checks']:
                need(check['stage'] in ('after_initial_pack','after_timing'),'linear source stage')
                for item in check['sources']:
                    role=item['source'];need(role in ('key','value','slots'),'linear role')
                    width=128 if role=='slots' else 8192
                    need(item['guards_exact'] and item['all_bytes_compared']==f.PAGES*width,'linear source byte coverage')
                    chunks=[]
                    for start in range(0,f.PAGES,128):
                        count=min(128,f.PAGES-start)
                        data=(np.arange(start*16,(start+count)*16,dtype='<i8').tobytes() if role=='slots' else cold[start:start+count,0 if role=='key' else 1].tobytes())
                        chunks.append(dict(start=start,bytes=len(data),sha256=sha(data)))
                    need(item['chunks']==chunks,'independently regenerated full linear source')
'''
 anchor="            for check in rec['initial_source_readonly_checks']:need(check['chunks']==expected['bf16',None],'regenerated source digests')"
 text=change(text,anchor,anchor+extra)
 ns=dict(__name__='cjj_existing_fullstep_compact_adapter',__file__=str(AUDIT),COMPACT_PLAN=expected_plan,COMPACT_WRITER=RAW/'out/frozen-sources/biv150_konly_compact_writer.py',COMPACT_REMAP=c.compact_chunk)
 original_argv=sys.argv;sys.argv=[str(AUDIT),'--raw-root',str(RAW),'--source-root',str(PARENT),'--out',str(a.out)]+(['--regenerate'] if a.regenerate else [])
 try:exec(compile(text,str(AUDIT)+'::<compact address/source adapter>','exec'),ns);ns['main']()
 finally:sys.argv=original_argv
 report=json.loads(a.out.read_text());f=old.f
 for rec in summary['records']:
  n,L,s,pitch=f.CASES[rec['case']]
  assert len(rec['writer_bindings'])==3
  for phase,b in enumerate(rec['writer_bindings']):
   assert b==dict(cache_pages=f.PAGES,format=c.FORMAT,grid=[n,2],key_row_stride=256,key_rows=n,launch=True,live_gpu_mapping_certified=False,requires_finite_valid_K_external=True,requires_gpu_slots_equal_snapshot=True,slot_count=n,slot_epoch=phase,value_row_stride=256,value_rows=n)
  assert rec['compact_roles']==dict(format=c.FORMAT,page_bytes=12544,pages=f.PAGES,V_consumer_offset=4352,writer_three_full_bases=True,frozen_views_sha256=c.FROZEN['views'][1])
  assert [v['stage'] for v in rec['compact_initial_source_checks']]==['after_initial_pack','after_timing']
  assert all({v['source'] for v in check['sources']}=={'key','value','slots'} for check in rec['compact_initial_source_checks'])
  for role in ('initial_encoder','k8_append'):assert rec['actual_compiled'][role]['source_sha256']==rec['actual_compiled']['k8_append']['source_sha256']
  for arm in ('bf16','k8'):assert rec['warm'][arm]['replays']==128 and rec['warm'][arm]['unroll']==8
  assert all(key in rec['initial'] for key in ('compact_initial_key_linear_pack','compact_initial_value_linear_pack','compact_initial_slots','K8_GPU_full_encode','source_GPU_upload','CPU_full_counter_and_oracle_total_seconds'))
 cleanup_source=c.adapted_source()[0]
 assert 'del initial_backs,initial_values;torch.cuda.empty_cache()' in cleanup_source
 ideal={r['case']:sum(not v['metrics']['ideal_partial_diagnostic']['passed'] for v in r['numeric']) for r in summary['records']}
 report.update(compact_frontend_source_commit=COMMIT,compact_frontend_sha256=sha(SOURCE/'tools/probe_biv150_native_k8_compact_fullstep.py'),independent_adapter_sha256=sha(__file__),original_auditor_sha256=sha(AUDIT),compact_consumer_independently_replayed=actual,compact_source_transform=c.adapted_source()[1],actual_dc5_writer_and_views_byte_exact=True,compact_writer_commit=c.WRITER_COMMIT,compact_writer_sources=c.FROZEN,compact_layout=dict(page_bytes=12544,Kcode=[0,4096],FP32scale=[4096,4224],reserved=[4224,4352],V_BF16=[4352,12544],nonzero_layer_base_bytes=256),compact_remap_applied_to_all_initial_and_appended_gold_chunks=True,full_linear_initial_K_V_slot_source_hashes_regenerated=a.regenerate,ideal_partial_diagnostic_failures_preserved=ideal,cleanup_evidence=dict(original_owner_terminal=True,case_source_deletes_allocations_and_calls_empty_cache=True,complete_summary_reached_after_case_cleanup=True,postrun_GPU_idle_or_other_process_state_not_in_archive=True),scope_limits=['Fixed one-layer synthetic arenas with34276 pages; no42-layer capacity/lifetime/model qualification.','Timed newKV pitch256; actual service pitch2560 is not covered.','BF16 control is oldaace component, not latest main.','No paired comparison with new8b86; no latest-main or model gain claim.','GPU cache/input bodies not archived; CPU reconstructed expected chunk hashes corroborate runtime full-byte/readonly comparisons, not a new GPU readback.','Archived pyc files are hashed for transport completeness but never imported.'],capacity_or_model_or_service_verified=False,relative_to_current8b86_gain_verified=False)
 a.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(report=str(a.out),sha256=sha(a.out),passed=report['passed'],compact_cache_remap=True,service_or_capacity=False),sort_keys=True))
if __name__=='__main__':main()
