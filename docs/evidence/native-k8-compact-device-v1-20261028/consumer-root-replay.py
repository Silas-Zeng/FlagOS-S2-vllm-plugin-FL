"""Finite CPU replay of compact100 saved arrays using existing 51da fixtures."""
import functools,gzip,hashlib,io,json,math,sys,time
from pathlib import Path
SOURCE=Path('/private/tmp/flagos-cjj-native-k8-compact-consumer-frozen-51d-20261028')
ROOT=Path('/private/tmp/cjj-native-k8-compact-consumer-v1-raw-20261028');OUT=ROOT/'out'
COMPILE=Path('/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out')
sys.path.insert(0,str(SOURCE/'tools'))
import probe_biv150_native_k8_compact_consumer as frontend
import biv150_native_k8_compact_consumer_contract as contract
import biv150_native_k8_consumer_v2_fixture as f
from biv150_native_k8_codec_receipt import require_codec
import numpy as np
start=time.monotonic();sha=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_text())
manifest=read(ROOT/'download-manifest.json');d=read(OUT/'consumer-summary.json');owner=read(next((ROOT/'exports').glob('*owner.json')))
assert (manifest['raw_files'],manifest['raw_bytes'])==(138,32787475)
assert manifest['original_owner']==dict(pid=14175,start_ticks=353418626,same_start=True,state='Z') and not manifest['model_rerun']
assert owner['pid']==14175 and owner['start_ticks']==353418626 and owner['source_commit']=='51da6fd73e7fff1513081fb330323ce72c00856c'
assert not owner['official_score'] and '--gpu' in owner['command'] and '--root-owned-exclusive' in owner['command']
assert len(manifest['files'])==138 and sum(x['bytes'] for x in manifest['files'])==32787475
for row in manifest['files']:
 raw=(ROOT/row['relative']).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
for row in d['artifact_manifest']:contract.artifact(OUT,row)
assert all(d[k] is True for k in ('completed','implementation_verified','full_field_bitwise_verified','Graph_verified','side_stream_verified','eager_numeric_passed','original_BF16_proxy_passed')) and not d['errors']
assert all(d[k] is False for k in ('precision_gate','model_verified','service_installed','capacity_reclaimed','N40_supported','thresholds_changed')) and d['timing'] is None
assert (d['attention_launches'],d['eager_producer_calls'],d['eager_reducer_calls'],d['capture_pairs'],d['graph_replay_pairs'],d['reducer_launches'],d['encoder_launches'])==(32,24,24,8,8,32,0)
assert d['target']==['corex',71,64] and d['cases']==[list(x) for x in f.CASES]
assert d['source_proof']==frontend.factory.verify() and d['reference']==f.plan()
for name,digest in {**d['frontend_sources'],**d['reference_source_pins']}.items():assert sha((SOURCE/'tools'/name).read_bytes())==digest
assert sha(f.REDUCER_FILE.read_bytes())==f.REDUCER_SHA
assert (OUT/'compile-summary-input.json').read_bytes()==(COMPILE/'compile-summary.json').read_bytes()
review_path=OUT/'actual-so-review-input.json';review_sha=sha(review_path.read_bytes());assert review_sha=='3b3e94c9af183b77df5be794746f2b783c1b925f643b5d1d450e13da3a3ddc28'
admission=contract.admission(COMPILE/'compile-summary.json',review_path,review_sha,COMPILE/'build/cjj_native_k8_compact_probe.so');assert admission==d['compact_admission']
codecpath=Path('/private/tmp/cjj-native-k8-codec-v1-raw-20261028/out/codec-summary.json')
assert codecpath.read_bytes()==(OUT/'original-codec-summary-input.json').read_bytes()
assert require_codec(codecpath,'3c9733ab79e69c77f70068cff2d3da553ad9c2b81b08ea3c396c6e0788b4f263')==d['old_codec_admission']
assert d['original_library_sha256']==f.LIBRARY_SHA and sha(Path('/private/tmp/cjj-native-k8-compile-v1-raw-20261028/out/build/cjj_native_k8_qk_probe.so').read_bytes())==f.LIBRARY_SHA
@functools.lru_cache(maxsize=12)
def unpack(path):
 packed=(OUT/path).read_bytes();return packed,gzip.decompress(packed)
def load(meta):
 assert meta['encoding']=='gzip';packed,raw=unpack(meta['path']);assert (len(packed),sha(packed),len(raw),sha(raw))==(meta['stored_bytes'],meta['stored_sha256'],meta['bytes'],meta['sha256']);return raw
fields=('partial','maximum','summation','output');shapes=dict(partial=(64,16,16,128),maximum=(64,16,16),summation=(64,16,16),output=(64,16,128))
stages=('A-producer','A-after','graphB-after','eagerB-producer','eagerB-after','sideC-after')
pattern_by_stage={'A-before':0,'A-producer':0,'A-after':0,'graphB-before':1,'graphB-after':1,'eagerB-before':1,'eagerB-producer':1,'eagerB-after':1,'sideC-before':2,'sideC-after':2}
metric_checks=proxy_checks=guard_checks=cross_arm_checks=readonly_checks=gold_arrays=0;ideal_failures=0;gold_bitexact=True;gold_max_abs=0.;metric_max_abs=0.;metric_max_rel=0.;results=[]
def equal_metric(a,b):
 assert set(a)==set(b)
 for key in a:
  if type(a[key]) is float:assert math.isclose(a[key],b[key],rel_tol=1e-9,abs_tol=1e-12),(key,a[key],b[key])
  else:assert a[key]==b[key],(key,a[key],b[key])
for record in d['records']:
 n,s,pitch=record['case'];assert (n,s,pitch) in f.CASES and record['completed']
 print('CPU array replay',record['case'],flush=True)
 cases={p:f.fixture(n,s,pitch,p) for p in (0,1,2)};gold={p:f.golden(cases[p]) for p in cases};expected={p:f.input_bytes(cases[p]) for p in cases}
 for phase,g in gold.items():
  saved=record['golden'][str(phase)];assert set(g)==set(saved)
  for field,wanted in g.items():
   if not isinstance(wanted,np.ndarray):assert saved[field]==wanted;continue
   actual=np.load(io.BytesIO(load(saved[field])),allow_pickle=False);assert actual.shape==wanted.shape and actual.dtype==wanted.dtype
   exact=np.array_equal(actual,wanted,equal_nan=True);gold_bitexact &= exact;gold_arrays+=1
   assert np.allclose(actual,wanted,rtol=1e-12,atol=1e-11,equal_nan=True)
   if wanted.dtype!=np.bool_:gold_max_abs=max(gold_max_abs,float(np.nanmax(np.abs(actual-wanted))))
 for arm,ar in record['arms'].items():
  assert arm in ('original','compact') and ar['completed']
  inputs={p:dict(expected[p]) for p in cases}
  if arm=='compact':
   for p in cases:inputs[p]['cache']=frontend.compact_bytes(inputs[p]['cache'])
  assert set(ar['snapshots'])==set(pattern_by_stage)
  for p in cases:
   for name,value in inputs[p].items():assert load(ar['upload_readbacks'][str(p)][name])==value
  for stage,phase in pattern_by_stage.items():
   for name,value in inputs[phase].items():assert load(ar['snapshots'][stage][name])==value;readonly_checks+=1
  binding=ar['binding'];capture=ar['graph_capture_binding'];k=binding['key'];v=binding['value'];scale=binding['scale'];cache=binding['inputs']['cache']
  step,vs,offset,ss=(12544,6272,4352,3136) if arm=='compact' else (16384,8192,8192,4096)
  assert k['strides']==[step,256,128,1] and v['strides']==[vs,256,128,1] and scale['shape']==[29*ss]
  assert k['pointer']==scale['pointer']==cache['pointer']+128 and v['pointer']==k['pointer']+offset
  assert len(inputs[0]['cache'])==128+29*step+128==cache['storage_bytes']
  for role in ('query','key','scale','value'):assert capture[role]==binding[role]
  for field in fields:assert capture[field]==binding['scratch'][field]
  assert binding['query']['strides']==[pitch,128,1]
  assert binding['scratch']['partial']['strides']==[32768,2048,128,1]
  assert binding['scratch']['maximum']['strides']==binding['scratch']['summation']['strides']==[256,16,1]
  if arm=='compact':
   assert ar['closed_epoch_rejected'] and len(ar['epoch_records'])==3
   for phase,item in enumerate(ar['epoch_records']):
    receipt=item['admission'];assert item['pattern']==phase and receipt['epoch']==phase+1 and receipt['raw']==cache
    assert receipt['live_cache_readback_sha256']==sha(inputs[phase]['cache']) and receipt['lease_is_process_local'] and receipt['layout_checked']
    roles={name:{**value,'shape':tuple(value['shape']),'strides':tuple(value['strides'])} for name,value in receipt['roles'].items()}
    core_receipt={key:receipt[key] for key in ('format','epoch','purpose','storage_id','layer_base','pages','layer_bytes','layout_checked')}
    frontend.factory.admit_fixture(core_receipt,roles,epoch=phase+1)
  assert [x['stage'] for x in ar['checks']]==list(stages)
  for item in ar['checks']:
   stage,phase=item['stage'],item['pattern'];assert phase==pattern_by_stage[stage] and all(item['checks'].values())
   calculated={name+'_readonly':True for name in inputs[phase]};metrics={};snapshot=ar['snapshots'][stage]
   for field,shape in shapes.items():
    output=field=='output';bits=np.frombuffer(load(snapshot[field]),dtype='<u2' if output else '<u4');marker=f.BF16_GUARD if output else f.F32_GUARD
    calculated[field+'_outside_guards']=bool((bits[:64]==marker).all() and (bits[-64:]==marker).all());core=bits[64:-64].reshape(shape);layout=np.zeros(shape,dtype=bool)
    if output:layout[:n]=True;active=core[:n]
    else:layout[:n,:,:s]=True;active=core[:n,:,:s]
    calculated[field+'_physical_padding_guard']=bool((core[~layout]==marker).all());guard_checks+=2
    if output and stage.endswith('-producer'):
     calculated['output_still_poison']=bool((active==f.BF16_POISON).all());continue
    got=(active.astype(np.uint32)<<16).view('<f4').astype(np.float64) if output else active.view('<f4').astype(np.float64)
    if output:
     metrics[field]=f.metrics(got,gold[phase][field]);metrics['original_BF16_proxy']=f.metrics(got,gold[phase]['original_BF16_output']);assert metrics['original_BF16_proxy']['passed'];proxy_checks+=1
     zero=np.array(cases[phase]['lengths'][64:-64])==0;calculated['zero_length_output_positive_zero']=bool((active[zero]==0).all())
    else:
     mask=np.broadcast_to(gold[phase]['active'][:,None,:,None] if field=='partial' else gold[phase]['active'][:,None,:],active.shape)
     calculated[field+'_inactive_poison']=bool((active[~mask]==f.F32_POISON).all());metrics[field]=f.metrics(got[mask],gold[phase][field][mask])
     if field=='partial':metrics['ideal_partial_diagnostic']=f.metrics(got[mask],gold[phase]['ideal_partial_diagnostic'][mask]);ideal_failures+=not metrics['ideal_partial_diagnostic']['passed']
    calculated[field+'_FP64']=metrics[field]['passed'];assert metrics[field]['passed'];metric_checks+=1
    metric_max_abs=max(metric_max_abs,metrics[field]['maximum_absolute_error']);metric_max_rel=max(metric_max_rel,metrics[field]['relative_RMSE'])
   assert calculated==item['checks'] and all(calculated.values())
   assert set(metrics)==set(item['metrics'])
   for key,value in metrics.items():equal_metric(value,item['metrics'][key])
  # Complete byte comparisons include readonly input data, inactive poison, physical padding and guards.
  for name in ar['snapshots']['graphB-after']:assert load(ar['snapshots']['graphB-after'][name])==load(ar['snapshots']['eagerB-after'][name])
  for phase in ('A','eagerB'):
   for field in fields[:3]:assert load(ar['snapshots'][phase+'-producer'][field])==load(ar['snapshots'][phase+'-after'][field])
  for name in inputs[0]:assert load(ar['snapshots']['A-before'][name])!=load(ar['snapshots']['graphB-before'][name])
  assert load(ar['snapshots']['A-after']['output'])!=load(ar['snapshots']['graphB-after']['output'])
  for stage in ('A-before','graphB-before','eagerB-before','sideC-before'):
   for field,shape in shapes.items():
    bits=np.frombuffer(load(ar['snapshots'][stage][field]),dtype='<u2' if field=='output' else '<u4')[64:-64].reshape(shape)
    active=bits[:n] if field=='output' else bits[:n,:,:s];assert np.all(active==(f.BF16_POISON if field=='output' else f.F32_POISON))
  for meta in ar['reducer']['artifacts'].values():load(meta)
 for stage in stages:
  for field in fields:
   assert load(record['arms']['original']['snapshots'][stage][field])==load(record['arms']['compact']['snapshots'][stage][field]);cross_arm_checks+=1
   assert record['cross_arm_full_bitwise'][stage][field] is True
 for key in ('hash','n_regs','n_spills','source_sha256','artifacts'):assert record['arms']['original']['reducer'][key]==record['arms']['compact']['reducer'][key]
 results.append(dict(case=record['case'],three_input_patterns_regenerated=True,all_uploaded_readbacks_and_saved_inputs_equal=True,both_arms_full_OML_final_bitwise=True,changed_B_Graph_first_equals_fresh_eager_B_all_arrays=True,side_C_saved_arrays_numeric_checked=True,pointer_layout_and_epoch_records_checked=True,reducer_OML_readonly=True,compact_closed_epoch_rejection_recorded=True))
assert len(results)==4 and metric_checks==176 and proxy_checks==32 and cross_arm_checks==96
assert not {'torch','triton'}.intersection(sys.modules)
report=dict(CPU_only=True,source_commit=owner['source_commit'],summary_sha256=sha((OUT/'consumer-summary.json').read_bytes()),raw_files=138,raw_bytes=32787475,all_raw_and_internal_artifact_hashes_checked=True,owner=manifest['original_owner'],compile_and_root_review_binding_verified=True,compact_library_sha256=admission['library_sha256'],original_library_sha256=f.LIBRARY_SHA,original_codec_full_bit_replay_passed=True,cases=results,golden_arrays_regenerated=gold_arrays,saved_gold_arrays_bitexact_to_local_regeneration=gold_bitexact,saved_gold_maximum_absolute_difference=gold_max_abs,saved_gold_tolerance='rtol1e-12/atol1e-11; tolerates host FP64 exp/BLAS last-bit differences only',full_field_cross_arm_byte_comparisons=cross_arm_checks,full_array_numeric_hard_checks=metric_checks,original_BF16_final_proxy_checks=proxy_checks,readonly_snapshot_checks=readonly_checks,guard_and_physical_padding_checks=guard_checks,maximum_hard_metric_absolute_error=metric_max_abs,maximum_hard_metric_relative_RMSE=metric_max_rel,ideal_partial_diagnostic_failures_preserved=ideal_failures,reference='Unchanged pagewise BF16 P v2 for partials; unchanged ideal quantized-score FP64 final/M/L; separate original BF16 final proxy; all original .005 thresholds preserved',bounded_component_implementation_verified=True,bounded_Graph_and_side_stream_results_verified=True,scope=dict(pages=29,maximum_context=193,query_pitches=[2048,2560],layer_offset_bytes=128,scratch_backing_batch=64,scratch_backing_segments=16,patterns=3,one_synthetic_layer_only=True),limitations=['Graph and side-stream execution identity comes from hash-bound frontend source and actual run records; saved arrays are independently replayed, not a new GPU run.','Epoch JSON is evidence from the original live gate, not an authorization token or a reconstructed live allocation.','No device writer/encoder,42-layer model,allocator-capacity reclamation,actual service,long-context,performance or timing qualification.','Page compactness changes synthetic physical allocation only; it does not prove service memory-capacity gain.'],encoder_launches=0,writer_verified=False,model_verified=False,precision_gate=False,service_verified=False,capacity_reclaimed=False,performance_verified=False,timing=None,reviewer_GPU_SSH_Torch_Triton_used=False,script_sha256=sha(Path(__file__).read_bytes()),CPU_seconds=time.monotonic()-start)
path=Path('/private/tmp/cjj-compact100-consumer-actual-review-20261028.json');path.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(report=str(path),sha256=sha(path.read_bytes()),metric_checks=metric_checks,cross_arm_checks=cross_arm_checks,CPU_seconds=report['CPU_seconds']),indent=2))
