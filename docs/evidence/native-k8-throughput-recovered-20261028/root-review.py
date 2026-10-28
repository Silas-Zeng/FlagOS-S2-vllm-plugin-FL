"""Finite local-only replay of the completed original PR98 throughput records."""
import ast, hashlib, json, math, subprocess
from pathlib import Path
from collections import Counter
root=Path('/private/tmp/cjj-native-k8-service-throughput-v1-raw-20261028')
source=Path('/private/tmp/flagos-cjj-native-k8-service-frozen-a97-20261028')
report_path=Path('/private/tmp/cjj-resumed-native98-throughput-review-20261028.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
d=read(root/'download-manifest.json')
assert d['raw_files']==len(d['files'])==66
assert d['raw_bytes']==sum(x['bytes'] for x in d['files'])==4039331
assert len({x['relative'] for x in d['files']})==66
for f in d['files']:
 p=root/f['relative'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['relative']
m=read(root/'out/manifest.json');state=read(root/'out/state.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==m['source_commit']=='a975e8715beb752d6a76b456f417ba2011ea917b'
assert len(m['source_hashes'])==252
for rel,h in m['source_hashes'].items(): assert sha(source/rel)==h,rel
assert sha(root/'out/executed-controller.py')==m['controller_sha256']=='4d597cd4b8d632a8ea928bbdc6be607dc02ed02f03ba241bcc1d47ddd6059a63'
assert sha(Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py'))==m['helper_sha256']=='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
text=(source/'tools/run_biv150_normalization_accuracy.py').read_text();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('benchmark','throughput')]
protocol='\n'.join(''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno]) for n in nodes)
assert protocol==m['throughput_protocol_source']
assert hashlib.sha256(protocol.encode()).hexdigest()==m['throughput_protocol_function_sha256']=='e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d'
expected_service=['/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
assert m['command']==expected_service
assert m['actual_resolution']['origin']==m['source']+'/vllm_fl/__init__.py'
assert m['actual_resolution']['package_executed'] is False
assert all(state[k] is True for k in ('completed','smoke_passed','throughput_measured','cleanup_complete'))
assert state['errors']==[] and state['phase']=='completed'
assert all(state[k] is False for k in ('model_precision_verified','native_quality_covered','native_model_precision_verified'))
assert m['full_acceptance'] is False
cleanup=read(root/'out/owned-cleanup.json');shutdown=read(root/'out/graceful-api-shutdown.json')
assert cleanup['complete'] and cleanup['remaining_owned']==[] and cleanup['unobserved_same_session']==[] and cleanup['signal_errors']==[]
assert shutdown['identity_verified'] and shutdown['API_exited'] and shutdown['returncode']==0
scores=read(root/'out/scores.json');assert scores==state['throughput']
warmups=read(root/'out/warmups.json');score_checks={};warm_checks={}
for length,count in ((4096,256),(16384,128)):
 key=str(length);labels=[f'warm-{length}-{i}' for i in range(3)]+[f'full-warm-{length}',f'score-{length}']
 collected=[]
 for idx,label in enumerate(labels):
  output,num,seed=(128,64,920000+idx) if idx<3 else (1024,count,0)
  command=['vllm','bench','serve','--backend','vllm','--model','minicpm','--tokenizer','/workspace/MiniCPM5-2B','--endpoint','/v1/completions','--host','127.0.0.1','--port','9031','--dataset-name','random','--ignore-eos','--random-input-len',key,'--random-output-len',str(output),'--max-concurrency','64','--num-prompts',str(num),'--seed',str(seed),'--disable-tqdm','--save-result','--result-dir',m['source'].rsplit('/code/',1)[0]+'/'+d['tag']+'/'+label,'--label',label]
  assert read(root/'out'/label/'command.json')==command,label
  paths=list((root/'out'/label).glob(label+'-*.json'));assert len(paths)==1
  result=read(paths[0]);assert result['label']==label and result['completed']==num and result['failed']==0 and result['num_prompts']==num and result['max_concurrency']==64
  assert result['total_input_tokens']==length*num and result['total_output_tokens']==output*num
  for metric,numerator in [('total_token_throughput',(length+output)*num),('output_throughput',output*num),('request_throughput',num)]:
   assert math.isclose(result[metric],numerator/result['duration'],rel_tol=1e-12),label
  assert all(math.isfinite(result[k]) and result[k]>0 for k in ['duration','total_token_throughput','mean_ttft_ms','mean_tpot_ms'])
  collected.append(result)
 assert collected[:3]==warmups[key][:3]
 assert dict(collected[3],full_workload=True,scored=False)==warmups[key][3]
 assert collected[-1]==scores[key]
 r=collected[-1]
 score_checks[key]={k:r[k] for k in ['date','label','duration','completed','failed','total_input_tokens','total_output_tokens','total_token_throughput','mean_ttft_ms','mean_tpot_ms','max_concurrency','max_concurrent_requests']}
 score_checks[key]['tokens_duration_arithmetic_verified']=True
 warm_checks[key]={'labels':[v['label'] for v in collected[:4]],'full_warm_total_token_throughput':collected[3]['total_token_throughput'],'full_warm_not_scored':True}
log=(root/'out/server.log').read_text()
parse_json=lambda marker:[json.loads(line.split(marker+' ',1)[1]) for line in log.splitlines() if marker+' ' in line]
runtime=m['actual_runtime_after_throughput'];c3=parse_json('BIV150_COMPILE_SEMANTICS')
assert len(c3)==2 and c3==runtime['actual_compile_semantics'] and all(v==m['expected_compile_semantics'] for v in c3)
assert not any(x in log for x in ['BIV150_COMPUTE_ORDER_SCOPE_REJECTED','BIV150_PREFILL_MULTI2_FALLBACK'])
installs=parse_json('BIV150_KONLY_NATIVE_DECODE_INSTALL')
assert installs==runtime['native_decode']['installations'] and len(installs)==1
install=installs[0];assert install['installed'] and install['loaded_library_confirmed'] and install['library']==m['native_decode_library']
assert install['module_sha256']==sha(source/'vllm_fl/patches/biv150_konly_encoded_native.py')
events=[]
for marker in ('BIV150_KONLY_NATIVE_DECODE_SELECTED','BIV150_KONLY_NATIVE_DECODE_CAPTURED'):
 events.extend({'marker':marker,'event':e} for e in parse_json(marker))
# Manifests preserve the same marker-group ordering as the controller.
assert events==runtime['native_decode']['events'] and len(events)==168
scopes=[]
for marker in ('BIV150_KONLY_NATIVE_DECODE_SELECTED','BIV150_KONLY_NATIVE_DECODE_CAPTURED'):
 for n,s in ((64,2),(32,4)):
  es=[x['event'] for x in events if x['marker']==marker and x['event']['shape']==n and x['event']['segments']==s]
  assert len(es)==42 and {e['entry_generation'] for e in es}==set(range(1,43))
  assert all(e['library']==m['native_decode_library'] and e['native_launch_returned'] and e['physical_page_bytes']==16384 and e['query']['strides']==[2048,128,1] for e in es)
  scopes.append({'marker':marker,'rows':n,'segments':s,'distinct_layer_generations':42,'query_pitch':2048})
assert 'GPU KV cache size: 548,320 tokens' in log
for marker,values in runtime['preserved_parent_installations'].items():
 parsed=[ast.literal_eval(line.split(marker+' ',1)[1]) for line in log.splitlines() if marker+' ' in line]
 assert json.loads(json.dumps(parsed))==values and all(x['installed'] is True for x in values),marker
cache={}
for key in scores:
 obs=read(root/f'out/score-{key}-cache-observation.json')
 cache[key]={k:obs[k] for k in ('new_metadata','changed_metadata','selected_tuning_rows_changed','scope')}
oldroot=Path('/private/tmp/cjj-main-b3-fullwarm-service-records-raw-v1-20261028')
old=read(oldroot/'out/scores.json');oldmanifest=read(oldroot/'out/manifest.json')
assert oldmanifest['native_sha256']==m['native_sha256'] and oldmanifest['native_gqa_library_sha256']==m['native_gqa_library_sha256']
comparisons={}
for key,r in scores.items():
 prior=old[key];comparisons[key]={'old_b3_total_tok_s':prior['total_token_throughput'],'old_b3_mean_ttft_ms':prior['mean_ttft_ms'],'descriptive_throughput_delta_percent':100*(r['total_token_throughput']/prior['total_token_throughput']-1),'descriptive_ttft_delta_percent':100*(r['mean_ttft_ms']/prior['mean_ttft_ms']-1),'would_pass_1pct_ttft_gate_against_this_old_b3_point':r['mean_ttft_ms']<=1.01*prior['mean_ttft_ms']}
main=Path('/Users/chenjunjie/Documents/flagos/repos/FlagOS-S2-vllm-plugin-FL')
prdoc=main/'docs/zh/biv150-native-gqa-v-overlap.md';prtext=prdoc.read_text()
assert all(x in prtext for x in ('6627.816667','4832.660000','103/105','c8a725da','14f4457a'))
pr105={'source':'local latest main documentation; upstream raw run records were not inspected by this local-only review','document':str(prdoc),'document_sha256':sha(prdoc),'main_commit_at_read':subprocess.check_output(['git','rev-parse','HEAD'],cwd=main,text=True).strip(),'statistics':'four official rounds, discard first, mean of last three','precision':'103/105 reported','GQA_SO_prefix':'c8a725da','W8A8_SO_prefix':'14f4457a','KV_capacity_tokens':{'4096':548336,'16384':548288},'scores':{'4096':6627.816667,'16384':4832.66},'ttft_not_in_local_doc':True}
pr105['descriptive_PR98_minus_PR105_percent']={k:100*(scores[k]['total_token_throughput']/v-1) for k,v in pr105['scores'].items()}
report={'review_schema':1,'local_cpu_only':True,'ssh_used':False,'gpu_used':False,'raw_root':str(root),'download_manifest_sha256':sha(root/'download-manifest.json'),'raw_integrity':{'files':66,'bytes':4039331,'all_sha256_and_sizes_verified':True},'source_commit':m['source_commit'],'frozen_source_hashes_verified':252,'controller_sha256':m['controller_sha256'],'helper_sha256':m['helper_sha256'],'benchmark_function_sha256':m['throughput_protocol_function_sha256'],'fixed_service_command_verified':m['command'],'all_ten_benchmark_commands_verified':True,'original_owner_at_collection':d['original_owner'],'terminal_state':{'phase':state['phase'],'updated_utc':state['updated_utc'],'completed':True,'errors':[],'cleanup_complete':True,'graceful_API_returncode':0},'scores':score_checks,'warmups':warm_checks,'libraries_recorded_in_worker':{'W8A8':m['native_sha256'],'BF16_GQA':m['native_gqa_library_sha256'],'private_K8':m['native_decode_library']['sha256']},'worker_evidence':{'compile_semantics_records_matching_expected':2,'parent_installers_verified':len(runtime['preserved_parent_installations']),'native_library_load_and_import_record_verified':True,'native_events':168,'scopes':scopes,'KV_capacity_tokens':548320,'private_page_bytes':16384,'Graph_replay_or_request_frequency_proven':False,'model_precision_verified':False,'burst64_native_runtime_window_events':len(m['native_burst64_window']['native_events'])},'cache_observations':cache,'old_same_libraries_b3_comparison':comparisons,'PR105_latest_main_document':pr105,'precision_gate':False,'full_acceptance':False,'merge_qualified':False,'limitations':['This verifies completed original diagnostic throughput records, not model precision.','One scored sample per geometry; warmup results are explicitly excluded from scored results.','The old b3 4K score had a large unexplained fullwarm-to-score drop; its descriptive delta is not a stable gain estimate.','The latest PR105 uses different W8A8/GQA library artifacts, capacities, and four-round discard-first statistics; no controlled comparison or cross-config combination is valid.','PR105 TTFT originals were not provided in the local document; no latest-main relative TTFT qualification is claimed.','Against the old local b3 point, 4K TTFT exceeds +1%; this is distinct from any competition baseline working cap.','Recorded max_concurrent_requests (69/65) exceeds requested CLI concurrency64; this field is preserved, not rewritten. CLI commands and max_concurrency64 are verified, but request-level arrays are not saved.','Host SELECTED/CAPTURED records establish launch/capture identities, not actual Graph replay counts or model accuracy bucket coverage.','Generated caches and native library binaries were excluded from this archive; binary identities here are worker/controller records, not fresh rehashes of remote binaries.','Selected tuning row hashes cover the controller-selected scope, not all typed SQLite rows; metadata scope includes client preflight, not isolated score timing.']}
report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report_path)
print(sha(report_path))
print(json.dumps({'scores':score_checks,'latest_descriptive_deltas':pr105['descriptive_PR98_minus_PR105_percent'],'old_ttft':comparisons,'precision_gate':False,'merge_qualified':False},ensure_ascii=False,indent=2))
