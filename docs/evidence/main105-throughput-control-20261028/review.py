"""Finite local CPU review, derived from original PR98 throughput checks."""
import ast,base64,hashlib,importlib.util,json,math,re,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
R=Path('/private/tmp/cjj-main105-throughput-v1-raw-20261028');O=R/'out'
S=Path('/private/tmp/flagos-cjj-main105-frozen-1cb-20261028')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
a=read(R/'download-manifest.json');m=read(O/'manifest.json');state=read(O/'state.json');owner=read(next((R/'exports').glob('*owner.json')))
assert (a['raw_files'],a['raw_bytes'])==(30,869640) and len(a['files'])==30
assert sum(x['bytes'] for x in a['files'])==869640 and len({x['relative'] for x in a['files']})==30
for x in a['files']:
 p=R/x['relative'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
assert a['original_owner']==dict(pid=14598,start_ticks=353505395,same_start=True,state='Z') and not a['model_rerun']
assert (owner['pid'],owner['start_ticks'])==(14598,353505395) and owner['official_score'] is False
commit='1cbac531ca9afaa0e9a68624d5d10cb1d0412055'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()==m['source_commit']==owner['source_commit']==commit
assert len(m['source_hashes'])==238
for rel,h in m['source_hashes'].items():assert sha(S/rel)==h,rel
controller=O/'executed-controller.py';assert sha(controller)==m['controller_sha256']=='145ae5002af7769e8ae4777e23f33febe7ed01366dca53f16046bf5727adb9b9'
assert sha(Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py'))==m['helper_sha256']=='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
launch=read(Path('/private/tmp/cjj-main105-throughput-v1-20261028-launch.json'))
assert base64.b64decode(launch['controller_base64'])==controller.read_bytes()
expected_args=[x.replace('{SOURCE}',m['source']).replace('{OUT}',owner['out']) for x in launch['args']]
assert owner['command']==['/usr/local/bin/python',launch['script'],*expected_args]
assert '--baseline' in expected_args and expected_args[-2:]==['--warmup-plan','full-only']
assert all(x not in expected_args for x in ('--native-k8-library','--q32-hoist','--graph-coverage-bootstrap-dir','--model105','--supplemental-math105-parallel64'))
spec=importlib.util.spec_from_file_location('reviewed_v4_controller',controller);ctrl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctrl)
assert m['warmup_plan_metadata']==ctrl.warmup_plan_metadata(SimpleNamespace(warmup_plan='full-only',throughput=True,full_workload_warmup=True,graph_coverage_bootstrap_dir=None))
assert m['private_enabled'] is False and m['native_decode_requested'] is False and m['native_decode_library'] is None and m['fresh_q32_hoist_requested'] is False
assert m['expected_compile_semantics']==m['off_compile_semantics'] and 'konly_encoded' not in m['expected_compile_semantics']
assert m['graph_coverage_bootstrap']==dict(enabled=False,formal_score_eligible=False,quality_covered=False)
env=ctrl.native_env({ctrl.ENABLE:'0',ctrl.K8_ENABLE:'1',ctrl.K8_PATH:'/obsolete',ctrl.K8_HASH:'stale',ctrl.GRAPH_FLAG:'1'})
assert env[ctrl.ENABLE]=='0' and env[ctrl.K8_ENABLE]=='0' and not any(x in env for x in (ctrl.K8_PATH,ctrl.K8_HASH,ctrl.GRAPH_FLAG))
assert m['actual_resolution']['origin']==m['source']+'/vllm_fl/__init__.py' and not m['actual_resolution']['package_executed']
assert m['abi']==dict(torch='2.10.0',cxx11_abi=False)
service=['/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
assert m['command']==service and m['official_service_parameters_unchanged'] is True
text=(S/'tools/run_biv150_normalization_accuracy.py').read_text();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('benchmark','throughput')]
protocol='\n'.join(''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno]) for n in nodes)
assert protocol==m['throughput_protocol_source'] and hashlib.sha256(protocol.encode()).hexdigest()==m['throughput_protocol_function_sha256']=='e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d'
assert all(state[k] is True for k in ('completed','smoke_passed','throughput_measured','cleanup_complete')) and state['phase']=='completed' and not state['errors']
assert all(state[k] is False for k in ('model_precision_verified','native_quality_covered','native_model_precision_verified')) and m['full_acceptance'] is False
cleanup=read(O/'owned-cleanup.json');shutdown=read(O/'graceful-api-shutdown.json')
assert cleanup['complete'] and cleanup['remaining_owned']==cleanup['unobserved_same_session']==cleanup['signal_errors']==[]
assert cleanup['pid_reuse_never_signalled'] and not cleanup['group_signal_used']
assert shutdown['identity_verified'] and shutdown['API_exited'] and shutdown['returncode']==0
scores=read(O/'scores.json');warms=read(O/'warmups.json');assert scores==state['throughput']
score_checks={};warm_checks={};cache={}
for length,count in ((4096,256),(16384,128)):
 key=str(length);collected=[]
 for label in (f'full-warm-{length}',f'score-{length}'):
  command=['vllm','bench','serve','--backend','vllm','--model','minicpm','--tokenizer','/workspace/MiniCPM5-2B','--endpoint','/v1/completions','--host','127.0.0.1','--port','9031','--dataset-name','random','--ignore-eos','--random-input-len',key,'--random-output-len','1024','--max-concurrency','64','--num-prompts',str(count),'--seed','0','--disable-tqdm','--save-result','--result-dir',owner['out']+'/'+label,'--label',label]
  assert read(O/label/'command.json')==command
  paths=list((O/label).glob(label+'-*.json'));assert len(paths)==1
  result=read(paths[0]);assert result['label']==label and result['completed']==result['num_prompts']==count and result['failed']==0 and result['max_concurrency']==64
  assert result['total_input_tokens']==length*count and result['total_output_tokens']==1024*count
  assert result['model_id']=='minicpm' and result['tokenizer_id']=='/workspace/MiniCPM5-2B' and result['request_rate']=='inf'
  for metric,num in [('total_token_throughput',(length+1024)*count),('output_throughput',1024*count),('request_throughput',count)]:assert math.isclose(result[metric],num/result['duration'],rel_tol=1e-12)
  assert all(math.isfinite(result[k]) and result[k]>0 for k in ('duration','total_token_throughput','mean_ttft_ms','mean_tpot_ms'))
  assert ctrl.validate_score(result,length,1024,count)['result_arithmetic_verified']
  collected.append(result)
 assert len(warms[key])==1 and dict(collected[0],full_workload=True,scored=False)==warms[key][0]
 assert collected[1]==scores[key] and m['official_score_commands'][key]==read(O/f'score-{key}/command.json')
 score_checks[key]={k:collected[1][k] for k in ('date','duration','completed','failed','total_input_tokens','total_output_tokens','total_token_throughput','output_throughput','mean_ttft_ms','mean_tpot_ms','max_concurrency','max_concurrent_requests')}
 warm_checks[key]={k:collected[0][k] for k in ('date','duration','completed','failed','total_token_throughput','mean_ttft_ms')};warm_checks[key]['scored']=False
 obs=read(O/f'score-{key}-cache-observation.json');before,after=obs['before'],obs['after'];assert before==read(O/f'score-{key}-cache-before.json')
 assert obs['new_metadata']==sorted(after['triton_json_sha256'].keys()-before['triton_json_sha256'].keys())
 assert obs['changed_metadata']==sorted(k for k in after['triton_json_sha256'].keys()&before['triton_json_sha256'].keys() if after['triton_json_sha256'][k]!=before['triton_json_sha256'][k])
 assert obs['selected_tuning_rows_changed']==(before['selected_tuning_rows_sha256']!=after['selected_tuning_rows_sha256'])
 cache[key]={k:obs[k] for k in ('new_metadata','changed_metadata','selected_tuning_rows_changed','scope')}
log=(O/'server.log').read_text();runtime=ctrl.evidence(log,m['expected_compile_semantics'],native_gqa_sha=m['native_gqa_library_sha256'])
assert json.loads(json.dumps(runtime))==m['actual_runtime_after_throughput']
assert len(runtime['actual_compile_semantics'])==2 and len(runtime['preserved_parent_installations'])==9 and runtime['stock_main_control']
assert runtime['native_decode']==dict(enabled=False,quality_covered=False)
assert all(x not in log for x in ('BIV150_KONLY_NATIVE_DECODE_INSTALL ','BIV150_KONLY_ENCODED_SELECTED ','BIV150_NATIVE_GRAPH_COVERAGE '))
gqa=runtime['native_gqa'];assert len(gqa['installs'])==1 and len(gqa['events'])==4 and not gqa['request_frequency_proven']
inst=gqa['installs'][0];assert inst['installed'] and inst['overlap_v'] and not inst['compile_at_runtime'] and inst['extra_KV_bytes']==0
assert inst['library_sha256']==m['native_gqa_library_sha256']=='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8'
assert inst['supported_shapes']==[[64,2],[32,4]] and inst['q_strides']==[[2048,128,1],[2560,128,1]]
for marker in ('BIV150_NATIVE_GQA_SELECTED','BIV150_NATIVE_GQA_CAPTURED'):
 for n,s in ((64,2),(32,4)):
  rows=[x for x in gqa['events'] if x['marker']==marker and x['rows']==n and x['segments']==s];assert len(rows)==1
  x=rows[0];assert x['grid']==[n,2,s] and x['threads']==64 and x['q_stride']==[2048,128,1] and x['k_stride']==x['v_stride']==[8192,256,128,1] and x['overlap_v'] and x['extra_KV_bytes']==0
assert m['native_shape_scope']==dict(required=[[32,4],[64,2]],observed=[[32,4],[64,2]],request_frequency_proven=False)
assert 'GPU KV cache size: 548,416 tokens' in log and m['native_sha256']=='5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
assert m['native_burst64_window']['native_events']==[] and not m['native_burst64_window']['Graph_replays_counted']
assert sha(O/'native-burst64-runtime-window.log')==m['native_burst64_window']['sha256']
# Existing local native-GQA build archive supplies a fresh binary hash; no remote process is contacted.
B=Path('/private/tmp/cjj-main105-native-gqa-build-v1-raw-20261028');bm=read(B/'download-manifest.json');assert (bm['raw_files'],bm['raw_bytes'])==(8,471741)
for row in bm['files']:
 p=B/row['relative'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
assert sha(B/'out/corex_native_gqa_s2_probe.so')==m['native_gqa_library_sha256']
old=read(Path('/private/tmp/cjj-native-k8-service-throughput-v1-raw-20261028/out/scores.json'))
descriptive={k:dict(original_a975_PR98_total_tok_s=old[k]['total_token_throughput'],original_a975_PR98_mean_TTFT_ms=old[k]['mean_ttft_ms'],PR98_minus_this_main105_throughput_percent=100*(old[k]['total_token_throughput']/scores[k]['total_token_throughput']-1),PR98_minus_this_main105_TTFT_percent=100*(old[k]['mean_ttft_ms']/scores[k]['mean_ttft_ms']-1)) for k in scores}
report=dict(local_CPU_only=True,raw_root=str(R),raw_files=30,raw_bytes=869640,all_archive_SHA_and_sizes_verified=True,source_commit=commit,source_hashes_verified=238,controller_version='v4/full-only',controller_sha256=m['controller_sha256'],helper_sha256=m['helper_sha256'],source_and_executed_controller_and_launch_config_bound=True,fixed_service_CLI=service,benchmark_function_sha256=m['throughput_protocol_function_sha256'],all_four_complete_warm_score_commands_and_token_duration_arithmetic_verified=True,warmup_plan=m['warmup_plan_metadata'],scores=score_checks,warmups=warm_checks,private_flags=dict(encoded_K8=False,native_K8=False,q32_hoist=False,Graph_coverage_observer=False,compiled_fingerprint_matches_OFF=True,native_enabled_in_C3_is_parent_native_path_not_private_K8=True),libraries=dict(W8A8_SHA_recorded=m['native_sha256'],BF16_GQA_SHA_binary_rehashed=m['native_gqa_library_sha256'],BF16_GQA_build_archive_files_rehashed=8,tuning_snapshot_source='v6 per exact owner CLI',tuning_snapshot_records=m['tuning_snapshot']),runtime=dict(compile_semantics_records=2,parent_installers=9,BF16_native_GQA_events=gqa['events'],KV_capacity_tokens=548416,actual_request_frequency_proven=False,actual_Graph_replay_count_proven=False),cache_observations=cache,original_owner_at_collection=a['original_owner'],terminal_state=dict(updated_utc=state['updated_utc'],completed=True,cleanup_complete=True,API_returncode=0,remaining_owned=[],errors=[]),reported_concurrent_archive_activity=dict(provenance='Root task report; not independently timed or resource-profiled in these30 raw files',during_4K='Readonly download of old fullstep large archive;240-second timeout',stopped_before_16K_score=True,other_GPU_workload_reported=False,zero_CPU_IO_network_effect_proven=False,noisy_or_negligible_effect_conclusion_supported=False),historical_descriptive_comparison_only=descriptive,limitations=['One complete warm and one score per geometry; no variance estimate or sufficient-warmth guarantee.','4K cache observation found4 new index metadata and selected tuning rows changed;16K had no new/changed selected metadata. Scope includes benchmark-client preflight, not isolated timed cost.','Parent reports concurrent old fullstep archive read during4K. Absence of another GPU task does not prove zero CPU/IO/network impact; no noise/significance attribution is made.','SELECTED/CAPTURED events establish native implementation selection and capture identities, not actual request frequency or replay coverage.','max_concurrent_requests68/65 is preserved alongside exact CLI max-concurrency64; request-level latency/token arrays were not saved.','W8A8 and tuning snapshot hashes are controller/run records; their binary/database files are excluded from this archive. The separate local GQA archive allowed fresh0c66 binary rehash.','PR98 a975 descriptive comparison changes source,GQA library,physical capacity,warm scheduling and timing; it is not a controlled causal gain or merge qualification.','This throughput-only control does not run Math105 or establish model precision. No PR105 documentation average or other-library results are substituted.'],precision_gate=False,model_verified=False,full_acceptance=False,merge_qualified=False,SSH_or_GPU_or_PR_used=False,script_sha256=sha(Path(__file__)))
assert not {'torch','triton'}.intersection(sys.modules)
path=Path('/private/tmp/cjj-main105-throughput-actual-review-20261028.json');path.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(report=str(path),sha256=sha(path),scores=score_checks,descriptive=descriptive),indent=2))
