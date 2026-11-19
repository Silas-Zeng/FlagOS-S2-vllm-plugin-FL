"""Bounded CPU-only replay of PR107 original formal-protocol throughput records."""
import ast,base64,hashlib,importlib.util,json,math,posixpath,subprocess,sys
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace
R=Path('/private/tmp/cjj-native107-main105-throughput-v1-raw-20261028');O=R/'out';S=Path('/private/tmp/flagos-cjj-native107-service-frozen-b5d-20261028')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_text())
a=read(R/'download-manifest.json');m=read(O/'manifest.json');state=read(O/'state.json');owner=read(next((R/'exports').glob('*owner.json')))
assert (a['raw_files'],a['raw_bytes'])==(48,3954804) and len(a['files'])==48 and sum(x['bytes'] for x in a['files'])==3954804
assert len({x['relative'] for x in a['files']})==48
for row in a['files']:
 p=R/row['relative'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
assert a['original_owner']==dict(pid=30958,start_ticks=354031040,same_start=True,state='Z') and not a['model_rerun']
assert (owner['pid'],owner['start_ticks'])==(30958,354031040) and owner['official_score'] is False
commit='b5d1a241f45efd21517d04e25a7dc5d44ac1ca06'
git=lambda *args:subprocess.check_output(['git',*args],cwd=S)
assert git('rev-parse','HEAD').decode().strip()==m['source_commit']==owner['source_commit']==commit
assert not git('status','--porcelain')
# Verify immutable objects, resolving symlinks INSIDE THE SAME tested commit.
tree={}
for line in git('ls-tree','-r',commit).decode().splitlines():
 meta,path=line.split('\t',1);mode,kind,oid=meta.split();tree[path]=(mode,kind,oid)
needed=set(m['source_hashes'])|{'tools/run_biv150_normalization_accuracy.py'}
for path in list(needed):
 if tree[path][0]=='120000':needed.add(posixpath.normpath(posixpath.join(posixpath.dirname(path),git('cat-file','blob',tree[path][2]).decode())))
ids=sorted({tree[path][2] for path in needed});proc=subprocess.run(['git','cat-file','--batch'],cwd=S,input=('\n'.join(ids)+'\n').encode(),stdout=subprocess.PIPE,check=True);data=proc.stdout;at=0;blobs={}
for oid in ids:
 end=data.index(b'\n',at);header=data[at:end].decode().split();assert header[0]==oid and header[1]=='blob';size=int(header[2]);blobs[oid]=data[end+1:end+1+size];at=end+size+2
assert at==len(data);symlinks=[]
def committed(path):
 mode,_,oid=tree[path]
 if mode!='120000':return blobs[oid]
 target=blobs[oid].decode();resolved=posixpath.normpath(posixpath.join(posixpath.dirname(path),target));assert not target.startswith('/') and not resolved.startswith('../')
 assert tree[resolved][0]!='120000';assert (S/path).is_symlink() and str((S/path).readlink())==target
 symlinks.append(dict(path=path,target=target,resolved_in_tested_commit=resolved,link_blob_oid=oid,resolved_blob_oid=tree[resolved][2]));return blobs[tree[resolved][2]]
assert len(m['source_hashes'])==252
for path,h in m['source_hashes'].items():assert digest(committed(path))==h==sha(S/path),path
controller=O/'executed-controller.py';assert sha(controller)==m['controller_sha256']=='558579df4ebd3ac359167a429e9b3c6905952509de5910ebf843162ba0346600'
assert sha(Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py'))==m['helper_sha256']=='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
launch=read(Path('/private/tmp/cjj-native107-main105-throughput-v1-20261028-launch.json'));assert base64.b64decode(launch['controller_base64'])==controller.read_bytes()
args=[x.replace('{SOURCE}',m['source']).replace('{OUT}',owner['out']) for x in launch['args']];assert owner['command']==['/usr/local/bin/python',launch['script'],*args]
assert all(x in args for x in ('--q32-hoist','--native-k8-library','--native-k8-sha256','--throughput','--full-workload-warmup')) and args[-2:]==['--warmup-plan','full-only']
assert '--baseline' not in args and '--graph-coverage-bootstrap-dir' not in args
spec=importlib.util.spec_from_file_location('reviewed_controller',controller);ctrl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctrl)
assert m['warmup_plan_metadata']==ctrl.warmup_plan_metadata(SimpleNamespace(warmup_plan='full-only',throughput=True,full_workload_warmup=True,graph_coverage_bootstrap_dir=None))
assert all(m[k] is True for k in ('private_enabled','native_decode_requested','fresh_q32_hoist_requested','official_service_parameters_unchanged'))
assert m['graph_coverage_bootstrap']==dict(enabled=False,formal_score_eligible=False,quality_covered=False)
assert m['expected_compile_semantics']!=m['off_compile_semantics'] and 'konly_encoded' not in m['off_compile_semantics']
assert m['expected_compile_semantics']['konly_encoded']['q32_hoist_enabled']
assert m['actual_resolution']['origin']==m['source']+'/vllm_fl/__init__.py' and not m['actual_resolution']['package_executed'] and m['abi']==dict(torch='2.10.0',cxx11_abi=False)
service=['/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072'];assert m['command']==service
text=committed('tools/run_biv150_normalization_accuracy.py').decode();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('benchmark','throughput')]
protocol='\n'.join(''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno]) for n in nodes)
assert protocol==m['throughput_protocol_source'] and digest(protocol.encode())==m['throughput_protocol_function_sha256']=='e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d'
assert all(state[k] is True for k in ('completed','smoke_passed','throughput_measured','cleanup_complete')) and state['phase']=='completed' and not state['errors']
assert all(state[k] is False for k in ('model_precision_verified','native_quality_covered','native_model_precision_verified')) and m['full_acceptance'] is False
cleanup=read(O/'owned-cleanup.json');shutdown=read(O/'graceful-api-shutdown.json');assert cleanup['complete'] and cleanup['remaining_owned']==cleanup['unobserved_same_session']==cleanup['signal_errors']==[]
assert cleanup['pid_reuse_never_signalled'] and not cleanup['group_signal_used'];assert shutdown['identity_verified'] and shutdown['API_exited'] and shutdown['returncode']==0
scores=read(O/'scores.json');warms=read(O/'warmups.json');assert scores==state['throughput'];scored={};warmed={};cache={}
for length,count in ((4096,256),(16384,128)):
 key=str(length);collected=[]
 for label in (f'full-warm-{key}',f'score-{key}'):
  command=['vllm','bench','serve','--backend','vllm','--model','minicpm','--tokenizer','/workspace/MiniCPM5-2B','--endpoint','/v1/completions','--host','127.0.0.1','--port','9031','--dataset-name','random','--ignore-eos','--random-input-len',key,'--random-output-len','1024','--max-concurrency','64','--num-prompts',str(count),'--seed','0','--disable-tqdm','--save-result','--result-dir',owner['out']+'/'+label,'--label',label]
  assert read(O/label/'command.json')==command;paths=list((O/label).glob(label+'-*.json'));assert len(paths)==1;r=read(paths[0]);assert r['label']==label and r['completed']==r['num_prompts']==count and r['failed']==0 and r['max_concurrency']==64
  assert r['total_input_tokens']==length*count and r['total_output_tokens']==1024*count
  for metric,num in [('total_token_throughput',(length+1024)*count),('output_throughput',1024*count),('request_throughput',count)]:assert math.isclose(r[metric],num/r['duration'],rel_tol=1e-12)
  assert all(math.isfinite(r[k]) and r[k]>0 for k in ('duration','total_token_throughput','mean_ttft_ms','mean_tpot_ms')) and ctrl.validate_score(r,length,1024,count)['result_arithmetic_verified'];collected.append(r)
 assert len(warms[key])==1 and dict(collected[0],full_workload=True,scored=False)==warms[key][0] and collected[1]==scores[key]
 assert m['official_score_commands'][key]==read(O/f'score-{key}/command.json')
 scored[key]={k:collected[1][k] for k in ('date','duration','completed','failed','total_input_tokens','total_output_tokens','total_token_throughput','output_throughput','mean_ttft_ms','mean_tpot_ms','max_concurrency','max_concurrent_requests')}
 warmed[key]={k:collected[0][k] for k in ('date','duration','completed','failed','total_token_throughput','mean_ttft_ms')};warmed[key]['scored']=False
 obs=read(O/f'score-{key}-cache-observation.json');before,after=obs['before'],obs['after'];assert before==read(O/f'score-{key}-cache-before.json')
 assert obs['new_metadata']==sorted(after['triton_json_sha256'].keys()-before['triton_json_sha256'].keys())
 assert obs['changed_metadata']==sorted(k for k in after['triton_json_sha256'].keys()&before['triton_json_sha256'].keys() if after['triton_json_sha256'][k]!=before['triton_json_sha256'][k])
 assert obs['selected_tuning_rows_changed']==(before['selected_tuning_rows_sha256']!=after['selected_tuning_rows_sha256']);cache[key]={k:obs[k] for k in ('new_metadata','changed_metadata','selected_tuning_rows_changed','scope')}
log=(O/'server.log').read_text();runtime=ctrl.evidence(log,m['expected_compile_semantics'],native_gqa_sha=m['native_gqa_library_sha256'],native_library=m['native_decode_library']['path'],native_identity=m['native_identity'])
assert json.loads(json.dumps(runtime))==m['actual_runtime_after_throughput'] and len(runtime['actual_compile_semantics'])==2 and len(runtime['preserved_parent_installations'])==9
native=runtime['native_decode'];assert native['enabled'] and native['actual_imports_verified'] and native['host_or_capture_only'] and not native['actual_request_frequency_proven'] and not native['Graph_replay_observed']
assert len(native['installations'])==1 and len(native['events'])==168
assert m['native_identity']['module_sha256']==sha(S/'vllm_fl/patches/biv150_konly_encoded_native.py')
scopes=[]
for marker in ('BIV150_KONLY_NATIVE_DECODE_SELECTED','BIV150_KONLY_NATIVE_DECODE_CAPTURED'):
 for n,s in ((64,2),(32,4)):
  es=[x['event'] for x in native['events'] if x['marker']==marker and x['event']['shape']==n and x['event']['segments']==s]
  assert len(es)==42 and {x['entry_generation'] for x in es}==set(range(1,43))
  assert all(x['library']==m['native_decode_library'] and x['native_launch_returned'] and x['physical_page_bytes']==16384 and x['cache_pages']==34270 and x['query']['strides']==[2048,128,1] for x in es)
  scopes.append(dict(marker=marker,rows=n,segments=s,distinct_entry_generations=42,query_pitch=2048))
gqa=runtime['native_gqa'];assert len(gqa['installs'])==1 and gqa['installs'][0]['library_sha256']==m['native_gqa_library_sha256'] and gqa['installs'][0]['overlap_v'] and gqa['events']==[]
assert any(x['profile']=='q32' and x['I8_fragment_hoist_verified'] is True for x in runtime['selected'])
assert 'BIV150_NATIVE_GRAPH_COVERAGE ' not in log and m['native_burst64_window']['native_events']==[] and not m['native_burst64_window']['Graph_replays_counted'] and sha(O/'native-burst64-runtime-window.log')==m['native_burst64_window']['sha256']
assert 'GPU KV cache size: 548,320 tokens' in log
libpaths=dict(BF16_GQA=Path('/private/tmp/cjj-main105-native-gqa-build-v1-raw-20261028/out/corex_native_gqa_s2_probe.so'),Native_K8=Path('/private/tmp/cjj-native-k8-v-overlap-compile-v1-raw-20261028/out/build/cjj_native_k8_v_overlap_probe.so'))
assert sha(libpaths['BF16_GQA'])==m['native_gqa_library_sha256']=='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8'
assert sha(libpaths['Native_K8'])==m['native_decode_library']['sha256']==ctrl.K8_SHA=='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
baseR=Path('/private/tmp/cjj-main105-throughput-v1-raw-20261028');bm=read(baseR/'out/manifest.json');base=read(baseR/'out/scores.json');baseline_report=Path('/private/tmp/cjj-main105-throughput-actual-review-20261028.json')
assert sha(baseline_report)=='92eb5d3b001763673e669295d7a52bed7cbc92aae469515afc6d257a63c02405'
for row in read(baseR/'download-manifest.json')['files']:
 p=baseR/row['relative'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
assert bm['native_sha256']==m['native_sha256']=='5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
assert bm['native_gqa_library_sha256']==m['native_gqa_library_sha256'] and bm['tuning_snapshot']==m['tuning_snapshot'] and bm['warmup_plan_metadata']==m['warmup_plan_metadata']
official=read(Path('/private/tmp/flagos-track2-official-baseline-review-20261028.json'));assert sha(Path(official['files'][-1]['path']))==official['files'][-1]['sha256']
comparisons={}
for key,r in scores.items():
 prior=base[key];rule=next(x for x in official['rows'] if str(x['prefill_tokens'])==key);gain=100*(r['total_token_throughput']/prior['total_token_throughput']-1);ttft=100*(r['mean_ttft_ms']/prior['mean_ttft_ms']-1)
 comparisons[key]=dict(main105_total_tok_s=prior['total_token_throughput'],main105_mean_TTFT_ms=prior['mean_ttft_ms'],throughput_delta_percent=gain,TTFT_delta_percent=ttft,throughput_within_1pct_band=abs(gain)<=1,latest_main_TTFT_within_plus1pct=r['mean_ttft_ms']<=prior['mean_ttft_ms']*1.01,official_TP_floor=rule['derived_99pct_throughput_floor'],official_TTFT_ceiling_ms=rule['derived_101pct_TTFT_ceiling_ms'],official_table_TP_gate=Decimal(str(r['total_token_throughput']))>=Decimal(rule['derived_99pct_throughput_floor']),official_table_TTFT_gate=Decimal(str(r['mean_ttft_ms']))<=Decimal(rule['derived_101pct_TTFT_ceiling_ms']))
mathpath=Path('/private/tmp/cjj-native107-main105-math105-independent-review-v2-20261028.json');mj=read(mathpath);mm=read(Path('/private/tmp/cjj-native107-main105-math105-v1-raw-20261028/out/manifest.json'))
assert mj['source_commit']==mm['source_commit']==commit and mj['controller_sha256']==m['controller_sha256'] and mm['source_hashes']==m['source_hashes'] and mm['native_decode_library']==m['native_decode_library']
math_summary={k:{field:v[field] for field in ('correct','incorrect','exact_accuracy','max_token_truncation_count','metric_gate_095_passed')} for k,v in mj['results'].items()}
coverage=mj['coverage'];buckets=coverage['per_evaluation_buckets']['supplemental_parallel64']
assert buckets['64']['unique_questions_observed']==105 and buckets['32']['unique_questions_observed']==32 and coverage['final_summary_records']==[]
report=dict(local_CPU_only=True,raw_files=48,raw_bytes=3954804,raw_manifest_sha256=sha(R/'download-manifest.json'),run_classification='Official CLI/protocol diagnostic; original owner official_score=false, not an organizer-issued score',all_original_SHA_and_sizes_verified=True,source_commit=commit,immutable_git_source_files_verified=252,worktree_clean_and_matching_tested_commit=True,symlinks_resolved_in_same_immutable_commit=symlinks,controller_sha256=m['controller_sha256'],helper_sha256=m['helper_sha256'],benchmark_function_sha256=m['throughput_protocol_function_sha256'],all_four_official_parameter_commands_and_token_duration_arithmetic_verified=True,fixed_service_CLI=service,flags=dict(encoded_K8=True,native_K8=True,q32_hoist=True,Graph_observer=False),warmup_plan=m['warmup_plan_metadata'],scores=scored,warmups=warmed,comparison_to_reviewed_main105=comparisons,libraries=dict(W8A8_SHA_run_record=m['native_sha256'],BF16_GQA_SHA_fresh_local_rehash=sha(libpaths['BF16_GQA']),Native_K8_SHA_fresh_local_rehash=sha(libpaths['Native_K8']),v6_snapshot_records_equal_to_main105=True),runtime=dict(C3_records=2,parent_installers=9,native_installations=1,native_events=168,scopes=scopes,physical_page_bytes=16384,KV_capacity_tokens=548320,BF16_parent_installed_but_no_BF16_native_selection_events=True,q32_actual_hoist_selected=True,request_frequency_proven=False,Graph_replays_counted=False),cache_observations=cache,owner=a['original_owner'],terminal=dict(completed=True,cleanup_complete=True,errors=[],API_returncode=0,updated_utc=state['updated_utc']),separate_existing_model_review=dict(path=str(mathpath),sha256=sha(mathpath),source_and_controller_and_all252_source_hashes_match=True,metrics=math_summary,positive_supplemental_questions_with_bucket64=105,positive_supplemental_questions_with_bucket32=32,coverage_lower_bound_only=True,final_observer_summary_present=False,all_tokens_native_is_not_required_or_claimed=True,full_model_raw_not_independently_rerun_in_this_throughput_review=True),Root_reported_execution_context=dict(no_large_archive_download_parallel_to_candidate_scoring=True,not_independently_CPU_IO_profiled=True,latest_remote_main_at2141='1cbac531ca9afaa0e9a68624d5d10cb1d0412055',PR107_draft_and_clean=True,CI_checks_skipped=13,CI_checks_passed_not_inferred=True),official_table_performance_gate_all_passed=all(x['official_table_TP_gate'] and x['official_table_TTFT_gate'] for x in comparisons.values()),controller_precision_and_full_acceptance_flags_preserved_false=True,this_report_does_not_grant_merge=True,limitations=['4K throughput+0.050164283% lies inside the1% band and is treated as unchanged; this is not a statistically estimated noise distribution.','16K shows a single-run+7.275904824% gain against same5c/0c66/v6 main105 control; variance and causal attribution have not been established.','4K TTFT is higher than latest main by11.145%; it passes the independently verified official fixed table ceiling but fails a separate latest-main+1% regression comparison. Both facts are retained; no hard gate was edited.','4K adds6 index metadata records and selected tuning rows change;16K reports no new/changed metadata. This observation includes client preflight,not isolated timed scoring.','Main1054K had a reported parallel readonly archive timeout; candidate scoring did not according to Root. Retrospective files do not quantify CPU/IO impact or explain observed differences.','Throughput runs have no Graph observer;168 SELECTED/CAPTURED entries are not replay or request counts. Separate Math105 positive request coverage is a lower bound and has no final observer summary.','Each geometry has one full warm and one scored sample;warmup is not a proof of complete cache stability.','W8A8/tuning database binaries are not in this archive; those identities remain run records. Native K8 and BF16 GQA original local SOs were freshly rehashed.','CLI concurrency64 and reported max_concurrent_requests68/65 are preserved; no per-request raw latency arrays were saved.','Latest remote main/PR/CI and no-parallel-archive statements are Root reports,not a new network audit.13 skipped CI checks are not13 passes.'],SSH_GPU_PR_or_Torch_Triton_used=False,script_sha256=sha(Path(__file__)))
assert not {'torch','triton'}.intersection(sys.modules)
path=Path('/private/tmp/cjj-native107-throughput-actual-review-20261028.json');path.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(report=str(path),sha256=sha(path),comparisons=comparisons,source_files=252,native_events=168),indent=2))
