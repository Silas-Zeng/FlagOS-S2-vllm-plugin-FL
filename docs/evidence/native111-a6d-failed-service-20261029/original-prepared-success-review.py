"""Prepared CPU-only PR111 service review. Requires completed genuine raw archive; no live access."""
import argparse,ast,collections,hashlib,importlib.util,json,math,posixpath,subprocess,sys
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--repo',type=Path,default=Path('/private/tmp/flagos-cjj-compact-service-n48-qualified-20261029'))
p.add_argument('--launch',type=Path,default=Path('/private/tmp/cjj-native111-q32-n48-throughput-v1-20261029-launch.json'))
p.add_argument('--base-so',type=Path,default=Path('/private/tmp/cjj-native110-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_v_overlap_probe.so'))
p.add_argument('--n48-so',type=Path,default=Path('/private/tmp/cjj-native112-n48-compile-v1-raw-20261029/out/build/cjj_native_k8_compact_n48_s4_probe.so'))
a_cli=p.parse_args();R=a_cli.raw;O=R/'out';S=a_cli.repo;assert not a_cli.out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
commit='a6d654a7444bba72cbbd48d3d9bfd33172cfd129';controller_pin='ae99bfb55c69dc1bf9464ffe0a1b7e627882bd4bc9d185d4177f9a902c29ea56'
launch=read(a_cli.launch);assert launch['commit']==commit and launch['controller_sha256']==controller_pin
m=read(O/'manifest.json');state=read(O/'state.json');a=read(R/'download-manifest.json');owner=read(R/'exports'/(launch['tag']+'-owner.json'))
assert a['raw_files']==len(a['files'])==len({x['relative'] for x in a['files']}) and a['raw_bytes']==sum(x['bytes'] for x in a['files'])
for row in a['files']:
 q=Path(row['relative']);assert not q.is_absolute() and '..' not in q.parts
 path=R/q;assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
assert a['original_owner']['same_start'] is True and a['original_owner']['state'] in ('Z','X') and a['model_rerun'] is False
assert (a['original_owner']['pid'],a['original_owner']['start_ticks'])==(62772,356555561)
assert (owner['pid'],owner['start_ticks'])==(62772,356555561) and owner['official_score'] is False
assert m['source_commit']==owner['source_commit']==commit
git=lambda *args:subprocess.check_output(['git',*args],cwd=S)
# Verify immutable objects, resolving symlinks INSIDE THE SAME tested commit.
tree={}
for line in git('ls-tree','-r',commit).decode().splitlines():
 meta,path=line.split('\t',1);mode,kind,oid=meta.split();tree[path]=(mode,kind,oid)
needed=set(m['source_hashes'])|{'tools/run_biv150_normalization_accuracy.py',launch['script']}
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
assert set(m['source_hashes'])=={p for p in tree if p.startswith(('vllm_fl/','csrc/'))}
for path,h in m['source_hashes'].items():assert digest(committed(path))==h==sha(S/path),path
controller=O/'executed-controller.py';assert sha(controller)==m['controller_sha256']==controller_pin and controller.read_bytes()==committed(launch['script'])
helper=S/'docs/evidence/compact-service-q32-combination-20261029/actual-owned-service-helper94db.py'
assert sha(helper)==m['helper_sha256']=='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
args=[x.replace('{SOURCE}',m['source']).replace('{OUT}',owner['out']) for x in launch['args']]
assert owner['command']==['/usr/local/bin/python',m['source']+'/'+launch['script'],*args]
assert all(x in args for x in ('--native-k8-library','--native-k8-n48-library','--q32-hoist','--throughput','--full-workload-warmup'))
assert args[args.index('--warmup-plan')+1]=='full-only' and not set(args)&{'--baseline','--math105','--supplemental-native-math105-64','--graph-coverage-bootstrap-dir'}
ctrl=load('reviewed111_controller',controller)
assert m['warmup_plan_metadata']==ctrl.warmup_plan_metadata(SimpleNamespace(warmup_plan='full-only',throughput=True,full_workload_warmup=True,graph_coverage_bootstrap_dir=None))
assert all(m[k] is True for k in ('private_enabled','native_decode_requested','native_n48_requested','fresh_q32_hoist_requested','official_service_parameters_unchanged'))
assert m['graph_coverage_bootstrap']==dict(enabled=False,formal_score_eligible=False,quality_covered=False)
assert m['expected_compile_semantics']!=m['off_compile_semantics'] and 'konly_compact' not in m['off_compile_semantics']
expected=m['expected_compile_semantics'];assert expected['konly_compact']['q32_hoist_enabled'] is True
config=expected['konly_compact']['native_decode'];assert config['enabled'] is True
q=load('reviewed111_qualification',S/'vllm_fl/compact_native_qualification.py')
assert config['qualification']==q.pin() and config['n48']['qualification']==q.pin('n48')
assert sha(a_cli.base_so)==config['sha256']==m['native_decode_library']['sha256']==ctrl.K8_SHA
assert sha(a_cli.n48_so)==config['n48']['sha256']==m['native_n48_library_sha256']==ctrl.N48_SHA
assert config['path']==m['native_decode_library']['path'] and config['n48']['path']==m['native_n48_library']
assert m['actual_resolution']['origin']==m['source']+'/vllm_fl/__init__.py' and not m['actual_resolution']['package_executed'] and m['abi']==dict(torch='2.10.0',cxx11_abi=False)
service=['/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072'];assert m['command']==service
text=committed('tools/run_biv150_normalization_accuracy.py').decode();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('benchmark','throughput')]
protocol='\n'.join(''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno]) for n in nodes)
assert protocol==m['throughput_protocol_source'] and digest(protocol.encode())==m['throughput_protocol_function_sha256']=='e236338934098fda2ea0a795b2ade845015123d8ea052806ca0062cc3035a66d'
assert all(state[k] is True for k in ('completed','smoke_passed','throughput_measured','cleanup_complete')) and state['phase']=='completed' and not state['errors']
assert all(state[k] is False for k in ('model_precision_verified','native_quality_covered','native_model_precision_verified')) and m['full_acceptance'] is False
assert not m['supplemental_Math105_parallel64_requested'] and not (O/'accuracy-verdict.json').exists()
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
 scored[key]={k:collected[1][k] for k in ('date','duration','completed','failed','total_input_tokens','total_output_tokens','total_token_throughput','output_throughput','mean_ttft_ms','mean_tpot_ms','median_itl_ms','mean_itl_ms','median_tpot_ms','max_concurrency','max_concurrent_requests')}
 warmed[key]={k:collected[0][k] for k in ('date','duration','completed','failed','total_token_throughput','mean_ttft_ms')};warmed[key]['scored']=False
 obs=read(O/f'score-{key}-cache-observation.json');before,after=obs['before'],obs['after'];assert before==read(O/f'score-{key}-cache-before.json')
 assert obs['new_metadata']==sorted(after['triton_json_sha256'].keys()-before['triton_json_sha256'].keys())
 assert obs['changed_metadata']==sorted(k for k in after['triton_json_sha256'].keys()&before['triton_json_sha256'].keys() if after['triton_json_sha256'][k]!=before['triton_json_sha256'][k])
 assert obs['selected_tuning_rows_changed']==(before['selected_tuning_rows_sha256']!=after['selected_tuning_rows_sha256']);cache[key]={k:obs[k] for k in ('new_metadata','changed_metadata','selected_tuning_rows_changed','scope')}
log=(O/'server.log').read_text();runtime=ctrl.evidence(log,expected,native_gqa_sha=m['native_gqa_library_sha256'],native_library=m['native_decode_library']['path'],native_identity=m['native_identity'])
assert json.loads(json.dumps(runtime))==m['actual_runtime_after_throughput']
assert m['native_identity']['module_sha256']==digest(committed('vllm_fl/patches/biv150_konly_compact_native.py'))
native=runtime['native_decode'];assert native['enabled'] and native['actual_imports_verified'] and native['host_or_capture_only'] and not native['actual_request_frequency_proven'] and not native['Graph_replay_observed']
assert native['installations'] and {e['event']['shape'] for e in native['events']}=={32,48,64}
scopes=collections.defaultdict(list)
for e in native['events']:scopes[(e['marker'],e['event']['shape'],e['event']['segments'])].append(e['event'])
scope_summary=[dict(marker=k[0],rows=k[1],segments=k[2],host_events=len(v),entry_generations=sorted({x['entry_generation'] for x in v}),query_pitches=sorted({x['query']['strides'][0] for x in v}),cache_pages=sorted({x['cache_pages'] for x in v})) for k,v in sorted(scopes.items())]
assert 'BIV150_NATIVE_GRAPH_COVERAGE ' not in log
hoisted_events=[e for e in runtime['selected'] if e['profile']=='q32' and e['I8_fragment_hoist_verified'] is True];assert hoisted_events
allocation=runtime['compact_allocation_and_fallback'];assert allocation['allocation_metadata_only'] and not allocation['Graph_replays_counted']
for alloc in allocation['allocations']:
 assert alloc['recorded_layers']==42 and alloc['raw_storage_bytes']==42*alloc['configured_num_blocks']*12544
 assert len(alloc['records'])==42 and all(x['carrier_strides']==[6272,2176,256,128,1] for x in alloc['records'])
# Saved smoke and burst responses are not benchmark scores or Math105.
smoke=read(O/'smoke-requests.json');assert len(smoke)==3
requests=list(smoke);bursts=[];lograw=(O/'server.log').read_bytes();library={k:v for k,v in config.items() if k!='enabled'}
for record in m['native_bursts']:
 n=record['concurrency_requested'];r=read(O/f'native-burst{n}-requests.json');window=record['runtime_window']
 assert n in (32,64,40,48) and r['completed']==len(r['records'])==n and {x['index'] for x in r['records']}==set(range(n))
 assert r['diagnostic_only'] and not r['quality_covered'] and not window['Graph_replays_counted']
 raw=(O/f'native-burst{n}-runtime-window.log').read_bytes();assert digest(raw)==window['sha256'] and len(raw)==window['bytes'] and lograw[window['log_byte_offset']:window['log_byte_offset']+len(raw)]==raw
 for x in r['records']:assert x['request']['ignore_eos'] is True and x['response']['usage']['completion_tokens']==16
 requests.extend(r['records']);bursts.append(dict(requests=n,scored=False,window_sha256=digest(raw),host_event_count=len(window['native_events'])))
assert len(bursts)==4 and len(requests)==187 and len({x['response']['id'] for x in requests})==187
for x in requests:
 request,response=x['request'],x['response'];usage=response['usage']
 assert request['model']==response['model']=='minicpm' and request['max_tokens']==16 and len(response['choices'])==1
 assert 0<usage['completion_tokens']<=16 and usage['prompt_tokens']>0 and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'] and 'error' not in response
# Fresh preserved Q32 objects: replay existing proposal and native grammar, not a new compiler.
proposal=load('reviewed111_proposal',S/'vllm_fl/patches/biv150_konly_compact_hoist_proposal.py')
ng=load('reviewed111_native_grammar',S/'vllm_fl/patches/biv150_konly_encoded_hoist_native.py')
prepared=[json.loads(line.split('BIV150_KONLY_Q32_HOIST_PREPARED ',1)[1]) for line in log.splitlines() if 'BIV150_KONLY_Q32_HOIST_PREPARED ' in line];assert prepared
fresh=[]
for event in prepared:
 remote=Path(event['artifact_directory']);relative=remote.relative_to(Path(owner['out']));folder=O/relative
 binding=read(folder/'binding.json');assert binding['artifact_directory']==str(remote) and binding['implementation_sha256']==expected['konly_compact']['implementation_sha256']
 assert all(binding[k] is False for k in ('numerical_verified','Graph_verified','precision_gate','performance_verified','service_verified'))
 for arm in ('unhoisted','hoisted'):
  snap=binding[arm];assert read(folder/arm/'record.json')==snap and snap['launch_geometry']==[16,1,64]
  for ext,h in snap['artifacts'].items():assert sha(folder/arm/('actual.'+ext))==h
  assert sha(folder/arm/'actual.jit_python_source.py')==snap['source_digest']
  resource=binding['resources'][arm];assert type(resource['n_regs']) is int and resource['n_regs']>0 and resource['n_spills']==0 and type(resource['shared']) is int and 0<=resource['shared']<=65536
  assert [resource['num_warps'],resource['num_stages'],resource['warp_size']]==[16,1,64]
 assert all(binding['unhoisted'][k]==binding['hoisted'][k] for k in ('source_digest','function_name','ABI','launch_geometry'))
 assert (folder/'unhoisted/actual.ttir').read_bytes()==(folder/'hoisted/actual.ttir').read_bytes()
 override,proof=proposal.propose((folder/'unhoisted/actual.ttgir').read_text(),binding['actual_metadata'])
 assert override==(folder/'override.ttgir').read_text() and digest(override.encode())==binding['override_sha256']==event['override_sha256']
 assert json.loads(json.dumps(proof))==binding['proposal']
 nativeproof=ng.audit(override,*[(folder/'hoisted'/('actual.'+ext)).read_text() for ext in ('ttgir','llir','asm')]);assert json.loads(json.dumps(nativeproof))==binding['native']
 assert event['loaded_resources']==binding['resources'] and event['hoisted_hash']==binding['hoisted']['hash'] and event['unhoisted_hash']==binding['unhoisted']['hash']
 fresh.append(dict(artifact_directory=str(relative),binding_sha256=sha(folder/'binding.json'),hoisted_hash=event['hoisted_hash'],unhoisted_hash=event['unhoisted_hash'],resources=binding['resources'],proposal_and_native_IR_replayed=True,numerical_or_service_qualification=False))
assert {x['hash'] for x in hoisted_events}<={x['hoisted_hash'] for x in fresh}
# Official fixed table is an archived source, distinct from latest-main regression comparisons.
table_path=S/'docs/evidence/official-iluvatar-baseline-20261028/source-review.json';table=read(table_path);thresholds={}
for row in table['rows']:
 k=str(row['prefill_tokens']);floor=Decimal(row['total_tok_s'])*Decimal('.99');ceiling=Decimal(row['mean_TTFT_ms'])*Decimal('1.01')
 assert str(floor)==row['derived_99pct_throughput_floor'] and str(ceiling)==row['derived_101pct_TTFT_ceiling_ms']
 thresholds[k]=dict(official_TP_floor=str(floor),official_TTFT_ceiling_ms=str(ceiling),TP_pass=Decimal(str(scores[k]['total_token_throughput']))>=floor,TTFT_pass=Decimal(str(scores[k]['mean_ttft_ms']))<=ceiling)
report=dict(schema='cjj-native111-service-independent-CPU-v1',audit_passed=True,source_commit=commit,controller_sha256=controller_pin,helper_sha256=m['helper_sha256'],benchmark_function_sha256=m['throughput_protocol_function_sha256'],source_hashes_verified=len(m['source_hashes']),source_symlinks=symlinks,
 raw_files=a['raw_files'],raw_bytes=a['raw_bytes'],download_manifest_sha256=sha(R/'download-manifest.json'),owner=a['original_owner'],launch_sha256=sha(a_cli.launch),service_command=service,
 variant=dict(compact=True,page_bytes=12544,base_SO=sha(a_cli.base_so),N48_SO=sha(a_cli.n48_so),Q32_requested=True,Math105_run=False),warmup_plan=m['warmup_plan_metadata'],warmups=warmed,scores=scored,cache_observations=cache,benchmark_scored_requests=384,benchmark_warm_requests=384,
 fixed_official_table_performance=thresholds,official_table_review_sha256=sha(table_path),diagnostic_requests=dict(count=187,bursts=bursts,model_precision_verified=False),runtime_host_wiring=dict(C3=m['expected_compile_semantics'],native_scopes=scope_summary,allocation=allocation,actual_request_frequency_proven=False,Graph_replay_frequency_proven=False),fresh_Q32_artifacts=fresh,
 final_stats_present=m['compact_final_export_found'],final_stats_files=m['compact_final_exports'],terminal=dict(completed=True,cleanup_complete=True,errors=state['errors'],same_start_closed=True,API_returncode=shutdown['returncode']),
 model_precision_verified=False,precision_gate=False,full_acceptance=False,latest_main_gain_verified=False,local_CPU_only=True,review_script_sha256=sha(Path(__file__)),
 limitations=['Single full warm and single score per geometry; no repeat distribution or causality.', 'Math105 was not run and no prior model score transfers to this source.', 'Native SELECTED/CAPTURED and Q32 events are host/capture observations, not actual request or Graph replay frequencies.', 'Fresh Q32 preserves actual artifacts and replays original ABI/coherence/proposal/native gates; it adds no numerical or model qualification.', 'runtime n_spills and dynamic shared are reported separately; no assertion that every ELF scalar spill count is zero.', 'Only aggregate benchmark token/latency records are available unless saved raw explicitly includes per-request arrays.', 'Official table performance bounds are separate from latest-main comparison; full acceptance remains false.', 'No remote/GPU remeasurement or runtime changes were performed by this review.'])
assert not {'torch','triton','vllm','vllm_fl','numpy'}&set(sys.modules)
a_cli.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(report=str(a_cli.out),sha256=sha(a_cli.out),raw_files=a['raw_files'],precision_gate=False)))
