"""Bounded CPU-only replay of PR109 original formal-protocol throughput records."""
import argparse,ast,base64,collections,datetime,hashlib,importlib.util,json,math,posixpath,re,subprocess,sys
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,default=Path('/private/tmp/flagos-cjj-native109-throughput-evidence-20261029'))
parser.add_argument('--raw',type=Path,default=Path('/private/tmp/cjj-native109-throughput-v1-raw-20261029'))
parser.add_argument('--reference-raw',type=Path,default=Path('/private/tmp/cjj-native107-main105-throughput-v1-raw-20261028'))
parser.add_argument('--math-raw',type=Path,default=Path('/private/tmp/cjj-native109-math105-v1-raw-20261029'))
parser.add_argument('--launch',type=Path,default=Path('/private/tmp/cjj-root-native109-throughput-v1-20261029-launch.json'))
parser.add_argument('--dispatcher',type=Path,default=Path('/private/tmp/cjj-vllm-cudagraph-source-live-20261028/vllm/v1/cudagraph_dispatcher.py'))
parser.add_argument('--gqa-so',type=Path,default=Path('/private/tmp/cjj-main105-native-gqa-build-v1-raw-20261028/out/corex_native_gqa_s2_probe.so'))
parser.add_argument('--compact-so',type=Path,default=Path('/private/tmp/cjj-native-k8-compact-compile-v1-raw-20261028/out/build/cjj_native_k8_compact_probe.so'))
parser.add_argument('--out',type=Path,default=Path('/private/tmp/cjj-native109-throughput-independent-review-v1-20261029.json'))
args_cli=parser.parse_args();R=args_cli.raw;O=R/'out';S=args_cli.repo

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();digest=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(p.read_text())
a=read(R/'download-manifest.json');m=read(O/'manifest.json');state=read(O/'state.json');owner=read(next((R/'exports').glob('*owner.json')))
assert (a['raw_files'],a['raw_bytes'])==(36,3077775) and len(a['files'])==36 and sum(x['bytes'] for x in a['files'])==3077775
assert len({x['relative'] for x in a['files']})==36
for row in a['files']:
 p=R/row['relative'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
assert a['original_owner']==dict(pid=53242,start_ticks=355306781,same_start=True,state='Z') and not a['model_rerun']
assert (owner['pid'],owner['start_ticks'])==(53242,355306781) and owner['official_score'] is False
commit='fc25c86a13fc273bb7368b83af4ca264524b4e12'
git=lambda *args:subprocess.check_output(['git',*args],cwd=S)
assert m['source_commit']==owner['source_commit']==commit
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
assert len(m['source_hashes'])==268
for path,h in m['source_hashes'].items():assert digest(committed(path))==h==sha(S/path),path
controller=O/'executed-controller.py';assert sha(controller)==m['controller_sha256']=='d8a5bd435845bdb3dd1e4d3741f4ca38fe85b4cc152036517cbcf034f0eb4495'
assert sha(S/'docs/evidence/native109-math105-v1-20261029/dependencies/cjj-actual-owned-service-helper-94db-20261028.py')==m['helper_sha256']=='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976'
launch=read(args_cli.launch);assert base64.b64decode(launch['controller_base64'])==controller.read_bytes()
args=[x.replace('{SOURCE}',m['source']).replace('{OUT}',owner['out']) for x in launch['args']];assert owner['command']==['/usr/local/bin/python',launch['script'],*args]
assert all(x in args for x in ('--native-k8-library','--native-k8-sha256','--throughput','--full-workload-warmup')) and args[-2:]==['--warmup-plan','full-only']
assert '--baseline' not in args and '--graph-coverage-bootstrap-dir' not in args and '--q32-hoist' not in args
spec=importlib.util.spec_from_file_location('reviewed_controller',controller);ctrl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctrl)
assert m['warmup_plan_metadata']==ctrl.warmup_plan_metadata(SimpleNamespace(warmup_plan='full-only',throughput=True,full_workload_warmup=True,graph_coverage_bootstrap_dir=None))
assert all(m[k] is True for k in ('private_enabled','native_decode_requested','official_service_parameters_unchanged'))
assert m['graph_coverage_bootstrap']==dict(enabled=False,formal_score_eligible=False,quality_covered=False)
assert m['expected_compile_semantics']!=m['off_compile_semantics'] and 'konly_compact' not in m['off_compile_semantics']
assert m['expected_compile_semantics']['konly_compact']['q32_hoist_enabled'] is False and m['fresh_q32_hoist_requested'] is False
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
 scored[key]={k:collected[1][k] for k in ('date','duration','completed','failed','total_input_tokens','total_output_tokens','total_token_throughput','output_throughput','mean_ttft_ms','mean_tpot_ms','median_itl_ms','mean_itl_ms','median_tpot_ms','max_concurrency','max_concurrent_requests')}
 warmed[key]={k:collected[0][k] for k in ('date','duration','completed','failed','total_token_throughput','mean_ttft_ms')};warmed[key]['scored']=False
 obs=read(O/f'score-{key}-cache-observation.json');before,after=obs['before'],obs['after'];assert before==read(O/f'score-{key}-cache-before.json')
 assert obs['new_metadata']==sorted(after['triton_json_sha256'].keys()-before['triton_json_sha256'].keys())
 assert obs['changed_metadata']==sorted(k for k in after['triton_json_sha256'].keys()&before['triton_json_sha256'].keys() if after['triton_json_sha256'][k]!=before['triton_json_sha256'][k])
 assert obs['selected_tuning_rows_changed']==(before['selected_tuning_rows_sha256']!=after['selected_tuning_rows_sha256']);cache[key]={k:obs[k] for k in ('new_metadata','changed_metadata','selected_tuning_rows_changed','scope')}
log=(O/'server.log').read_text();runtime=ctrl.evidence(log,m['expected_compile_semantics'],native_gqa_sha=m['native_gqa_library_sha256'],native_library=m['native_decode_library']['path'],native_identity=m['native_identity'])
assert json.loads(json.dumps(runtime))==m['actual_runtime_after_throughput'] and len(runtime['actual_compile_semantics'])==2 and len(runtime['preserved_parent_installations'])==9
native=runtime['native_decode'];assert native['enabled'] and native['actual_imports_verified'] and native['host_or_capture_only'] and not native['actual_request_frequency_proven'] and not native['Graph_replay_observed']
assert len(native['installations'])==1 and len(native['events'])==168
assert m['native_identity']['module_sha256']==sha(S/'vllm_fl/patches/biv150_konly_compact_native.py')
scopes=[]
for marker in ('BIV150_KONLY_NATIVE_DECODE_SELECTED','BIV150_KONLY_NATIVE_DECODE_CAPTURED'):
 for n,s in ((64,2),(32,4)):
  es=[x['event'] for x in native['events'] if x['marker']==marker and x['event']['shape']==n and x['event']['segments']==s]
  assert len(es)==42 and {x['entry_generation'] for x in es}==set(range(1,43))
  assert all({k:x['library'][k] for k in ('path','sha256')}==m['native_decode_library'] and x['native_launch_returned'] and x['physical_page_bytes']==12544 and x['cache_pages']==44769 and x['query']['strides']==[2048,128,1] for x in es)
  scopes.append(dict(marker=marker,rows=n,segments=s,distinct_entry_generations=42,query_pitch=2048))
gqa=runtime['native_gqa'];assert len(gqa['installs'])==1 and gqa['installs'][0]['library_sha256']==m['native_gqa_library_sha256'] and gqa['installs'][0]['overlap_v'] and gqa['events']==[]
assert any(x['profile']=='q32' for x in runtime['selected']) and all(x['I8_fragment_hoist_verified'] is False for x in runtime['selected'])
assert 'BIV150_NATIVE_GRAPH_COVERAGE ' not in log and m['compact_final_export_found'] is False and m['compact_final_exports']==[]
assert 'GPU KV cache size: 716,304 tokens' in log
libpaths=dict(BF16_GQA=args_cli.gqa_so,Native_compact=args_cli.compact_so)
assert sha(libpaths['BF16_GQA'])==m['native_gqa_library_sha256']=='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8'
assert sha(libpaths['Native_compact'])==m['native_decode_library']['sha256']==ctrl.K8_SHA=='40f9b17c634d2344ca54b9f1202e4e10932bf3f3f632cd41d793ddafd0934453'
allocation=runtime['compact_allocation_and_fallback'];assert len(allocation['allocations'])==1
alloc=allocation['allocations'][0]
assert alloc['initialized'] and alloc['recorded_layers']==42 and alloc['configured_num_blocks']==44769 and alloc['page_bytes']==12544
assert alloc['raw_storage_bytes']==42*44769*12544 and len(alloc['records'])==42
for record in alloc['records']:
 assert record['physical_pages']==44769 and record['storage_bytes']==44769*12544 and record['carrier_strides']==[6272,2176,256,128,1]
assert allocation['physical_tokens_per_layer']==[716304] and allocation['allocation_metadata_only'] and not allocation['Graph_replays_counted']
# Every saved non-benchmark request is separately checked; these are NOT scored.
requests=list(read(O/'smoke-requests.json'));assert len(requests)==3
bursts=[];lograw=(O/'server.log').read_bytes()
for record in m['native_bursts']:
 n=record['concurrency_requested'];result=read(O/f'native-burst{n}-requests.json');window=record['runtime_window']
 assert n in (32,64,40,48) and result['completed']==len(result['records'])==n and result['concurrency_requested']==n
 assert {x['index'] for x in result['records']}==set(range(n)) and result['diagnostic_only'] and not result['quality_covered']
 assert not window['native_events'] and not window['Graph_replays_counted']
 raw=(O/f'native-burst{n}-runtime-window.log').read_bytes()
 assert digest(raw)==window['sha256'] and len(raw)==window['bytes'] and lograw[window['log_byte_offset']:window['log_byte_offset']+len(raw)]==raw
 assert all(x['request']['ignore_eos'] is True for x in result['records'])
 requests.extend(result['records']);bursts.append(dict(requests=n,scored=False,window_SHA=digest(raw),no_native_events_in_saved_window=True))
assert len(requests)==187 and len({x['response']['id'] for x in requests})==187
for row in requests:
 request,response=row['request'],row['response'];usage=response['usage']
 assert request['model']==response['model']=='minicpm' and request['max_tokens']==16 and len(response['choices'])==1
 assert usage['completion_tokens']==16 and usage['prompt_tokens']>0 and usage['total_tokens']==usage['prompt_tokens']+16
 assert 'error' not in response and response['choices'][0]['finish_reason']=='length'
# Same official protocol, different implementations. Rehash prior raw; do not
# re-run its already preserved source/model review or manufacture a new control.
baseR=args_cli.reference_raw;bm=read(baseR/'out/manifest.json');base=read(baseR/'out/scores.json')
base_manifest=read(baseR/'download-manifest.json')
assert (base_manifest['raw_files'],base_manifest['raw_bytes'])==(48,3954804)
for row in base_manifest['files']:
 p=baseR/row['relative'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
baseline_review=S/'docs/evidence/native107-throughput-v1-20261028/review.json'
assert sha(baseline_review)=='57bb07087627709f2ccc27b73d4def3d78793c5a562b5fb0d7a89aa96bd0c218'
assert bm['source_commit']=='b5d1a241f45efd21517d04e25a7dc5d44ac1ca06'
assert bm['native_sha256']==m['native_sha256']=='5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
assert bm['native_gqa_library_sha256']==m['native_gqa_library_sha256'] and bm['tuning_snapshot']==m['tuning_snapshot']
assert bm['warmup_plan_metadata']==m['warmup_plan_metadata'] and bm['command']==m['command']
assert bm['throughput_protocol_source']==m['throughput_protocol_source'] and bm['throughput_protocol_function_sha256']==m['throughput_protocol_function_sha256']
assert bm['fresh_q32_hoist_requested'] is True and bm['native_decode_library']['sha256']=='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
comparisons={}
for key,result in scores.items():
 prior=base[key];count={4096:256,16384:128}[int(key)]
 assert prior['completed']==count and prior['failed']==0 and prior['total_input_tokens']==int(key)*count and prior['total_output_tokens']==1024*count
 assert math.isclose(prior['total_token_throughput'],(int(key)+1024)*count/prior['duration'],rel_tol=1e-12)
 oldcmd=read(baseR/f'out/score-{key}/command.json');newcmd=read(O/f'score-{key}/command.json')
 assert len(oldcmd)==len(newcmd) and all(x==y or (i>0 and newcmd[i-1]=='--result-dir') for i,(x,y) in enumerate(zip(oldcmd,newcmd)))
 comparisons[key]={metric:dict(PR107=prior[metric],PR109=result[metric],delta_percent=100*(result[metric]/prior[metric]-1)) for metric in ('total_token_throughput','mean_ttft_ms','mean_tpot_ms','median_tpot_ms','mean_itl_ms','median_itl_ms','duration')}
# Related archived SDK code, NOT newly collected as part of this TP run.
# Extract only two source methods and the tested plugin's selector, no SDK import.
dispatch_text=args_cli.dispatcher.read_text();dispatch_tree=ast.parse(dispatch_text)
methods=[n for n in ast.walk(dispatch_tree) if isinstance(n,ast.FunctionDef) and n.name in ('_compute_bs_to_padded_graph_size','_create_padded_batch_descriptor')]
assert len(methods)==2
namespace=dict(CUDAGraphMode=SimpleNamespace(NONE='NONE',FULL='FULL'),BatchDescriptor=lambda **kw:kw)
exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'<two-archived-SDK-methods>','exec'),namespace)
gqa_source=committed('vllm_fl/patches/biv150_gqa_decode.py').decode();gqa_tree=ast.parse(gqa_source)
selector=next(n for n in gqa_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pick_segments')
constants=[n for n in gqa_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TARGET_PROGRAMS' for t in n.targets)]
assert len(constants)==1
# TARGET_PROGRAMS uses the existing environment default256; record the source
# expression and run this explicitly conditional default, never change a service.
assert '256' in ast.unparse(constants[0]);sel={'TARGET_PROGRAMS':256}
exec(compile(ast.Module(body=[selector],type_ignores=[]),'<tested-plugin-selector>','exec'),sel)
log_pattern=re.compile(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([0-9.]+) tokens/s, Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%')
sampling={}
for tag,root,result_set in [('109',R,scores),('107',baseR,base)]:
 lines=(root/'out/server.log').read_text().splitlines()
 configline=next(line for line in lines if 'Initializing a V1 LLM engine' in line)
 captures=ast.literal_eval(re.search(r"'cudagraph_capture_sizes': (\[[^\]]+\])",configline)[1]);assert captures==sorted(captures) and all(n in captures for n in (32,40,48,56,64))
 mock=SimpleNamespace(compilation_config=SimpleNamespace(max_cudagraph_capture_size=max(captures),cudagraph_capture_sizes=captures,compile_sizes=[]),cudagraph_mode=SimpleNamespace(has_mode=lambda mode:mode=='FULL'),vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=64)),uniform_decode_query_len=1)
 namespace['_compute_bs_to_padded_graph_size'](mock)
 mapping={}
 for live in (31,32,40,41,42,43,48,56,64):
  desc=namespace['_create_padded_batch_descriptor'](mock,live,True,False)
  n=desc['num_tokens'];segments=sel['_pick_segments'](n,2,16)
  mapping[str(live)]=dict(conditional_padded_rows=n,segments=segments,existing_native_supported=(n,segments) in ((32,4),(64,2)))
 sampling[tag]={}
 for key,result in result_set.items():
  end=datetime.datetime.strptime(result['date'],'%Y%m%d-%H%M%S');start=end-datetime.timedelta(seconds=result['duration']);rows=[]
  for number,line in enumerate(lines,1):
   match=log_pattern.search(line)
   if not match:continue
   stamp=datetime.datetime.strptime('2026-'+match[1],'%Y-%m-%d %H:%M:%S')
   if start<=stamp<=end:
    rows.append(dict(line=number,timestamp=stamp.isoformat(),prompt_tok_s=float(match[2]),generation_tok_s=float(match[3]),running=int(match[4]),waiting=int(match[5]),KV_usage_percent=float(match[6]),raw_line=line))
  zero=[x for x in rows if x['prompt_tok_s']==0 and x['generation_tok_s']>0 and x['running']>0]
  sampling[tag][key]=dict(approximate_start=start.isoformat(),result_second_resolution_end=end.isoformat(),samples=rows,
    sample_count=len(rows),running_histogram=dict(collections.Counter(x['running'] for x in rows)),zero_displayed_prompt_positive_generation_lines=[x['line'] for x in zero],zero_displayed_prompt_sample_count=len(zero),zero_displayed_prompt_running_histogram=dict(collections.Counter(x['running'] for x in zero)),conditional_padding=mapping,capture_sizes=captures)
assert sampling['109']['16384']['zero_displayed_prompt_running_histogram']=={42:9,41:5,3:1,2:1}
assert sampling['107']['16384']['zero_displayed_prompt_running_histogram']=={32:7,31:2,25:1,1:1}
# Reuse committed independent Math evidence and its complete archive: no regrade.
mathpath=S/'docs/evidence/native109-math105-v1-20261029/independent-review.json';mj=read(mathpath)
assert sha(mathpath)=='daa1948b4df1eecbbfda6f73e6d90ede3c03023a631b4a41ea260f0691dfe9cc'
mathm=read(args_cli.math_raw/'out/manifest.json');assert mj['source_commit']==mathm['source_commit']==commit
assert mathm['source_hashes']==m['source_hashes'] and mathm['controller_sha256']==m['controller_sha256'] and mathm['native_decode_library']==m['native_decode_library']
assert mathm['expected_compile_semantics']==m['expected_compile_semantics']
math_manifest=read(args_cli.math_raw/'download-manifest.json')
assert (math_manifest['raw_files'],math_manifest['raw_bytes'])==(41,11198363)
for row in math_manifest['files']:
 p=args_cli.math_raw/row['relative'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
mathsummary={k:{field:v[field] for field in ('correct','incorrect','exact_accuracy','max_token_truncation_count','metric_gate_095_passed')} for k,v in mj['results'].items()}
assert mathsummary['official_parallel8']['correct']==103 and mathsummary['supplemental_parallel64']['correct']==102
assert not mj['native_request_quality_gate'] and mj['official_Math105_precision_gate'] and not mj['full_acceptance']
limits=[
 'One full warm and one scored round per geometry; no repeat distribution or causal isolation. Both current throughput deltas are negative; they are not discarded as noise.',
 'PR109 compact12544/40f9/Q32-hoist-OFF vs PR107 encoded16384/8b86/Q32-hoist-ON changes multiple implementation properties, native binary, capacity and shape mix. The result does not isolate any one cause.',
 '4K adds six index metadata entries and selected tuning rows change.16K reports none. The observation includes client preflight; it neither measures isolated scoring compile cost nor proves complete cache stability.',
 'Logger windows use result.date minus measured duration to result.date. End time has one-second resolution and any result-serialization delay is unmeasured; these are approximate score windows, not per-request traces.',
 'The logger shows snapshots/interval averages every~10s. Displayed0.0 prompt throughput is rounded. Running is live request count, not scheduled token count, padded tensor rows or native calls.',
 'Conditional padding uses an archived2026-10-28 SDK source snapshot, source SHA recorded separately, not a fresh SDK snapshot inside this109 TP collection. It assumes uniform FULL decode, query_len1, scheduler cap>=64, matching dispatch and scheduled counts. Real path selection per request is not proved.',
 'The16K TPOT/ITL changes and41/42 Running samples are consistent with decode and the unsupported padded48/S4 path being worth testing. They do not prove actual Native frequency, a latency decomposition, or causality.',
 '168 SELECTED/CAPTURED events are host/capture observations, not Graph replays or request counts. Final compact stats are absent; lifetime counters/overflow/errors remain unknown.',
 'Benchmark results contain aggregate token totals and latency summaries; per-request benchmark lengths/latency arrays were not saved. Separate187 smoke/burst replies are complete but not scored or model-quality evidence.',
 'W8A8 and tuning database binary identities are run records. Local compact40f9 and BF16GQA0c66 SOs were freshly rehashed without loading. No new remote FS/GPU-idle observation is claimed.',
 'The separate Math105 review is reused by exact SHA with same268 source hashes/controller/C3/library. Its original1/0 truncations and absent Native request-coverage qualification remain unchanged; no answer regrading was performed.',
 'Original controller official_score=false denotes a team run using official protocol, not an organizer-issued score. Full acceptance and native coverage flags remain false. No source/ABI/performance qualification is transferred to111 or110.',
]
report=dict(schema='cjj-native109-throughput-independent-CPU-v1',audit_passed=True,local_CPU_only=True,source_commit=commit,source_hashes_verified=268,source_symlinks=symlinks,
 raw_files=36,raw_bytes=3077775,download_manifest_sha256=sha(R/'download-manifest.json'),controller_sha256=m['controller_sha256'],helper_sha256=m['helper_sha256'],benchmark_function_sha256=m['throughput_protocol_function_sha256'],
 original_owner=a['original_owner'],original_owner_official_score=owner['official_score'],service_command=service,actual_source_resolution=m['actual_resolution'],ABI=m['abi'],
 variant=dict(compact=True,page_bytes=12544,native_compact=True,encoded16KiB=False,native16KiB=False,q32_hoist=False,graph_observer=False),
 libraries=dict(W8A8_run_record=m['native_sha256'],BF16_GQA_local_SHA=sha(libpaths['BF16_GQA']),compact_native_local_SHA=sha(libpaths['Native_compact']),tuning_snapshot=m['tuning_snapshot']),
 warmup_plan=m['warmup_plan_metadata'],four_complete_commands_and_arithmetic_verified=True,scores=scored,warmups=warmed,comparison_to_PR107=comparisons,
 scored_requests=384,warm_requests=384,total_benchmark_input_tokens=sum(v['total_input_tokens'] for v in scores.values())*2,total_benchmark_output_tokens=sum(v['total_output_tokens'] for v in scores.values())*2,
 diagnostic_requests=dict(count=187,unique_API_IDs=187,all_output_tokens16=True,all_usage_adds_up=True,smoke_count=3,bursts=bursts,scored=False),
 cache_observations=cache,runtime=dict(C3_records=2,C3=m['expected_compile_semantics'],preserved_parent_installers=9,native_installations=1,native_events=168,scopes=scopes,fallback_events=len(allocation['fallback_events']),fallback_shapes=allocation['observed_fallback_shapes'],q32_selected_but_unhoisted=True,allocation=dict(page_bytes=12544,blocks=44769,recorded_layers=42,recorded_bytes=alloc['raw_storage_bytes'],physical_tokens_per_layer=716304,metadata_only=True),actual_native_frequency_proven=False,Graph_replays_observed=False,final_stats_present=False),
 running_samples=sampling,conditional_source_reasoning=dict(dispatcher_sha256=sha(args_cli.dispatcher),dispatcher_is_related_archived_snapshot=True,SDK_functions={n.name:dict(line=n.lineno,source=ast.get_source_segment(dispatch_text,n)) for n in methods},tested_gqa_source_sha256=digest(gqa_source.encode()),selector_source=ast.get_source_segment(gqa_source,selector),target_programs_default=256,comparison_is_conditional_not_frequency=True),
 separate_math_dependency=dict(path='docs/evidence/native109-math105-v1-20261029/independent-review.json',sha256=sha(mathpath),all268_sources_controller_C3_library_equal=True,results=mathsummary,official_Math105_precision_gate=True,native_request_quality_gate=False,raw_files_rehashed=41,full_regrade_rerun=False),
 baseline_dependency=dict(path='docs/evidence/native107-throughput-v1-20261028/review.json',sha256=sha(baseline_review),source_commit=bm['source_commit'],raw48_rehashed=True,original_math_review_not_rerun=True),
 terminal=dict(completed=state['completed'],cleanup_complete=state['cleanup_complete'],errors=state['errors'],phase=state['phase'],updated_utc=state['updated_utc'],API_returncode=shutdown['returncode'],same_start_owner_closed=True),
 recommendation='Record the negative throughput comparison and close PR109 without merging; preserve it as evidence for separately qualified iteration.',full_acceptance=False,controller_native_quality_covered=False,limitations=limits,
 dependencies=dict(launch_sha256=sha(args_cli.launch),SDK_snapshot_sha256=sha(args_cli.dispatcher),compact_SO_sha256=sha(args_cli.compact_so),GQA_SO_sha256=sha(args_cli.gqa_so)),SSH_GPU_retest_or_network_write=False,review_script_sha256=sha(Path(__file__)))
assert not {'torch','triton','vllm','vllm_fl','numpy'}.intersection(sys.modules)
args_cli.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(report=str(args_cli.out),sha256=sha(args_cli.out),comparisons=comparisons,raw_files=36,source_files=268),indent=2))
