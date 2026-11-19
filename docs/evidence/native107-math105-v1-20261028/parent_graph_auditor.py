"""Read-only CPU linkage audit; no framework/device imports or raw writes."""
from pathlib import Path
import collections,datetime,hashlib,importlib.util,json,re
ROOT=Path('/private/tmp/cjj-native98-main105-graph-smoke-v2-raw-20261028');OUT=ROOT/'out'
DEST=Path('/private/tmp/cjj-native98-graph-smoke-v2-cpu-review-20261028.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise AssertionError(why)
def load(p):return json.loads(p.read_text())
def events(lines,marker):
 result=[]
 for line_no,line in enumerate(lines,1):
  if marker+' ' in line:
   payload=line.split(marker+' ',1)[1];value,end=json.JSONDecoder().raw_decode(payload)
   need(not payload[end:].strip(),'trailing event bytes')
   result.append((line_no,value))
 return result
def utc(ns):return datetime.datetime.fromtimestamp(ns/1e9,datetime.timezone.utc).isoformat()
dm=load(ROOT/'download-manifest.json');need(len(dm['files'])==dm['raw_files']==32,'raw count')
need(sum(x['bytes'] for x in dm['files'])==dm['raw_bytes']==3317913,'raw bytes')
for row in dm['files']:
 p=ROOT/row['relative'];need(p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'download SHA '+row['relative'])
need(dm['original_owner']==dict(pid=9250,start_ticks=353247596,same_start=True,state='Z'),'owner')
m=load(OUT/'manifest.json');state=load(OUT/'state.json');cleanup=load(OUT/'owned-cleanup.json');graceful=load(OUT/'graceful-api-shutdown.json')
need(state['completed'] and state['smoke_passed'] and state['cleanup_complete'] and not state['errors'],'controller terminal')
need(not state['model_precision_verified'] and not state['native_model_precision_verified'] and not state['throughput_measured'] and not state['native_quality_covered'],'scope')
need(cleanup['complete'] and not cleanup['remaining_owned'] and not cleanup['signal_errors'] and not cleanup['group_signal_used'],'cleanup')
need(graceful['attempted'] and graceful['identity_verified'] and graceful['signal']=='SIGTERM' and graceful['API_exited'] and graceful['returncode']==0,'graceful owned API')
need(m['source_commit']=='63836c71d9fb2dae819da9b6218fd9d26c1d6b08','source')
need(sha(OUT/'executed-controller.py')==m['controller_sha256']=='d5d866350f6bccd5cc211b2c1108670ccfb3c504ac17d05fd0fc9b37ddbd7135','controller freeze')
B=Path('/private/tmp/cjj-native-graph-coverage-bootstrap-v2-20261028')
need(sha(B/'bootstrap-sources.json')==m['graph_coverage_bootstrap']['manifest_sha256']=='c6e795f59999e4214d11ca9dbb1b368064c3da7adca2eba5a7f5c4a0b1ae9f8b','bootstrap freeze')
for name,pin in m['graph_coverage_bootstrap']['source_hashes'].items():need(sha(B/name)==pin,'bootstrap file '+name)
spec=importlib.util.spec_from_file_location('graph_probe_cpu_readonly',B/'cjj_native_graph_coverage_probe_20261028.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
raw=(OUT/'server.log').read_bytes();lines=raw.decode().splitlines();all_events=events(lines,'BIV150_NATIVE_GRAPH_COVERAGE');kinds=collections.Counter(v['kind'] for _,v in all_events)
need(kinds==dict(observer_install=1,native_capture_certificate=2,real_request_native_full_graph_replay_returned=14),'observer events')
install_line,install=next((i,v) for i,v in all_events if v['kind']=='observer_install')
need(install['source_pins']==probe.PINS and install['probe_sha256']==sha(B/'cjj_native_graph_coverage_probe_20261028.py'),'observer source')
for name,pin in probe.PINS.items():
 if name.startswith('vllm_fl.'):
  need(m['source_hashes'][name.replace('.','/')+'.py']==pin,'source inventory '+name)
  need(install['source_paths'][name].startswith(m['source']+'/'),'source path '+name)
need(install['pid']==9294,'engine PID')
ni=events(lines,'BIV150_KONLY_NATIVE_DECODE_INSTALL');need(len(ni)==1,'native install count');native=ni[0][1]
need(native['installed'] and native['enabled'] and native['loaded_library_confirmed'] and not native['runtime_compile'],'loaded native')
need(native['library']==m['native_decode_library'] and native['library']['sha256']==probe.SHA and native['operator']==probe.OP,'native library')
need(native['module_sha256']==probe.PINS['vllm_fl.patches.biv150_konly_encoded_native'],'native module')
stock=events(lines,'BIV150_NATIVE_GQA_INSTALL');need(len(stock)==1 and stock[0][1]['overlap_v'] and stock[0][1]['library_sha256']==m['native_gqa_library_sha256']=='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8','stock overlap library')
need(all(x==m['expected_compile_semantics'] for x in m['actual_runtime']['actual_compile_semantics']),'worker fingerprint')
cap_payloads=events(lines,'BIV150_KONLY_NATIVE_DECODE_CAPTURED');need(len(cap_payloads)==84,'source native captured count')
certificates={};cert_report=[]
for no,event in all_events:
 if event['kind']!='native_capture_certificate':continue
 name=Path(event['full_certificate_file']).name;p=OUT/'native-graph-coverage'/name;need(sha(p)==event['full_certificate_sha256']==p.stem,'certificate SHA')
 cert=load(p);need({k:v for k,v in cert.items() if k!='native_payloads'}=={k:v for k,v in event.items() if k not in ('full_certificate_file','full_certificate_sha256','native_payload_sha256')},'certificate console/full metadata')
 need(event['native_payload_sha256']==[hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest() for x in cert['native_payloads']],'payload SHA list')
 n=cert['bucket']['num_tokens'];s=2 if n==64 else 4;need(n in (32,64) and cert['bucket']==dict(has_lora=False,num_active_loras=0,num_reqs=n,num_tokens=n,uniform=True),'bucket')
 need(cert['pid']==9294 and cert['native_count']==len(cert['native_payloads'])==42 and cert['native_generations']==list(range(1,43)),'42 distinct cache epochs')
 need([x['entry_generation'] for x in cert['native_payloads']]==list(range(1,43)),'ordered 42 cache owners')
 for payload in cert['native_payloads']:
  need(payload in [x for _,x in cap_payloads],'payload not in raw Native print')
  need(payload['shape']==n and payload['segments']==s and payload['capture'] and payload['native_launch_returned'],'native captured geometry')
  need(payload['library']==native['library'] and payload['operator']==probe.OP and payload['cache_pages']==34276 and payload['physical_page_bytes']==16384 and payload['format']=='cjj-konly-rn32-rne-v2','native physical cache')
  need(payload['query']==dict(device='cuda:0',dtype='torch.bfloat16',offset=0,shape=[n,16,128],strides=[2048,128,1]),'actual query stride')
  need(payload['table']['shape']==[n,8192] and payload['table']['strides']==[8192,1] and payload['table']['dtype']=='torch.int32','table ABI')
  need(payload['partial']['shape']==[64,16,s,128] and payload['partial']['strides']==[32768,2048,128,1],'width16 scratch')
  need(all(payload[q]['shape']==[64,16,s] and payload[q]['strides']==[256,16,1] for q in ('maximum','summation')),'scalar scratch')
  need(not payload['service_numeric_verified'],'component/model qualification')
 key=tuple(cert[k] for k in ('pid','wrapper_id','entry_id','graph_id'));need(key not in certificates,'duplicate graph certificate');certificates[key]=cert
 cert_report.append(dict(file=str(p),sha256=sha(p),server_line=no,bucket=n,segments=s,query_pitch=2048,cache_pages=34276,native_count=42,entry_generations=list(range(1,43)),capture_utc=utc(cert['wall_ns']),component_qualifications=sorted({x['component_qualification'] for x in cert['native_payloads']}),explicit_layer_names_saved=False))
burst=load(OUT/'native-burst64-requests.json');need(burst['completed']==burst['concurrency_requested']==len(burst['records'])==64,'HTTP burst completed')
need({x['index'] for x in burst['records']}==set(range(64)),'HTTP row indices')
responses={x['response']['id']:x for x in burst['records']};need(len(responses)==64,'HTTP response IDs')
need(all(x['response']['usage']==dict(x['response']['usage'],prompt_tokens=35,completion_tokens=16,total_tokens=51) and len(x['response']['choices'])==1 and x['response']['choices'][0]['finish_reason']=='length' for x in burst['records']),'HTTP complete usages')
window=(OUT/'native-burst64-runtime-window.log').read_bytes();w=m['native_burst64_window'];need(len(window)==w['bytes'] and sha(OUT/'native-burst64-runtime-window.log')==w['sha256'] and raw[w['log_byte_offset']:w['log_byte_offset']+w['bytes']]==window,'HTTP window raw identity')
window_replays=[v for _,v in events(window.decode().splitlines(),'BIV150_NATIVE_GRAPH_COVERAGE') if v['kind']=='real_request_native_full_graph_replay_returned']
replays=[(i,v) for i,v in all_events if v['kind']=='real_request_native_full_graph_replay_returned'];need(window_replays==[v for _,v in replays],'all replays in HTTP window')
previous={};all_engine_ids={};report=[]
for no,e in replays:
 key=tuple(e[k] for k in ('pid','wrapper_id','entry_id','graph_id'));need(key in certificates,'replay graph identity');c=certificates[key]
 need(e['bucket']==c['bucket'] and e['native_count']==42 and e['native_generations']==c['native_generations'],'replay bucket/epochs')
 need(e['native_library_sha256']==probe.SHA and e['operator']==probe.OP and e['replay_returned'] and not e['GPU_completion_or_precision_proven'],'replay identity')
 need(e['monotonic_ns']>c['monotonic_ns'] and e['wall_ns']>c['wall_ns'],'capture precedes replay')
 need(e['bucket']['num_tokens']==e['bucket']['num_reqs']==64 and e['valid_requests']==len(e['requests']) and e['padding_rows']==64-e['valid_requests'],'valid/padded sequence-token count')
 ids=[r['request_id'] for r in e['requests']];need(len(set(ids))==len(ids),'request uniqueness')
 for row in e['requests']:
  rid=row['request_id'];matches=[x for x in responses if rid.startswith(x+'-') and re.fullmatch('[0-9a-f]{8}',rid[len(x)+1:])];need(len(matches)==1,'unique HTTP id prefix '+rid)
  all_engine_ids[rid]=matches[0];need(type(row['computed_tokens']) is int and row['prompt_tokens']==35 and 35<=row['computed_tokens']<51,'completed prompt/valid decode')
  if rid in previous:need(row['computed_tokens']==previous[rid]+1,'nonstale request progress')
  previous[rid]=row['computed_tokens']
 report.append(dict(server_line=no,bucket_tokens=64,bucket_sequences=64,valid_decode_sequences=e['valid_requests'],valid_query_tokens=e['valid_requests'],padding=e['padding_rows'],native_cache_epochs=42,computed_range=[min(x['computed_tokens'] for x in e['requests']),max(x['computed_tokens'] for x in e['requests'])],utc=utc(e['wall_ns'])))
need(len(all_engine_ids)==len(set(all_engine_ids.values()))==64,'64 actual HTTP requests linked')
need(collections.Counter(e['valid_requests'] for _,e in replays)=={64:13,63:1},'replay census')
error_lines=[i for i,l in enumerate(lines,1) if any(t in l for t in ('OBSERVER_ERROR','"kind": "observer_error"','OBSERVER_SUMMARY_FAILED'))]
summary_files=list((OUT/'native-graph-coverage').glob('worker-summary-*.json'));summary_events=[(i,v) for i,v in all_events if v['kind']=='observer_exit_summary']
need(not error_lines and not summary_files and not summary_events,'reported observer terminal observation changed')
shutdown=next(i for i,l in enumerate(lines,1) if '[shutdown] API server: shutdown triggered' in l)
term=next(i for i,l in enumerate(lines,1) if 'trigger received signal=SIGTERM' in l)
engine_dead=[i for i,l in enumerate(lines,1) if 'EngineDeadError' in l];need(engine_dead and all(i>shutdown and i>term for i in engine_dead),'request-period EngineDead')
http200=[i for i,l in enumerate(lines,1) if 'POST /v1/chat/completions HTTP/1.1" 200 OK' in l]
need(len(http200)==67 and max(http200)<shutdown and max(i for i,_ in replays)<shutdown,'HTTP completion before shutdown')
result=dict(readonly_CPU=True,raw_root=str(ROOT),raw_manifest_sha256=sha(ROOT/'download-manifest.json'),all_32_raw_file_SHA_verified=True,raw_bytes=dm['raw_bytes'],source_commit=m['source_commit'],controller_sha256=m['controller_sha256'],worker_fingerprint=m['expected_compile_semantics'],source_pins_matched=9,stock_overlap_library_sha256=m['native_gqa_library_sha256'],private_native_library=native['library'],certificates=cert_report,observed_replay_records=report,HTTP_completed=64,HTTP_tokens=dict(prompt=2240,completion=1024),HTTP_ID_prefix_engine_suffix_one_to_one=True,actual_request_ids_linked=64,actual_request_state_advances_each_observed_step=True,observed_replay_lower_bound=14,observed_native64_decode_query_row_lower_bound=895,native32_actual_replay_observed=False,observed_observer_errors=0,finalizer_summary_persisted=False,final_error_count_verified=False,final_overflow_verified=False,complete_recording_integrity_verified=False,observer_terminal_limitation='No worker-summary or observer_exit_summary in complete diagnostic record collection; absence of first-error marker is not a persisted final zero counter.',shutdown=dict(all_67_HTTP200_before_cleanup=True,shutdown_start_server_line=shutdown,engine_SIGTERM_server_line=term,EngineDeadError_server_lines=engine_dead,EngineDeadError_only_after_owned_cleanup_started=True,graceful_API_returncode=0,owned_cleanup_complete=True,finalizer_bypass_cause_not_proven=True),coverage_positive_lower_bound_verified=True,all_native_shapes_or_full_request_frequency_verified=False,Math105_or_precision_accepted=False,formal_score_eligible=False,quality_covered=False,raw_files_modified=False)
DEST.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(report=str(DEST),sha256=sha(DEST),raw32_SHA_pass=True,certificates=2,each_certificate_native_cache_entries=42,replays=14,request_rows=895,all64_HTTP_IDs_linked=True,finalizer_summary=False,EngineDead_after_cleanup_only=True,quality_covered=False),sort_keys=True,indent=2))
