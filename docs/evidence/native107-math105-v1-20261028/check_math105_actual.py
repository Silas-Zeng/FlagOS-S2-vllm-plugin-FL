"""Bounded CPU adaptation of the existing Math105 and FL Graph raw auditors."""
import argparse,ast,collections,hashlib,importlib.util,json,re,subprocess,posixpath
from pathlib import Path
SOURCE=Path('/private/tmp/flagos-cjj-native-k8-v-overlap-service-20261028')
BUNDLE=Path('/private/tmp/cjj-native-v-overlap-graph-coverage-bootstrap-v3-20261028')
MATH_AUDITOR=Path('/private/tmp/cjj-native-k8-service-math105-root-review-20261028.py')
GRAPH_AUDITOR=Path('/private/tmp/check_cjj_native98_graph_smoke_v2_actual_20261028.py')
COMMIT='b5d1a241f45efd21517d04e25a7dc5d44ac1ca06'
LIB='8b86cb688ad72dcf9859994ed2092620312cdd5ae96d194904b2bde7926aacc5'
OP='cjj_native_k8_v_overlap_probe::produce'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def need(ok,why):
 if not ok:raise AssertionError(why)
def replace(s,a,b):
 need(s.count(a)==1,'finite old-source adapter '+a);return s.replace(a,b)
def events(lines,marker):
 values=[]
 for line_no,line in enumerate(lines,1):
  if marker+' ' in line:
   payload=line.split(marker+' ',1)[1];v,end=json.JSONDecoder().raw_decode(payload);need(not payload[end:].strip(),'event trailing bytes');values.append((line_no,v))
 return values

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-root',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
 if a.raw_root is None:
  print(json.dumps(dict(prepared_only=True,CPU_only=True,requires_complete_original_manifest=True,source_commit=COMMIT,library_sha256=LIB,operator=OP,no_fixture_or_actual_result_claimed=True,quality_covered=False)));return
 need(a.out is not None and not a.out.exists(),'fresh output required')
 root=a.raw_root.resolve();out=root/'out';d=read(root/'download-manifest.json');m=read(out/'manifest.json');state=read(out/'state.json')
 need(d['read_only'] and d['model_rerun'] is False,'transport scope');need(d['original_owner']==dict(pid=21979,start_ticks=353729022,same_start=True,state='Z'),'original owner')
 need(len(d['files'])==d['raw_files']==55 and sum(r['bytes'] for r in d['files'])==d['raw_bytes']==44782164,'raw inventory')
 seen=set()
 for r in d['files']:
  rel=Path(r['relative']);need(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen,'raw path');seen.add(str(rel));q=root/rel
  need(q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'raw SHA '+str(rel))
 need({str(q.relative_to(root)) for q in root.rglob('*') if q.is_file() and q.name!='download-manifest.json'}==seen,'full diagnostic file set')
 need(m['source_commit']==COMMIT,'source commit')
 # The shared development checkout may advance after launch. Verify every
 # recorded file against the immutable tested Git object, never another HEAD.
 tree={}
 for record in subprocess.check_output(['git','-C',str(SOURCE),'ls-tree','-rz',COMMIT]).split(b'\0'):
  if record:
   info,name=record.split(b'\t',1);tree[name.decode()]=info.split()[0].decode()
 source_symlinks={}
 for rel,h in m['source_hashes'].items():
  target=rel
  for _ in range(8):
   blob=subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':'+target])
   if tree[target]!='120000':break
   target=posixpath.normpath(posixpath.join(posixpath.dirname(target),blob.decode()));need(not target.startswith('../') and not target.startswith('/'),'Git symlink escape')
  else:raise AssertionError('Git symlink cycle')
  if target!=rel:source_symlinks[rel]=target
  need(hashlib.sha256(blob).hexdigest()==h,'immutable tested source '+rel)
 need(sha(out/'executed-controller.py')==m['controller_sha256']=='558579df4ebd3ac359167a429e9b3c6905952509de5910ebf843162ba0346600','controller')
 need(m['helper_sha256']==sha('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py')=='94dbbb0fbc8f3c8837ea239ec18fc0ec4872a2b0049a1843d96a3e6a9701e976','helper')
 need(m['native_decode_library']['sha256']==LIB and m['native_identity']['module_sha256']=='90c9604f6a8d8932080df53a1e03c65a676e2f5a814389310fbed81f7e4ee306','native identity')
 need(m['native_gqa_library_sha256']=='0c66098b0f293c7e02b098517fae9a995402f7dc74f629e78c36a830903e87b8','stock library')
 service=['/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072'];need(m['command']==service,'fixed service')
 raw=(out/'server.log').read_bytes();log=raw.decode();lines=log.splitlines();c3=[v for _,v in events(lines,'BIV150_COMPILE_SEMANTICS')]
 need(c3 and all(v==m['expected_compile_semantics'] for v in c3),'actual worker fingerprint')
 for key in ('actual_runtime','actual_runtime_after_math','actual_runtime_after_supplemental64'):
  need(m[key]['actual_compile_semantics']==c3 and m[key]['native_decode']['actual_imports_verified'] and m[key]['native_decode']['quality_covered'] is False,'actual identity receipt')
 need(not any(x in log for x in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED','BIV150_PREFILL_MULTI2_FALLBACK')),'source/C3 scope reject')
 # Keep the old evaluator's complete per-question, command/config, grade,
 # token-usage and uniqueness checks. Only remove its old observed answers and
 # truncation counts; these are recomputed, never targets for a new candidate.
 old=MATH_AUDITOR.read_text();block=old[old.index("base=m['source']"):old.index("coverage=parse('BIV150_NATIVE_GRAPH_COVERAGE')")]
 block=replace(block,' assert correct==(103 if parallel==8 else 100)',' # Candidate correctness is observed, never a fixed historical expectation.')
 block=replace(block,' assert stopped==limit_rows and len(stopped)==(0 if parallel==8 else 2)',' assert stopped==limit_rows')
 block=replace(block," assert Counter(x['stop_reason'] for x in rows)==({'stop':105} if parallel==8 else {'stop':103,'max_tokens':2})"," assert sum(Counter(x['stop_reason'] for x in rows).values())==105")
 # The original prompt-content comparison remains unchanged. Generated
 # assistant messages and per-run message IDs must not be cross-run targets.
 ns=dict(m=m,d=d,out=out,root=root,read=read,sha=sha,json=json,re=re,ast=ast,Counter=collections.Counter,hashlib=hashlib)
 math_error=None
 try:exec(compile(block,str(MATH_AUDITOR)+'::<candidate counts observed>','exec'),ns)
 except Exception as e:math_error=repr(e)
 results=ns.get('results',{});maps=ns.get('question_maps',{})
 if math_error is None:
  for q,v in maps['official_parallel8'].items():
   other=maps['supplemental_parallel64'][q]
   prompt=lambda messages:[{k:z for k,z in row.items() if k not in ('id','perf_metrics')} for row in messages if row.get('source')!='generate']
   need(prompt(v['messages'])==prompt(other['messages']),'same input prompts across evaluations')
 # Always retain an independent basic error/truncation census if strict
 # evaluator consistency fails; never promote a partial parse to a pass.
 census={}
 for name,folder in [('official_parallel8',out/'math500-level3'),('supplemental_parallel64',out/'supplemental-native-math105-parallel64/math500-level3')]:
  paths=list(folder.glob('*/predictions/minicpm/*.jsonl'));rows=[];bad=[]
  for q in paths:
   for no,line in enumerate(q.read_text().splitlines(),1):
    if line.strip():
     try:rows.append(json.loads(line))
     except Exception as e:bad.append(dict(file=str(q),line=no,error=repr(e)))
  census[name]=dict(prediction_files=len(paths),rows=len(rows),parse_errors=bad,request_errors=sum(v.get('model_output',{}).get('error') is not None for v in rows),stop_reasons=dict(collections.Counter(c.get('stop_reason') for v in rows for c in v.get('model_output',{}).get('choices',[]))))
 graph={};graph_error=None
 try:
  spec=importlib.util.spec_from_file_location('actual_v_overlap_graph_probe_readonly',BUNDLE/'cjj_native_graph_coverage_probe_20261028.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
  need(sha(BUNDLE/'bootstrap-sources.json')==m['graph_coverage_bootstrap']['manifest_sha256']=='905aaf38eff0b5d604c42751cac0ee3cc35f528053a75f2d8cc0553ddeb353c6','v3 bundle')
  for name,h in m['graph_coverage_bootstrap']['source_hashes'].items():need(sha(BUNDLE/name)==h,'bundle source')
  ev=events(lines,'BIV150_NATIVE_GRAPH_COVERAGE');install=[(i,v) for i,v in ev if v['kind']=='observer_install'];need(len(install)==1,'observer install')
  need(install[0][1]['source_pins']==probe.PINS and install[0][1]['probe_sha256']==sha(BUNDLE/'cjj_native_graph_coverage_probe_20261028.py'),'actual FL GraphWrapper/probe source')
  for name,h in probe.PINS.items():
   if name.startswith('vllm_fl.'):need(m['source_hashes'][name.replace('.','/')+'.py']==h,'pin source '+name)
  captures=events(lines,'BIV150_KONLY_NATIVE_DECODE_CAPTURED');certs={};cert_info=[]
  for no,e in ev:
   if e['kind']!='native_capture_certificate':continue
   q=out/'native-graph-coverage'/Path(e['full_certificate_file']).name;need(sha(q)==e['full_certificate_sha256']==q.stem,'full certificate SHA');c=read(q)
   need({k:v for k,v in c.items() if k!='native_payloads'}=={k:v for k,v in e.items() if k not in ('full_certificate_file','full_certificate_sha256','native_payload_sha256')},'full/console metadata')
   need(e['native_payload_sha256']==[hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest() for x in c['native_payloads']],'payload hashes')
   n=c['bucket']['num_tokens'];s=2 if n==64 else 4;need(n in (32,64) and c['bucket']==dict(num_tokens=n,num_reqs=n,uniform=True,has_lora=False,num_active_loras=0),'FULL bucket')
   need(c['native_count']==len(c['native_payloads'])==42 and len(set(c['native_generations']))==42 and sorted(x['entry_generation'] for x in c['native_payloads'])==c['native_generations'],'42 unique current cache epochs')
   for x in c['native_payloads']:
    need(x in [v for _,v in captures] and x['capture'] and x['native_launch_returned'] and x['shape']==n and x['segments']==s,'source-bound captured payload')
    need(x['library']==m['native_decode_library'] and x['operator']==OP and x['physical_page_bytes']==16384 and x['format']=='cjj-konly-rn32-rne-v2','new8b86 typed cache')
    need(x['query']['shape']==[n,16,128] and x['query']['strides'] in ([2048,128,1],[2560,128,1]) and x['table']['shape']==[n,8192],'actual ABI')
   key=tuple(c[k] for k in ('pid','wrapper_id','entry_id','graph_id'));need(key not in certs,'duplicate certificate identity');certs[key]=c
   cert_info.append(dict(file=str(q.relative_to(root)),sha256=sha(q),bucket=n,native_count=42,cache_entry_generations=c['native_generations'],server_line=no))
  api={}
  for phase,qmap in maps.items():
   for q,v in qmap.items():
    rid=v['model_output']['id'];need(type(rid)is str and rid not in api,'unique separate evaluator response IDs');api[rid]=(phase,q,v)
  need(set(maps)=={'official_parallel8','supplemental_parallel64'} and len(api)==210,'two distinct 105 runs')
  window=(out/'supplemental-native-math105-parallel64-runtime-window.log').read_bytes();w=m['supplemental64_request_window'];need(len(window)==w['bytes'] and hashlib.sha256(window).hexdigest()==w['sha256'] and raw[w['log_byte_offset']:w['log_byte_offset']+w['bytes']]==window,'supplemental HTTP raw window')
  window_ids={v['monotonic_ns'] for _,v in events(window.decode().splitlines(),'BIV150_NATIVE_GRAPH_COVERAGE') if v['kind']=='real_request_native_full_graph_replay_returned'}
  buckets={phase:{str(n):dict(replay_lower_bound=0,valid_query_row_lower_bound=0,question_ids=set()) for n in (32,64)} for phase in maps};unassigned=0;per_question=collections.defaultdict(lambda:collections.Counter());prior={};joined=0
  for no,e in ev:
   if e['kind']!='real_request_native_full_graph_replay_returned':continue
   key=tuple(e[k] for k in ('pid','wrapper_id','entry_id','graph_id'));need(key in certs,'same captured graph');c=certs[key];n=e['bucket']['num_tokens']
   need(e['bucket']==c['bucket'] and e['native_count']==42 and e['native_generations']==c['native_generations'] and e['native_library_sha256']==LIB and e['operator']==OP and e['replay_returned'],'replay source/epoch identity')
   need(e['monotonic_ns']>c['monotonic_ns'] and e['valid_requests']==len(e['requests'])==len({r['request_id'] for r in e['requests']}) and e['padding_rows']==n-e['valid_requests'],'valid query/sequence rows vs padding')
   matched=[]
   for row in e['requests']:
    base,sep,suffix=row['request_id'].rpartition('-');match=api.get(base) if sep and re.fullmatch('[0-9a-f]{8}',suffix) else None
    if match is None:matched.append(None);continue
    phase,q,v=match;u=v['model_output']['usage'];need(row['prompt_tokens']==u['input_tokens'] and type(row['computed_tokens']) is int and row['prompt_tokens']<=row['computed_tokens']<row['prompt_tokens']+u['output_tokens'],'real evaluated prompt/decode token bounds')
    t=(phase,row['request_id']);need(t not in prior or row['computed_tokens']>prior[t],'strict CPU request progress');prior[t]=row['computed_tokens'];matched.append((phase,q))
   phases={x[0] for x in matched if x is not None}
   if None in matched or len(phases)!=1:unassigned+=1;continue
   phase=next(iter(phases));need(phase!='supplemental_parallel64' or e['monotonic_ns'] in window_ids,'supplemental evaluation HTTP window')
   b=buckets[phase][str(n)];b['replay_lower_bound']+=1;b['valid_query_row_lower_bound']+=len(matched);b['question_ids'].update(q for _,q in matched);joined+=1
   for _,q in matched:per_question[phase,q][str(n)]+=1
  for phase,groups in buckets.items():
   for b in groups.values():b['question_ids']=sorted(b['question_ids']);b['unique_questions_observed']=len(b['question_ids'])
  errors=[(i,v) for i,v in ev if v['kind']=='observer_error'];summaries=list((out/'native-graph-coverage').glob('worker-summary-*.json'));receipts=[(i,v) for i,v in ev if v['kind']=='observer_exit_summary'];terminal=[]
  for q in summaries:
   s=read(q);need(s['source_pins']==probe.PINS,'summary source');matches=[v for _,v in receipts if Path(v['summary_file']).name==q.name and v['summary_sha256']==sha(q)];need(len(matches)==1,'summary SHA receipt');terminal.append(dict(file=str(q.relative_to(root)),sha256=sha(q),observer_error_count=s['observer_error_count'],overflow=s['overflow'],recording_integrity=s['recording_integrity']))
  graph=dict(observer_event_counts=dict(collections.Counter(v['kind'] for _,v in ev)),source_pins=probe.PINS,certificate_records=cert_info,all_native_replays_source_and_epoch_linked=True,per_evaluation_buckets=buckets,joined_evaluator_replay_lower_bound=joined,other_unassigned_replay_records=unassigned,per_question_bucket_replay_lower_bounds=[dict(evaluation=phase,question_id=q,bucket_replays=dict(v)) for (phase,q),v in sorted(per_question.items())],observed_error_records=errors,final_summary_records=terminal,final_error_count_or_overflow_verified=bool(terminal),complete_recording_integrity_verified=bool(terminal) and all(v['recording_integrity'] for v in terminal),coverage_is_lower_bound_only=True,capture_is_not_request=True,quality_covered=False)
 except Exception as e:graph_error=repr(e)
 cleanup=read(out/'owned-cleanup.json');shutdown=read(out/'graceful-api-shutdown.json');terminal_ok=cleanup['complete'] and not cleanup['remaining_owned'] and not cleanup['unobserved_same_session'] and not cleanup['signal_errors'] and shutdown['identity_verified'] and shutdown['API_exited'] and shutdown['returncode']==0
 starts=[i for i,l in enumerate(lines,1) if '[shutdown] API server: shutdown triggered' in l];dead=[i for i,l in enumerate(lines,1) if 'EngineDeadError' in l]
 result=dict(local_CPU_only=True,raw_root=str(root),raw_manifest_sha256=sha(root/'download-manifest.json'),all_raw_SHA_verified=True,raw_files=55,raw_bytes=44782164,source_commit=COMMIT,source_files_verified=len(m['source_hashes']),source_verified_against_immutable_tested_git_object=True,source_symlinks_resolved_in_same_commit=source_symlinks,library_sha256=LIB,controller_sha256=m['controller_sha256'],old_math_auditor_sha256=sha(MATH_AUDITOR),old_graph_auditor_sha256=sha(GRAPH_AUDITOR),script_sha256=sha(__file__),math_strict_consistency_error=math_error,results=results,basic_census=census,graph_audit_error=graph_error,coverage=graph,terminal=dict(original_owner=d['original_owner'],state_completed=state['completed'],state_errors=state['errors'],cleanup_verified=terminal_ok,shutdown_start_lines=starts,EngineDeadError_lines=dead,EngineDeadError_only_after_shutdown=bool(starts) and all(i>starts[0] for i in dead)),official_metric_gate=math_error is None and results.get('official_parallel8',{}).get('metric_gate_095_passed',False),supplemental_metric_gate=math_error is None and results.get('supplemental_parallel64',{}).get('metric_gate_095_passed',False),controller_quality_covered_unchanged=state['native_quality_covered'],actual_full_Native_quality_covered=False,formal_throughput_or_merge_qualified=False,limitations=['Accuracy is recomputed from saved per-question evaluator grades, not independent mathematical regrading.','Official8 and supplemental64 are separate runs; temperature1.0 differences are not causally attributed.','Positive joined replay records prove only the saved 32/64 request/epoch lower bounds, not all tokens or GPU completion.','Without a SHA-bound final summary no final zero-error/zero-overflow or complete recording claim is made.','Old22e capture evidence is never used; no archived raw data is altered.'])
 a.out.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(report=str(a.out),sha256=sha(a.out),math_error=math_error,graph_error=graph_error,official=result['official_metric_gate'],supplemental=result['supplemental_metric_gate'],quality_covered=False),sort_keys=True))
if __name__=='__main__':main()
