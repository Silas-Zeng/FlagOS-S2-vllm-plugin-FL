"""Finite local raw-record replay of a975 official and supplemental Math105."""
import hashlib,json,subprocess
from pathlib import Path
from collections import Counter
import re,ast
root=Path('/private/tmp/cjj-native-k8-service-math105-v1-raw-20261028');out=root/'out'
source=Path('/private/tmp/flagos-cjj-native-k8-service-frozen-a97-20261028')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
d=read(root/'download-manifest.json');assert d['raw_files']==len(d['files'])==53
assert d['raw_bytes']==sum(x['bytes'] for x in d['files'])==12312819
assert len({x['relative'] for x in d['files']})==53
for row in d['files']:
 p=root/row['relative'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['relative']
m=read(out/'manifest.json');state=read(out/'state.json');tp=read(Path('/private/tmp/cjj-native-k8-service-throughput-v1-raw-20261028/out/manifest.json'))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==m['source_commit']==tp['source_commit']=='a975e8715beb752d6a76b456f417ba2011ea917b'
assert m['source_hashes']==tp['source_hashes'] and len(m['source_hashes'])==252
for rel,h in m['source_hashes'].items():assert sha(source/rel)==h,rel
assert sha(out/'executed-controller.py')==m['controller_sha256']=='4a35a01a38ddd7015dde533a24caed3ed914086f65a506c96c305e7c31e498aa'
assert m['helper_sha256']==sha(Path('/private/tmp/cjj-actual-owned-service-helper-94db-20261028.py'))
service=['/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
assert m['command']==service==tp['command']
assert m['expected_compile_semantics']==tp['expected_compile_semantics']
for k in ['native_sha256','native_gqa_library_sha256','native_decode_library','tuning_snapshot']:
 assert m[k]==tp[k],k
assert m['actual_resolution']['origin']==m['source']+'/vllm_fl/__init__.py'
log=(out/'server.log').read_text()
parse=lambda marker:[json.loads(line.split(marker+' ',1)[1]) for line in log.splitlines() if marker+' ' in line]
c3=parse('BIV150_COMPILE_SEMANTICS');assert len(c3)==2 and all(x==m['expected_compile_semantics'] for x in c3)
for key in ['actual_runtime','actual_runtime_after_math','actual_runtime_after_supplemental64']:
 assert m[key]['actual_compile_semantics']==c3
 assert m[key]['native_decode']['actual_imports_verified']
 assert m[key]['native_decode']['quality_covered'] is False
assert not any(x in log for x in ['BIV150_COMPUTE_ORDER_SCOPE_REJECTED','BIV150_PREFILL_MULTI2_FALLBACK'])
base=m['source'].rsplit('/code/',1)[0]+'/'+d['tag']
cli=['evalscope','eval','--model','minicpm','--api-url','http://127.0.0.1:9031/v1/chat/completions','--api-key','EMPTY','--eval-type','openai_api','--datasets','math_500','--dataset-args','{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}','--eval-batch-size','8','--timeout','3600','--generation-config','{"temperature":1.0,"top_p":0.95,"max_tokens":32768}','--work-dir',base+'/math500-level3','--ignore-errors']
assert read(out/'math500-level3/command.json')==cli
cli64=list(cli);cli64[15]='64';cli64[21]=base+'/supplemental-native-math105-parallel64/math500-level3'
assert read(out/'supplemental-native-math105-parallel64/math500-level3/command.json')==cli64
adapt=read(out/'supplemental-native-math105-parallel64/supplemental-command-adaptation.json')
assert adapt['original_official_command']==cli and adapt['actual_supplemental_command']==cli64 and adapt['changed_indices']==[15,21]
assert adapt['diagnostic_only'] is True and adapt['quality_covered'] is False
results={};question_maps={}
def text_content(content):
 if isinstance(content,str):return content
 return ''.join(x.get('text','') for x in content or [] if x.get('type')=='text')
for name,folder,parallel in [('official_parallel8',out/'math500-level3',8),('supplemental_parallel64',out/'supplemental-native-math105-parallel64/math500-level3',64)]:
 def one(glob):
  paths=list(folder.glob(glob));assert len(paths)==1;return paths[0]
 pp=one('*/predictions/minicpm/*.jsonl');rp=one('*/reviews/minicpm/*.jsonl');reportp=one('*/reports/minicpm/math_500.json');configp=one('*/configs/task_config.yaml')
 preds=[json.loads(x) for x in pp.read_text().splitlines() if x.strip()];reviews=[json.loads(x) for x in rp.read_text().splitlines() if x.strip()]
 assert len(preds)==len(reviews)==105
 predmap={v['metadata']['question_id']:v for v in preds};reviewmap={v['sample_score']['sample_metadata']['question_id']:v for v in reviews}
 assert len(predmap)==len(reviewmap)==105 and predmap.keys()==reviewmap.keys()
 assert len({v['index'] for v in preds})==len({v['index'] for v in reviews})==105
 report=read(reportp);config=configp.read_text()
 assert re.findall(r'^eval_batch_size: (.*)$',config,re.M)==[str(parallel)]
 block=re.findall(r'^generation_config:\n((?:  .*\n)+)',config,re.M);assert len(block)==1
 generation={key:ast.literal_eval(value) for key,value in (line.strip().split(': ',1) for line in block[0].splitlines())}
 assert generation==dict(batch_size=parallel,max_tokens=32768,temperature=1.0,timeout=3600.0,top_p=.95)
 assert re.findall(r'^limit: (.*)$',config,re.M)==['null'] and re.findall(r'^repeats: (.*)$',config,re.M)==['1']
 assert 'datasets:\n- math_500\n' in config
 assert 'dataset_args:\n  math_500:\n    dataset_id: /workspace/evalscope-datasets/math_500\n    subset_list:\n    - Level 3\n' in config
 rows=[]
 for q in sorted(predmap):
  p=predmap[q];r=reviewmap[q];s=r['sample_score'];v=s['score'];mo=p['model_output']
  assert p['index']==r['index']==s['sample_id']==s['group_id'] and s['generation_index']==0
  assert p['messages']==r['messages'] and p['metadata']==s['sample_metadata']
  assert p['model']=='minicpm' and mo['model']=='minicpm' and len(mo['choices'])==1
  assert v['status']=='success' and set(v['value'])=={'accuracy'} and v['value']['accuracy'] in (0.0,1.0)
  choice=mo['choices'][0];usage=mo['usage'];assert usage['total_tokens']==usage['input_tokens']+usage['output_tokens']
  rows.append({'question_id':q,'index':p['index'],'correct':int(v['value']['accuracy']),'review_status':v['status'],'request_error':mo['error'],'stop_reason':choice['stop_reason'],'input_tokens':usage['input_tokens'],'output_tokens':usage['output_tokens'],'empty_final_text':not text_content(choice['message']['content']).strip()})
 correct=sum(x['correct'] for x in rows);accuracy=correct/105
 assert correct==(103 if parallel==8 else 100)
 assert report['metrics'][0]['num']==report['num']==105 and report['metrics'][0]['score']==round(accuracy,4)
 assert report['execution_summary']['requested']==report['execution_summary']['succeeded']==105 and report['execution_summary']['errored']==0 and report['execution_summary']['incomplete'] is False
 assert not [x for x in rows if x['request_error'] is not None or x['empty_final_text']]
 usage=report['perf_metrics']['summary']['usage']
 assert usage['total_input_tokens']==sum(x['input_tokens'] for x in rows)
 assert usage['total_output_tokens']==sum(x['output_tokens'] for x in rows)
 stopped=[x for x in rows if x['stop_reason']=='max_tokens'];limit_rows=[x for x in rows if x['output_tokens']==32768]
 assert stopped==limit_rows and len(stopped)==(0 if parallel==8 else 2)
 assert Counter(x['stop_reason'] for x in rows)==({'stop':105} if parallel==8 else {'stop':103,'max_tokens':2})
 verdict=read(folder.parent/'accuracy-verdict.json');assert verdict['score']==round(accuracy,4) and verdict['metric_count']==verdict['predictions']==verdict['unique_questions']==105 and verdict['failed_predictions']==[]
 results[name]={'parallel':parallel,'official_protocol':parallel==8,'diagnostic_only':parallel!=8,'predictions':105,'reviews':105,'unique_questions':105,'correct':correct,'incorrect':105-correct,'exact_accuracy':accuracy,'reported_rounded_accuracy':report['metrics'][0]['score'],'metric_gate_095_passed':accuracy>=.95,'request_error_count':0,'empty_final_text_count':0,'max_token_truncation_count':len(stopped),'truncated_questions':stopped,'incorrect_questions':[x for x in rows if not x['correct']],'question_ids_sha256':hashlib.sha256(json.dumps(sorted(predmap),ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'files':{k:{'relative':str(p.relative_to(root)),'sha256':sha(p)} for k,p in [('predictions',pp),('reviews',rp),('report',reportp),('config',configp)]},'per_question':rows}
 question_maps[name]=predmap
assert question_maps['official_parallel8'].keys()==question_maps['supplemental_parallel64'].keys()
for q,p in question_maps['official_parallel8'].items():
 other=question_maps['supplemental_parallel64'][q]
 assert p['index']==other['index'] and p['metadata']==other['metadata']
 assert p['messages'][0]['content']==other['messages'][0]['content']
coverage=parse('BIV150_NATIVE_GRAPH_COVERAGE');assert len(coverage)==1 and coverage[0]['kind']=='observer_install' and coverage[0]['quality_covered'] is False
assert m['supplemental64_request_window']['native_events']==[] and m['supplemental64_request_window']['Graph_replays_counted'] is False
probe=Path('/private/tmp/cjj-native-graph-coverage-bootstrap-v1-20261028/cjj_native_graph_coverage_probe_20261028.py')
assert sha(probe)==m['graph_coverage_bootstrap']['source_hashes'][probe.name]
assert "self.graph.CUDAGraphWrapper.__dict__['__call__']" in probe.read_text()
runner=(source/'vllm_fl/worker/model_runner.py').read_text();assert 'from vllm_fl.compilation.graph import GraphWrapper' in runner and 'self.model = GraphWrapper(' in runner
cleanup=read(out/'owned-cleanup.json');shutdown=read(out/'graceful-api-shutdown.json')
assert cleanup['complete'] and cleanup['remaining_owned']==cleanup['unobserved_same_session']==cleanup['signal_errors']==[]
assert shutdown['identity_verified'] and shutdown['API_exited'] and shutdown['returncode']==0
assert state['completed'] and state['cleanup_complete'] and state['errors']==[] and state['phase']=='completed'
assert state['native_quality_covered'] is False and state['native_model_precision_verified'] is False and m['full_acceptance'] is False
report={'schema':1,'local_cpu_only':True,'SSH_GPU_PR_or_production_mutation':False,'raw_root':str(root),'download_manifest_sha256':sha(root/'download-manifest.json'),'raw_integrity':{'files':53,'bytes':12312819,'all_hashes_and_sizes_verified':True},'source_commit':m['source_commit'],'source_hashes_verified':252,'same_production_source_and_libraries_as_original_throughput_a975':True,'source_qualification_transfer_to_638_or_latest_main':False,'controller_sha256':m['controller_sha256'],'helper_sha256':m['helper_sha256'],'official_service_command_verified':service,'official_evaluation_command_verified':cli,'supplemental_evaluation_command_verified':cli64,'only_supplemental_command_changes':['eval-batch-size 8 -> 64','work-dir'], 'worker_C3_records_matching_expected':2,'results':results,'identical_105_question_set_and_prompts_across_runs':True,'coverage':{'observer_install_seen':True,'observer_event_counts':dict(Counter(x['kind'] for x in coverage)),'native_request_replay_coverage_verified':False,'supplemental64_runtime_native_events':0,'observer_attached_to':'vllm.compilation.cuda_graph.CUDAGraphWrapper','production_wrapper_in_source':'vllm_fl.compilation.graph.GraphWrapper','source_mismatch_explains_missing_measurement_not_proof_of_no_native_execution':True,'actual_native_quality_covered':False},'terminal':{'original_owner_at_collection':d['original_owner'],'phase':state['phase'],'updated_utc':state['updated_utc'],'cleanup_complete':True,'remaining_owned':[],'errors':[],'API_returncode':0},'official_math105_metric_gate':True,'native_quality_gate':False,'full_acceptance':False,'merge_qualified':False,'limitations':['Correct counts recomputed from all saved evaluator per-question accuracy grades, not an independent mathematical regrade.','Supplemental parallel64 is diagnostic and has two max-token truncations; failed_predictions=[] does not mean no truncation.','Native host/capture records and observer readiness do not establish native Graph replay/request coverage.','The wrong GraphWrapper class was instrumented, so lack of replay events is an observer limitation, not proof that native code did not execute.','Sampling temperature remains 1.0; differences between 103/105 and 100/105 are not attributed causally to parallelism or kernel accuracy.','Source and library records match a975 throughput, but this run enables diagnostic instrumentation and does not qualify later 638/latest-main sources.','No new throughput, capacity or latest-main comparison was performed by this review.']}
path=Path('/private/tmp/cjj-native-k8-service-math105-root-review-20261028.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(path);print(sha(path));print(json.dumps({k:{n:v[n] for n in ['correct','predictions','exact_accuracy','max_token_truncation_count','request_error_count']} for k,v in results.items()},indent=2))
