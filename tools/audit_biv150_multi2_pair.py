"""Independent CPU audit of the frozen main2412/e08 pair; never runs a model."""
import argparse
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import re

BASELINE='241220723a09061c182764f8f738e5ec01190367'
CANDIDATE='e08c79456f9b943dbc4d9d02d0a4e030b4eceac1'
LIBRARY='5cbb8853a81a4621efc936fa60b952391bed1b56408db1bad62d0031be573dbe'
DRIVER='6f741f12fab105150f8c8b5dc7af7233e6e7865cb27f86706eeaa77ae402c96b'
CONTROLLER='eccb67a6e7a9c60b08b8e0a900102bde40810cb8ee057a3e3266ad59f1d0c87e'
LIMITS={'4096':(256,4809.36,4770.3209),'16384':(128,3364.03,156938.85)}
SERVE=['/usr/local/bin/vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
       '--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}',
       '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
REQUESTS={'decode':'90e8284043b131ead2842c04cf83b5e3af3aa2f79358bee412769b76f61574c2',
          'prefill':'ecf8e319aa48e3c66e8b40019df2dff483ace79dbfdd61de41d436bb3961a42b'}


class AuditError(ValueError):pass


def need(condition,message):
    if not condition:raise AuditError(message)


def reject_constant(value):raise AuditError('Nonfinite JSON value: '+value)


def read(path):return json.loads(Path(path).read_text(),parse_constant=reject_constant)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):return type(value) in (float,int) and math.isfinite(value)
def close(a,b):return finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)


def events(log,marker):
    result=[]
    for line in log.splitlines():
        if marker+' ' not in line:continue
        raw=line.split(marker+' ',1)[1]
        try:value=json.loads(raw,parse_constant=reject_constant)
        except json.JSONDecodeError:value=ast.literal_eval(raw)
        need(isinstance(value,dict),'Non-object event '+marker)
        result.append(value)
    return result


def precision(folder,state):
    reports=list(folder.glob('math500-level3/*/reports/minicpm/math_500.json'))
    need(len(reports)==1,'Expected one original105 report')
    report=reports[0];work=report.parents[2]
    review=work/'reviews/minicpm/math_500_Level 3.jsonl'
    prediction=work/'predictions/minicpm/math_500_Level 3.jsonl'
    reviews=[json.loads(x,parse_constant=reject_constant) for x in review.read_text().splitlines() if x]
    predictions=[json.loads(x,parse_constant=reject_constant) for x in prediction.read_text().splitlines() if x]
    scores={x['sample_score']['sample_metadata']['question_id']:x['sample_score']['score']['value']['accuracy'] for x in reviews}
    ids=[x['metadata']['question_id'] for x in predictions]
    need(len(reviews)==len(predictions)==len(scores)==len(set(ids))==105,'Incomplete or duplicated105 questions')
    need(set(scores)==set(ids),'Review/prediction question mismatch')
    need(all(x in (0,1) for x in scores.values()),'Nonbinary review accuracy')
    need(all(x['sample_score']['score']['status']=='success' for x in reviews),'Failed review status')
    need(all(not x['model_output'].get('error') and x['model_output'].get('choices') for x in predictions),'Failed prediction')
    metrics=[x for x in read(report)['metrics'] if x['identity']['name']=='accuracy']
    correct=sum(scores.values());fraction=correct/105;passed=fraction>=.95
    need(len(metrics)==1 and metrics[0]['num']==105,'Wrong accuracy metric count')
    need(close(metrics[0]['score'],round(fraction,4)),'Report score differs from105 original reviews')
    verdict=read(folder/'accuracy-verdict.json')
    need(verdict['metric_count']==verdict['predictions']==verdict['unique_questions']==105 and not verdict['failed_predictions'],'Invalid stored105 verdict')
    need(close(verdict['score'],metrics[0]['score']),'Stored score changed')
    need(state['accuracy']==verdict and state['precision_passed'] is passed and state['precision_gate'] is passed,'Precision flag does not match original105')
    need(state.get('near_miss_diagnostic') is (not passed),'Near-miss diagnostic flag mismatch')
    return dict(correct=correct,count=105,fraction=fraction,passed=passed,question_scores=scores,
                raw_files=[dict(path=str(p),sha256=sha(p)) for p in (report,review,prediction)])


def visible(text):
    if '<think>' in text:
        if '</think>' not in text:return ''
        text=text.rsplit('</think>',1)[1]
    return text.strip()


def semantics(folder,stored):
    request=read(folder/'requests.json')
    need(all(request[k]['sha256']==v for k,v in REQUESTS.items()),'Peer request fixture identity changed')
    decoded=[];chunks=[];done=False
    for line in (folder/'decode-sse.txt').read_text().splitlines():
        if not line.startswith('data: '):continue
        raw=line[6:].strip()
        if raw=='[DONE]':done=True;continue
        need(not done,'SSE event after DONE')
        event=json.loads(raw,parse_constant=reject_constant);need(not event.get('error'),'SSE error')
        decoded.append(event)
        for choice in event.get('choices',[]):
            text=choice.get('delta',{}).get('content')
            if text:chunks.append(text)
    need(done and decoded,'Incomplete raw SSE')
    transport=read(folder/'decode-response.json');prefill=read(folder/'prefill-response.json')
    need(not prefill.get('error') and prefill.get('choices'),'Prefill transport error')
    need(transport['events']==decoded and transport['content']==chunks,'SSE transport/decoded evidence differs')
    decode_text=visible(''.join(chunks))
    prefill_text=visible(prefill['choices'][0].get('message',{}).get('content') or '')
    need(re.findall(r'(?<!\d)\d+(?!\d)',decode_text)==[str(i) for i in range(1,129)],'Raw decode semantic failure')
    answers=re.findall(r'(?<!\d)\d+(?!\d)',prefill_text)
    need(answers and answers[-1]=='85','Raw prefill semantic failure')
    summary=read(folder/'result.json')
    need(summary==stored,'Stored mixed summary changed')
    need(all(summary[k] is True for k in ('completed','all_correct','decode_correct','prefill_correct','http_overlap')),'Mixed semantic/overlap gate false')
    need(summary['decode_visible']==decode_text and summary['prefill_visible']==prefill_text,'Visible summary differs from raw responses')
    need(summary['prefill_usage']==prefill.get('usage'),'Prefill usage differs')
    start,first,end=(transport[k] for k in ('started','first_content','finished'))
    ps,pe=summary['prefill_started'],summary['prefill_finished']
    need(all(finite(x) for x in (start,first,end,ps,pe)) and 0<start<=first<=ps<end and ps<=pe,'Invalid transport/overlap timing')
    need(summary['decode_started']==start and summary['decode_finished']==end,'Decode timing changed')
    return dict(raw_sse_recomputed=True,decode_correct=True,prefill_correct=True,http_overlap=True,
                files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})


def benchmark(folder,name,length,output,count,seed):
    run=folder/name;command=read(run/'command.json')
    expected=['vllm','bench','serve','--backend','vllm','--model','minicpm','--tokenizer','/workspace/MiniCPM5-2B',
        '--endpoint','/v1/completions','--host','127.0.0.1','--port','9031','--dataset-name','random','--ignore-eos',
        '--random-input-len',str(length),'--random-output-len',str(output),'--max-concurrency','64','--num-prompts',str(count),
        '--seed',str(seed),'--disable-tqdm','--save-result','--result-dir',command[command.index('--result-dir')+1],'--label',name]
    need(command==expected and Path(command[command.index('--result-dir')+1]).name==name,'Official benchmark command changed: '+name)
    files=[p for p in run.glob('*.json') if p.name!='command.json']
    need(len(files)==1,'Expected one original benchmark result: '+name)
    value=read(files[0])
    required=dict(num_prompts=count,max_concurrency=64,completed=count,failed=0,
                  total_input_tokens=length*count,total_output_tokens=output*count)
    need(all(value.get(k)==v for k,v in required.items()),'Official request/token count failure: '+name)
    need(all(finite(value.get(k)) and value[k]>0 for k in ('duration','total_token_throughput','mean_ttft_ms')),'Nonfinite/nonpositive timing: '+name)
    need(close(value['total_token_throughput'],(value['total_input_tokens']+value['total_output_tokens'])/value['duration']),'Throughput/duration/token arithmetic differs: '+name)
    return value,dict(path=str(files[0]),sha256=sha(files[0]),command_sha256=sha(run/'command.json'))


def working_gate(score,length):
    count,baseline,limit=LIMITS[length]
    requests=all(score.get(k)==v for k,v in dict(num_prompts=count,max_concurrency=64,completed=count,failed=0,
                 total_input_tokens=int(length)*count,total_output_tokens=1024*count).items())
    tps=score['total_token_throughput'];ttft=score['mean_ttft_ms']
    throughput=finite(tps) and tps>=baseline*.99;latency=finite(ttft) and 0<ttft<=limit
    return dict(requests_ok=requests,throughput_ok=throughput,ttft_ok=latency,passed=requests and throughput and latency,
        baseline_total_tokens_per_s=baseline,throughput_floor=baseline*.99,working_mean_ttft_limit_ms=limit,
        goal_20_percent=finite(tps) and tps>=baseline*1.20)


def runtime(log,candidate):
    for marker in ('BIV150_COMPUTE_ORDER_SCOPE_REJECTED','BIV150_PREFILL_MULTI2_FALLBACK','BIV150_PREFILL_Q32_FALLBACK'):
        need(marker not in log,'Runtime rejected/fell back: '+marker)
    q32=events(log,'BIV150_PREFILL_Q32_INSTALL')
    need(q32 and all(x['installed'] is True for x in q32),'Original Q32 parent not installed')
    order=events(log,'BIV150_COMPUTE_ORDER_FIRST_SELECTED')
    need(order,'No actual C3 selection')
    for row in order:
        need(row['original_model_forward_returned'] is True and row['runtime_mode']=='NONE'
             and row['inverse_logits_order']=='original_request_rows','C3 model/ordering protocol changed')
        need(row['sampler_rows']==row['N'] and row['M']>=row['N']>0
             and 0<=row['Dprefix']<row['Dall']<=row['N']
             and row['outside']==row['Dall']-row['Dprefix']>0,'C3 selection geometry invalid')
    installed=events(log,'BIV150_PREFILL_MULTI2_INSTALL');selected=events(log,'BIV150_PREFILL_MULTI2_SELECTED')
    if not candidate:
        need(not installed and not selected,'Baseline contains MULTI2 implementation')
    else:
        need(installed and selected and all(x['installed'] is True for x in installed),'Candidate MULTI2 not installed/selected')
        records={x['candidate_hash']:x for x in events(log,'BIV150_Q_FRAGMENT_SELECTED') if x.get('mode')=='selected'}
        for row in selected:
            need(row['num_sequences']==2 and row['rows']>=1024 and tuple(row['grid'])==(row['rows']//32+2,2),'Wrong generic MULTI2 scope/grid')
            need(tuple(row[k] for k in ('block_m','block_q','num_warps','tile_size','num_stages'))==(256,32,16,64,1),'Wrong MULTI2 configuration')
            need(row['hash'] in records,'Selected kernel has no actual compiler proof')
            record=records[row['hash']];proof=record['mechanism'];meta=record['candidate_resources']['metadata']
            need(proof['mechanism_pass'] is True and proof['ttgir_q_load_outside_loop'] is True
                 and proof['llvm_q_memory_reads_in_loop']==0 and proof['llvm_q_memory_reads_total']>0,'Actual lowered proof failed')
            need((meta['num_warps'],meta['num_stages'])==(16,1),'Actual compiler options differ')
    return dict(q32_parent_installed=True,q32_selected_events_allowed=len(events(log,'BIV150_PREFILL_Q32_SELECTED')),
                c3=order,multi2_installations=installed,multi2_selections=selected)


def arm(folder,commit,candidate):
    state=read(folder/'state.json');manifest=read(folder/'manifest.json');log=(folder/'server.log').read_text()
    need(state['phase']=='completed' and state['complete'] is True and state['cleanup_complete'] is True,'Arm not completely cleaned up')
    need(state['cleanup_remaining']==[] and state['port_9031_closed'] is True and state['owned_service_members'],'Owned cleanup evidence missing')
    need(all(type(x['pid']) is int and x['pid']>0 and type(x['start_ticks']) is int and x['start_ticks']>0 for x in state['owned_service_members']),'Invalid owned process identities')
    need(state['commit']==manifest['commit']==commit,'Model source commit mismatch')
    need(state['semantic_passed'] is True and state['post_score_semantic_passed'] is True,'State semantic gate false')
    need(manifest['command']==SERVE and manifest['production_defaults'] is True,'Official/default serving command changed')
    need(manifest['native_mode']=='on' and manifest['stable_small_o']==manifest['out_lowering']==manifest['large_linear']=='1','Team production paths disabled')
    need(all(manifest[k] is True for k in ('run_accuracy','run_throughput','full_workload_warmup','post_score_semantic_check')),'Missing complete acceptance protocol')
    need(manifest['gate_library_sha256']==LIBRARY and manifest['driver_sha256']==DRIVER,'Frozen native library/driver identity mismatch')
    need(manifest['prefill_multi2_mode']=='default' and manifest['prefill_multi2_available'] is candidate,'Wrong default/baseline-module mode')
    env=manifest['env']
    need(env['PYTHONPATH']==manifest['source'] if 'PYTHONPATH' in env else True,'Loaded-source PYTHONPATH mismatch')
    need(not any(k in env for k in ('VLLM_FL_BIV150_PREFILL_MULTI2','VLLM_FL_BIV150_PREFILL_Q32',
         'VLLM_FL_BIV150_SMALL_OUT_LOWERING','VLLM_FL_BIV150_LARGE_LINEAR')),'Manual default feature env present')
    need(env['VLLM_PLUGINS']=='fl' and env['VLLM_FL_BIV150_NATIVE_W8A8']=='1' and env['VLLM_FL_BIV150_STABLE_SMALL_O']=='1','Plugin/native setup differs')
    actual=events(log,'BIV150_COMPILE_SEMANTICS');expected=manifest['expected_source_fingerprint']
    need(expected.get('schema')==1 and expected.get('native_enabled') is True
         and re.fullmatch(r'[0-9a-f]{64}',expected.get('sha256','')),'Invalid expected source fingerprint')
    need(manifest['loaded_source_fingerprint_verified'] is True and actual and all(x==expected for x in actual),'Actual source fingerprint mismatch')
    accuracy=precision(folder,state)
    need(state['diagnostic_scores_only'] is (not accuracy['passed']),'Diagnostic score labeling mismatch')
    initial=semantics(folder/'mixed-requests',state['diagnostic'])
    post=semantics(folder/'post-score-semantic/mixed-requests',state['post_score_diagnostic'])
    warmups=read(folder/'warmups.json');stored=read(folder/'scores.json');scores={};gates={};files={}
    need({p.name for p in folder.glob('score-*') if p.is_dir()}=={'score-4096','score-16384'},'Extra/missing scored rounds')
    for length,(count,_,_) in LIMITS.items():
        need(len(warmups[length])==4 and warmups[length][-1].get('full_workload') is True,'Incomplete warmup protocol')
        for index in range(3):
            value,_=benchmark(folder,f'warm-{length}-{index}',int(length),128,64,920000+index)
            need(value==warmups[length][index],'Short-warm summary changed')
        value,_=benchmark(folder,f'full-warm-{length}',int(length),1024,count,0)
        need(dict(value,full_workload=True)==warmups[length][-1],'Full-warm summary changed')
        score,raw=benchmark(folder,'score-'+length,int(length),1024,count,0)
        need(score==stored[length]==state['scores'][length],'Score summary differs from original result')
        gate=working_gate(score,length)
        need(gate==state['working_gates'][length],'Stored working gate differs from independent recomputation')
        scores[length]=score;gates[length]=gate;files[length]=raw
    path_proof=runtime(log,candidate)
    need(manifest['prefill_multi2_selected']==path_proof['multi2_selections'],'Manifest selection events changed')
    if candidate:need(manifest['compute_order_preserved'] is True,'Missing C3 preservation verdict')
    need(manifest['tuning_snapshot'] and all(re.fullmatch(r'[0-9a-f]{64}',x['sha256']) for x in manifest['tuning_snapshot']),'Invalid initial snapshot manifest')
    return dict(state=state,manifest=manifest,accuracy=accuracy,scores=scores,working_gates=gates,
                semantics=dict(initial=initial,post_score=post),runtime=path_proof,raw_score_files=files,
                evidence_sha256={p.name:sha(p) for p in (folder/'state.json',folder/'manifest.json',folder/'server.log')})


def audit(root,baseline_commit=BASELINE,candidate_commit=CANDIDATE):
    root=Path(root);control=read(root/'state.json')
    need(control['phase']=='completed','Controller completion is required, but is not an acceptance gate')
    need(control['baseline_commit']==baseline_commit and control['candidate_commit']==candidate_commit,'Controller model source mismatch')
    need(control['tool_sha256']==DRIVER and control['controller_sha256']==CONTROLLER and control['returncode']==0,'Frozen controller/driver identity or exit mismatch')
    baseline=arm(root/'baseline',baseline_commit,False);candidate=arm(root/'candidate',candidate_commit,True)
    command=control['command']
    need(command[command.index('--source')+1]==candidate['manifest']['source']
         and command[command.index('--commit')+1]==candidate_commit,'Controller child source differs')
    for label,value in (('baseline',baseline),('candidate',candidate)):
        need(control['results'][label]['state']==value['state'] and control['results'][label]['manifest']==value['manifest'],'Controller embedded arm snapshot differs')
    for key in ('command','tuning_snapshot','gate_library_sha256','driver_sha256','diagnostic_sha256'):
        need(baseline['manifest'][key]==candidate['manifest'][key],'Matched-arm identity differs: '+key)
    need(baseline['manifest']['expected_source_fingerprint']!=candidate['manifest']['expected_source_fingerprint'],'Different frozen source trees share a fingerprint')
    need(set(baseline['accuracy']['question_scores'])==set(candidate['accuracy']['question_scores']),'Arms used different105 question sets')
    comparison={}
    for length in LIMITS:
        a=baseline['scores'][length]['total_token_throughput'];b=candidate['scores'][length]['total_token_throughput'];change=b/a-1
        stored=control['comparison'][length]
        need(close(stored['baseline_tps'],a) and close(stored['candidate_tps'],b) and close(stored['observed_relative_change'],change),'Controller comparison differs from raw scores')
        comparison[length]=dict(baseline_tps=a,candidate_tps=b,observed_relative_change=change,higher=b>a,
            baseline_mean_ttft_ms=baseline['scores'][length]['mean_ttft_ms'],candidate_mean_ttft_ms=candidate['scores'][length]['mean_ttft_ms'])
    precision_gate=all(x['accuracy']['passed'] for x in (baseline,candidate))
    formal=precision_gate and all(g['passed'] for x in (baseline,candidate) for g in x['working_gates'].values())
    eligible=formal and all(x['higher'] for x in comparison.values())
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),cpu_only=True,
        structural_integrity_passed=True,precision_gate=precision_gate,formal_gates_passed=formal,
        eligible_for_agent_merge_review_on_frozen_pair=eligible,automatic_merge_performed=False,
        baseline=baseline,candidate=candidate,comparison=comparison,
        paired_question_changes=[dict(question_id=k,baseline=v,candidate=candidate['accuracy']['question_scores'][k])
            for k,v in baseline['accuracy']['question_scores'].items() if v!=candidate['accuracy']['question_scores'][k]],
        caveats=['Completed/exit0 is not acceptance;99/105 remains diagnostic and cannot accept.',
            'One pair is not statistical significance or a stable effect estimate.',
            'Host/driver declarations and raw records are checked; no live process/GPU inspection or model rerun is performed.',
            'Fresh team-main synchronization and human/agent final merge review remain separate.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline-commit',default=BASELINE);p.add_argument('--candidate-commit',default=CANDIDATE)
    args=p.parse_args()
    try:
        result=audit(args.root,args.baseline_commit,args.candidate_commit)
    except Exception as error:
        result=dict(cpu_only=True,structural_integrity_passed=False,precision_gate=False,
                    formal_gates_passed=False,eligible_for_agent_merge_review_on_frozen_pair=False,error=repr(error))
    result['audit_script_sha256']=sha(Path(__file__))
    with args.out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result.get(k) for k in ('structural_integrity_passed','precision_gate','formal_gates_passed','eligible_for_agent_merge_review_on_frozen_pair','error')}))
    raise SystemExit(0 if result['eligible_for_agent_merge_review_on_frozen_pair'] else 1)
