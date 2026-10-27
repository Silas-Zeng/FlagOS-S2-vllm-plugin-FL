"""Synthetic v2 audit regression including actual Python repr tuple grids; no GPU/model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import audit_biv150_multi2_pair_v2 as audit
import audit_biv150_multi2_pair as audit_v1

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def mixed(folder):
    text=' '.join(str(i) for i in range(1,129))
    event={'choices':[{'delta':{'content':text}}]}
    folder.mkdir(parents=True)
    write(folder/'requests.json',{k:dict(sha256=v,payload={}) for k,v in audit.REQUESTS.items()})
    (folder/'decode-sse.txt').write_text('data: '+json.dumps(event)+'\n\ndata: [DONE]\n')
    write(folder/'decode-response.json',dict(events=[event],content=[text],started=1.,first_content=2.,finished=4.))
    usage=dict(prompt_tokens=100,completion_tokens=1)
    write(folder/'prefill-response.json',dict(choices=[dict(message=dict(content='85'))],usage=usage))
    summary=dict(completed=True,all_correct=True,decode_correct=True,prefill_correct=True,
        decode_visible=text,prefill_visible='85',prefill_usage=usage,prefill_started=2.1,
        prefill_finished=3.,decode_started=1.,decode_finished=4.,http_overlap=True)
    write(folder/'result.json',summary)
    return summary


def score(folder,name,length,output,count,seed,tps):
    value=dict(num_prompts=count,max_concurrency=64,completed=count,failed=0,
        total_input_tokens=length*count,total_output_tokens=output*count,
        duration=(length+output)*count/tps,total_token_throughput=tps,mean_ttft_ms=1000.,mean_tpot_ms=1.)
    command=['vllm','bench','serve','--backend','vllm','--model','minicpm','--tokenizer','/workspace/MiniCPM5-2B',
        '--endpoint','/v1/completions','--host','127.0.0.1','--port','9031','--dataset-name','random','--ignore-eos',
        '--random-input-len',str(length),'--random-output-len',str(output),'--max-concurrency','64','--num-prompts',str(count),
        '--seed',str(seed),'--disable-tqdm','--save-result','--result-dir','/remote/'+folder.name+'/'+name,'--label',name]
    write(folder/name/'command.json',command);write(folder/name/'original.json',value)
    return value


def accuracy(folder,correct):
    work=folder/'math500-level3'/'timestamp'
    reports=work/'reports/minicpm/math_500.json'
    metric=dict(identity=dict(name='accuracy'),num=105,score=round(correct/105,4))
    write(reports,dict(metrics=[metric]))
    reviews=[dict(sample_score=dict(sample_metadata=dict(question_id=str(i)),
        score=dict(value=dict(accuracy=int(i<correct)),status='success'))) for i in range(105)]
    predictions=[dict(metadata=dict(question_id=str(i)),model_output=dict(choices=[dict(message=dict(content='synthetic'))])) for i in range(105)]
    for path,rows in ((work/'reviews/minicpm/math_500_Level 3.jsonl',reviews),
                      (work/'predictions/minicpm/math_500_Level 3.jsonl',predictions)):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    verdict=dict(metric_count=105,predictions=105,unique_questions=105,failed_predictions=[],score=metric['score'],report=str(reports))
    write(folder/'accuracy-verdict.json',verdict)
    return verdict


def make_arm(folder,candidate):
    folder.mkdir()
    manifest=dict(commit=audit.CANDIDATE if candidate else audit.BASELINE,source='/remote/e08' if candidate else '/remote/main2412',
        command=audit.SERVE,production_defaults=True,native_mode='on',stable_small_o='1',out_lowering='1',large_linear='1',
        run_accuracy=True,run_throughput=True,full_workload_warmup=True,post_score_semantic_check=True,
        gate_library_sha256=audit.LIBRARY,driver_sha256=audit.DRIVER,diagnostic_sha256='3'*64,
        prefill_multi2_mode='default',prefill_multi2_available=candidate,
        env=dict(VLLM_PLUGINS='fl',VLLM_FL_BIV150_NATIVE_W8A8='1',VLLM_FL_BIV150_STABLE_SMALL_O='1'),
        loaded_source_fingerprint_verified=True,expected_source_fingerprint=dict(schema=1,sha256=('2' if candidate else '1')*64,native_enabled=True),
        tuning_snapshot=[dict(name='initial.db',sha256='4'*64)],prefill_multi2_selected=[])
    initial=mixed(folder/'mixed-requests');post=mixed(folder/'post-score-semantic/mixed-requests')
    warmups={};scores={};gates={}
    for length,(count,_,_) in audit.LIMITS.items():
        tps=6100. if candidate else 6000.
        warmups[length]=[score(folder,f'warm-{length}-{i}',int(length),128,64,920000+i,tps) for i in range(3)]
        value=score(folder,f'full-warm-{length}',int(length),1024,count,0,tps)
        warmups[length].append(dict(value,full_workload=True))
        scores[length]=score(folder,'score-'+length,int(length),1024,count,0,tps)
        gates[length]=audit.working_gate(scores[length],length)
    write(folder/'warmups.json',warmups);write(folder/'scores.json',scores)
    verdict=accuracy(folder,100)
    state=dict(phase='completed',complete=True,cleanup_complete=True,cleanup_remaining=[],port_9031_closed=True,
        owned_service_members=[dict(pid=101,start_ticks=123,sid=101)],commit=manifest['commit'],
        semantic_passed=True,post_score_semantic_passed=True,accuracy=verdict,precision_passed=True,
        precision_gate=True,near_miss_diagnostic=False,diagnostic_scores_only=False,
        diagnostic=initial,post_score_diagnostic=post,scores=scores,working_gates=gates)
    lines=[('BIV150_COMPILE_SEMANTICS',manifest['expected_source_fingerprint']),
        ('BIV150_PREFILL_Q32_INSTALL',dict(installed=True)),('BIV150_PREFILL_Q32_SELECTED',dict(rows=2048)),
        ('BIV150_COMPUTE_ORDER_FIRST_SELECTED',dict(original_model_forward_returned=True,runtime_mode='NONE',
            inverse_logits_order='original_request_rows',sampler_rows=3,N=3,M=1026,Dprefix=0,Dall=1,outside=1))]
    if candidate:
        selected=dict(hash='a'*64,num_sequences=2,rows=1026,grid=[34,2],block_m=256,block_q=32,num_warps=16,tile_size=64,num_stages=1)
        manifest['prefill_multi2_selected']=[selected];manifest['compute_order_preserved']=True
        lines.extend([('BIV150_PREFILL_MULTI2_INSTALL',dict(installed=True)),
            ('BIV150_PREFILL_MULTI2_SELECTED',selected),
            ('BIV150_Q_FRAGMENT_SELECTED',dict(mode='selected',candidate_hash=selected['hash'],
                mechanism=dict(mechanism_pass=True,ttgir_q_load_outside_loop=True,llvm_q_memory_reads_in_loop=0,llvm_q_memory_reads_total=4),
                candidate_resources=dict(metadata=dict(num_warps=16,num_stages=1))))])
    (folder/'server.log').write_text(''.join(marker+' '+json.dumps(value)+'\n' for marker,value in lines))
    write(folder/'state.json',state);write(folder/'manifest.json',manifest)


def controller(root):
    states={name:dict(state=audit.read(root/name/'state.json'),manifest=audit.read(root/name/'manifest.json')) for name in ('baseline','candidate')}
    comparison={s:dict(baseline_tps=states['baseline']['state']['scores'][s]['total_token_throughput'],
        candidate_tps=states['candidate']['state']['scores'][s]['total_token_throughput'],
        observed_relative_change=states['candidate']['state']['scores'][s]['total_token_throughput']/states['baseline']['state']['scores'][s]['total_token_throughput']-1) for s in audit.LIMITS}
    write(root/'state.json',dict(phase='completed',baseline_commit=audit.BASELINE,candidate_commit=audit.CANDIDATE,
        tool_sha256=audit.DRIVER,controller_sha256=audit.CONTROLLER,returncode=0,results=states,comparison=comparison,
        command=['python','driver','--source',states['candidate']['manifest']['source'],'--commit',audit.CANDIDATE]))


def replace_event(root,marker,change):
    path=root/'candidate/server.log';lines=[]
    for line in path.read_text().splitlines():
        if line.startswith(marker+' '):
            value=json.loads(line[len(marker)+1:]);change(value);line=marker+' '+json.dumps(value)
        lines.append(line)
    path.write_text('\n'.join(lines)+'\n')


def main(actual_manifest=None,actual_log=None):
    rejected=[]
    actual_fixture=None
    with tempfile.TemporaryDirectory(prefix='multi2-final-audit-cpu-') as temporary:
        base=Path(temporary)/'valid';base.mkdir();make_arm(base/'baseline',False);make_arm(base/'candidate',True);controller(base)
        valid=audit.audit(base)
        assert valid['eligible_for_agent_merge_review_on_frozen_pair']
        assert valid['baseline']['runtime']['q32_selected_events_allowed']==1
        cli=subprocess.run([sys.executable,str(ROOT/'tools/audit_biv150_multi2_pair_v2.py'),
            '--root',str(base),'--out',str(Path(temporary)/'valid-audit.json')],capture_output=True,text=True)
        assert cli.returncode==0
        # The actual service uses print(dict), whose grid is a tuple. Manifest is JSON.
        repr_root=Path(temporary)/'python-repr';shutil.copytree(base,repr_root)
        marker='BIV150_PREFILL_MULTI2_SELECTED '
        path=repr_root/'candidate/server.log';lines=[]
        for line in path.read_text().splitlines():
            if line.startswith(marker):
                value=json.loads(line[len(marker):]);value['grid']=tuple(value['grid'])
                line=marker+repr(value)
            lines.append(line)
        path.write_text('\n'.join(lines)+'\n')
        try:audit_v1.audit(repr_root)
        except audit_v1.AuditError as error:assert str(error)=='Manifest selection events changed'
        else:raise AssertionError('v1 failed to reproduce the actual tuple/list mismatch')
        assert audit.audit(repr_root)['eligible_for_agent_merge_review_on_frozen_pair']
        cli=subprocess.run([sys.executable,str(ROOT/'tools/audit_biv150_multi2_pair_v2.py'),
            '--root',str(repr_root),'--out',str(Path(temporary)/'repr-audit.json')],capture_output=True,text=True)
        assert cli.returncode==0
        representation_rejected=[]
        for fault in ('log-grid','log-hash','log-extra','manifest-grid','manifest-hash','manifest-extra','manifest-only-field'):
            root=Path(temporary)/fault;shutil.copytree(repr_root,root)
            folder=root/'candidate'
            if fault.startswith('log-'):
                path=folder/'server.log';lines=[]
                for line in path.read_text().splitlines():
                    if line.startswith(marker):
                        value=audit_v1.ast.literal_eval(line[len(marker):])
                        if fault=='log-grid':value['grid']=(33,2)
                        elif fault=='log-hash':value['hash']='b'*64
                        else:value['unexpected_extra']=1
                        line=marker+repr(value)
                    lines.append(line)
                path.write_text('\n'.join(lines)+'\n')
            else:
                manifest=audit.read(folder/'manifest.json');selection=manifest['prefill_multi2_selected'][0]
                if fault=='manifest-grid':selection['grid']=[33,2]
                elif fault=='manifest-hash':selection['hash']='b'*64
                elif fault=='manifest-extra':selection['unexpected_extra']=1
                else:selection['block_m']=128
                write(folder/'manifest.json',manifest);controller(root)
            try:audit.audit(root)
            except audit.AuditError:pass
            else:raise AssertionError('Representation mutant accepted: '+fault)
            representation_rejected.append(fault)
        cases=['precision99','forged-precision99','ttft','working-gate','count','grid','missing-selected','baseline-multi2',
               'fingerprint','manual-env','cleanup','duplicate105','raw-sse','snapshot','serve-cli','comparison']
        for case in cases:
            root=Path(temporary)/case;shutil.copytree(base,root)
            folder=root/'candidate';state=audit.read(folder/'state.json');manifest=audit.read(folder/'manifest.json')
            if case in ('precision99','forged-precision99'):
                verdict=accuracy(folder,99);state.update(accuracy=verdict,precision_passed=False,precision_gate=False,
                    near_miss_diagnostic=True,diagnostic_scores_only=True)
                if case=='forged-precision99':state.update(precision_passed=True,precision_gate=True,near_miss_diagnostic=False,diagnostic_scores_only=False)
            elif case=='ttft':
                value=audit.read(folder/'score-4096/original.json');value['mean_ttft_ms']=1e9
                write(folder/'score-4096/original.json',value);stored=audit.read(folder/'scores.json');stored['4096']=value
                write(folder/'scores.json',stored);state['scores']['4096']=value;state['working_gates']['4096']=audit.working_gate(value,'4096')
            elif case=='working-gate':state['working_gates']['4096']['passed']=False
            elif case=='count':
                value=audit.read(folder/'score-4096/original.json');value['total_input_tokens']-=1;write(folder/'score-4096/original.json',value)
            elif case=='grid':replace_event(root,'BIV150_PREFILL_MULTI2_SELECTED',lambda e:e.update(grid=[33,2]))
            elif case=='missing-selected':
                path=folder/'server.log';path.write_text('\n'.join(x for x in path.read_text().splitlines() if not x.startswith('BIV150_PREFILL_MULTI2_SELECTED '))+'\n')
            elif case=='baseline-multi2':
                with (root/'baseline/server.log').open('a') as f:f.write('BIV150_PREFILL_MULTI2_INSTALL {"installed":true}\n')
            elif case=='fingerprint':replace_event(root,'BIV150_COMPILE_SEMANTICS',lambda e:e.update(sha256='9'*64))
            elif case=='manual-env':manifest['env']['VLLM_FL_BIV150_PREFILL_MULTI2']='1'
            elif case=='cleanup':state['cleanup_complete']=False
            elif case=='duplicate105':
                p=next(folder.glob('math500-level3/*/reviews/minicpm/*.jsonl'));rows=p.read_text().splitlines();rows[-1]=rows[0];p.write_text('\n'.join(rows)+'\n')
            elif case=='raw-sse':
                p=folder/'mixed-requests/decode-sse.txt';p.write_text(p.read_text().replace('127 128','127'))
            elif case=='snapshot':manifest['tuning_snapshot'][0]['sha256']='f'*64
            elif case=='serve-cli':manifest['command']=[*manifest['command'],'--max-num-seqs','32']
            write(folder/'state.json',state);write(folder/'manifest.json',manifest);controller(root)
            if case=='comparison':
                d=audit.read(root/'state.json');d['comparison']['4096']['observed_relative_change']=.99;write(root/'state.json',d)
            try:result=audit.audit(root)
            except (audit.AuditError,KeyError):pass
            else:assert not result['eligible_for_agent_merge_review_on_frozen_pair'],case
            if case=='precision99':
                out=Path(temporary)/'near99-audit.json'
                cli=subprocess.run([sys.executable,str(ROOT/'tools/audit_biv150_multi2_pair_v2.py'),
                    '--root',str(root),'--out',str(out)],capture_output=True,text=True)
                assert cli.returncode==1 and audit.read(out)['precision_gate'] is False
            rejected.append(case)
    if actual_manifest is not None or actual_log is not None:
        assert actual_manifest is not None and actual_log is not None
        manifest_sha=hashlib.sha256(actual_manifest.read_bytes()).hexdigest()
        log_sha=hashlib.sha256(actual_log.read_bytes()).hexdigest()
        assert manifest_sha=='8ca944da28bf9a9a7906b91b3033664fc778d7071a1ae7b28a486820e4af6dba'
        assert log_sha=='5036196e7211db93f651c485fd4f1c77cfa45b5706d08793d3f7dc766ae52241'
        manifest=audit.read(actual_manifest);proof=audit.runtime(actual_log.read_text(),True)
        assert len(proof['multi2_selections'])==1
        assert type(proof['multi2_selections'][0]['grid']) is tuple
        assert type(manifest['prefill_multi2_selected'][0]['grid']) is list
        # Execute exactly the manifest-selection comparison statements from each source,
        # after the unchanged runtime() validates the complete actual selection/proof.
        for version,module in ((1,audit_v1),(2,audit)):
            tree=ast.parse(Path(module.__file__).read_text())
            arm=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arm')
            nodes=[]
            for node in arm.body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='validated_selections' for t in node.targets):nodes.append(node)
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and any(isinstance(a,ast.Constant) and a.value=='Manifest selection events changed' for a in node.value.args):nodes.append(node)
            assert len(nodes)==(1 if version==1 else 2)
            namespace=dict(manifest=manifest,path_proof=proof,need=module.need)
            try:exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-selection-comparison>','exec'),namespace)
            except module.AuditError as error:
                assert version==1 and str(error)=='Manifest selection events changed'
            else:assert version==2
        assert hashlib.sha256(actual_manifest.read_bytes()).hexdigest()==manifest_sha
        assert hashlib.sha256(actual_log.read_bytes()).hexdigest()==log_sha
        actual_fixture=dict(manifest_sha256=manifest_sha,server_log_sha256=log_sha,
            runtime_strict_checks_passed=True,v1_mismatch_reproduced=True,v2_complete_dict_comparison_passed=True,
            selections=1,original_raw_unchanged=True,full_pair_audited=False)
    result=dict(passed=True,cpu_only=True,gpu_executed=False,synthetic_fixtures=True,precision_gate=False,
        audit_version=2,actual_fixture=actual_fixture,minimal_valid_fixture_accepted=True,existing_baseline_q32_allowed=True,rejected=rejected,
        original_json_positive=True,python_repr_tuple_positive=True,v1_actual_mismatch_reproduced=True,
        representation_rejected=representation_rejected,
        cli_valid_exits0_and_near99_exits1=True,
        model_evaluation_performed=False,model_source_frozen=audit.CANDIDATE,
        limitation='Synthetic archive protocol/fault tests only; no real run, model precision or throughput claim.')
    write(ROOT/'docs/evidence/prefill-multi2-20261027/final-auditor-v2-cpu.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actual-manifest',type=Path)
    p.add_argument('--actual-log',type=Path)
    args=p.parse_args()
    main(args.actual_manifest,args.actual_log)
