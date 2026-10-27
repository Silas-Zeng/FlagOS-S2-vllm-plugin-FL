#!/usr/bin/env python3
"""Independent CPU record/protocol/score/runtime replay; never execute a model/cache."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'docs/evidence/stock-qk8-service-diagnostic-20261027/model105-records-v3'
SOURCE='a48cf1c82f8665b7e70e3e70fea5e0845eeee504'


def checked(data,item):
    assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];return data


def run(folder=FOLDER):
    index=json.loads((folder/'index.json').read_text());assert index['source_commit']==SOURCE
    files={}
    for part in index['archives']:
        data=checked((folder/part['file']).read_bytes(),part);assert len(data)<=2*1024*1024
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
            members=tar.getmembers();assert len(members)==part['members']
            for member in members:
                name=PurePosixPath(member.name)
                assert member.isfile() and not name.is_absolute() and '..' not in name.parts and member.name not in files
                files[member.name]=tar.extractfile(member).read()
    assert len(files)==index['payload_files']
    download=json.loads(checked(files['raw/download-manifest.json'],index['original_download_manifest']))
    assert download['raw_files']==index['original_raw_files']==len(download['files'])==18
    assert download['raw_bytes']==index['original_raw_bytes']==sum(v['bytes'] for v in download['files'])==4325172
    assert download['original_owner']==index['original_owner']==dict(pid=629199,start_ticks=344628210,same_start=True,state='Z')
    expected={'raw/download-manifest.json'}
    for item in download['files']:
        name='raw/'+item['relative'];checked(files[name],item);expected.add(name)
    for item in index['critical_git_source_files']:
        name='source/'+item['path'];checked(files[name],item);expected.add(name)
        assert files[name]==subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+item['path']])
    assert set(files)==expected and index['omitted_record_files']==[]
    state=json.loads(files['raw/out/state.json']);manifest=json.loads(files['raw/out/manifest.json'])
    verdict=json.loads(files['raw/out/accuracy-verdict.json']);cleanup=json.loads(files['raw/out/owned-cleanup.json'])
    assert state['completed'] and state['passed'] and state['valid_pure_decode_diagnostic'] and state['pure_decode_math105_gate_passed']
    assert not state['errors'] and state['cleanup_complete'] and cleanup['complete']
    assert not cleanup['remaining_owned'] and not cleanup['unobserved_same_session'] and not cleanup['signal_errors']
    assert cleanup['pid_reuse_never_signalled'] and cleanup['group_signal_used'] is False
    assert not any(state[k] for k in ('precision_gate','full_prefill_precision_passed','encoded_cache_precision_passed','throughput_measured'))
    assert manifest['source_commit']==SOURCE and manifest['frozen_production']==index['production_base']
    assert manifest['production_defaults_preserved'] and manifest['diagnostic_only'] and not manifest['full_prefill_quantization']
    symlinks=[]
    for name,digest in manifest['source_hashes'].items():
        resolved=name
        for _ in range(8):
            data=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+resolved])
            mode=subprocess.check_output(['git','-C',str(ROOT),'ls-tree',SOURCE,'--',resolved],text=True).split()[0]
            if mode!='120000':break
            target=data.decode();assert not target.startswith('/')
            new=posixpath.normpath(posixpath.join(posixpath.dirname(resolved),target))
            assert not new.startswith('../') and not new.startswith('/')
            symlinks.append(dict(path=resolved,target=target,resolved=new));resolved=new
        else:raise AssertionError('Git source symlink cycle')
        # The controller hashes file bytes through tracked symlinks, not the
        # Git symlink-text blob. Resolve against this same frozen Git tree.
        assert hashlib.sha256(data).hexdigest()==digest
    controller=ROOT/'tools/run_biv150_stock_decode_105.py'
    assert controller.read_bytes()==files['source/tools/run_biv150_stock_decode_105.py']
    spec=importlib.util.spec_from_file_location('model105_frozen_cpu_replay',controller);tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    assert hashlib.sha256(controller.read_bytes()).hexdigest()==manifest['controller_sha256']
    assert hashlib.sha256(tool.protocol_source(ROOT).encode()).hexdigest()==manifest['protocol_function_sha256']==tool.PROTOCOL_SHA
    assert manifest['command']==tool.SERVICE_COMMAND and manifest['native_library_sha256']==tool.NATIVE_SHA
    assert manifest['expected_compile_semantics_on']!=manifest['expected_compile_semantics_off']
    assert manifest['actual_package_resolution']['origin']==str(Path(manifest['source'])/'vllm_fl/__init__.py')
    assert manifest['actual_PYTHONPATH']==manifest['source'] and manifest['env'][tool.ENABLE]=='1'
    for row in manifest['tuning_snapshot']:
        assert (row['sha256'],row['bytes'])==tool.SNAPSHOT[row['name']] and row['quick_check']=='ok'
    assert len(manifest['tuning_snapshot'])==2
    assert hashlib.sha256(files['raw/out/micro-certificate.json']).hexdigest()==manifest['micro_certificate_sha256']=='ee20e9c7879b18ee4a6e28f85878cd71677aa9b75b05566060d425580e466d8b'
    owner=json.loads(next(data for name,data in files.items() if name.endswith('-owner.json')))
    assert (owner['pid'],owner['start_ticks'],owner['source_commit'])==(629199,344628210,SOURCE)
    command=json.loads(files['raw/out/math500-level3/command.json'])
    assert command==manifest['accuracy_command']==tool.accuracy_command(Path(owner['out']))
    actual=json.loads(json.dumps(tool.runtime_evidence(files['raw/out/server.log'].decode(),manifest['expected_compile_semantics_on'])))
    assert actual==manifest['runtime_after_eval'] and actual['diagnostic_on_bit_verified']
    assert actual['selected_host_submissions_lower_bound']==4096 and len(actual['actual_kernel_hashes'])==4
    predname=next(n for n in files if '/predictions/' in n and n.endswith('.jsonl'))
    reviewname=next(n for n in files if '/reviews/' in n and n.endswith('.jsonl'))
    reportname=next(n for n in files if '/reports/minicpm/' in n and n.endswith('math_500.json'))
    predictions=[json.loads(s) for s in files[predname].decode().splitlines() if s.strip()]
    reviews=[json.loads(s) for s in files[reviewname].decode().splitlines() if s.strip()]
    assert len(predictions)==len(reviews)==105
    indices={r['index'] for r in predictions};questions={r['metadata']['question_id'] for r in predictions}
    assert indices==set(range(105)) and len(questions)==105
    assert {r['index'] for r in reviews}==indices
    by_index={r['index']:r for r in reviews};stop=Counter();truncated=[];correct=0
    for row in predictions:
        output=row['model_output'];assert not output.get('error') and len(output.get('choices',[]))==1
        choice=output['choices'][0];stop[choice['stop_reason']]+=1
        review=by_index[row['index']]['sample_score'];assert review['sample_metadata']['question_id']==row['metadata']['question_id']
        assert review['score']['status']=='success';grade=review['score']['value']['accuracy'];assert grade in (0.,1.);correct+=int(grade)
        if choice['stop_reason']=='max_tokens':
            assert output['usage']['output_tokens']==32768
            truncated.append(dict(index=row['index'],question_id=row['metadata']['question_id'],output_tokens=output['usage']['output_tokens']))
    assert correct==102 and stop==Counter(stop=104,max_tokens=1)
    report=json.loads(files[reportname]);accuracy=[m for m in report['metrics'] if m['identity']['name']=='accuracy']
    assert len(accuracy)==1 and accuracy[0]['num']==105 and accuracy[0]['score']==round(correct/105,4)==.9714
    tool.validate_verdict(verdict);assert verdict['score']==.9714 and verdict['metric_count']==verdict['predictions']==verdict['unique_questions']==105 and not verdict['failed_predictions']
    inventory=json.loads(files['raw/out/prediction-inventory.json'])
    assert inventory['count']==inventory['unique_question_ids']==inventory['unique_indices']==105 and not inventory['failed_rows'] and not inventory['parse_errors']
    assert not {'torch','triton','numpy','vllm','evalscope'}.intersection(sys.modules)
    return dict(cpu_replay_passed=True,source_commit=SOURCE,team_base=index['team_base'],raw_files=18,raw_bytes=4325172,
        correct=102,total=105,reported_score=.9714,transport_failures=[],unique_indices=105,unique_questions=105,
        stop_reasons=dict(stop),max_tokens_outputs=truncated,all_outputs_untruncated=False,
        official_protocol_exact=True,actual_installation_on_fingerprint_and_selection=True,
        selected_host_submissions_lower_bound=4096,actual_kernel_hashes=actual['actual_kernel_hashes'],
        graph_replays_not_counted=True,owned_cleanup_complete=True,native_library_sha256=tool.NATIVE_SHA,
        source_git_hashes_verified=len(manifest['source_hashes']),tracked_symlinks_resolved=symlinks,
        records_only_not_generated_caches=True,
        old_base_stock_bf16_cache_pure_decode_diagnostic_passed=True,precision_gate=False,
        full_prefill_precision_verified=False,encoded_cache_verified=False,latest_dac_verified=False,
        throughput_measured=False,gpu_executed_by_replayer=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args();report=run()
    if args.out:args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
