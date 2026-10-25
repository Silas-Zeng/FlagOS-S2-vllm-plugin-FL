"""Throughput diagnostic after the unchanged first accuracy run.

Three predetermined warmup rounds on both sides; retain every round. Scored
commands are delegated unchanged to the frozen acceptance driver. No accuracy
resampling or credit from the throughput-only control.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


def idle():
    deadline=time.monotonic()+120
    while True:
        with socket.socket() as port:free=port.connect_ex(('127.0.0.1',9031))!=0
        gpu='No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if free and gpu:return
        if time.monotonic()>deadline:raise RuntimeError('GPU/port occupied; never interrupt unowned work')
        time.sleep(5)


def source_identity(args):
    root=args.source.resolve()
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    assert actual==args.commit
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip()
    reference=json.loads((args.accuracy_run/'manifest.json').read_text())
    accuracy=json.loads((args.accuracy_run/'accuracy-verdict.json').read_text())
    assert reference['git_commit']==actual and Path(reference['source']).resolve()==root
    assert accuracy['predictions']==accuracy['unique_questions']==accuracy['metric_count']==105
    assert not accuracy['failed_predictions']
    assert reference['env']['VLLM_FL_BIV150_LIVE_CARRY_GATE']=='1'
    library=root/'build-native-w8a8/biv150_native_w8a8.so'
    assert hashlib.sha256(library.read_bytes()).hexdigest()==reference['native_library_sha256']
    for path,digest in reference['source_hashes'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
    return root,reference,accuracy


def run_side(args):
    root,reference,accuracy=source_identity(args);idle()
    file=root/'tools/run_biv150_gate_team_acceptance.py'
    spec=importlib.util.spec_from_file_location('frozen_acceptance',file)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    protocol=dict(diagnostic_throughput_only=True,accuracy_resampled=False,
        accuracy_evaluated_here=False,linked_first_accuracy=accuracy,
        linked_first_accuracy_file=str(args.accuracy_run/'accuracy-verdict.json'),
        side=args.side,runtime_commit=args.commit,
        frozen_driver_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        diagnostic_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        warmup_rounds=3,warmup_output_len=128,
        warmup_requests={'4096':128,'16384':64},timing_rounds_discarded=0,
        warmup_timing_used_to_choose_scored_start=False,
        scored_command_unchanged=True,scored_rounds_per_length=1)
    with args.out.with_suffix('.protocol.json').open('x') as stream:json.dump(protocol,stream,indent=2)
    def throughput(folder):
        scores={};warmups={}
        for length,count in ((4096,256),(16384,128)):
            warm_count=128 if length==4096 else 64;history=[]
            for index in range(3):
                value=runner.benchmark(folder,f'warm-{length}-{index}',length,128,warm_count,920000+index)
                assert value['completed']==warm_count and value['failed']==0
                assert value['total_input_tokens']==length*warm_count
                assert value['total_output_tokens']==128*warm_count
                history.append(value['mean_tpot_ms']);warmups[str(length)]=history
                (folder/'warmups.json').write_text(json.dumps(warmups,indent=2))
            scores[str(length)]=runner.benchmark(folder,f'score-{length}',length,1024,count,0)
            (folder/'scores.json').write_text(json.dumps(scores,indent=2))
        return scores
    runner.throughput=throughput
    runner.main(SimpleNamespace(source=root,out=args.out,mode='throughput',live_carry_control=args.side=='control'))


def main(args):
    if args.side!='both':return run_side(args)
    root,reference,accuracy=source_identity(args);idle();args.out.mkdir(parents=True,exist_ok=False)
    state=dict(driver_pid=os.getpid(),runtime_commit=args.commit,source=str(root),
        first_accuracy=accuracy,precision_gate=accuracy['score']>=.95,
        control_accuracy_evaluated=False,scored_rounds_per_side_per_length=1,
        fixed_warmup_rounds=3,stable_gain_proven=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for side in ('candidate','control'):
        idle();command=[sys.executable,'-u',str(Path(__file__).resolve()),'--source',str(root),
            '--commit',args.commit,'--accuracy-run',str(args.accuracy_run),'--out',str(args.out/side),'--side',side]
        with (args.out/(side+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=root,env=dict(os.environ,PYTHONPATH=str(root),PYTHONUNBUFFERED='1'),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=side,pid=process.pid,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(') ',1)[1].split()[19],command=command)
            (args.out/'state.json').write_text(json.dumps(state,indent=2));code=process.wait()
        state['exit_code']=code;(args.out/'state.json').write_text(json.dumps(state,indent=2))
        if code:raise SystemExit(code)
        state.pop('exit_code')
    candidate=json.loads((args.out/'candidate/manifest.json').read_text())
    control=json.loads((args.out/'control/manifest.json').read_text())
    for item in (candidate,control):
        assert item['git_commit']==args.commit and item['command']==reference['command']
        assert item['native_library_sha256']==reference['native_library_sha256']
        assert item['source_hashes']==reference['source_hashes']
    allowed={'VLLM_FL_BIV150_LIVE_CARRY_GATE','TORCHINDUCTOR_CACHE_DIR'}
    assert {k:v for k,v in candidate['env'].items() if k not in allowed}=={k:v for k,v in control['env'].items() if k not in allowed}
    assert candidate['env']['VLLM_FL_BIV150_LIVE_CARRY_GATE']=='1'
    assert control['env']['VLLM_FL_BIV150_LIVE_CARRY_GATE']=='0'
    state['matched_source_library_cli_other_flags_and_warmup']=True
    a=json.loads((args.out/'candidate/scores.json').read_text());b=json.loads((args.out/'control/scores.json').read_text())
    state['comparison']={length:dict(candidate_toks=a[length]['total_token_throughput'],control_toks=b[length]['total_token_throughput'],
        relative_gain=a[length]['total_token_throughput']/b[length]['total_token_throughput']-1,
        candidate_ttft_ms=a[length]['mean_ttft_ms'],control_ttft_ms=b[length]['mean_ttft_ms']) for length in a}
    state.update(stage='finished',completed=True);(args.out/'state.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--accuracy-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--side',choices=('both','candidate','control'),default='both')
    main(p.parse_args())
