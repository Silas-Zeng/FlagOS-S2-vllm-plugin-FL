"""Resolve borderline TTFT with two fresh services, one warmed4K score each.

Waits for the explicitly owned predecessor. Never interrupts an existing service.
The original IQ-then-throughput result is retained as a separate observation.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace/experiments/minicpm-s2')
BASE='3283672b92fa30f93f652b242193999836a95f88'


def idle_gpu():
    for _ in range(4):
        text=subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
        if 'No running processes found' in text:return
        time.sleep(15)
    raise RuntimeError('GPU occupied; do not overlap or stop another process')


def main(args):
    print(json.dumps(dict(waiting_for_owned_predecessor=args.wait_pid)),flush=True)
    deadline=time.monotonic()+7200
    while True:
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:break
        if time.monotonic()>deadline:raise TimeoutError('Predecessor still alive')
        time.sleep(15)
    full=json.loads((ROOT/'stream-kv-20261023/full/acceptance.json').read_text())
    assert full.get('precision_gate') and full.get('throughput_completed')
    idle_gpu()
    args.out.mkdir(parents=True,exist_ok=False)
    code=ROOT/'code/kv-probes-synced-20261023'
    baseline=ROOT/'code/stream-kv-baseline-pair-20261023'
    subprocess.run(['git','-C',str(code),'worktree','add','--detach',str(baseline),BASE],check=True)
    candidate=ROOT/'code/stream-kv-20261023'
    sys.path.insert(0,str(code/'tools'))
    import run_biv150_normalization_accuracy as runner

    def one_case(out):
        warm=[]
        for index in range(4):
            result=runner.benchmark(out,f'warm-4096-{index}',4096,128,64,920000+index)
            warm.append(result['mean_tpot_ms'])
            (out/'warmups.json').write_text(json.dumps({'4096':warm},indent=2))
            if len(warm)>=2 and warm[-2]>0 and abs(warm[-1]/warm[-2]-1)<.03:break
        else:raise RuntimeError('Warmup not stable; no scored run')
        score=runner.benchmark(out,'score-4096',4096,1024,256,0)
        result={'4096':score};(out/'scores.json').write_text(json.dumps(result,indent=2))
        return result

    runner.throughput=one_case
    result=dict(reason='TTFT near1% working-reference boundary; original trial followed IQ',
                order=['candidate','baseline'],baseline_ref=BASE,sources={},arms={})
    # Reverse the earlier baseline-before-candidate chronology; each arm still
    # gets its own new process, identical cache settings, warmup and seed.
    for name,source in [('candidate',candidate),('baseline',baseline)]:
        idle_gpu()
        result['sources'][name]=dict(path=str(source),commit=subprocess.check_output(
            ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
            gqa_sha256=hashlib.sha256((source/'vllm_fl/patches/biv150_gqa_decode.py').read_bytes()).hexdigest())
        runner.main(argparse.Namespace(source=source,out=args.out/name,mode='throughput'))
        arm=json.loads((args.out/name/'scores.json').read_text())['4096']
        states=[]
        marker='BIV150_GQA_DECODE_INSTALL '
        for line in (args.out/name/'server.log').read_text().splitlines():
            if marker in line:states.append(ast.literal_eval(line.split(marker,1)[1]))
        assert states and states[-1].get('installed')
        if name=='candidate':assert states[-1].get('kv_load_policies')=={'batch_le_8':'.cg','batch_gt_8':'.cv'}
        result['arms'][name]=dict(score=arm,gqa_installation=states[-1])
        (args.out/'pair.json').write_text(json.dumps(result,indent=2))
    b=result['arms']['baseline']['score'];c=result['arms']['candidate']['score']
    result['throughput_ratio']=c['total_token_throughput']/b['total_token_throughput']
    result['ttft_ratio']=c['mean_ttft_ms']/b['mean_ttft_ms']
    result['within_paired_ttft_1pct']=result['ttft_ratio']<=1.01
    result['within_initial_working_ttft_1pct']=c['mean_ttft_ms']<=4723.09*1.01
    result['completed']=True
    (args.out/'pair.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['arms','sources']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,required=True);parser.add_argument('--out',type=Path,required=True);main(parser.parse_args())
