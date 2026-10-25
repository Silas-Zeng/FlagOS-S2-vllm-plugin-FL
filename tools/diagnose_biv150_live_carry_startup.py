"""Owned startup-only diagnostic, unchanged service CLI, zero scored requests."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

def main(args):
    with socket.socket() as p:assert p.connect_ex(('127.0.0.1',9031))!=0
    assert 'No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(args.manifest.read_text());source=Path(manifest['source'])
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('VLLM_FL_BIV150_','BIV150_NATIVE_')):env.pop(k)
    env.update(manifest['env']);env.update(
        PYTHONPATH=str(Path(__file__).resolve().parent)+':'+str(source),
        VLLM_FL_BIV150_LIVE_CARRY_GATE=str(args.flag),BIV150_PICKLE_DIAGNOSTIC='1',
        TORCHINDUCTOR_CACHE_DIR=str(args.out/'inductor-cache'),TORCH_LOGS='graph_code')
    report=dict(diagnostic_only=True,accuracy_evaluated=False,throughput_evaluated=False,
        source=str(source),source_commit=manifest['git_commit'],live_carry_flag=args.flag,
        command=manifest['command'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (args.out/'server.log').open('x') as log:
        child=subprocess.Popen(manifest['command'],cwd=source,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        report['pid']=child.pid;report['start_ticks']=Path(f'/proc/{child.pid}/stat').read_text().split()[21]
        (args.out/'state.json').write_text(json.dumps(report,indent=2))
        try:
            deadline=time.monotonic()+360
            while time.monotonic()<deadline and child.poll() is None:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:9031/v1/models',timeout=2) as response:
                        if response.status==200:report['ready']=True;break
                except Exception:pass
                time.sleep(2)
            report.setdefault('ready',False);report['server_exit_before_cleanup']=child.poll()
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            report['exit_after_cleanup']=child.poll()
            (args.out/'state.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--flag',type=int,choices=[0,1],required=True)
    main(p.parse_args())
