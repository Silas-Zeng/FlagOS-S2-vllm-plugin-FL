"""Owned component driver: clean frozen source, CPU checks, then GPU probe."""

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def main(args):
    source=args.source.resolve()
    assert not subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip()
    with socket.socket() as port:
        assert port.connect_ex(("127.0.0.1",9031))!=0
    assert "No running processes found" in subprocess.check_output(["/usr/local/corex/bin/ixsmi"],text=True)
    args.out.mkdir(parents=True,exist_ok=False)
    state=dict(queue_pid=os.getpid(),source=str(source),component_only=True,
        commit=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip())
    steps=[("cpu-check",[sys.executable,str(source/"tools/check_biv150_bf16_splitk_cpu.py"),
                         "--out",str(args.out/"cpu-check.json")]),
           ("out-baseline-audit",[sys.executable,"-u",str(source/"tools/bench_biv150_bf16_real_oracle.py"),
                          "--baseline","out","--out",str(args.out/"out-baseline-audit")]),
           ("bf16-full42",[sys.executable,"-u",str(source/"tools/bench_biv150_bf16_real_oracle.py"),
                          "--baseline","out","--all-layers","--out",str(args.out/"full42")])]
    for stage,command in steps:
        if stage!='cpu-check':
            deadline=time.monotonic()+30
            while True:
                with socket.socket() as port:port_free=port.connect_ex(('127.0.0.1',9031))!=0
                gpu_free='No running processes found' in subprocess.check_output(['/usr/local/corex/bin/ixsmi'],text=True)
                if port_free and gpu_free:break
                if time.monotonic()>=deadline:raise RuntimeError('GPU/port still busy; no component started')
                time.sleep(2)
        with (args.out/f"{stage}.log").open("x") as log:
            process=subprocess.Popen(command,cwd=source,env=dict(os.environ,PYTHONPATH=str(source)),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=stage,pid=process.pid,command=command,
                         started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (args.out/"state.json").write_text(json.dumps(state,indent=2))
            code=process.wait()
        state["exit_code"]=code
        (args.out/"state.json").write_text(json.dumps(state,indent=2))
        if code: raise SystemExit(code)
        state.pop("exit_code")
        if stage=="out-baseline-audit":
            audit=json.loads((args.out/"out-baseline-audit/result.json").read_text())
            rows=audit['records']
            candidates=[dict(layer=r['layer'],geometry=r['geometry'],m=r['m'],config=r['config'],
                             min_round_speedup=r['min_round_speedup']) for r in rows
                if r['candidate_cpu_all_passed'] and r.get('graph_cpu_passed')
                and r.get('timing_candidate_cpu',{}).get('passed') and r.get('min_round_speedup',0)>1]
            state['out_baseline_positive_records']=candidates
            (args.out/"state.json").write_text(json.dumps(state,indent=2))
            if not candidates:
                state.update(completed=True,full42_skipped="no complete numeric and consistent timing gain against actual out baseline")
                (args.out/"state.json").write_text(json.dumps(state,indent=2))
                return
    state["completed"]=True
    (args.out/"state.json").write_text(json.dumps(state,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
