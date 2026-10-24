"""Owned component driver: clean frozen source, CPU checks, then GPU probe."""

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
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
           ("bf16-probe",[sys.executable,"-u",str(source/"tools/bench_biv150_bf16_small_splitk.py"),
                          "--out",str(args.out/"probe")])]
    for stage,command in steps:
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
    state["completed"]=True
    (args.out/"state.json").write_text(json.dumps(state,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
