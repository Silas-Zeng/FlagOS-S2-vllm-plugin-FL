"""Explore throughput of the known 99/105 candidate; never report acceptance."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path


RUNTIME="8a1fc9bb05689b4b781ba8b065416bdfb459bcb5"
ORIGINAL=Path("/workspace/experiments/minicpm-s2/residual-cache-service-v2-20261024")
PR31=Path("/workspace/experiments/minicpm-s2/ragged-gqa-service-20261024")


def main(args):
    source=args.source.resolve()
    assert not subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip()
    assert not subprocess.check_output(["git","-C",str(source),"diff",RUNTIME,"--","csrc","vllm_fl"],text=True)
    with socket.socket() as port:
        assert port.connect_ex(("127.0.0.1",9031))!=0
    assert "No running processes found" in subprocess.check_output(["/usr/local/corex/bin/ixsmi"],text=True)
    known=json.loads((ORIGINAL/"full/accuracy-verdict.json").read_text())
    assert known["score"]==.9429 and known["predictions"]==known["unique_questions"]==105
    assert not known["failed_predictions"]
    manifest=json.loads((ORIGINAL/"full/manifest.json").read_text())
    assert manifest["git_commit"]==RUNTIME
    for name,sha in manifest["source_hashes"].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
    baseline_state=json.loads((PR31/"state.json").read_text())
    assert baseline_state.get("completed") is True
    assert baseline_state["commit"]=="e972997522a8623178d932e5ec79e30e6724d69c"
    baseline=json.loads((PR31/"full/scores.json").read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    original_source=Path(manifest["source"])
    library=original_source/"build-native-w8a8/biv150_native_w8a8.so"
    assert hashlib.sha256(library.read_bytes()).hexdigest()==manifest["native_library_sha256"]
    target=source/"build-native-w8a8/biv150_native_w8a8.so"
    target.parent.mkdir(exist_ok=False)
    shutil.copyfile(library,target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==manifest["native_library_sha256"]
    # Reuse the exact previously tested private SO; no build, install, shared
    # source mutation or repeat of unchanged operator tests is needed.
    metadata=dict(diagnostic_only=True,eligible_for_merge=False,precision_gate=False,
        known_accuracy=known,original_runtime=RUNTIME,source=str(source),
        commit=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip(),
        native_library_sha256=manifest["native_library_sha256"],
        source_hashes=manifest["source_hashes"],baseline_commit=baseline_state["commit"],
        baseline_comparison="Separate warmed single rounds; not a paired significance test",
        protocol="fixed service/benchmark, TPOT-stable warmup, one full round per length")
    (args.out/"known-accuracy.json").write_text(json.dumps(known,indent=2))
    (args.out/"diagnostic.json").write_text(json.dumps(metadata,indent=2))
    command=[sys.executable,"-u",str(source/"tools/run_biv150_gate_team_acceptance.py"),
             "--source",str(source),"--out",str(args.out/"full"),"--mode","throughput"]
    with (args.out/"throughput.log").open("x") as log:
        process=subprocess.Popen(command,cwd=source,env=dict(os.environ,PYTHONPATH=str(source)),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state=dict(driver_pid=os.getpid(),pid=process.pid,command=command,
                   diagnostic_only=True,eligible_for_merge=False,precision_gate=False,
                   completed=False,commit=metadata["commit"],stage="throughput")
        (args.out/"state.json").write_text(json.dumps(state,indent=2))
        code=process.wait()
    state["exit_code"]=code
    if code:
        (args.out/"state.json").write_text(json.dumps(state,indent=2))
        raise SystemExit(code)
    scores=json.loads((args.out/"full/scores.json").read_text())
    spec=importlib.util.spec_from_file_location("followup_acceptance",source/"tools/run_biv150_gate_team_acceptance.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    metadata.update(scores=scores,performance_only_gates=module.performance_gates(scores),
        reference_delta={length:scores[length]["total_token_throughput"]/baseline[length]["total_token_throughput"]-1
                         for length in ("4096","16384")})
    server=(args.out/"full/server.log").read_text(errors="replace")
    metadata["scope_rejections"]=[line for line in server.splitlines() if "BIV150_COMPUTE_ORDER_SCOPE_REJECTED" in line]
    metadata["q_fragment_fallback"]=[line for line in server.splitlines() if "BIV150_Q_FRAGMENT_FALLBACK" in line]
    (args.out/"diagnostic.json").write_text(json.dumps(metadata,indent=2))
    state.update(diagnostic_completed=True,throughput_completed=True,stage="finished")
    (args.out/"state.json").write_text(json.dumps(state,indent=2))
    print(json.dumps(dict(diagnostic_only=True,eligible_for_merge=False,
                         reference_delta=metadata["reference_delta"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args())
