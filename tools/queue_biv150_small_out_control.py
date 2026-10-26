"""Run the exact PR58 acceptance command with only out lowering disabled.

CPU-only while the recorded candidate owners live. Never terminates another job.
The control repeats full105 and the same workload warmups, not only scoring.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]), state=fields[0])
    except FileNotFoundError:
        return None


def same_live(expected):
    current = identity(expected["pid"])
    return bool(current and current["start_ticks"] == expected["start_ticks"]
                and current["state"] != "Z")


def control_command(queue, candidate, out):
    command = list(queue["command"])
    assert queue["source_commit"] == candidate["commit"]
    for flag in ("--run-accuracy", "--run-throughput", "--full-workload-warmup",
                 "--post-score-semantic-check"):
        assert flag in command, flag
    assert command[command.index("--out-lowering") + 1] == "1"
    assert command[command.index("--native-mode") + 1] == "on"
    assert command[command.index("--stable-small-o") + 1] == "1"
    assert command[command.index("--commit") + 1] == candidate["commit"]
    command[command.index("--out-lowering") + 1] = "0"
    if '--large-linear' in command:
        assert command[command.index('--large-linear')+1] in ('0','1')
        command[command.index('--large-linear')+1]='0'
    command[command.index("--out") + 1] = str(out)
    return command


def main(args):
    # Exclusive state creation prevents accidentally dispatching this queue twice.
    with args.state.open("x") as stream:
        json.dump(dict(phase="created", owner=identity(os.getpid())), stream)
    state = dict(owner=identity(os.getpid()), tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save(phase, **values):
        state.update(phase=phase, updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.state)

    try:
        upstream = [dict(pid=pid, start_ticks=ticks) for pid, ticks in args.upstream]
        assert upstream and all(same_live(owner) for owner in upstream), "Upstream identity is not live at queue creation"
        assert not args.out.exists() and not args.log.exists()
        save("waiting_candidate", upstream=upstream)
        while any(same_live(owner) for owner in upstream):
            time.sleep(5)
        queue = json.loads(args.candidate_queue.read_text())
        candidate = json.loads((args.candidate_out / "state.json").read_text())
        assert queue["phase"] in ("completed","direct_launch"), queue
        if queue['phase']=='direct_launch':
            assert any((owner['pid'],owner['start_ticks']) ==
                       (queue['child']['pid'],queue['child']['start_ticks'])
                       for owner in upstream),'Direct launch must wait its exact driver identity'
        assert candidate["complete"] and candidate["cleanup_complete"], candidate
        assert candidate["precision_passed"] and candidate["semantic_passed"]
        assert candidate["post_score_semantic_passed"]
        assert set(candidate["scores"]) == {"4096", "16384"}
        assert all(candidate["working_gates"][length]["passed"] for length in ("4096", "16384"))
        command = control_command(queue, candidate, args.out)
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1", 9031)) != 0, "Port occupied; leave other work untouched"
        assert "No running processes found" in subprocess.check_output(["/usr/local/corex/bin/ixsmi"], text=True), "GPU occupied; leave other work untouched"
        save("starting_control", command=command, candidate_commit=candidate["commit"])
        with args.log.open("x") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        save("control", child=identity(child.pid))
        code = child.wait()
        save("control_finished", returncode=code)
        assert code == 0, f"Control failed; inspect {args.log}"
        result = json.loads((args.out / "state.json").read_text())
        assert result["complete"] and result["cleanup_complete"]
        save("completed", control_result=result)
    except BaseException as error:
        save("failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", type=Path, required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--upstream", type=int, nargs=2, action="append", required=True, metavar=("PID", "START_TICKS"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    main(parser.parse_args())
