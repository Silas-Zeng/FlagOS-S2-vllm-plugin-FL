"""Owned serial packing checks, allocation evidence and unchanged service gates."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from probe_biv150_memory_ownership import require_idle, sha


def main(args):
    source = Path(__file__).resolve().parents[1]
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    require_idle()
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), source=str(source),
        commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        library=str(args.library), library_sha256=sha(args.library), completed=False)
    stages = [
        ("packing-equivalence", [sys.executable, str(source / "tools/check_biv150_weight_packing.py"),
             "--weights", str(args.weights), "--baseline-source", str(args.baseline_source),
             "--out", str(args.out / "packing-equivalence.json")]),
        ("memory-ownership", [sys.executable, str(source / "tools/probe_biv150_memory_ownership.py"),
             "--weights", str(args.weights), "--library", str(args.library),
             "--out", str(args.out / "memory")]),
        ("service", [sys.executable, str(source / "tools/run_biv150_gate_team_queue.py"),
             "--source", str(source), "--out", str(args.out / "service")]),
    ]
    for stage, command in stages:
        # Wait only for cleanup of the child that just exited; never stop others.
        deadline = time.monotonic() + 30
        while True:
            try:
                require_idle()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(2)
        with (args.out / f"{stage}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=source, env=dict(os.environ, PYTHONPATH=str(source)),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state.update(stage=stage, child_pid=process.pid, command=command,
                         started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (args.out / "state.json").write_text(json.dumps(state, indent=2))
            code = process.wait()
        state.update(exit_code=code)
        (args.out / "state.json").write_text(json.dumps(state, indent=2))
        if code:
            raise SystemExit(code)
    state.update(completed=True)
    (args.out / "state.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
